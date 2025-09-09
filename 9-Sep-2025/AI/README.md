# Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference 

**Title (ZH)**: 直接对齐完整的扩散轨迹与精细粒度的人类偏好 

**Authors**: Xiangwei Shen, Zhimin Li, Zhantao Yang, Shiyi Zhang, Yingfang Zhang, Donghao Li, Chunyu Wang, Qinglin Lu, Yansong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06942)  

**Abstract**: Recent studies have demonstrated the effectiveness of directly aligning diffusion models with human preferences using differentiable reward. However, they exhibit two primary challenges: (1) they rely on multistep denoising with gradient computation for reward scoring, which is computationally expensive, thus restricting optimization to only a few diffusion steps; (2) they often need continuous offline adaptation of reward models in order to achieve desired aesthetic quality, such as photorealism or precise lighting effects. To address the limitation of multistep denoising, we propose Direct-Align, a method that predefines a noise prior to effectively recover original images from any time steps via interpolation, leveraging the equation that diffusion states are interpolations between noise and target images, which effectively avoids over-optimization in late timesteps. Furthermore, we introduce Semantic Relative Preference Optimization (SRPO), in which rewards are formulated as text-conditioned signals. This approach enables online adjustment of rewards in response to positive and negative prompt augmentation, thereby reducing the reliance on offline reward fine-tuning. By fine-tuning the this http URL model with optimized denoising and online reward adjustment, we improve its human-evaluated realism and aesthetic quality by over 3x. 

**Abstract (ZH)**: 直接对齐：通过可微奖励直接将扩散模型与人类偏好对齐 

---
# Paper2Agent: Reimagining Research Papers As Interactive and Reliable AI Agents 

**Title (ZH)**: Paper2Agent: 重塑科研论文为互动可靠的人工智能代理 

**Authors**: Jiacheng Miao, Joe R. Davis, Jonathan K. Pritchard, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2509.06917)  

**Abstract**: We introduce Paper2Agent, an automated framework that converts research papers into AI agents. Paper2Agent transforms research output from passive artifacts into active systems that can accelerate downstream use, adoption, and discovery. Conventional research papers require readers to invest substantial effort to understand and adapt a paper's code, data, and methods to their own work, creating barriers to dissemination and reuse. Paper2Agent addresses this challenge by automatically converting a paper into an AI agent that acts as a knowledgeable research assistant. It systematically analyzes the paper and the associated codebase using multiple agents to construct a Model Context Protocol (MCP) server, then iteratively generates and runs tests to refine and robustify the resulting MCP. These paper MCPs can then be flexibly connected to a chat agent (e.g. Claude Code) to carry out complex scientific queries through natural language while invoking tools and workflows from the original paper. We demonstrate Paper2Agent's effectiveness in creating reliable and capable paper agents through in-depth case studies. Paper2Agent created an agent that leverages AlphaGenome to interpret genomic variants and agents based on ScanPy and TISSUE to carry out single-cell and spatial transcriptomics analyses. We validate that these paper agents can reproduce the original paper's results and can correctly carry out novel user queries. By turning static papers into dynamic, interactive AI agents, Paper2Agent introduces a new paradigm for knowledge dissemination and a foundation for the collaborative ecosystem of AI co-scientists. 

**Abstract (ZH)**: Paper2Agent：一种将研究论文转换为AI代理的自动化框架 

---
# Test-Time Scaling in Reasoning Models Is Not Effective for Knowledge-Intensive Tasks Yet 

**Title (ZH)**: 测试时缩放在推理模型中尚未证明对知识密集型任务有效 

**Authors**: James Xu Zhao, Bryan Hooi, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2509.06861)  

**Abstract**: Test-time scaling increases inference-time computation by allowing models to generate long reasoning chains, and has shown strong performance across many domains. However, in this work, we show that this approach is not yet effective for knowledge-intensive tasks, where high factual accuracy and low hallucination rates are essential. We conduct a comprehensive evaluation of test-time scaling using 12 reasoning models on two knowledge-intensive benchmarks. Our results reveal that increasing test-time computation does not consistently improve accuracy and, in many cases, it even leads to more hallucinations. We then analyze how extended reasoning affects hallucination behavior. We find that reduced hallucinations often result from the model choosing to abstain after thinking more, rather than from improved factual recall. Conversely, for some models, longer reasoning encourages attempts on previously unanswered questions, many of which result in hallucinations. Case studies show that extended reasoning can induce confirmation bias, leading to overconfident hallucinations. Despite these limitations, we observe that compared to non-thinking, enabling thinking remains beneficial. Code and data are available at this https URL 

**Abstract (ZH)**: 测试时扩展会增加推理时的计算量，允许模型生成长的推理链，并在许多领域显示出强大的性能。然而，在这项工作中，我们表明，在高事实准确性和低幻觉率至关重要的知识密集型任务中，这一方法尚未有效。我们使用12个推理模型在两个知识密集型基准上全面评估了测试时扩展。我们的结果表明，增加测试时的计算并不一致地提高准确性，在许多情况下甚至会导致更多的幻觉。然后我们分析了扩展推理如何影响幻觉行为。我们发现，幻觉减少通常是因为模型在思考更多后选择避免给出答案，而不是因为事实检索的改进。相反，对于一些模型，较长的推理会鼓励尝试更多之前未回答的问题，许多问题最终导致幻觉。案例研究表明，扩展推理可能导致确认偏见，从而导致过度自信的幻觉。尽管存在这些局限性，我们观察到，与不进行思考相比，使模型能够思考仍然有益。代码和数据可在以下链接获得。 

---
# RAFFLES: Reasoning-based Attribution of Faults for LLM Systems 

**Title (ZH)**: RAFFLES: 基于推理的LLM系统故障归因 

**Authors**: Chenyang Zhu, Spencer Hong, Jingyu Wu, Kushal Chawla, Charlotte Tang, Youbing Yin, Nathan Wolfe, Erin Babinsky, Daben Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06822)  

**Abstract**: We have reached a critical roadblock in the development and enhancement of long-horizon, multi-component LLM agentic systems: it is incredibly tricky to identify where these systems break down and why. Evaluation capabilities that currently exist today (e.g., single pass LLM-as-a-judge) are limited in that they often focus on individual metrics or capabilities, end-to-end outcomes, and are narrowly grounded on the preferences of humans. We argue that to match the agentic capabilities, evaluation frameworks must also be able to reason, probe, iterate, and understand the complex logic passing through these systems over long horizons. In this paper, we present RAFFLES - an evaluation architecture that incorporates reasoning and iterative refinement. Specifically, RAFFLES operates as an iterative, multi-component pipeline, using a central Judge to systematically investigate faults and a set of specialized Evaluators to assess not only the system's components but also the quality of the reasoning by the Judge itself, thereby building a history of hypotheses. We tested RAFFLES against several baselines on the Who&When dataset, a benchmark designed to diagnose the "who" (agent) and "when" (step) of a system's failure. RAFFLES outperforms these baselines, achieving an agent-step fault pair accuracy of over 43% on the Algorithmically-Generated dataset (a substantial increase from the previously published best of 16.6%) and over 20% on the Hand-Crafted dataset (surpassing the previously published best of 8.8%). These results demonstrate a key step towards introducing automated fault detection for autonomous systems over labor-intensive manual human review. 

**Abstract (ZH)**: 我们在长时 horizon、多组件大语言模型代理系统的发展与提升中遇到了一个关键障碍：很难确定这些系统在何处出现故障及其原因。目前存在的评估能力（例如，单次过的大语言模型作为法官）在聚焦单一指标或能力、端到端结果且紧密基于人类偏好方面受到限制。我们提出，要匹配代理能力，评估框架还必须能够推理、探索、迭代并理解这些系统在长时 horizon 中传递的复杂逻辑。在本文中，我们提出了 RAFFLES——一种结合推理与迭代完善的评估架构。具体而言，RAFFLES 作为迭代的多组件管道运行，采用一个中央法官系统地调查故障，并通过一组专门的评估器评估法官本身及其推理的质量，从而建立假说的历史。我们在 Who&When 数据集上对 RAFFLES 进行了与多个基线的测试，该数据集用于诊断系统的“谁”（代理）和“何时”（步骤）。RAFFLES 在算法生成数据集上达到超过 43% 的代理-步骤故障配对准确性（比之前出版的最佳结果 16.6% 有显著提升），在手工打造数据集上达到超过 20% 的准确性（超过之前出版的最佳结果 8.8%）。这些结果展示了向通过劳动密集型的手动人工审查引入自动故障检测的一步重要进展。 

---
# Another Turn, Better Output? A Turn-Wise Analysis of Iterative LLM Prompting 

**Title (ZH)**: 又一轮，更优输出？基于轮次的迭代LLM提示分析 

**Authors**: Shashidhar Reddy Javaji, Bhavul Gauri, Zining Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06770)  

**Abstract**: Large language models (LLMs) are now used in multi-turn workflows, but we still lack a clear way to measure when iteration helps and when it hurts. We present an evaluation framework for iterative refinement that spans ideation, code, and math. Our protocol runs controlled 12-turn conversations per task, utilizing a variety of prompts ranging from vague ``improve it'' feedback to targeted steering, and logs per-turn outputs. We score outcomes with domain-appropriate checks (unit tests for code; answer-equivalence plus reasoning-soundness for math; originality and feasibility for ideation) and track turn-level behavior with three families of metrics: semantic movement across turns, turn-to-turn change, and output size growth. Across models and tasks, gains are domain-dependent: they arrive early in ideas and code, but in math late turns matter when guided by elaboration. After the first few turns, vague feedback often plateaus or reverses correctness, while targeted prompts reliably shift the intended quality axis (novelty vs. feasibility in ideation; speed vs. readability in code; in math, elaboration outperforms exploration and drives late-turn gains). We also observe consistent domain patterns: ideation moves more in meaning across turns, code tends to grow in size with little semantic change, and math starts fixed but can break that path with late, elaborative this http URL, the framework and metrics make iteration measurable and comparable across models, and signal when to steer, stop, or switch strategies. 

**Abstract (ZH)**: 大规模语言模型（LLMs）现在用于多轮工作流程中，但我们仍然缺乏明确的方法来衡量迭代是助益还是有害。我们提出了一种涵盖创意、代码和数学的迭代 refinement 评估框架。我们的协议每个任务运行受控的12轮对话，利用从模糊的“改进它”反馈到目标导向引导等多种提示，并记录每轮输出。我们使用领域特定的检查（代码中的单元测试；数学中的答案等价性加上推理稳健性；创意中的原创性和可行性）来评分，并通过三类指标追踪每轮行为：跨轮次语义变化、轮次之间变化和输出大小增长。在不同模型和任务中，收益具有领域依赖性：在创意和代码中早期出现，在数学中只有在受到详细扩展引导时，后期轮次才重要。在最初的几轮后，模糊反馈往往导致正确性的停滞或逆转，而目标导向提示则可靠地改变预期质量轴（创意中的新颖性与可行性；代码中的速度与可读性；在数学中，详细扩展优于探索并驱动后期轮次的收益）。此外，我们还观察到一致的领域模式：创意在一轮次间更具意义变化，代码通常在大小上增长但语义变化不大，而数学从一开始就保持固定，但在后期详细的扩展引导下可以打破这种路径。该框架和指标使迭代在不同模型之间可测量和可比较，并指示何时需要导向、停止或切换策略。 

---
# VehicleWorld: A Highly Integrated Multi-Device Environment for Intelligent Vehicle Interaction 

**Title (ZH)**: VehicleWorld: 一种高度集成的多设备智能车辆交互环境 

**Authors**: Jie Yang, Jiajun Chen, Zhangyue Yin, Shuo Chen, Yuxin Wang, Yiran Guo, Yuan Li, Yining Zheng, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06736)  

**Abstract**: Intelligent vehicle cockpits present unique challenges for API Agents, requiring coordination across tightly-coupled subsystems that exceed typical task environments' complexity. Traditional Function Calling (FC) approaches operate statelessly, requiring multiple exploratory calls to build environmental awareness before execution, leading to inefficiency and limited error recovery. We introduce VehicleWorld, the first comprehensive environment for the automotive domain, featuring 30 modules, 250 APIs, and 680 properties with fully executable implementations that provide real-time state information during agent execution. This environment enables precise evaluation of vehicle agent behaviors across diverse, challenging scenarios. Through systematic analysis, we discovered that direct state prediction outperforms function calling for environmental control. Building on this insight, we propose State-based Function Call (SFC), a novel approach that maintains explicit system state awareness and implements direct state transitions to achieve target conditions. Experimental results demonstrate that SFC significantly outperforms traditional FC approaches, achieving superior execution accuracy and reduced latency. We have made all implementation code publicly available on Github this https URL. 

**Abstract (ZH)**: 智能车辆仪表板为API代理带来了独特挑战，需要协调紧密耦合的子系统，其复杂性远超典型任务环境。传统函数调用（FC）方法以无状态方式操作，要求在执行前进行多次探索性调用以建立环境意识，导致效率低下且错误恢复能力有限。我们引入了VehicleWorld，这是汽车领域首个全面的环境，包含30个模块、250个API和680个属性，提供了完全可执行的实现，并在代理执行过程中提供实时状态信息。该环境使得能够在多种具有挑战性的场景中精确评估车辆代理行为。通过系统分析，我们发现直接状态预测在环境控制方面优于函数调用。基于这一见解，我们提出了基于状态的函数调用（SFC）这一新颖方法，该方法保持明确的系统状态意识，并直接实现状态转换以达到目标条件。实验结果表明，SFC在执行准确性和降低延迟方面显著优于传统FC方法。我们已将所有实现代码公开发布在GitHub上：https://github.com/XXXXX。 

---
# Reinforcement Learning Foundations for Deep Research Systems: A Survey 

**Title (ZH)**: 深度研究系统 reinforcement 学习基础：一个综述 

**Authors**: Wenjun Li, Zhi Chen, Jingru Lin, Hannan Cao, Wei Han, Sheng Liang, Zhi Zhang, Kuicai Dong, Dexun Li, Chen Zhang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06733)  

**Abstract**: Deep research systems, agentic AI that solve complex, multi-step tasks by coordinating reasoning, search across the open web and user files, and tool use, are moving toward hierarchical deployments with a Planner, Coordinator, and Executors. In practice, training entire stacks end-to-end remains impractical, so most work trains a single planner connected to core tools such as search, browsing, and code. While SFT imparts protocol fidelity, it suffers from imitation and exposure biases and underuses environment feedback. Preference alignment methods such as DPO are schema and proxy-dependent, off-policy, and weak for long-horizon credit assignment and multi-objective trade-offs. A further limitation of SFT and DPO is their reliance on human defined decision points and subskills through schema design and labeled comparisons. Reinforcement learning aligns with closed-loop, tool-interaction research by optimizing trajectory-level policies, enabling exploration, recovery behaviors, and principled credit assignment, and it reduces dependence on such human priors and rater biases.
This survey is, to our knowledge, the first dedicated to the RL foundations of deep research systems. It systematizes work after DeepSeek-R1 along three axes: (i) data synthesis and curation; (ii) RL methods for agentic research covering stability, sample efficiency, long context handling, reward and credit design, multi-objective optimization, and multimodal integration; and (iii) agentic RL training systems and frameworks. We also cover agent architecture and coordination, as well as evaluation and benchmarks, including recent QA, VQA, long-form synthesis, and domain-grounded, tool-interaction tasks. We distill recurring patterns, surface infrastructure bottlenecks, and offer practical guidance for training robust, transparent deep research agents with RL. 

**Abstract (ZH)**: 深层研究系统的RL基础：从规划者、协调者和执行者的人工智能到层次化部署的研究 

---
# CogGuide: Human-Like Guidance for Zero-Shot Omni-Modal Reasoning 

**Title (ZH)**: CogGuide: 类人类的引导在零样本多模态推理中的应用 

**Authors**: Zhou-Peng Shou, Zhi-Qiang You, Fang Wang, Hai-Bo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06641)  

**Abstract**: Targeting the issues of "shortcuts" and insufficient contextual understanding in complex cross-modal reasoning of multimodal large models, this paper proposes a zero-shot multimodal reasoning component guided by human-like cognitive strategies centered on an "intent sketch". The component comprises a plug-and-play three-module pipeline-Intent Perceiver, Strategy Generator, and Strategy Selector-that explicitly constructs a "understand-plan-select" cognitive process. By generating and filtering "intent sketch" strategies to guide the final reasoning, it requires no parameter fine-tuning and achieves cross-model transfer solely through in-context engineering. Information-theoretic analysis shows that this process can reduce conditional entropy and improve information utilization efficiency, thereby suppressing unintended shortcut reasoning. Experiments on IntentBench, WorldSense, and Daily-Omni validate the method's generality and robust gains; compared with their respective baselines, the complete "three-module" scheme yields consistent improvements across different reasoning engines and pipeline combinations, with gains up to approximately 9.51 percentage points, demonstrating the practical value and portability of the "intent sketch" reasoning component in zero-shot scenarios. 

**Abstract (ZH)**: 针对多模态大型模型在复杂跨模态推理中“捷径”问题和不足的上下文理解能力，本文提出了一种以“意图素描”为中心的人类认知策略指导的零样本多模态推理组件。该组件包含可插拔的三模块管道——意图感知器、策略生成器和策略选择器，明确构建了一个“理解-规划-选择”的认知过程。通过生成和过滤“意图素描”策略来指导最终推理，无需参数微调，仅通过上下文工程实现跨模型迁移。信息论分析表明，该过程可以降低条件熵并提高信息利用效率，从而抑制无意中的捷径推理。在IntentBench、WorldSense和Daily-Omni上的实验验证了该方法的普适性和稳健性增益；与各自的基线相比，完整的“三模块”方案在不同推理引擎和管道组合中一致地表现出改进，增幅高达约9.51个百分点，展示了“意图素描”推理组件在零样本场景中的实用价值和可移植性。 

---
# An AI system to help scientists write expert-level empirical software 

**Title (ZH)**: 一种辅助科学家编写高水平 empirical 软件的AI系统 

**Authors**: Eser Aygün, Anastasiya Belyaeva, Gheorghe Comanici, Marc Coram, Hao Cui, Jake Garrison, Renee Johnston Anton Kast, Cory Y. McLean, Peter Norgaard, Zahra Shamsi, David Smalling, James Thompson, Subhashini Venugopalan, Brian P. Williams, Chujun He, Sarah Martinson, Martyna Plomecka, Lai Wei, Yuchen Zhou, Qian-Ze Zhu, Matthew Abraham, Erica Brand, Anna Bulanova, Jeffrey A. Cardille, Chris Co, Scott Ellsworth, Grace Joseph, Malcolm Kane, Ryan Krueger, Johan Kartiwa, Dan Liebling, Jan-Matthis Lueckmann, Paul Raccuglia, Xuefei, Wang, Katherine Chou, James Manyika, Yossi Matias, John C. Platt, Lizzie Dorfman, Shibl Mourad, Michael P. Brenner  

**Link**: [PDF](https://arxiv.org/pdf/2509.06503)  

**Abstract**: The cycle of scientific discovery is frequently bottlenecked by the slow, manual creation of software to support computational experiments. To address this, we present an AI system that creates expert-level scientific software whose goal is to maximize a quality metric. The system uses a Large Language Model (LLM) and Tree Search (TS) to systematically improve the quality metric and intelligently navigate the large space of possible solutions. The system achieves expert-level results when it explores and integrates complex research ideas from external sources. The effectiveness of tree search is demonstrated across a wide range of benchmarks. In bioinformatics, it discovered 40 novel methods for single-cell data analysis that outperformed the top human-developed methods on a public leaderboard. In epidemiology, it generated 14 models that outperformed the CDC ensemble and all other individual models for forecasting COVID-19 hospitalizations. Our method also produced state-of-the-art software for geospatial analysis, neural activity prediction in zebrafish, time series forecasting and numerical solution of integrals. By devising and implementing novel solutions to diverse tasks, the system represents a significant step towards accelerating scientific progress. 

**Abstract (ZH)**: 科学发现的周期常因支持计算实验的缓慢手动软件创建而受阻。为此，我们提出一个AI系统，该系统能创建专家级的科学软件，旨在最大化质量指标。该系统利用大型语言模型（LLM）和树搜索（TS）系统性地提高质量指标，并智能地导航可能解决方案的广阔空间。当系统探索并整合来自外部的复杂研究思想时，能够达到专家级结果。树搜索的有效性在一系列基准测试中得到验证。在生物信息学领域，系统发现40种新型单细胞数据分析方法，在公共排行榜上优于顶级的人工开发方法。在流行病学领域，系统生成了14个模型，在预测COVID-19住院人数方面优于CDC集成和其他个体模型。我们的方法还为地理空间分析、斑马鱼神经活动预测、时间序列预测和积分的数值解提供了最先进的软件。通过为多种任务设计并实施新颖的解决方案，该系统代表了加速科学进步的重要一步。 

---
# Scaling up Multi-Turn Off-Policy RL and Multi-Agent Tree Search for LLM Step-Provers 

**Title (ZH)**: 扩展多轮离策RL和多智能体树搜索在语言模型步骤证明中的应用 

**Authors**: Ran Xin, Zeyu Zheng, Yanchen Nie, Kun Yuan, Xia Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.06493)  

**Abstract**: The integration of Large Language Models (LLMs) into automated theorem proving has shown immense promise, yet is fundamentally constrained by challenges in scaling up both training-time reinforcement learning (RL) and inference-time compute. This paper introduces \texttt{BFS-Prover-V2}, a system designed to address this dual scaling problem. We present two primary innovations. The first is a novel multi-turn off-policy RL framework for continually improving the performance of LLM step-prover at training time. This framework, inspired by the principles of AlphaZero, utilizes a multi-stage expert iteration pipeline featuring adaptive tactic-level data filtering and periodic retraining to surmount the performance plateaus that typically curtail long-term RL in LLM-based agents. The second innovation is a planner-enhanced multi-agent search architecture that scales reasoning capabilities at inference time. This architecture employs a general reasoning model as a high-level planner to iteratively decompose complex theorems into a sequence of simpler subgoals. This hierarchical approach substantially reduces the search space, enabling a team of parallel prover agents to collaborate efficiently by leveraging a shared proof cache. We demonstrate that this dual approach to scaling yields state-of-the-art results on established formal mathematics benchmarks. \texttt{BFS-Prover-V2} achieves 95.08\% and 41.4\% on the MiniF2F and ProofNet test sets respectively. While demonstrated in the domain of formal mathematics, the RL and inference techniques presented in this work are of broader interest and may be applied to other domains requiring long-horizon multi-turn reasoning and complex search. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自动定理证明中的集成展现了巨大的潜力，但仍然受到扩展训练时强化学习（RL）和推理时计算能力双重挑战的限制。本文介绍了BFS-Prover-V2系统，旨在解决这一双重扩展问题。我们提出了两项主要创新。首先，提出了一种新颖的多轮次离策略RL框架，在训练时间内持续提升LLM步骤证明器的性能。该框架借鉴了AlphaZero的原理，采用多阶段专家迭代流水线，并结合自适应策略级数据过滤和定期重新训练，以克服基于LLM的代理在长期RL中通常会遇到的性能平台期。其次是增强计划者的多代理搜索架构，该架构在推理时扩展了推理能力。该架构使用一个通用推理模型作为高级计划器，通过迭代分解复杂的定理为一系列更简单的子目标。这种分层方法显著减少了搜索空间，从而使一个团队的并行证明代理能够通过共享证明缓存高效协作。我们展示了这种双重扩展方法在正式数学基准测试中取得了最先进的结果。BFS-Prover-V2分别在MiniF2F和ProofNet测试集上取得了95.08%和41.4%的性能。虽然在正式数学领域进行了演示，但本文提出的RL和推理技术具有更广泛的应用兴趣，并可能应用于其他需要远期多轮次推理和复杂搜索的领域。 

---
# MORSE: Multi-Objective Reinforcement Learning via Strategy Evolution for Supply Chain Optimization 

**Title (ZH)**: MORSE: 多目标强化学习在供应链优化中的策略进化方法 

**Authors**: Niki Kotecha, Ehecatl Antonio del Rio Chanona  

**Link**: [PDF](https://arxiv.org/pdf/2509.06490)  

**Abstract**: In supply chain management, decision-making often involves balancing multiple conflicting objectives, such as cost reduction, service level improvement, and environmental sustainability. Traditional multi-objective optimization methods, such as linear programming and evolutionary algorithms, struggle to adapt in real-time to the dynamic nature of supply chains. In this paper, we propose an approach that combines Reinforcement Learning (RL) and Multi-Objective Evolutionary Algorithms (MOEAs) to address these challenges for dynamic multi-objective optimization under uncertainty. Our method leverages MOEAs to search the parameter space of policy neural networks, generating a Pareto front of policies. This provides decision-makers with a diverse population of policies that can be dynamically switched based on the current system objectives, ensuring flexibility and adaptability in real-time decision-making. We also introduce Conditional Value-at-Risk (CVaR) to incorporate risk-sensitive decision-making, enhancing resilience in uncertain environments. We demonstrate the effectiveness of our approach through case studies, showcasing its ability to respond to supply chain dynamics and outperforming state-of-the-art methods in an inventory management case study. The proposed strategy not only improves decision-making efficiency but also offers a more robust framework for managing uncertainty and optimizing performance in supply chains. 

**Abstract (ZH)**: 在供应链管理中，决策往往需要平衡多个冲突的目标，如成本降低、服务水平提升和环境可持续性。传统的多目标优化方法，如线性规划和演化算法，难以实时适应供应链的动态性。本文提出了一种结合强化学习（RL）和多目标演化算法（MOEAs）的方法，以应对动态多目标优化过程中的不确定性挑战。该方法利用MOEAs搜索策略神经网络的参数空间，生成帕累托前沿的策略，为决策者提供多样化的策略群体，可以根据当前系统目标动态切换，确保实时决策的灵活性和适应性。此外，我们引入条件值-at-风险（CVaR）以纳入风险敏感决策，增强在不确定环境中的韧性。通过案例研究，我们证明了该方法的有效性，并在库存管理案例中表现出优于现有先进方法的能力。所提出的方法不仅提高了决策效率，还提供了一种更稳健的框架来管理不确定性并优化供应链性能。 

---
# MAS-Bench: A Unified Benchmark for Shortcut-Augmented Hybrid Mobile GUI Agents 

**Title (ZH)**: MAS-Bench: 一键增强混合移动GUI代理的统一基准 

**Authors**: Pengxiang Zhao, Guangyi Liu, Yaozhen Liang, Weiqing He, Zhengxi Lu, Yuehao Huang, Yaxuan Guo, Kexin Zhang, Hao Wang, Liang Liu, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06477)  

**Abstract**: To enhance the efficiency of GUI agents on various platforms like smartphones and computers, a hybrid paradigm that combines flexible GUI operations with efficient shortcuts (e.g., API, deep links) is emerging as a promising direction. However, a framework for systematically benchmarking these hybrid agents is still underexplored. To take the first step in bridging this gap, we introduce MAS-Bench, a benchmark that pioneers the evaluation of GUI-shortcut hybrid agents with a specific focus on the mobile domain. Beyond merely using predefined shortcuts, MAS-Bench assesses an agent's capability to autonomously generate shortcuts by discovering and creating reusable, low-cost workflows. It features 139 complex tasks across 11 real-world applications, a knowledge base of 88 predefined shortcuts (APIs, deep-links, RPA scripts), and 7 evaluation metrics. The tasks are designed to be solvable via GUI-only operations, but can be significantly accelerated by intelligently embedding shortcuts. Experiments show that hybrid agents achieve significantly higher success rates and efficiency than their GUI-only counterparts. This result also demonstrates the effectiveness of our method for evaluating an agent's shortcut generation capabilities. MAS-Bench fills a critical evaluation gap, providing a foundational platform for future advancements in creating more efficient and robust intelligent agents. 

**Abstract (ZH)**: MAS-Bench：面向移动领域的GUI-shortcut混合代理评估基准 

---
# Accelerate Scaling of LLM Alignment via Quantifying the Coverage and Depth of Instruction Set 

**Title (ZH)**: 通过量化指令集的覆盖面和深度加速大型语言模型的对齐缩放 

**Authors**: Chengwei Wu, Li Du, Hanyu Zhao, Yiming Ju, Jiapu Wang, Tengfei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.06463)  

**Abstract**: With the growing demand for applying large language models to downstream tasks, improving model alignment performance and efficiency has become crucial. Such a process involves selecting informative instructions from a candidate pool. However, due to the complexity of instruction set distributions, the key factors driving the performance of aligned models remain unclear. As a result, current instruction set refinement methods fail to improve performance as the instruction pool expands continuously. To address this issue, we first investigate the key factors that influence the relationship between instruction dataset distribution and aligned model performance. Based on these insights, we propose a novel instruction data selection method. We identify that the depth of instructions and the coverage of the semantic space are the crucial factors determining downstream performance, which could explain over 70\% of the model loss on the development set. We then design an instruction selection algorithm to simultaneously maximize the depth and semantic coverage of the selected instructions. Experimental results demonstrate that, compared to state-of-the-art baseline methods, it can sustainably improve model performance at a faster pace and thus achieve \emph{``Accelerated Scaling''}. 

**Abstract (ZH)**: 随着对将大规模语言模型应用于下游任务需求的增长，提高模型对齐性能和效率已成为关键。这一过程涉及到从候选集中选择具有信息性的指令。但由于指令集分布的复杂性，驱动对齐模型性能的关键因素仍不清楚。因此，当前的指令集精炼方法无法在指令池不断扩大的情况下改善性能。为解决这一问题，我们首先研究影响指令数据集分布与对齐模型性能之间关系的关键因素。基于这些洞见，我们提出了一种新的指令数据选择方法。我们发现指令的深度和语义空间的覆盖范围是决定下游性能的关键因素，可以解释开发集上超过70%的模型损失。然后，我们设计了一种指令选择算法，同时最大化所选指令的深度和语义覆盖范围。实验结果表明，与最先进的基线方法相比，它能够更快地持续提升模型性能，从而实现“加速扩展”的目标。 

---
# HyFedRAG: A Federated Retrieval-Augmented Generation Framework for Heterogeneous and Privacy-Sensitive Data 

**Title (ZH)**: HyFedRAG：一种用于异构和隐私敏感数据的联邦检索增强生成框架 

**Authors**: Cheng Qian, Hainan Zhang, Yongxin Tong, Hong-Wei Zheng, Zhiming Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.06444)  

**Abstract**: Centralized RAG pipelines struggle with heterogeneous and privacy-sensitive data, especially in distributed healthcare settings where patient data spans SQL, knowledge graphs, and clinical notes. Clinicians face difficulties retrieving rare disease cases due to privacy constraints and the limitations of traditional cloud-based RAG systems in handling diverse formats and edge devices. To address this, we introduce HyFedRAG, a unified and efficient Federated RAG framework tailored for Hybrid data modalities. By leveraging an edge-cloud collaborative mechanism, HyFedRAG enables RAG to operate across diverse data sources while preserving data privacy. Our key contributions are: (1) We design an edge-cloud collaborative RAG framework built on Flower, which supports querying structured SQL data, semi-structured knowledge graphs, and unstructured documents. The edge-side LLMs convert diverse data into standardized privacy-preserving representations, and the server-side LLMs integrates them for global reasoning and generation. (2) We integrate lightweight local retrievers with privacy-aware LLMs and provide three anonymization tools that enable each client to produce semantically rich, de-identified summaries for global inference across devices. (3) To optimize response latency and reduce redundant computation, we design a three-tier caching strategy consisting of local cache, intermediate representation cache, and cloud inference cache. Experimental results on PMC-Patients demonstrate that HyFedRAG outperforms existing baselines in terms of retrieval quality, generation consistency, and system efficiency. Our framework offers a scalable and privacy-compliant solution for RAG over structural-heterogeneous data, unlocking the potential of LLMs in sensitive and diverse data environments. 

**Abstract (ZH)**: 集中式的RAG管道在处理异构和隐私敏感数据时存在困难，特别是在患者数据横跨SQL、知识图谱和临床笔记的分布式医疗环境中。临床医生因隐私限制和传统基于云的RAG系统在处理多样格式和边缘设备时的局限性，难以检索罕见疾病案例。为此，我们提出了HyFedRAG，这是一种针对混合数据模态统一且高效的联邦RAG框架。通过利用边缘-云协作机制，HyFedRAG能够在保护数据隐私的同时跨多种数据来源运行RAG。我们的主要贡献包括：（1）我们设计了一种基于Flower的边缘-云协作RAG框架，支持查询结构化SQL数据、半结构化知识图谱和非结构化文档。边缘侧的LLM将多种数据转换为标准化的隐私保护表示，服务器侧的LLM则将它们结合进行全局推理和生成。（2）我们集成了轻量级的本地检索器并与隐私保护的LLM集成，并提供了三种匿名化工具，使每个客户端能够在设备间生成语义丰富的去标识化摘要以供全局推理。（3）为了优化响应延迟并减少冗余计算，我们设计了一种三级缓存策略，包括本地缓存、中间表示缓存和云推理缓存。实验证明，HyFedRAG在检索质量、生成一致性和系统效率方面优于现有基线。我们的框架为在结构异构数据上实现RAG提供了可扩展且符合隐私要求的解决方案，并在敏感和多样化数据环境中释放了LLM的潜力。 

---
# Tree of Agents: Improving Long-Context Capabilities of Large Language Models through Multi-Perspective Reasoning 

**Title (ZH)**: 代理之树：通过多视角推理提升大型语言模型的长上下文能力 

**Authors**: Song Yu, Xiaofei Xu, Ke Deng, Li Li, Lin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2509.06436)  

**Abstract**: Large language models (LLMs) face persistent challenges when handling long-context tasks, most notably the lost in the middle issue, where information located in the middle of a long input tends to be underutilized. Some existing methods that reduce input have the risk of discarding key information, while others that extend context windows often lead to attention dispersion. To address these limitations, we propose Tree of Agents (TOA), a multi-agent reasoning framework that segments the input into chunks processed by independent agents. Each agent generates its local cognition, then agents dynamically exchange information for collaborative reasoning along tree-structured paths. TOA enables agents to probe different reasoning orders for multi-perspective understanding, effectively mitigating position bias and reducing hallucinations. To improve processing efficiency, we incorporate prefix-hash caching and adaptive pruning strategies, achieving significant performance improvements with comparable API overhead. Experiments show that TOA, powered by compact LLaMA3.1-8B, significantly outperforms multiple baselines and demonstrates comparable performance to the latest and much larger commercial models, such as Gemini1.5-pro, on various long-context tasks. Code is available at this https URL. 

**Abstract (ZH)**: Large语言模型（LLMs）在处理长上下文任务时面临持续挑战，最 notably的问题是中间信息丢失，即长输入中间部分的信息往往被充分利用不足。一些减少输入长度的方法存在舍弃关键信息的风险，而延长上下文窗口的方法往往会引发注意力分散。为了解决这些限制，我们提出了代理树（TOA）这一多代理推理框架，该框架将输入分割成由独立代理处理的片段。每个代理生成其局部认知，然后代理通过基于树结构的路径动态交换信息进行合作推理。TOA使代理能够探索不同的推理顺序以实现多视角理解，从而有效减轻位置偏见并减少幻觉。为了提高处理效率，我们结合了前缀哈希缓存和自适应剪枝策略，在相近的API开销下实现了显著的性能提升。实验结果表明，由紧凑的LLaMA3.1-8B驱动的TOA在多种长上下文任务上显著优于多个基线模型，并展示了与最新且更大规模的商业模型（如Gemini1.5-pro）相当的性能。代码可在以下链接获取：这个 https URL。 

---
# Teaching AI Stepwise Diagnostic Reasoning with Report-Guided Chain-of-Thought Learning 

**Title (ZH)**: 基于报告引导的链式思考教学：逐步诊断推理的AI教学 

**Authors**: Yihong Luo, Wenwu He, Zhuo-Xu Cui, Dong Liang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06409)  

**Abstract**: This study presents DiagCoT, a multi-stage framework that applies supervised fine-tuning to general-purpose vision-language models (VLMs) to emulate radiologists' stepwise diagnostic reasoning using only free-text reports. DiagCoT combines contrastive image-report tuning for domain alignment, chain-of-thought supervision to capture inferential logic, and reinforcement tuning with clinical reward signals to enhance factual accuracy and fluency. On the MIMIC-CXR benchmark, DiagCoT improved zero-shot disease classification AUC from 0.52 to 0.76 (absolute gain of 0.24), pathology grounding mIoU from 0.08 to 0.31 (absolute gain of 0.23), and report generation BLEU from 0.11 to 0.33 (absolute gain of 0.22). It outperformed state-of-the-art models including LLaVA-Med and CXR-LLAVA on long-tailed diseases and external datasets. By converting unstructured clinical narratives into structured supervision, DiagCoT offers a scalable approach for developing interpretable and diagnostically competent AI systems for radiology. 

**Abstract (ZH)**: DiagCoT：一种用于模仿放射科医生逐步诊断推理的多阶段框架，仅使用自由文本报告对通用视觉-语言模型进行监督微调 

---
# A data-driven discretized CS:GO simulation environment to facilitate strategic multi-agent planning research 

**Title (ZH)**: 基于数据驱动的 discretized CS:GO 模拟环境，以促进战略性多Agent规划研究 

**Authors**: Yunzhe Wang, Volkan Ustun, Chris McGroarty  

**Link**: [PDF](https://arxiv.org/pdf/2509.06355)  

**Abstract**: Modern simulation environments for complex multi-agent interactions must balance high-fidelity detail with computational efficiency. We present DECOY, a novel multi-agent simulator that abstracts strategic, long-horizon planning in 3D terrains into high-level discretized simulation while preserving low-level environmental fidelity. Using Counter-Strike: Global Offensive (CS:GO) as a testbed, our framework accurately simulates gameplay using only movement decisions as tactical positioning -- without explicitly modeling low-level mechanics such as aiming and shooting. Central to our approach is a waypoint system that simplifies and discretizes continuous states and actions, paired with neural predictive and generative models trained on real CS:GO tournament data to reconstruct event outcomes. Extensive evaluations show that replays generated from human data in DECOY closely match those observed in the original game. Our publicly available simulation environment provides a valuable tool for advancing research in strategic multi-agent planning and behavior generation. 

**Abstract (ZH)**: 现代复杂多智能体交互的仿真环境需平衡高保真细节与计算效率。我们提出了DECOY多智能体仿真器，将其在3D地形上的战略、长时规划抽象为高层次离散化仿真，同时保留低层级环境保真度。使用《反恐精英：全球进攻》（CS:GO）作为实验平台，我们的框架仅通过移动决策进行战术位置模拟，即可准确再现 gameplay，无需明确建模低层级机制如瞄准和射击。我们方法的核心在于一个路径点系统，该系统简化并离散化了连续状态和动作，并配以基于真实CS:GO锦标赛数据训练的神经预测和生成模型，以重建事件结果。广泛评估表明，DECOY生成的重放与原始游戏中的观察结果高度一致。我们公开提供的仿真环境为战略多智能体规划和行为生成的研究提供了有价值的工具。 

---
# Evaluating Multi-Turn Bargain Skills in LLM-Based Seller Agent 

**Title (ZH)**: 基于LLM的卖家代理多轮讨价还价技能评估 

**Authors**: Issue Yishu Wang, Kakam Chong, Xiaofeng Wang, Xu Yan, DeXin Kong, Chen Ju, Ming Chen, Shuai Xiao, Shuguang Han, jufeng chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.06341)  

**Abstract**: In online second-hand marketplaces, multi-turn bargaining is a crucial part of seller-buyer interactions. Large Language Models (LLMs) can act as seller agents, negotiating with buyers on behalf of sellers under given business constraints. A critical ability for such agents is to track and accurately interpret cumulative buyer intents across long negotiations, which directly impacts bargaining effectiveness. We introduce a multi-turn evaluation framework for measuring the bargaining ability of seller agents in e-commerce dialogues. The framework tests whether an agent can extract and track buyer intents. Our contributions are: (1) a large-scale e-commerce bargaining benchmark spanning 622 categories, 9,892 products, and 3,014 tasks; (2) a turn-level evaluation framework grounded in Theory of Mind (ToM) with annotated buyer intents, moving beyond outcome-only metrics; and (3) an automated pipeline that extracts reliable intent from massive dialogue data. 

**Abstract (ZH)**: 在线二手市场中，多轮讨价还价是卖家与买家交互的重要组成部分。大规模语言模型可以作为卖家代理，在给定的商业约束下与买家进行谈判。这种代理的一个关键能力是跟踪和准确解释长时间谈判中累积的买家意图，这直接影响谈判效果。我们提出了一种多轮谈判评估框架，用于衡量电子商务对话中卖家代理的谈判能力。该框架测试代理是否能够提取和跟踪买家意图。我们的贡献包括：（1）涵盖622个类别、9,892个产品和3,014个任务的大规模电子商务谈判基准；（2）基于理论心智（ToM）的回合级评估框架，并标注了买家意图，超越了仅关注结果的指标；以及（3）一个自动提取大规模对话数据中可靠意图的管道。 

---
# Large Language Models as Virtual Survey Respondents: Evaluating Sociodemographic Response Generation 

**Title (ZH)**: 大型语言模型作为虚拟调查受访者：社会 demographic 特征应答生成评估 

**Authors**: Jianpeng Zhao, Chenyu Yuan, Weiming Luo, Haoling Xie, Guangwei Zhang, Steven Jige Quan, Zixuan Yuan, Pengyang Wang, Denghui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06337)  

**Abstract**: Questionnaire-based surveys are foundational to social science research and public policymaking, yet traditional survey methods remain costly, time-consuming, and often limited in scale. This paper explores a new paradigm: simulating virtual survey respondents using Large Language Models (LLMs). We introduce two novel simulation settings, namely Partial Attribute Simulation (PAS) and Full Attribute Simulation (FAS), to systematically evaluate the ability of LLMs to generate accurate and demographically coherent responses. In PAS, the model predicts missing attributes based on partial respondent profiles, whereas FAS involves generating complete synthetic datasets under both zero-context and context-enhanced conditions. We curate a comprehensive benchmark suite, LLM-S^3 (Large Language Model-based Sociodemographic Survey Simulation), that spans 11 real-world public datasets across four sociological domains. Our evaluation of multiple mainstream LLMs (GPT-3.5/4 Turbo, LLaMA 3.0/3.1-8B) reveals consistent trends in prediction performance, highlights failure modes, and demonstrates how context and prompt design impact simulation fidelity. This work establishes a rigorous foundation for LLM-driven survey simulations, offering scalable and cost-effective tools for sociological research and policy evaluation. Our code and dataset are available at: this https URL 

**Abstract (ZH)**: 基于问卷的调查是社会科学研究和公共政策制定的基础，但传统的调查方法仍然昂贵、耗时且规模有限。本文探讨了一种新的范式：使用大规模语言模型（LLMs）模拟虚拟调查受访者。我们介绍了两种新的模拟设置，即部分属性模拟（PAS）和全程属性模拟（FAS），以系统评估LLMs生成准确且人口统计学一致的响应的能力。在PAS中，模型根据部分受访者的画像预测缺失的属性，而在FAS中，则在零上下文和增强上下文条件下生成完整的合成数据集。我们编纂了一个全面的基准套件LLM-S^3（基于大规模语言模型的社会人口统计学调查模拟），涵盖了四个社会学领域内的11个真实世界公共数据集。对多个主流LLM（GPT-3.5/4 Turbo、LLaMA 3.0/3.1-8B）的评估揭示了预测性能的一致趋势，指出了失效模式，并展示了上下文和提示设计如何影响模拟精度。本项工作为基于LLM的调查模拟奠定了严格的理论基础，提供了社会学研究和政策评价的可扩展且成本效益高的工具。我们的代码和数据集可在以下网址获得：this https URL 

---
# Can AI Make Energy Retrofit Decisions? An Evaluation of Large Language Models 

**Title (ZH)**: AI能为能源改造决策提供帮助吗？大型语言模型的评估 

**Authors**: Lei Shu, Dong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.06307)  

**Abstract**: Conventional approaches to building energy retrofit decision making suffer from limited generalizability and low interpretability, hindering adoption in diverse residential contexts. With the growth of Smart and Connected Communities, generative AI, especially large language models (LLMs), may help by processing contextual information and producing practitioner readable recommendations. We evaluate seven LLMs (ChatGPT, DeepSeek, Gemini, Grok, Llama, and Claude) on residential retrofit decisions under two objectives: maximizing CO2 reduction (technical) and minimizing payback period (sociotechnical). Performance is assessed on four dimensions: accuracy, consistency, sensitivity, and reasoning, using a dataset of 400 homes across 49 US states. LLMs generate effective recommendations in many cases, reaching up to 54.5 percent top 1 match and 92.8 percent within top 5 without fine tuning. Performance is stronger for the technical objective, while sociotechnical decisions are limited by economic trade offs and local context. Agreement across models is low, and higher performing models tend to diverge from others. LLMs are sensitive to location and building geometry but less sensitive to technology and occupant behavior. Most models show step by step, engineering style reasoning, but it is often simplified and lacks deeper contextual awareness. Overall, LLMs are promising assistants for energy retrofit decision making, but improvements in accuracy, consistency, and context handling are needed for reliable practice. 

**Abstract (ZH)**: 基于生成式AI的大型语言模型在住宅能效改造决策中的潜力：超越传统方法的限制 

---
# From Implicit Exploration to Structured Reasoning: Leveraging Guideline and Refinement for LLMs 

**Title (ZH)**: 从隐式探索到结构化推理：利用指南和精炼提升语言模型 

**Authors**: Jiaxiang Chen, Zhuo Wang, Mingxi Zou, Zhucong Li, Zhijian Zhou, Song Wang, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06284)  

**Abstract**: Large language models (LLMs) have advanced general-purpose reasoning, showing strong performance across diverse tasks. However, existing methods often rely on implicit exploration, where the model follows stochastic and unguided reasoning paths-like walking without a map. This leads to unstable reasoning paths, lack of error correction, and limited learning from past experience. To address these issues, we propose a framework that shifts from implicit exploration to structured reasoning through guideline and refinement. First, we extract structured reasoning patterns from successful trajectories and reflective signals from failures. During inference, the model follows these guidelines step-by-step, with refinement applied after each step to correct errors and stabilize the reasoning process. Experiments on BBH and four additional benchmarks (GSM8K, MATH-500, MBPP, HumanEval) show that our method consistently outperforms strong baselines across diverse reasoning tasks. Structured reasoning with stepwise execution and refinement improves stability and generalization, while guidelines transfer well across domains and flexibly support cross-model collaboration, matching or surpassing supervised fine-tuning in effectiveness and scalability. 

**Abstract (ZH)**: 大型语言模型（LLMs）在通用推理方面取得了进展，展现了在多种任务上的强大性能。然而，现有方法往往依赖于隐式的探索，模型沿着随机且不受指导的推理路径进行推理，如同没有地图的行走。这导致了不稳定的推理路径、缺乏错误纠正以及有限的学习能力。为解决这些问题，我们提出了一种框架，通过指南和修正转向结构化的推理。首先，我们从成功的轨迹和失败的反思信号中提取结构化的推理模式。在推理过程中，模型按照这些指南逐步执行，并在每一步后进行修正以纠正错误并稳定推理过程。实验结果表明，我们的方法在BBH及四个附加基准（GSM8K、MATH-500、MBPP、HumanEval）上的一致性地优于强baseline，在多种推理任务上表现出色。逐步执行和修正的结构化推理提高了稳定性和泛化能力，而指南在不同领域之间传递良好，并灵活支持跨模型协作，在效果和扩展性上可匹及或超越监督微调。 

---
# SFR-DeepResearch: Towards Effective Reinforcement Learning for Autonomously Reasoning Single Agents 

**Title (ZH)**: SFR-DeepResearch: 向自主推理单个代理的有效强化学习方向 

**Authors**: Xuan-Phi Nguyen, Shrey Pandit, Revanth Gangi Reddy, Austin Xu, Silvio Savarese, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2509.06283)  

**Abstract**: Equipping large language models (LLMs) with complex, interleaved reasoning and tool-use capabilities has become a key focus in agentic AI research, especially with recent advances in reasoning-oriented (``thinking'') models. Such capabilities are key to unlocking a number of important applications. One such application is Deep Research (DR), which requires extensive search and reasoning over many sources. Our work in this paper focuses on the development of native Autonomous Single-Agent models for DR featuring minimal web crawling and Python tool integration. Unlike multi-agent systems, where agents take up pre-defined roles and are told what to do at each step in a static workflow, an autonomous single-agent determines its next action dynamically based on context, without manual directive. While prior work has proposed training recipes for base or instruction-tuned LLMs, we focus on continual reinforcement learning (RL) of reasoning-optimized models to further enhance agentic skills while preserving reasoning ability. Towards this end, we propose a simple RL recipe with entirely synthetic data, which we apply to various open-source LLMs. Our best variant SFR-DR-20B achieves up to 28.7% on Humanity's Last Exam benchmark. In addition, we conduct key analysis experiments to provide more insights into our methodologies. 

**Abstract (ZH)**: 装备有复杂交错推理和工具使用能力的大规模语言模型（LLMs）已成为有目的的AI研究中的关键焦点，尤其是在推理导向（“思考”）模型 recently 的进展之后。此类能力对于开启多种重要应用至关重要。其中一个应用是深度研究（DR），其需要在多种来源上进行广泛的搜索和推理。本文的工作集中在开发原生自主单智能体模型（DR）的开发上，这些模型具备最小限度的网页抓取和Python工具集成。与多智能体系统不同，在多智能体系统中，智能体承担预定义的角色，并在其静态工作流中的每一步被告知做什么，自主单智能体会根据上下文动态确定其下一步行动，无需人工指令。尽管先前的工作已经提出了针对基础或指令调优语言模型的训练方法，但我们专注于通过持续强化学习（RL）来进一步提升智能体技能，同时保持推理能力。为此，我们提出了一种基于完全合成数据的简单RL方法，并将其应用于多种开源LLM。我们的最佳变体SFR-DR-20B在人类的最后一试基准测试中达到了最高28.7%的成绩。此外，我们还进行了关键分析实验，以更深入地了解我们的方法。 

---
# TableMind: An Autonomous Programmatic Agent for Tool-Augmented Table Reasoning 

**Title (ZH)**: TableMind: 一种用于工具增强表格推理的自主程序化代理 

**Authors**: Chuang Jiang, Mingyue Cheng, Xiaoyu Tao, Qingyang Mao, Jie Ouyang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06278)  

**Abstract**: Table reasoning is crucial for leveraging structured data in domains such as finance, healthcare, and scientific research. While large language models (LLMs) show promise in multi-step reasoning, purely text-based methods often struggle with the complex numerical computations and fine-grained operations inherently required in this task. Tool-integrated reasoning improves computational accuracy via explicit code execution, yet existing systems frequently rely on rigid patterns, supervised imitation, and lack true autonomous adaptability. In this paper, we present TableMind, an LLM-driven table reasoning agent that (i) autonomously performs multi-turn tool invocation, (ii) writes and executes data-analyzing code in a secure sandbox environment for data analysis and precise numerical reasoning, and (iii) exhibits high-level capabilities such as planning and self-reflection to adapt strategies. To realize these capabilities, we adopt a two-stage fine-tuning paradigm built on top of a powerful pre-trained language model: supervised fine-tuning on high-quality reasoning trajectories to establish effective tool usage patterns, followed by reinforcement fine-tuning to optimize multi-objective strategies. In particular, we propose Rank-Aware Policy Optimization (RAPO), which increases the update weight of high-quality trajectories when their output probabilities are lower than those of low-quality ones, thereby guiding the model more consistently toward better and more accurate answers. Extensive experiments on several mainstream benchmarks demonstrate that TableMind achieves superior performance compared to competitive baselines, yielding substantial gains in both reasoning accuracy and computational precision. 

**Abstract (ZH)**: TableMind：基于大型语言模型的自主表推理代理 

---
# REMI: A Novel Causal Schema Memory Architecture for Personalized Lifestyle Recommendation Agents 

**Title (ZH)**: REMI: 一种新型因果模式记忆架构的个性化生活方式推荐代理 

**Authors**: Vishal Raman, Vijai Aravindh R, Abhijith Ragav  

**Link**: [PDF](https://arxiv.org/pdf/2509.06269)  

**Abstract**: Personalized AI assistants often struggle to incorporate complex personal data and causal knowledge, leading to generic advice that lacks explanatory power. We propose REMI, a Causal Schema Memory architecture for a multimodal lifestyle agent that integrates a personal causal knowledge graph, a causal reasoning engine, and a schema based planning module. The idea is to deliver explainable, personalized recommendations in domains like fashion, personal wellness, and lifestyle planning. Our architecture uses a personal causal graph of the user's life events and habits, performs goal directed causal traversals enriched with external knowledge and hypothetical reasoning, and retrieves adaptable plan schemas to generate tailored action plans. A Large Language Model orchestrates these components, producing answers with transparent causal explanations. We outline the CSM system design and introduce new evaluation metrics for personalization and explainability, including Personalization Salience Score and Causal Reasoning Accuracy, to rigorously assess its performance. Results indicate that CSM based agents can provide more context aware, user aligned recommendations compared to baseline LLM agents. This work demonstrates a novel approach to memory augmented, causal reasoning in personalized agents, advancing the development of transparent and trustworthy AI lifestyle assistants. 

**Abstract (ZH)**: 个性化AI助手往往难以整合复杂的个人数据和因果知识，导致提供的建议缺乏解释力。我们提出了一种因果模式记忆架构REMI，该架构用于多模态生活方式代理，集成了个人因果知识图、因果推理引擎和基于模式的规划模块。目的是在时尚、个人健康和生活方式规划等领域提供可解释的个性化推荐。该架构使用用户的生平事件和个人习惯的个人因果图，进行目标导向的因果遍历，结合外部知识和假设推理，并检索适应性计划模式生成定制化的行动计划。大规模语言模型协调这些组件，生成具有透明因果解释的答案。我们概述了CSM系统设计，并引入了新的个性化和解释性评估指标，包括个性化显著性评分和个人因果推理准确性，以严格评估其性能。结果表明，基于CSM的代理可以比基线语言模型代理提供更具上下文关联和个人导向的建议。这项工作展示了增强记忆和因果推理的新颖方法在个性化代理中的应用，推动了透明和可信赖的生活方式AI助手的发展。 

---
# Proof2Silicon: Prompt Repair for Verified Code and Hardware Generation via Reinforcement Learning 

**Title (ZH)**: Proof2Silicon: 通过强化学习进行验证代码和硬件生成的提示修复 

**Authors**: Manvi Jha, Jiaxin Wan, Deming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.06239)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in automated code generation but frequently produce code that fails formal verification, an essential requirement for hardware and safety-critical domains. To overcome this fundamental limitation, we previously proposed PREFACE, a model-agnostic framework based on reinforcement learning (RL) that iteratively repairs the prompts provided to frozen LLMs, systematically steering them toward generating formally verifiable Dafny code without costly fine-tuning. This work presents Proof2Silicon, a novel end-to-end synthesis framework that embeds the previously proposed PREFACE flow to enable the generation of correctness-by-construction hardware directly from natural language specifications. Proof2Silicon operates by: (1) leveraging PREFACE's verifier-driven RL agent to optimize prompt generation iteratively, ensuring Dafny code correctness; (2) automatically translating verified Dafny programs into synthesizable high-level C using Dafny's Python backend and PyLog; and (3) employing Vivado HLS to produce RTL implementations. Evaluated rigorously on a challenging 100-task benchmark, PREFACE's RL-guided prompt optimization consistently improved Dafny verification success rates across diverse LLMs by up to 21%. Crucially, Proof2Silicon achieved an end-to-end hardware synthesis success rate of up to 72%, generating RTL designs through Vivado HLS synthesis flows. These results demonstrate a robust, scalable, and automated pipeline for LLM-driven, formally verified hardware synthesis, bridging natural-language specification and silicon realization. 

**Abstract (ZH)**: Large Language Models (LLMs)在自动生成代码方面的表现令人印象深刻，但经常生成无法进行形式验证的代码，而形式验证是硬件和安全关键领域的一项基本要求。为克服这一根本性限制，我们之前提出了PREFACE，这是一种基于强化学习的模型无关框架，该框架通过迭代修复冻结的LLM的提示，有系统地引导其生成形式可验证的Dafny代码，而无需昂贵的微调。本文介绍了一种名为Proof2Silicon的新颖端到端综合框架，该框架嵌入了PREFACE流程，能够直接从自然语言规范生成构造正确的硬件。Proof2Silicon通过以下步骤工作：(1)利用PREFACE的验证驱动RL代理优化提示生成，确保Dafny代码的正确性；(2)使用Dafny的Python后端和PyLog自动将验证通过的Dafny程序翻译为可综合的高层C代码；(3)使用Vivado HLS生成寄存器传输级实现。在一项具有挑战性的100任务基准测试中，PREFACE的RL引导提示优化一致地提高了各种LLM的Dafny验证成功率多达21%。关键的是，Proof2Silicon实现了从端到端硬件综合的成功率高达72%，通过Vivado HLS综合流程生成了寄存器传输级设计。这些结果表明，PREFACE为LLM驱动的形式验证硬件综合提供了一个稳健、可扩展和自动化的管道，将自然语言规范与硅实现连接起来。 

---
# PillagerBench: Benchmarking LLM-Based Agents in Competitive Minecraft Team Environments 

**Title (ZH)**: PillagerBench: 在竞争性Minecraft团队环境中文本生成代理的基准测试 

**Authors**: Olivier Schipper, Yudi Zhang, Yali Du, Mykola Pechenizkiy, Meng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06235)  

**Abstract**: LLM-based agents have shown promise in various cooperative and strategic reasoning tasks, but their effectiveness in competitive multi-agent environments remains underexplored. To address this gap, we introduce PillagerBench, a novel framework for evaluating multi-agent systems in real-time competitive team-vs-team scenarios in Minecraft. It provides an extensible API, multi-round testing, and rule-based built-in opponents for fair, reproducible comparisons. We also propose TactiCrafter, an LLM-based multi-agent system that facilitates teamwork through human-readable tactics, learns causal dependencies, and adapts to opponent strategies. Our evaluation demonstrates that TactiCrafter outperforms baseline approaches and showcases adaptive learning through self-play. Additionally, we analyze its learning process and strategic evolution over multiple game episodes. To encourage further research, we have open-sourced PillagerBench, fostering advancements in multi-agent AI for competitive environments. 

**Abstract (ZH)**: 基于LLM的代理在多种协作与策略性推理任务中展现了潜力，但它们在竞争性多代理环境中的有效性仍待深入探索。为填补这一空白，我们引入了PillagerBench，一种用于在Minecraft中评估实时竞争性团队对团队多代理系统的新型框架。该框架提供了可扩展的API、多轮测试和基于规则的内置对手，以实现公平和可重复的比较。此外，我们还提出了TactiCrafter，一种基于LLM的多代理系统，通过可读性战术促进团队协作，学习因果依赖关系，并适应对手策略。我们的评估表明，TactiCrafter在基准方法中表现更优，并通过自我对弈展示了自适应学习能力。另外，我们还分析了其在多个游戏回合中的学习过程和战略演变。为了促进进一步研究，我们开源了PillagerBench，推动竞争性环境中的多代理AI技术进步。 

---
# From Long to Short: LLMs Excel at Trimming Own Reasoning Chains 

**Title (ZH)**: 从长到短：大规模语言模型擅长精简自身的推理链 

**Authors**: Wei Han, Geng Zhan, Sicheng Yu, Chenyu Wang, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2509.06174)  

**Abstract**: O1/R1 style large reasoning models (LRMs) signal a substantial leap forward over conventional instruction-following LLMs. By applying test-time scaling to generate extended reasoning paths, they establish many SOTAs across a wide range of complex reasoning tasks. However, recent studies show that LRMs are prone to suffer from overthinking -- the tendency to overcomplicate simple problems, leading to excessive strategy switching and long, convoluted reasoning traces that hinder their interpretability. To mitigate this issue, we conduct a systematic investigation into the reasoning efficiency of a broad set of LRMs and uncover a common dilemma: the difficulty in balancing multiple generation objectives such as correctness and brevity. Based on this discovery, we propose a test-time scaling method, EDIT (Efficient Dynamic Inference Trimming), which efficiently guides LRMs to identify the shortest correct reasoning paths at test time. EDIT employs constraint-guided generation while jointly tracking length and answer distributions under varying constraints, allowing it to select responses that strike an optimal balance between conciseness and correctness. Extensive experiments across diverse models and datasets show that EDIT substantially enhance the reasoning efficiency, producing compact yet informative outputs that improve readability and user experience. 

**Abstract (ZH)**: O1/R1风格大规模推理模型的推理效率研究：EDIT方法克服过度思考问题 

---
# Reverse-Engineered Reasoning for Open-Ended Generation 

**Title (ZH)**: 逆向工程推理以实现开放生成 

**Authors**: Haozhe Wang, Haoran Que, Qixin Xu, Minghao Liu, Wangchunshu Zhou, Jiazhan Feng, Wanjun Zhong, Wei Ye, Tong Yang, Wenhao Huang, Ge Zhang, Fangzhen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06160)  

**Abstract**: While the ``deep reasoning'' paradigm has spurred significant advances in verifiable domains like mathematics, its application to open-ended, creative generation remains a critical challenge. The two dominant methods for instilling reasoning -- reinforcement learning (RL) and instruction distillation -- falter in this area; RL struggles with the absence of clear reward signals and high-quality reward models, while distillation is prohibitively expensive and capped by the teacher model's capabilities. To overcome these limitations, we introduce REverse-Engineered Reasoning (REER), a new paradigm that fundamentally shifts the approach. Instead of building a reasoning process ``forwards'' through trial-and-error or imitation, REER works ``backwards'' from known-good solutions to computationally discover the latent, step-by-step deep reasoning process that could have produced them. Using this scalable, gradient-free approach, we curate and open-source DeepWriting-20K, a large-scale dataset of 20,000 deep reasoning trajectories for open-ended tasks. Our model, DeepWriter-8B, trained on this data, not only surpasses strong open-source baselines but also achieves performance competitive with, and at times superior to, leading proprietary models like GPT-4o and Claude 3.5. 

**Abstract (ZH)**: 虽然“深度推理”范式在可验证领域如数学取得了显著进展，其在开放性和创造性生成任务中的应用仍然是一个关键挑战。传统的两种增强推理的方法——强化学习（RL）和指令蒸馏——在这个领域遇到了困难；RL因缺乏清晰的奖励信号和高质量的奖励模型而举步维艰，而蒸馏则由于成本高昂且受到教师模型能力的限制而显得力不从心。为克服这些局限，我们提出了反向工程推理（REER）这一新的范式，从根本上改变了方法论。REER不再通过试错或模仿来构建推理过程，而是从已知良好的解决方案出发，向后发现能够生成这些解决方案的潜在的、步骤化的深度推理过程。借助这一可扩展且无梯度的方法，我们精选并开源了包含20,000个开放任务深度推理轨迹的DeepWriting-20K大数据集。基于此数据训练的DeepWriter-8B模型不仅超越了强大的开源基线，还在某些方面优于领先的专业模型，如GPT-4o和Claude 3.5。 

---
# Rethinking Reasoning Quality in Large Language Models through Enhanced Chain-of-Thought via RL 

**Title (ZH)**: 通过增强链式思考的RL促进大型语言模型中推理质量的重新思考 

**Authors**: Haoyang He, Zihua Rong, Kun Ji, Chenyang Li, Qing Huang, Chong Xia, Lan Yang, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06024)  

**Abstract**: Reinforcement learning (RL) has recently become the dominant paradigm for strengthening the reasoning abilities of large language models (LLMs). Yet the rule-based reward functions commonly used on mathematical or programming benchmarks assess only answer format and correctness, providing no signal as to whether the induced Chain-of-Thought (CoT) actually improves the answer. Furthermore, such task-specific training offers limited control over logical depth and therefore may fail to reveal a model's genuine reasoning capacity. We propose Dynamic Reasoning Efficiency Reward (DRER) -- a plug-and-play RL reward framework that reshapes both reward and advantage signals. (i) A Reasoning Quality Reward assigns fine-grained credit to those reasoning chains that demonstrably raise the likelihood of the correct answer, directly incentivising the trajectories with beneficial CoT tokens. (ii) A Dynamic Length Advantage decays the advantage of responses whose length deviates from a validation-derived threshold, stabilising training. To facilitate rigorous assessment, we also release Logictree, a dynamically constructed deductive reasoning dataset that functions both as RL training data and as a comprehensive benchmark. Experiments confirm the effectiveness of DRER: our 7B model attains GPT-o3-mini level performance on Logictree with 400 trianing steps, while the average confidence of CoT-augmented answers rises by 30%. The model further exhibits generalisation across diverse logical-reasoning datasets, and the mathematical benchmark AIME24. These results illuminate how RL shapes CoT behaviour and chart a practical path toward enhancing formal-reasoning skills in large language models. All code and data are available in repository this https URL. 

**Abstract (ZH)**: 动态推理效率奖励：增强大型语言模型推理能力的插件式 reinforcement 学习奖励框架 

---
# MapAgent: A Hierarchical Agent for Geospatial Reasoning with Dynamic Map Tool Integration 

**Title (ZH)**: MapAgent：一种集成动态地图工具的分级代理地理空间推理模型 

**Authors**: Md Hasebul Hasan, Mahir Labib Dihan, Mohammed Eunus Ali, Md Rizwan Parvez  

**Link**: [PDF](https://arxiv.org/pdf/2509.05933)  

**Abstract**: Agentic AI has significantly extended the capabilities of large language models (LLMs) by enabling complex reasoning and tool use. However, most existing frameworks are tailored to domains such as mathematics, coding, or web automation, and fall short on geospatial tasks that require spatial reasoning, multi-hop planning, and real-time map interaction. To address these challenges, we introduce MapAgent, a hierarchical multi-agent plug-and-play framework with customized toolsets and agentic scaffolds for map-integrated geospatial reasoning. Unlike existing flat agent-based approaches that treat tools uniformly-often overwhelming the LLM when handling similar but subtly different geospatial APIs-MapAgent decouples planning from execution. A high-level planner decomposes complex queries into subgoals, which are routed to specialized modules. For tool-heavy modules-such as map-based services-we then design a dedicated map-tool agent that efficiently orchestrates related APIs adaptively in parallel to effectively fetch geospatial data relevant for the query, while simpler modules (e.g., solution generation or answer extraction) operate without additional agent overhead. This hierarchical design reduces cognitive load, improves tool selection accuracy, and enables precise coordination across similar APIs. We evaluate MapAgent on four diverse geospatial benchmarks-MapEval-Textual, MapEval-API, MapEval-Visual, and MapQA-and demonstrate substantial gains over state-of-the-art tool-augmented and agentic baselines. We open-source our framwork at this https URL. 

**Abstract (ZH)**: 代理型AI显著扩展了大语言模型（LLMs）的能力，通过实现复杂的推理和工具使用。然而，现有的大多数框架主要针对数学、编码或网页自动化等领域，而在需要空间推理、多跳规划和实时地图交互的地理空间任务上表现不足。为应对这些挑战，我们提出了一种分级多代理插件式框架——MapAgent，该框架具有定制化的工具集和地理空间推理的代理式支架。与现有的平铺代理方法不同，后者在处理相似但细微不同的地理空间API时往往会压倒LLM，MapAgent将规划与执行分离。高级规划器将复杂查询分解为子目标，这些子目标被路由到专业模块中。对于工具密集型模块，如基于地图的服务，我们设计了一个专用的地图工具代理，能够有效地并行协调相关API，以适配地获取查询相关的地理空间数据，而较简单的模块（如解决方案生成或答案提取）则无需额外的代理开销。这种分级设计减轻了认知负担，提高了工具选择的准确性，并允许跨类似API进行精确协调。我们在四个不同的地理空间基准测试上评估了MapAgent——MapEval-Textual、MapEval-API、MapEval-Visual 和 MapQA，并展示了与现有最先进的工具增强和代理式基线相比的显著改进。我们已在以下网址开源了该框架：this https URL。 

---
# Chatbot To Help Patients Understand Their Health 

**Title (ZH)**: Chatbot 以帮助患者理解其健康状况 

**Authors**: Won Seok Jang, Hieu Tran, Manav Mistry, SaiKiran Gandluri, Yifan Zhang, Sharmin Sultana, Sunjae Kown, Yuan Zhang, Zonghai Yao, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05818)  

**Abstract**: Patients must possess the knowledge necessary to actively participate in their care. We present NoteAid-Chatbot, a conversational AI that promotes patient understanding via a novel 'learning as conversation' framework, built on a multi-agent large language model (LLM) and reinforcement learning (RL) setup without human-labeled data. NoteAid-Chatbot was built on a lightweight LLaMA 3.2 3B model trained in two stages: initial supervised fine-tuning on conversational data synthetically generated using medical conversation strategies, followed by RL with rewards derived from patient understanding assessments in simulated hospital discharge scenarios. Our evaluation, which includes comprehensive human-aligned assessments and case studies, demonstrates that NoteAid-Chatbot exhibits key emergent behaviors critical for patient education, such as clarity, relevance, and structured dialogue, even though it received no explicit supervision for these attributes. Our results show that even simple Proximal Policy Optimization (PPO)-based reward modeling can successfully train lightweight, domain-specific chatbots to handle multi-turn interactions, incorporate diverse educational strategies, and meet nuanced communication objectives. Our Turing test demonstrates that NoteAid-Chatbot surpasses non-expert human. Although our current focus is on healthcare, the framework we present illustrates the feasibility and promise of applying low-cost, PPO-based RL to realistic, open-ended conversational domains, broadening the applicability of RL-based alignment methods. 

**Abstract (ZH)**: 患者必须具备积极参与其治疗过程所需的知识。我们介绍了NoteAid-Chatbot，这是一种通过新颖的“学习即对话”框架促进患者理解的对话型AI，基于多代理大型语言模型和强化学习设置，无需人工标注数据。NoteAid-Chatbot基于经过两阶段训练的轻量级LaMA 3.2 3B模型构建：初始阶段是监督微调，使用医疗对话策略合成对话数据生成，随后是使用源自模拟出院场景中患者理解评估的奖励进行强化学习。我们的评估包括全面的人类对齐评估和案例研究，表明NoteAid-Chatbot即使没有明确为这些属性提供监督，也表现出关键的新兴行为，如清晰度、相关性和结构化对话。我们的结果表明，即使简单的基于PPO的奖励建模也能成功训练轻量级、领域特定的聊天机器人，处理多轮交互，融入多样化的教育策略，并满足精细的沟通目标。我们的图灵测试表明，NoteAid-Chatbot超越了非专家人类。尽管我们当前的重点是医疗保健，但我们提出的方法框架展示了将低成本、基于PPO的强化学习应用于现实的开放性对话领域可行性和潜力，扩展了基于RL对齐方法的应用范围。 

---
# Decision-Focused Learning Enhanced by Automated Feature Engineering for Energy Storage Optimisation 

**Title (ZH)**: 基于自动特征工程的决策导向学习促进能量存储优化 

**Authors**: Nasser Alkhulaifi, Ismail Gokay Dogan, Timothy R. Cargan, Alexander L. Bowler, Direnc Pekaslan, Nicholas J. Watson, Isaac Triguero  

**Link**: [PDF](https://arxiv.org/pdf/2509.05772)  

**Abstract**: Decision-making under uncertainty in energy management is complicated by unknown parameters hindering optimal strategies, particularly in Battery Energy Storage System (BESS) operations. Predict-Then-Optimise (PTO) approaches treat forecasting and optimisation as separate processes, allowing prediction errors to cascade into suboptimal decisions as models minimise forecasting errors rather than optimising downstream tasks. The emerging Decision-Focused Learning (DFL) methods overcome this limitation by integrating prediction and optimisation; however, they are relatively new and have been tested primarily on synthetic datasets or small-scale problems, with limited evidence of their practical viability. Real-world BESS applications present additional challenges, including greater variability and data scarcity due to collection constraints and operational limitations. Because of these challenges, this work leverages Automated Feature Engineering (AFE) to extract richer representations and improve the nascent approach of DFL. We propose an AFE-DFL framework suitable for small datasets that forecasts electricity prices and demand while optimising BESS operations to minimise costs. We validate its effectiveness on a novel real-world UK property dataset. The evaluation compares DFL methods against PTO, with and without AFE. The results show that, on average, DFL yields lower operating costs than PTO and adding AFE further improves the performance of DFL methods by 22.9-56.5% compared to the same models without AFE. These findings provide empirical evidence for DFL's practical viability in real-world settings, indicating that domain-specific AFE enhances DFL and reduces reliance on domain expertise for BESS optimisation, yielding economic benefits with broader implications for energy management systems facing similar challenges. 

**Abstract (ZH)**: 基于不确定性的能源管理决策受到未知参数的困扰，尤其是在电池储能系统（BESS）操作中。预测然后优化（PTO）方法将预测和优化视为独立的过程，这会使得预测误差积累成次优决策。新兴的决策导向学习（DFL）方法通过将预测和优化整合起来克服了这一局限，但它们相对新颖，主要在合成数据集或小型问题上进行了测试，实际可行性证据有限。实际的BESS应用增加了更多挑战，包括由于数据收集限制和运营限制导致的大得多的变异性及数据稀缺性。鉴于这些挑战，本工作利用自动特征工程（AFE）提取更丰富的表示，以改进DFL的初步方法。我们提出了一种适用于小型数据集的AFE-DFL框架，该框架既预测电价和需求，又优化BESS操作以最小化成本。我们在一个新型的英国房产实际数据集上验证了其有效性。评价包括将DFL方法与PTO方法（有和无AFE）进行比较。结果显示，DFL平均降低了运营成本，而添加AFE进一步提高了DFL方法的性能，相比无AFE模型提高了22.9%-56.5%。这些发现为DFL的实际可行性提供了实证证据，表明领域特定的AFE可以增强DFL，并减少对领域专业知识的依赖，对于面临类似挑战的能源管理系统具有经济利益和更广泛的含义。 

---
# DRF: LLM-AGENT Dynamic Reputation Filtering Framework 

**Title (ZH)**: DRF: LLM-AGENT动态声誉过滤框架 

**Authors**: Yuwei Lou, Hao Hu, Shaocong Ma, Zongfei Zhang, Liang Wang, Jidong Ge, Xianping Tao  

**Link**: [PDF](https://arxiv.org/pdf/2509.05764)  

**Abstract**: With the evolution of generative AI, multi - agent systems leveraging large - language models(LLMs) have emerged as a powerful tool for complex tasks. However, these systems face challenges in quantifying agent performance and lack mechanisms to assess agent credibility. To address these issues, we introduce DRF, a dynamic reputation filtering framework. DRF constructs an interactive rating network to quantify agent performance, designs a reputation scoring mechanism to measure agent honesty and capability, and integrates an Upper Confidence Bound - based strategy to enhance agent selection efficiency. Experiments show that DRF significantly improves task completion quality and collaboration efficiency in logical reasoning and code - generation tasks, offering a new approach for multi - agent systems to handle large - scale tasks. 

**Abstract (ZH)**: 基于大型语言模型的多agent系统的动态声誉过滤框架：提升复杂任务完成质量和协作效率 

---
# Hyperbolic Large Language Models 

**Title (ZH)**: 双曲大语言模型 

**Authors**: Sarang Patil, Zeyong Zhang, Yiran Huang, Tengfei Ma, Mengjia Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05757)  

**Abstract**: Large language models (LLMs) have achieved remarkable success and demonstrated superior performance across various tasks, including natural language processing (NLP), weather forecasting, biological protein folding, text generation, and solving mathematical problems. However, many real-world data exhibit highly non-Euclidean latent hierarchical anatomy, such as protein networks, transportation networks, financial networks, brain networks, and linguistic structures or syntactic trees in natural languages. Effectively learning intrinsic semantic entailment and hierarchical relationships from these raw, unstructured input data using LLMs remains an underexplored area. Due to its effectiveness in modeling tree-like hierarchical structures, hyperbolic geometry -- a non-Euclidean space -- has rapidly gained popularity as an expressive latent representation space for complex data modeling across domains such as graphs, images, languages, and multi-modal data. Here, we provide a comprehensive and contextual exposition of recent advancements in LLMs that leverage hyperbolic geometry as a representation space to enhance semantic representation learning and multi-scale reasoning. Specifically, the paper presents a taxonomy of the principal techniques of Hyperbolic LLMs (HypLLMs) in terms of four main categories: (1) hyperbolic LLMs through exp/log maps; (2) hyperbolic fine-tuned models; (3) fully hyperbolic LLMs, and (4) hyperbolic state-space models. We also explore crucial potential applications and outline future research directions. A repository of key papers, models, datasets, and code implementations is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务上取得了显著的成功，并在自然语言处理（NLP）、天气预报、生物蛋白质折叠、文本生成和解决数学问题等方面展示了优越的性能。然而，许多现实世界数据表现出高度非欧几里得的潜在分层结构，如蛋白质网络、交通网络、金融网络、脑网络以及自然语言中的句法树结构。使用LLMs从这些原始的无结构输入数据中有效学习内在语义蕴含和分层关系仍是一个未充分探索的领域。由于其在建模树状分层结构方面的有效性，双曲几何——一种非欧几里得空间——迅速成为跨图形、图像、语言和多模态数据建模的表达性潜在表示空间的热门选择。本文综述了利用双曲几何作为表示空间以增强语义表示学习和多尺度推理的LLMs的最新进展。具体而言，论文按照四个主要类别对双曲几何L大型语言模型（HypLLMs）的主要技术进行分类阐述：（1）基于exp/log映射的双曲几何L大型语言模型；（2）微调的双曲几何模型；（3）完全双曲几何L大型语言模型，以及（4）双曲几何状态空间模型。此外，本文还探讨了关键潜在应用并概述了未来的研究方向。有关关键论文、模型、数据集和代码实现的资源，请访问此网址：https://xxxxxx。 

---
# Towards Meta-Cognitive Knowledge Editing for Multimodal LLMs 

**Title (ZH)**: 向元认知知识编辑迈向多模态LLM 

**Authors**: Zhaoyu Fan, Kaihang Pan, Mingze Zhou, Bosheng Qin, Juncheng Li, Shengyu Zhang, Wenqiao Zhang, Siliang Tang, Fei Wu, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05714)  

**Abstract**: Knowledge editing enables multimodal large language models (MLLMs) to efficiently update outdated or incorrect information. However, existing benchmarks primarily emphasize cognitive-level modifications while lacking a focus on deeper meta-cognitive processes. To bridge this gap, we introduce CogEdit, a novel benchmark designed to evaluate MLLMs' meta-cognitive knowledge editing abilities across three levels: (1) Counterfactual-Driven Editing, assessing self-awareness of knowledge correctness changes; (2) Boundary Constraint Editing, ensuring appropriate generalization without unintended interference; and (3) Noise-Robust Editing, promoting reflective evaluation of uncertain information. To advance meta-cognitive editing, we propose MIND (Meta-cognitive INtegrated Dynamic Knowledge Editing), a framework that constructs a meta-knowledge memory for self-awareness, employs game-theoretic interactions to monitor knowledge activation, and incorporates label refinement for noise-robust updates. Extensive experiments show that MIND significantly outperforms existing cognitive editing approaches, achieving strong performance on both traditional and meta-cognitive knowledge editing benchmarks. 

**Abstract (ZH)**: 知识编辑使多模态大型语言模型（MLLMs）能够高效地更新过时或不正确的信息。然而，现有的基准主要侧重于认知层面的修改，缺乏对更深层次元认知过程的关注。为填补这一缺口，我们引入了CogEdit，一个旨在评估MLLMs在三个层面的元认知知识编辑能力的新基准：（1）反事实驱动编辑，评估知识正确性变化的自我意识；（2）边界约束编辑，确保适当的泛化而不产生意外干扰；（3）噪声鲁棒编辑，促进对不确定信息的反思性评估。为了推进元认知编辑，我们提出了一种MIND（元认知集成动态知识编辑）框架，该框架构建了一个元知识记忆以提供自我意识，采用博弈论交互来监控知识激活，并结合标签精炼来实现噪声鲁棒更新。大量实验表明，MIND 在认知编辑方法中表现出显著的优势，并在传统和元认知知识编辑基准上取得了优异的表现。 

---
# MSRFormer: Road Network Representation Learning using Multi-scale Feature Fusion of Heterogeneous Spatial Interactions 

**Title (ZH)**: MSRFormer：利用异质空间交互多尺度特征融合的路网表示学习 

**Authors**: Jian Yang, Jiahui Wu, Li Fang, Hongchao Fan, Bianying Zhang, Huijie Zhao, Guangyi Yang, Rui Xin, Xiong You  

**Link**: [PDF](https://arxiv.org/pdf/2509.05685)  

**Abstract**: Transforming road network data into vector representations using deep learning has proven effective for road network analysis. However, urban road networks' heterogeneous and hierarchical nature poses challenges for accurate representation learning. Graph neural networks, which aggregate features from neighboring nodes, often struggle due to their homogeneity assumption and focus on a single structural scale. To address these issues, this paper presents MSRFormer, a novel road network representation learning framework that integrates multi-scale spatial interactions by addressing their flow heterogeneity and long-distance dependencies. It uses spatial flow convolution to extract small-scale features from large trajectory datasets, and identifies scale-dependent spatial interaction regions to capture the spatial structure of road networks and flow heterogeneity. By employing a graph transformer, MSRFormer effectively captures complex spatial dependencies across multiple scales. The spatial interaction features are fused using residual connections, which are fed to a contrastive learning algorithm to derive the final road network representation. Validation on two real-world datasets demonstrates that MSRFormer outperforms baseline methods in two road network analysis tasks. The performance gains of MSRFormer suggest the traffic-related task benefits more from incorporating trajectory data, also resulting in greater improvements in complex road network structures with up to 16% improvements compared to the most competitive baseline method. This research provides a practical framework for developing task-agnostic road network representation models and highlights distinct association patterns of the interplay between scale effects and flow heterogeneity of spatial interactions. 

**Abstract (ZH)**: 使用深度学习将道路网络数据转换为向量表示在道路网络分析中已被证明有效。然而，城市道路网络的异构性和层次性对准确的表示学习提出了挑战。为应对这些问题，本文提出MSRFormer，一种通过解决空间流异构性和长距离依赖性的多尺度空间交互综合框架。它利用空间流卷积从大规模轨迹数据集中提取小尺度特征，并识别尺度相关的空间交互区域以捕获道路网络的空间结构和流异构性。通过采用图变换器，MSRFormer有效捕捉了多尺度的空间依赖关系。通过残差连接融合空间交互特征，并馈送到对比学习算法以获得最终的道路网络表示。在两个真实世界数据集上的验证表明，MSRFormer在两种道路网络分析任务中优于基准方法。MSRFormer的性能提升表明，与交通相关的任务可以从结合轨迹数据中获益更多，特别是在复杂道路网络结构中可获得高达16%的性能提升，超越最竞争的基准方法。该研究提供了一种适用于多种任务的道路网络表示模型的实际框架，并突出显示了尺度效应和空间交互流异构性交互的不同关联模式。 

---
# OccVLA: Vision-Language-Action Model with Implicit 3D Occupancy Supervision 

**Title (ZH)**: OccVLA：带有隐式3D占据监督的视觉-语言-动作模型 

**Authors**: Ruixun Liu, Lingyu Kong, Derun Li, Hang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.05578)  

**Abstract**: Multimodal large language models (MLLMs) have shown strong vision-language reasoning abilities but still lack robust 3D spatial understanding, which is critical for autonomous driving. This limitation stems from two key challenges: (1) the difficulty of constructing accessible yet effective 3D representations without expensive manual annotations, and (2) the loss of fine-grained spatial details in VLMs due to the absence of large-scale 3D vision-language pretraining. To address these challenges, we propose OccVLA, a novel framework that integrates 3D occupancy representations into a unified multimodal reasoning process. Unlike prior approaches that rely on explicit 3D inputs, OccVLA treats dense 3D occupancy as both a predictive output and a supervisory signal, enabling the model to learn fine-grained spatial structures directly from 2D visual inputs. The occupancy predictions are regarded as implicit reasoning processes and can be skipped during inference without performance degradation, thereby adding no extra computational overhead. OccVLA achieves state-of-the-art results on the nuScenes benchmark for trajectory planning and demonstrates superior performance on 3D visual question-answering tasks, offering a scalable, interpretable, and fully vision-based solution for autonomous driving. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在视觉语言推理方面表现出色，但仍缺乏稳健的三维空间理解能力，这在自动驾驶中至关重要。这一限制源于两大关键挑战：（1）在没有昂贵的手动注释的情况下，难以构建可访问且有效的三维表示，以及（2）由于缺乏大规模三维视觉语言预训练，视觉语言模型中会丢失详细的空间细节。为解决这些问题，我们提出了一种新的框架OccVLA，该框架将三维占有率表示整合到统一的多模态推理过程中。与依赖显式三维输入的先前方法不同，OccVLA 将密集的三维占有率作为预测输出和监督信号来处理，使模型能够直接从二维视觉输入中学习精细的空间结构。占有率预测被视为隐式的推理过程，在推断过程中可以跳过而不会降低性能，从而不再增加额外的计算开销。OccVLA 在 nuScenes 轨迹规划基准上达到了最佳结果，并在三维视觉问答任务上表现出优越性能，提供了一种可扩展、可解析且完全基于视觉的自动驾驶解决方案。 

---
# TreeGPT: A Novel Hybrid Architecture for Abstract Syntax Tree Processing with Global Parent-Child Aggregation 

**Title (ZH)**: TreeGPT：一种新型全局父节点-子节点聚合的混合架构用于抽象语法树处理 

**Authors**: Zixi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.05550)  

**Abstract**: We introduce TreeGPT, a novel neural architecture that combines transformer-based attention mechanisms with global parent-child aggregation for processing Abstract Syntax Trees (ASTs) in neural program synthesis tasks. Unlike traditional approaches that rely solely on sequential processing or graph neural networks, TreeGPT employs a hybrid design that leverages both self-attention for capturing local dependencies and a specialized Tree Feed-Forward Network (TreeFFN) for modeling hierarchical tree structures through iterative message passing.
The core innovation lies in our Global Parent-Child Aggregation mechanism, formalized as: $$h_i^{(t+1)} = \sigma \Big( h_i^{(0)} + W_{pc} \sum_{(p,c) \in E_i} f(h_p^{(t)}, h_c^{(t)}) + b \Big)$$ where $h_i^{(t)}$ represents the hidden state of node $i$ at iteration $t$, $E_i$ denotes all parent-child edges involving node $i$, and $f(h_p, h_c)$ is an edge aggregation function. This formulation enables each node to progressively aggregate information from the entire tree structure through $T$ iterations.
Our architecture integrates optional enhancements including gated aggregation with learnable edge weights, residual connections for gradient stability, and bidirectional propagation for capturing both bottom-up and top-down dependencies. We evaluate TreeGPT on the ARC Prize 2025 dataset, a challenging visual reasoning benchmark requiring abstract pattern recognition and rule inference. Experimental results demonstrate that TreeGPT achieves 96\% accuracy, significantly outperforming transformer baselines (1.3\%), large-scale models like Grok-4 (15.9\%), and specialized program synthesis methods like SOAR (52\%) while using only 1.5M parameters. Our comprehensive ablation study reveals that edge projection is the most critical component, with the combination of edge projection and gating achieving optimal performance. 

**Abstract (ZH)**: TreeGPT：结合全局父节点-子节点聚合机制的变压器架构用于神经程序合成任务中的抽象语法树处理 

---
# From Image Generation to Infrastructure Design: a Multi-agent Pipeline for Street Design Generation 

**Title (ZH)**: 从图像生成到基础设施设计：面向街道设计生成的多智能体管道 

**Authors**: Chenguang Wang, Xiang Yan, Yilong Dai, Ziyi Wang, Susu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05469)  

**Abstract**: Realistic visual renderings of street-design scenarios are essential for public engagement in active transportation planning. Traditional approaches are labor-intensive, hindering collective deliberation and collaborative decision-making. While AI-assisted generative design shows transformative potential by enabling rapid creation of design scenarios, existing generative approaches typically require large amounts of domain-specific training data and struggle to enable precise spatial variations of design/configuration in complex street-view scenes. We introduce a multi-agent system that edits and redesigns bicycle facilities directly on real-world street-view imagery. The framework integrates lane localization, prompt optimization, design generation, and automated evaluation to synthesize realistic, contextually appropriate designs. Experiments across diverse urban scenarios demonstrate that the system can adapt to varying road geometries and environmental conditions, consistently yielding visually coherent and instruction-compliant results. This work establishes a foundation for applying multi-agent pipelines to transportation infrastructure planning and facility design. 

**Abstract (ZH)**: 现实街景的视觉渲染对于促进公众参与主动交通规划至关重要。传统方法劳动密集型，阻碍了集体讨论和合作决策。虽然AI辅助生成设计展现出变革性的潜力，能够快速生成设计方案，但现有生成方法通常需要大量领域特定的训练数据，并难以在复杂街道视图场景中实现精确的空间变化。我们提出了一种多agent系统，可以直接在真实的街景图像上编辑和重新设计自行车设施。该框架集成了车道定位、提示优化、设计生成和自动化评估，以合成现实且上下文适当的设计。在多种城市场景的实验中证明，该系统能够适应不同的道路几何形状和环境条件，始终保持视觉一致性和指令一致性。本研究为将多agent流水线应用于交通基础设施规划和设施设计奠定了基础。 

---
# Murphys Laws of AI Alignment: Why the Gap Always Wins 

**Title (ZH)**: AI对齐的墨菲定律：为什么差距总是胜出 

**Authors**: Madhava Gaikwad  

**Link**: [PDF](https://arxiv.org/pdf/2509.05381)  

**Abstract**: Large language models are increasingly aligned to human preferences through reinforcement learning from human feedback (RLHF) and related methods such as Direct Preference Optimization (DPO), Constitutional AI, and RLAIF. While effective, these methods exhibit recurring failure patterns i.e., reward hacking, sycophancy, annotator drift, and misgeneralization. We introduce the concept of the Alignment Gap, a unifying lens for understanding recurring failures in feedback-based alignment. Using a KL-tilting formalism, we illustrate why optimization pressure tends to amplify divergence between proxy rewards and true human intent. We organize these failures into a catalogue of Murphys Laws of AI Alignment, and propose the Alignment Trilemma as a way to frame trade-offs among optimization strength, value capture, and generalization. Small-scale empirical studies serve as illustrative support. Finally, we propose the MAPS framework (Misspecification, Annotation, Pressure, Shift) as practical design levers. Our contribution is not a definitive impossibility theorem but a perspective that reframes alignment debates around structural limits and trade-offs, offering clearer guidance for future design. 

**Abstract (ZH)**: 大型语言模型通过人类反馈强化学习（RLHF）及相关方法（如直接偏好优化（DPO）、宪法AI和RLAIF）越来越接近人类偏好。尽管有效，这些方法表现出重复出现的失败模式，包括奖励作弊、讨好行为、注释员偏差和误泛化。我们引入了对齐缺口的概念，这是一种统一的视角，用于理解基于反馈对齐中的重复失败。通过KL-倾斜的形式主义，我们解释了为什么优化压力倾向于放大代理奖励与真实人类意图之间的差异。我们将这些失败归类为AI对齐的墨菲定律，提出了对齐三难困境作为一种方式来界定优化强度、价值捕获和泛化的权衡。我们通过小规模的实证研究提供说明性的支持。最后，我们提出了MAPS框架（错误规定、注释、压力、变化）作为实际设计杠杆。我们的贡献不是一项完备的不可能定理，而是一种视角，重新框定了对齐辩论的结构性限制和权衡，为未来的设计提供了更清晰的指导。 

---
# Code Like Humans: A Multi-Agent Solution for Medical Coding 

**Title (ZH)**: 像人类一样编码：一种多代理医疗编码解决方案 

**Authors**: Andreas Motzfeldt, Joakim Edin, Casper L. Christensen, Christian Hardmeier, Lars Maaløe, Anna Rogers  

**Link**: [PDF](https://arxiv.org/pdf/2509.05378)  

**Abstract**: In medical coding, experts map unstructured clinical notes to alphanumeric codes for diagnoses and procedures. We introduce Code Like Humans: a new agentic framework for medical coding with large language models. It implements official coding guidelines for human experts, and it is the first solution that can support the full ICD-10 coding system (+70K labels). It achieves the best performance to date on rare diagnosis codes (fine-tuned discriminative classifiers retain an advantage for high-frequency codes, to which they are limited). Towards future work, we also contribute an analysis of system performance and identify its `blind spots' (codes that are systematically undercoded). 

**Abstract (ZH)**: 人类like的医疗编码：一种基于大语言模型的自主医疗编码框架 

---
# Characterizing Fitness Landscape Structures in Prompt Engineering 

**Title (ZH)**: characterizing 优化提示工程中适应度景观结构的研究 

**Authors**: Arend Hintze  

**Link**: [PDF](https://arxiv.org/pdf/2509.05375)  

**Abstract**: While prompt engineering has emerged as a crucial technique for optimizing large language model performance, the underlying optimization landscape remains poorly understood. Current approaches treat prompt optimization as a black-box problem, applying sophisticated search algorithms without characterizing the landscape topology they navigate. We present a systematic analysis of fitness landscape structures in prompt engineering using autocorrelation analysis across semantic embedding spaces. Through experiments on error detection tasks with two distinct prompt generation strategies -- systematic enumeration (1,024 prompts) and novelty-driven diversification (1,000 prompts) -- we reveal fundamentally different landscape topologies. Systematic prompt generation yields smoothly decaying autocorrelation, while diversified generation exhibits non-monotonic patterns with peak correlation at intermediate semantic distances, indicating rugged, hierarchically structured landscapes. Task-specific analysis across 10 error detection categories reveals varying degrees of ruggedness across different error types. Our findings provide an empirical foundation for understanding the complexity of optimization in prompt engineering landscapes. 

**Abstract (ZH)**: 而提示工程作为一个优化大规模语言模型性能的关键技术已经崭露头角，但其背后的优化景观仍不甚明了。当前的方法将提示优化视为一个黑盒问题，应用复杂的搜索算法而不表征他们导航的景观拓扑结构。我们利用自相关分析跨语义嵌入空间系统分析提示工程中的适应度景观结构。通过在两种不同的提示生成策略（系统枚举1,024个提示和 novelty-driven 多样化1,000个提示）下进行错误检测任务的实验，我们揭示了根本不同的景观拓扑结构。系统提示生成产生平滑衰减的自相关性，而多样化生成表现出非单调模式，其在中间语义距离处出现峰值相关性，表明这些景观具有崎岖且分层结构。针对10个错误检测类别进行的任务特定分析揭示了不同类型错误在崎岖度上的差异。我们的研究结果为理解提示工程景观中优化的复杂性提供了经验基础。 

---
# SasAgent: Multi-Agent AI System for Small-Angle Scattering Data Analysis 

**Title (ZH)**: SasAgent: 多Agent人工智能系统用于小角度散射数据分析 

**Authors**: Lijie Ding, Changwoo Do  

**Link**: [PDF](https://arxiv.org/pdf/2509.05363)  

**Abstract**: We introduce SasAgent, a multi-agent AI system powered by large language models (LLMs) that automates small-angle scattering (SAS) data analysis by leveraging tools from the SasView software and enables user interaction via text input. SasAgent features a coordinator agent that interprets user prompts and delegates tasks to three specialized agents for scattering length density (SLD) calculation, synthetic data generation, and experimental data fitting. These agents utilize LLM-friendly tools to execute tasks efficiently. These tools, including the model data tool, Retrieval-Augmented Generation (RAG) documentation tool, bump fitting tool, and SLD calculator tool, are derived from the SasView Python library. A user-friendly Gradio-based interface enhances user accessibility. Through diverse examples, we demonstrate SasAgent's ability to interpret complex prompts, calculate SLDs, generate accurate scattering data, and fit experimental datasets with high precision. This work showcases the potential of LLM-driven AI systems to streamline scientific workflows and enhance automation in SAS research. 

**Abstract (ZH)**: 基于大型语言模型的多agent系统SasAgent及其在小角散射数据自动分析中的应用 

---
# Benchmarking Large Language Models for Personalized Guidance in AI-Enhanced Learning 

**Title (ZH)**: 大型语言模型在人工智能增强学习个性化指导中的benchmarking研究 

**Authors**: Bo Yuan, Jiazi Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05346)  

**Abstract**: While Large Language Models (LLMs) are increasingly envisioned as intelligent assistants for personalized learning, systematic head-to-head evaluations within authentic learning scenarios remain limited. This study conducts an empirical comparison of three state-of-the-art LLMs on a tutoring task that simulates a realistic learning setting. Using a dataset comprising a student's answers to ten questions of mixed formats with correctness labels, each LLM is required to (i) analyze the quiz to identify underlying knowledge components, (ii) infer the student's mastery profile, and (iii) generate targeted guidance for improvement. To mitigate subjectivity and evaluator bias, we employ Gemini as a virtual judge to perform pairwise comparisons along various dimensions: accuracy, clarity, actionability, and appropriateness. Results analyzed via the Bradley-Terry model indicate that GPT-4o is generally preferred, producing feedback that is more informative and better structured than its counterparts, while DeepSeek-V3 and GLM-4.5 demonstrate intermittent strengths but lower consistency. These findings highlight the feasibility of deploying LLMs as advanced teaching assistants for individualized support and provide methodological guidance for future empirical research on LLM-driven personalized learning. 

**Abstract (ZH)**: 虽然大型语言模型（LLMs）越来越被视为个性化学习的智能助手，但在真实学习场景下系统的直接对比仍然有限。本研究通过一个模拟现实学习环境的辅导任务，对三种最先进的LLM进行了实证比较。使用包含一组学生对十道混合格式问题的回答及其正确性标签的数据集，每种LLM需完成以下任务：（i）分析测验以识别潜在的知识组件，（ii）推断学生的掌握情况，以及（iii）生成针对性的改进指导。为减少主观性和评价者偏见，我们利用Gemini作为虚拟评委，从准确度、清晰度、可行性和恰当性等多个维度进行两两对比。通过Bradley-Terry模型分析结果表明，GPT-4o通常更受欢迎，其反馈信息更丰富且结构更严谨，而DeepSeek-V3和GLM-4.5则显示出间歇性的优势但一致性较低。这些发现突显了部署LLM作为高级教学助手用于个性化支持的可行性，并为未来基于LLM的个性化学习实证研究提供了方法学指导。 

---
# MVRS: The Multimodal Virtual Reality Stimuli-based Emotion Recognition Dataset 

**Title (ZH)**: MVRS：基于多模态虚拟现实刺激的情绪识别数据集 

**Authors**: Seyed Muhammad Hossein Mousavi, Atiye Ilanloo  

**Link**: [PDF](https://arxiv.org/pdf/2509.05330)  

**Abstract**: Automatic emotion recognition has become increasingly important with the rise of AI, especially in fields like healthcare, education, and automotive systems. However, there is a lack of multimodal datasets, particularly involving body motion and physiological signals, which limits progress in the field. To address this, the MVRS dataset is introduced, featuring synchronized recordings from 13 participants aged 12 to 60 exposed to VR based emotional stimuli (relaxation, fear, stress, sadness, joy). Data were collected using eye tracking (via webcam in a VR headset), body motion (Kinect v2), and EMG and GSR signals (Arduino UNO), all timestamp aligned. Participants followed a unified protocol with consent and questionnaires. Features from each modality were extracted, fused using early and late fusion techniques, and evaluated with classifiers to confirm the datasets quality and emotion separability, making MVRS a valuable contribution to multimodal affective computing. 

**Abstract (ZH)**: 自动情绪识别随着AI的兴起变得越来越重要，特别是在医疗保健、教育和汽车系统等领域。然而，缺乏多模态数据集，特别是涉及身体运动和生理信号的数据集，限制了该领域的发展。为了解决这一问题，引入了MVRS数据集，该数据集包含了12至60岁之间的13名参与者在基于VR的情绪刺激（放松、恐惧、压力、悲伤、快乐）下的同步记录。数据通过VR头盔内置 webcam的眼动追踪、Kinect v2的身体运动以及Arduino UNO的EMG和GSR信号采集，并且时间戳对齐。参与者遵循统一的协议并在参与前签署了知情同意书并填写了问卷。每个模态的特征被提取，并使用早期融合和晚期融合技术进行融合，然后通过分类器进行评估以确认数据集的质量和情绪可分性，使MVRS成为多模态情感计算的重要贡献。 

---
# SynDelay: A Synthetic Dataset for Delivery Delay Prediction 

**Title (ZH)**: SynDelay: 用于配送延迟预测的合成数据集 

**Authors**: Liming Xu, Yunbo Long, Alexandra Brintrup  

**Link**: [PDF](https://arxiv.org/pdf/2509.05325)  

**Abstract**: Artificial intelligence (AI) is transforming supply chain management, yet progress in predictive tasks -- such as delivery delay prediction -- remains constrained by the scarcity of high-quality, openly available datasets. Existing datasets are often proprietary, small, or inconsistently maintained, hindering reproducibility and benchmarking. We present SynDelay, a synthetic dataset designed for delivery delay prediction. Generated using an advanced generative model trained on real-world data, SynDelay preserves realistic delivery patterns while ensuring privacy. Although not entirely free of noise or inconsistencies, it provides a challenging and practical testbed for advancing predictive modelling. To support adoption, we provide baseline results and evaluation metrics as initial benchmarks, serving as reference points rather than state-of-the-art claims. SynDelay is publicly available through the Supply Chain Data Hub, an open initiative promoting dataset sharing and benchmarking in supply chain AI. We encourage the community to contribute datasets, models, and evaluation practices to advance research in this area. All code is openly accessible at this https URL. 

**Abstract (ZH)**: 人工智能（AI）正在变革供应链管理，但在预测任务（如交付延迟预测）方面，由于高质量、公开可用的数据集匮乏，进展仍然受限。现有数据集往往具有专有性、规模小或维护不一致，阻碍了可再现性和基准测试。我们提出了SynDelay，这是一个用于交付延迟预测的合成数据集。该数据集使用基于实际数据训练的高级生成模型生成，保留了现实的交付模式并确保了隐私性。尽管不完全无噪声和无一致性，SynDelay 仍提供了一个具有挑战性和实用性的测试平台，用于推进预测建模。为了促进采用，我们提供了基准结果和评估指标作为初始基准，作为参考点而非前沿claim。SynDelay 通过供应链数据联盟公开提供，这是一个促进供应链AI领域数据集共享和基准测试的开放倡议。我们鼓励社区贡献数据集、模型和评估实践，以促进该领域的研究。所有代码均可在该网址访问。 

---
# Perception Graph for Cognitive Attack Reasoning in Augmented Reality 

**Title (ZH)**: 感知图 dla 认知攻击推理在增强现实中的应用 

**Authors**: Rongqian Chen, Shu Hong, Rifatul Islam, Mahdi Imani, G. Gary Tan, Tian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2509.05324)  

**Abstract**: Augmented reality (AR) systems are increasingly deployed in tactical environments, but their reliance on seamless human-computer interaction makes them vulnerable to cognitive attacks that manipulate a user's perception and severely compromise user decision-making. To address this challenge, we introduce the Perception Graph, a novel model designed to reason about human perception within these systems. Our model operates by first mimicking the human process of interpreting key information from an MR environment and then representing the outcomes using a semantically meaningful structure. We demonstrate how the model can compute a quantitative score that reflects the level of perception distortion, providing a robust and measurable method for detecting and analyzing the effects of such cognitive attacks. 

**Abstract (ZH)**: 增强现实（AR）系统在战术环境中越来越广泛地应用，但由于其对无缝人机交互的依赖，使其容易受到操纵用户感知的认知攻击，严重影响用户决策。为应对这一挑战，我们提出了感知图这一新型模型，该模型旨在在这些系统中推理人类感知。该模型首先模仿人类从MR环境中解读关键信息的过程，然后通过语义上有意义的结构表示结果。我们展示了该模型如何计算反映感知扭曲程度的量化得分，提供了一种 robust 和可测量的方法来检测和分析此类认知攻击的影响。 

---
# Attention of a Kiss: Exploring Attention Maps in Video Diffusion for XAIxArts 

**Title (ZH)**: 一顿热吻的关注：探索视频扩散中的注意力图在XAIxArts中的应用 

**Authors**: Adam Cole, Mick Grierson  

**Link**: [PDF](https://arxiv.org/pdf/2509.05323)  

**Abstract**: This paper presents an artistic and technical investigation into the attention mechanisms of video diffusion transformers. Inspired by early video artists who manipulated analog video signals to create new visual aesthetics, this study proposes a method for extracting and visualizing cross-attention maps in generative video models. Built on the open-source Wan model, our tool provides an interpretable window into the temporal and spatial behavior of attention in text-to-video generation. Through exploratory probes and an artistic case study, we examine the potential of attention maps as both analytical tools and raw artistic material. This work contributes to the growing field of Explainable AI for the Arts (XAIxArts), inviting artists to reclaim the inner workings of AI as a creative medium. 

**Abstract (ZH)**: 本文对视频扩散变换器中的注意力机制进行了艺术和技术研究。受早期视频艺术家通过操控模拟视频信号创造新视觉美学的启发，本研究提出了在生成视频模型中提取和可视化交叉注意力图的方法。基于开源的Wan模型，我们的工具为文本到视频生成中注意力的时空行为提供了可解释的窗口。通过探索性探针和艺术案例研究，我们探讨了注意力图作为分析工具和原始艺术材料的潜力。本文为艺术领域的可解释人工智能（XAIxArts）领域的发展做出了贡献，邀请艺术家重新掌握人工智能内部机制作为创作媒介的权利。 

---
# H$_{2}$OT: Hierarchical Hourglass Tokenizer for Efficient Video Pose Transformers 

**Title (ZH)**: H$_{2}$OT：分层_hourglass_令牌化器用于高效的视频姿态变换器 

**Authors**: Wenhao Li, Mengyuan Liu, Hong Liu, Pichao Wang, Shijian Lu, Nicu Sebe  

**Link**: [PDF](https://arxiv.org/pdf/2509.06956)  

**Abstract**: Transformers have been successfully applied in the field of video-based 3D human pose estimation. However, the high computational costs of these video pose transformers (VPTs) make them impractical on resource-constrained devices. In this paper, we present a hierarchical plug-and-play pruning-and-recovering framework, called Hierarchical Hourglass Tokenizer (H$_{2}$OT), for efficient transformer-based 3D human pose estimation from videos. H$_{2}$OT begins with progressively pruning pose tokens of redundant frames and ends with recovering full-length sequences, resulting in a few pose tokens in the intermediate transformer blocks and thus improving the model efficiency. It works with two key modules, namely, a Token Pruning Module (TPM) and a Token Recovering Module (TRM). TPM dynamically selects a few representative tokens to eliminate the redundancy of video frames, while TRM restores the detailed spatio-temporal information based on the selected tokens, thereby expanding the network output to the original full-length temporal resolution for fast inference. Our method is general-purpose: it can be easily incorporated into common VPT models on both seq2seq and seq2frame pipelines while effectively accommodating different token pruning and recovery strategies. In addition, our H$_{2}$OT reveals that maintaining the full pose sequence is unnecessary, and a few pose tokens of representative frames can achieve both high efficiency and estimation accuracy. Extensive experiments on multiple benchmark datasets demonstrate both the effectiveness and efficiency of the proposed method. Code and models are available at this https URL. 

**Abstract (ZH)**: 基于视频的3D人体姿态估计中高效Transformer的方法：分层即插即用剪裁与恢复框架（H$_{2}$OT） 

---
# Deep Reactive Policy: Learning Reactive Manipulator Motion Planning for Dynamic Environments 

**Title (ZH)**: 深度反应性策略：学习动态环境下的反应性操纵器运动规划 

**Authors**: Jiahui Yang, Jason Jingzhou Liu, Yulong Li, Youssef Khaky, Kenneth Shaw, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2509.06953)  

**Abstract**: Generating collision-free motion in dynamic, partially observable environments is a fundamental challenge for robotic manipulators. Classical motion planners can compute globally optimal trajectories but require full environment knowledge and are typically too slow for dynamic scenes. Neural motion policies offer a promising alternative by operating in closed-loop directly on raw sensory inputs but often struggle to generalize in complex or dynamic settings. We propose Deep Reactive Policy (DRP), a visuo-motor neural motion policy designed for reactive motion generation in diverse dynamic environments, operating directly on point cloud sensory input. At its core is IMPACT, a transformer-based neural motion policy pretrained on 10 million generated expert trajectories across diverse simulation scenarios. We further improve IMPACT's static obstacle avoidance through iterative student-teacher finetuning. We additionally enhance the policy's dynamic obstacle avoidance at inference time using DCP-RMP, a locally reactive goal-proposal module. We evaluate DRP on challenging tasks featuring cluttered scenes, dynamic moving obstacles, and goal obstructions. DRP achieves strong generalization, outperforming prior classical and neural methods in success rate across both simulated and real-world settings. Video results and code available at this https URL 

**Abstract (ZH)**: 在动态部分可观测环境中生成无碰撞运动路径是机器人操作臂面临的基本挑战。经典运动规划器可以计算全局最优轨迹，但需要完全了解环境，并且通常在动态场景下速度过慢。神经运动策略通过直接在原始感官输入上闭环运行提供了一种有前景的替代方案，但在复杂或动态环境中常常难以泛化。我们提出了Deep Reactive Policy (DRP) 神经运动策略，专门设计用于在多变的动态环境中实时生成运动，直接处理点云感知输入。其核心是IMPACT，一种基于转换器的预训练神经运动策略，在多样化的模拟场景中生成了100万条专家轨迹。我们进一步通过迭代的学生-教师微调方法改进了IMPACT的静态障碍物回避能力。此外，在推理阶段，我们使用DCP-RMP局部反应性目标提议模块增强策略的动态障碍物回避能力。我们对包含杂乱场景、动态移动障碍物和目标障碍的任务评估了DRP，DRP表现出强大的泛化能力，在模拟和真实世界设置中的成功率均优于先前的经典和神经方法。视频结果和代码可在以下链接查看：this https URL。 

---
# Interleaving Reasoning for Better Text-to-Image Generation 

**Title (ZH)**: 交错推理以提高文本到图像生成质量 

**Authors**: Wenxuan Huang, Shuang Chen, Zheyong Xie, Shaosheng Cao, Shixiang Tang, Yufan Shen, Qingyu Yin, Wenbo Hu, Xiaoman Wang, Yuntian Tang, Junbo Qiao, Yue Guo, Yao Hu, Zhenfei Yin, Philip Torr, Yu Cheng, Wanli Ouyang, Shaohui Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06945)  

**Abstract**: Unified multimodal understanding and generation models recently have achieve significant improvement in image generation capability, yet a large gap remains in instruction following and detail preservation compared to systems that tightly couple comprehension with generation such as GPT-4o. Motivated by recent advances in interleaving reasoning, we explore whether such reasoning can further improve Text-to-Image (T2I) generation. We introduce Interleaving Reasoning Generation (IRG), a framework that alternates between text-based thinking and image synthesis: the model first produces a text-based thinking to guide an initial image, then reflects on the result to refine fine-grained details, visual quality, and aesthetics while preserving semantics. To train IRG effectively, we propose Interleaving Reasoning Generation Learning (IRGL), which targets two sub-goals: (1) strengthening the initial think-and-generate stage to establish core content and base quality, and (2) enabling high-quality textual reflection and faithful implementation of those refinements in a subsequent image. We curate IRGL-300K, a dataset organized into six decomposed learning modes that jointly cover learning text-based thinking, and full thinking-image trajectories. Starting from a unified foundation model that natively emits interleaved text-image outputs, our two-stage training first builds robust thinking and reflection, then efficiently tunes the IRG pipeline in the full thinking-image trajectory data. Extensive experiments show SoTA performance, yielding absolute gains of 5-10 points on GenEval, WISE, TIIF, GenAI-Bench, and OneIG-EN, alongside substantial improvements in visual quality and fine-grained fidelity. The code, model weights and datasets will be released in: this https URL . 

**Abstract (ZH)**: 统一多模态理解和生成模型在图像生成能力方面取得了显著进步，但在指令遵循和细节保留方面仍与紧密耦合理解和生成的系统（如GPT-4o）存在较大差距。受交错推理最近进展的启发，我们探索此类推理是否能进一步改善文本到图像（T2I）生成。我们引入交错推理生成（IRG）框架，该框架交替进行基于文本的思考和图像合成：模型首先生成基于文本的思考以引导初始图像，然后反思结果，进一步细化图像的细节、视觉质量及美观性，同时保留语义。为了有效训练IRG，我们提出了交错推理生成学习（IRGL），该方法旨在实现两个子目标：（1）强化初始思考与生成阶段，以建立核心内容和基本质量；（2）在后续图像中实现高质量的文本反思及忠实的细节改进。我们精心策划了包含六种分解学习模式的IRGL-300K数据集，这些模式共同涵盖了基于文本的思考学习和完整的思考-图像轨迹。从一款原生输出交错文本-图像输出的统一基础模型出发，我们的两阶段训练首先构建稳健的思考与反思，然后高效地调整IRG管道在完整的思考-图像轨迹数据中。广泛实验表明，该方法在GenEval、WISE、TIIF、GenAI-Bench和OneIG-EN等基准测试中达到了最先进的性能，同时还显著提高了视觉质量和细节保真度。代码、模型权重和数据集将在以下链接发布：this https URL。 

---
# From Noise to Narrative: Tracing the Origins of Hallucinations in Transformers 

**Title (ZH)**: 从噪声到叙事：探究Transformer中幻觉的起源 

**Authors**: Praneet Suresh, Jack Stanley, Sonia Joseph, Luca Scimeca, Danilo Bzdok  

**Link**: [PDF](https://arxiv.org/pdf/2509.06938)  

**Abstract**: As generative AI systems become competent and democratized in science, business, and government, deeper insight into their failure modes now poses an acute need. The occasional volatility in their behavior, such as the propensity of transformer models to hallucinate, impedes trust and adoption of emerging AI solutions in high-stakes areas. In the present work, we establish how and when hallucinations arise in pre-trained transformer models through concept representations captured by sparse autoencoders, under scenarios with experimentally controlled uncertainty in the input space. Our systematic experiments reveal that the number of semantic concepts used by the transformer model grows as the input information becomes increasingly unstructured. In the face of growing uncertainty in the input space, the transformer model becomes prone to activate coherent yet input-insensitive semantic features, leading to hallucinated output. At its extreme, for pure-noise inputs, we identify a wide variety of robustly triggered and meaningful concepts in the intermediate activations of pre-trained transformer models, whose functional integrity we confirm through targeted steering. We also show that hallucinations in the output of a transformer model can be reliably predicted from the concept patterns embedded in transformer layer activations. This collection of insights on transformer internal processing mechanics has immediate consequences for aligning AI models with human values, AI safety, opening the attack surface for potential adversarial attacks, and providing a basis for automatic quantification of a model's hallucination risk. 

**Abstract (ZH)**: 随着生成式AI系统在科学、商业和政府领域的能力和普及程度不断提高，对它们的失败模式进行更深入的洞察现在变得尤为迫切。它们偶尔在行为上的波动性，如变压器模型的幻觉倾向，阻碍了在高风险领域对新兴AI解决方案的信任和采用。在本项工作中，我们通过稀疏自编码器捕获的概念表示，探讨了在输入空间实验控制下的不确定性情景中幻觉是如何在预训练变压器模型中出现的。系统性的实验表明，当输入信息变得越来越无结构时，变压器模型使用的语义概念数量会增加。面对输入空间日益增大的不确定性，变压器模型更容易激活与输入无关但具有一致性的语义特征，从而产生幻觉输出。在极端情况下，对于纯噪声输入，我们发现预训练变压器模型的中间激活中广泛存在被稳健触发的并具备意义的语义概念，并通过针对性的引导确认了其功能性完整性。我们还展示了变压器模型输出中的幻觉可以从嵌入在变压器层激活中的概念模式中可靠地预测。这些关于变压器内部处理机制的洞察对对齐AI模型与人类价值观、AI安全性、扩大潜在对抗攻击的攻击面以及提供模型幻觉风险自动量化基础等方面具有立竿见影的影响。 

---
# Neuro-Symbolic AI for Cybersecurity: State of the Art, Challenges, and Opportunities 

**Title (ZH)**: 神经符号人工智能在网络安全领域的现状、挑战与机遇 

**Authors**: Safayat Bin Hakim, Muhammad Adil, Alvaro Velasquez, Shouhuai Xu, Houbing Herbert Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.06921)  

**Abstract**: Traditional Artificial Intelligence (AI) approaches in cybersecurity exhibit fundamental limitations: inadequate conceptual grounding leading to non-robustness against novel attacks; limited instructibility impeding analyst-guided adaptation; and misalignment with cybersecurity objectives. Neuro-Symbolic (NeSy) AI has emerged with the potential to revolutionize cybersecurity AI. However, there is no systematic understanding of this emerging approach. These hybrid systems address critical cybersecurity challenges by combining neural pattern recognition with symbolic reasoning, enabling enhanced threat understanding while introducing concerning autonomous offensive capabilities that reshape threat landscapes. In this survey, we systematically characterize this field by analyzing 127 publications spanning 2019-July 2025. We introduce a Grounding-Instructibility-Alignment (G-I-A) framework to evaluate these systems, focusing on both cyber defense and cyber offense across network security, malware analysis, and cyber operations. Our analysis shows advantages of multi-agent NeSy architectures and identifies critical implementation challenges including standardization gaps, computational complexity, and human-AI collaboration requirements that constrain deployment. We show that causal reasoning integration is the most transformative advancement, enabling proactive defense beyond correlation-based approaches. Our findings highlight dual-use implications where autonomous systems demonstrate substantial capabilities in zero-day exploitation while achieving significant cost reductions, altering threat dynamics. We provide insights and future research directions, emphasizing the urgent need for community-driven standardization frameworks and responsible development practices that ensure advancement serves defensive cybersecurity objectives while maintaining societal alignment. 

**Abstract (ZH)**: 传统人工智能在网络安全中的根本局限性包括概念基础不足导致对新型攻击的不 robust 性；可指导性有限阻碍分析师引导的适应性；以及与网络安全目标的不一致。神经符号人工智能（NeSy AI）有可能彻底改变网络安全人工智能，但对其这一新兴方法的理解尚未系统化。这些混合系统通过结合神经模式识别与符号推理，以增强对威胁的理解并引入令人关切的自主攻击能力来重新定义威胁景观，从而应对关键的网络安全挑战。在本文综述中，我们通过分析2019年至2025年间的127篇出版物，系统化地 characterizes 这个领域。我们引入了一个扎根-可指导性-一致性的框架（G-I-A框架）来评估这些系统，重点关注网络安全性、恶意软件分析和网络操作中的网络防御和网络攻击。我们的分析表明，多代理神经符号架构具有优势，并确定了关键实施挑战，如标准差距、计算复杂性和人机协作要求，这些挑战限制了部署。我们表明，因果推理整合是最具变革性的进步，能够超越相关性方法实现主动防御。我们的发现指出，自主系统在零日攻击利用方面表现出色，同时显著降低成本，改变威胁动态。我们提供了见解并为未来研究方向提出了建议，强调了社区驱动的标准框架和负责任的发展实践的迫切需求，以确保技术进步服务于防御性网络安全目标的同时保持社会共识。 

---
# An Ethically Grounded LLM-Based Approach to Insider Threat Synthesis and Detection 

**Title (ZH)**: 基于伦理准则的LLM驱动的内部威胁合成与检测方法 

**Authors**: Haywood Gelman, John D. Hastings, David Kenley  

**Link**: [PDF](https://arxiv.org/pdf/2509.06920)  

**Abstract**: Insider threats are a growing organizational problem due to the complexity of identifying their technical and behavioral elements. A large research body is dedicated to the study of insider threats from technological, psychological, and educational perspectives. However, research in this domain has been generally dependent on datasets that are static and limited access which restricts the development of adaptive detection models. This study introduces a novel, ethically grounded approach that uses the large language model (LLM) Claude Sonnet 3.7 to dynamically synthesize syslog messages, some of which contain indicators of insider threat scenarios. The messages reflect real-world data distributions by being highly imbalanced (1% insider threats). The syslogs were analyzed for insider threats by both Claude Sonnet 3.7 and GPT-4o, with their performance evaluated through statistical metrics including precision, recall, MCC, and ROC AUC. Sonnet 3.7 consistently outperformed GPT-4o across nearly all metrics, particularly in reducing false alarms and improving detection accuracy. The results show strong promise for the use of LLMs in synthetic dataset generation and insider threat detection. 

**Abstract (ZH)**: 基于大型语言模型的伦理导向合成日志方法在内部威胁检测中的应用 

---
# Tackling the Noisy Elephant in the Room: Label Noise-robust Out-of-Distribution Detection via Loss Correction and Low-rank Decomposition 

**Title (ZH)**: 处理房间里的嘈杂大象：基于损失纠正和低秩分解的标签噪声鲁棒异分布检测 

**Authors**: Tarhib Al Azad, Shahana Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2509.06918)  

**Abstract**: Robust out-of-distribution (OOD) detection is an indispensable component of modern artificial intelligence (AI) systems, especially in safety-critical applications where models must identify inputs from unfamiliar classes not seen during training. While OOD detection has been extensively studied in the machine learning literature--with both post hoc and training-based approaches--its effectiveness under noisy training labels remains underexplored. Recent studies suggest that label noise can significantly degrade OOD performance, yet principled solutions to this issue are lacking. In this work, we demonstrate that directly combining existing label noise-robust methods with OOD detection strategies is insufficient to address this critical challenge. To overcome this, we propose a robust OOD detection framework that integrates loss correction techniques from the noisy label learning literature with low-rank and sparse decomposition methods from signal processing. Extensive experiments on both synthetic and real-world datasets demonstrate that our method significantly outperforms the state-of-the-art OOD detection techniques, particularly under severe noisy label settings. 

**Abstract (ZH)**: 稳健的离域分布（OOD）检测是现代人工智能（AI）系统不可或缺的组成部分，尤其是在涉及模型必须识别训练期间未见过的陌生类别的安全关键应用中。虽然在机器学习文献中对OOD检测进行了广泛研究，包括事后方法和训练基于方法，但在噪声训练标签下的有效性却未得到充分探索。近期研究表明，标签噪声可以显著劣化OOD性能，但针对这一问题的原理性解决方案仍然不足。在本文中，我们证明了直接将现有的标签噪声鲁棒方法与OOD检测策略结合是不足以解决这一关键挑战的。为克服这一问题，我们提出了一种将有噪标签学习文献中的损失校正技术与信号处理领域的低秩和稀疏分解方法相结合的稳健OOD检测框架。在合成数据集和真实世界数据集上的广泛实验表明，我们的方法在严重噪声标签设置下显著优于最先进的OOD检测技术。 

---
# Barlow-Swin: Toward a novel siamese-based segmentation architecture using Swin-Transformers 

**Title (ZH)**: Barlow-Swin：基于Swin Transformers的一种新型双目分割架构探究 

**Authors**: Morteza Kiani Haftlang, Mohammadhossein Malmir, Foroutan Parand, Umberto Michelucci, Safouane El Ghazouali  

**Link**: [PDF](https://arxiv.org/pdf/2509.06885)  

**Abstract**: Medical image segmentation is a critical task in clinical workflows, particularly for the detection and delineation of pathological regions. While convolutional architectures like U-Net have become standard for such tasks, their limited receptive field restricts global context modeling. Recent efforts integrating transformers have addressed this, but often result in deep, computationally expensive models unsuitable for real-time use. In this work, we present a novel end-to-end lightweight architecture designed specifically for real-time binary medical image segmentation. Our model combines a Swin Transformer-like encoder with a U-Net-like decoder, connected via skip pathways to preserve spatial detail while capturing contextual information. Unlike existing designs such as Swin Transformer or U-Net, our architecture is significantly shallower and competitively efficient. To improve the encoder's ability to learn meaningful features without relying on large amounts of labeled data, we first train it using Barlow Twins, a self-supervised learning method that helps the model focus on important patterns by reducing unnecessary repetition in the learned features. After this pretraining, we fine-tune the entire model for our specific task. Experiments on benchmark binary segmentation tasks demonstrate that our model achieves competitive accuracy with substantially reduced parameter count and faster inference, positioning it as a practical alternative for deployment in real-time and resource-limited clinical environments. The code for our method is available at Github repository: this https URL. 

**Abstract (ZH)**: 实时医用图像二元分割的轻量级端到端架构 

---
# UNH at CheckThat! 2025: Fine-tuning Vs Prompting in Claim Extraction 

**Title (ZH)**: UNH在CheckThat! 2025：细调 vs 填充文本在声明提取中的对比研究 

**Authors**: Joe Wilder, Nikhil Kadapala, Benji Xu, Mohammed Alsaadi, Aiden Parsons, Mitchell Rogers, Palash Agarwal, Adam Hassick, Laura Dietz  

**Link**: [PDF](https://arxiv.org/pdf/2509.06883)  

**Abstract**: We participate in CheckThat! Task 2 English and explore various methods of prompting and in-context learning, including few-shot prompting and fine-tuning with different LLM families, with the goal of extracting check-worthy claims from social media passages. Our best METEOR score is achieved by fine-tuning a FLAN-T5 model. However, we observe that higher-quality claims can sometimes be extracted using other methods, even when their METEOR scores are lower. 

**Abstract (ZH)**: 我们参与CheckThat! 任务2英语部分，探索各种提示方法和上下文学习方法，包括少量示例提示和不同大规模语言模型家族的微调，并旨在从社交媒体段落中提取值得验证的声明。我们的最好METEOR分数是由微调FLAN-T5模型获得的。然而，我们观察到，在某些情况下，即使METEOR分数较低，其他方法也可能提取出更高质量的声明。 

---
# AxelSMOTE: An Agent-Based Oversampling Algorithm for Imbalanced Classification 

**Title (ZH)**: 基于代理的过采样算法：AxelSMOTE 用于不平衡分类 

**Authors**: Sukumar Kishanthan, Asela Hevapathige  

**Link**: [PDF](https://arxiv.org/pdf/2509.06875)  

**Abstract**: Class imbalance in machine learning poses a significant challenge, as skewed datasets often hinder performance on minority classes. Traditional oversampling techniques, which are commonly used to alleviate class imbalance, have several drawbacks: they treat features independently, lack similarity-based controls, limit sample diversity, and fail to manage synthetic variety effectively. To overcome these issues, we introduce AxelSMOTE, an innovative agent-based approach that views data instances as autonomous agents engaging in complex interactions. Based on Axelrod's cultural dissemination model, AxelSMOTE implements four key innovations: (1) trait-based feature grouping to preserve correlations; (2) a similarity-based probabilistic exchange mechanism for meaningful interactions; (3) Beta distribution blending for realistic interpolation; and (4) controlled diversity injection to avoid overfitting. Experiments on eight imbalanced datasets demonstrate that AxelSMOTE outperforms state-of-the-art sampling methods while maintaining computational efficiency. 

**Abstract (ZH)**: 机器学习中的类别不平衡构成了一个重大挑战，因为偏斜的数据集通常会妨碍少数类别的性能。传统过采样技术虽然常被用于缓解类别不平衡，但也存在几个缺点：它们独立处理特征，缺乏基于相似性的控制，限制了样本多样性，并且无法有效地管理合成样本的多样性。为克服这些问题，我们提出了一种名为AxelSMOTE的新颖基于代理的方法，该方法将数据实例视作参与复杂交互的自主代理。基于Axelrod的文化传播模型，AxelSMOTE实现了四个关键创新：（1）基于特征的特质分组以保留相关性；（2）一种基于相似性的概率交换机制以实现有意义的交互；（3）使用Beta分布混合进行现实插值；（4）控制多样性注入以避免过拟合。在八个不平衡数据集上的实验表明，AxelSMOTE在保持计算效率的同时，优于最先进的采样方法。 

---
# floq: Training Critics via Flow-Matching for Scaling Compute in Value-Based RL 

**Title (ZH)**: FloQ: 通过流动匹配训练批评家以扩展值ベースRL的计算资源 

**Authors**: Bhavya Agrawalla, Michal Nauman, Khush Agarwal, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.06863)  

**Abstract**: A hallmark of modern large-scale machine learning techniques is the use of training objectives that provide dense supervision to intermediate computations, such as teacher forcing the next token in language models or denoising step-by-step in diffusion models. This enables models to learn complex functions in a generalizable manner. Motivated by this observation, we investigate the benefits of iterative computation for temporal difference (TD) methods in reinforcement learning (RL). Typically they represent value functions in a monolithic fashion, without iterative compute. We introduce floq (flow-matching Q-functions), an approach that parameterizes the Q-function using a velocity field and trains it using techniques from flow-matching, typically used in generative modeling. This velocity field underneath the flow is trained using a TD-learning objective, which bootstraps from values produced by a target velocity field, computed by running multiple steps of numerical integration. Crucially, floq allows for more fine-grained control and scaling of the Q-function capacity than monolithic architectures, by appropriately setting the number of integration steps. Across a suite of challenging offline RL benchmarks and online fine-tuning tasks, floq improves performance by nearly 1.8x. floq scales capacity far better than standard TD-learning architectures, highlighting the potential of iterative computation for value learning. 

**Abstract (ZH)**: 现代大规模机器学习技术的一个标志是使用提供密集监督的训练目标，例如在语言模型中通过教师强迫下一个token，在扩散模型中通过逐步去噪。这使得模型能够以可泛化的方式学习复杂的函数。受这一观察的启发，我们研究了迭代计算在强化学习（RL）中的时间差分（TD）方法中的益处。通常，它们以整体的方式表示值函数，不包含迭代计算。我们提出了floq（流匹配Q函数），这是一种使用速度场参数化Q函数并利用流匹配技术（通常用于生成建模）进行训练的方法。该速度场通过运行多次数值积分计算的目标速度场生成的值进行引导，采用TD学习目标进行训练。关键的是，floq通过适当设置积分步数，提供了比整体架构更精细的Q函数容量控制。在一系列具有挑战性的离线RL基准测试和在线微调任务中，floq将性能提高了近1.8倍。floq在Q函数容量扩展方面远远优于标准的TD学习架构，突显了迭代计算在值学习中的潜力。 

---
# Disentangling Interaction and Bias Effects in Opinion Dynamics of Large Language Models 

**Title (ZH)**: 分离交互作用和偏差效应在大型语言模型意见动力学中的影响 

**Authors**: Vincent C. Brockers, David A. Ehrlich, Viola Priesemann  

**Link**: [PDF](https://arxiv.org/pdf/2509.06858)  

**Abstract**: Large Language Models are increasingly used to simulate human opinion dynamics, yet the effect of genuine interaction is often obscured by systematic biases. We present a Bayesian framework to disentangle and quantify three such biases: (i) a topic bias toward prior opinions in the training data; (ii) an agreement bias favoring agreement irrespective of the question; and (iii) an anchoring bias toward the initiating agent's stance. Applying this framework to multi-step dialogues reveals that opinion trajectories tend to quickly converge to a shared attractor, with the influence of the interaction fading over time, and the impact of biases differing between LLMs. In addition, we fine-tune an LLM on different sets of strongly opinionated statements (incl. misinformation) and demonstrate that the opinion attractor shifts correspondingly. Exposing stark differences between LLMs and providing quantitative tools to compare them to human subjects in the future, our approach highlights both chances and pitfalls in using LLMs as proxies for human behavior. 

**Abstract (ZH)**: 大型语言模型越来越多地用于模拟人类意见动态，但真实的交互效果往往被系统性偏见所掩盖。我们提出了一种贝叶斯框架以分离和量化三种偏见：（i）主题偏见，倾向于训练数据中的先验观点；（ii）一致性偏见，倾向于一致而不论问题如何；（iii）锚定偏见，倾向于初始行动者的态度。将该框架应用于多步对话表明，意见轨迹往往会迅速向一个共享的吸引子收敛，交互影响随时间减弱，并且不同大型语言模型受偏见影响的方式不同。此外，我们对不同集强观点陈述（包括 misinformation）的大型语言模型进行微调，并证明了意见吸引子随之相应地变化。通过揭示大型语言模型之间明显的差异，并为将来将它们与人类受试者进行定量比较提供工具，我们的方法突显了使用大型语言模型作为人类行为代理的机遇与风险。 

---
# Automated Radiographic Total Sharp Score (ARTSS) in Rheumatoid Arthritis: A Solution to Reduce Inter-Intra Reader Variation and Enhancing Clinical Practice 

**Title (ZH)**: 基于放射影像的整体锐利度评分（ARTSS）在类风湿关节炎中的应用：减少阅片者间及阅片者内变异并提升临床实践 

**Authors**: Hajar Moradmand, Lei Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.06854)  

**Abstract**: Assessing the severity of rheumatoid arthritis (RA) using the Total Sharp/Van Der Heijde Score (TSS) is crucial, but manual scoring is often time-consuming and subjective. This study introduces an Automated Radiographic Sharp Scoring (ARTSS) framework that leverages deep learning to analyze full-hand X-ray images, aiming to reduce inter- and intra-observer variability. The research uniquely accommodates patients with joint disappearance and variable-length image sequences. We developed ARTSS using data from 970 patients, structured into four stages: I) Image pre-processing and re-orientation using ResNet50, II) Hand segmentation using UNet.3, III) Joint identification using YOLOv7, and IV) TSS prediction using models such as VGG16, VGG19, ResNet50, DenseNet201, EfficientNetB0, and Vision Transformer (ViT). We evaluated model performance with Intersection over Union (IoU), Mean Average Precision (MAP), mean absolute error (MAE), Root Mean Squared Error (RMSE), and Huber loss. The average TSS from two radiologists was used as the ground truth. Model training employed 3-fold cross-validation, with each fold consisting of 452 training and 227 validation samples, and external testing included 291 unseen subjects. Our joint identification model achieved 99% accuracy. The best-performing model, ViT, achieved a notably low Huber loss of 0.87 for TSS prediction. Our results demonstrate the potential of deep learning to automate RA scoring, which can significantly enhance clinical practice. Our approach addresses the challenge of joint disappearance and variable joint numbers, offers timesaving benefits, reduces inter- and intra-reader variability, improves radiologist accuracy, and aids rheumatologists in making more informed decisions. 

**Abstract (ZH)**: 利用Total Sharp/Van Der Heijde评分（TSS）评估类风湿关节炎（RA）的严重程度至关重要，但手动评分往往耗时且具主观性。本研究提出了一种基于深度学习的自动放射学Sharp评分（ARTSS）框架，旨在减少观察者间和观察者内的变异。该研究独特地处理了关节消失和图像序列长度不一的患者。我们使用970名患者的资料构建了ARTSS，分为四个阶段：I）使用ResNet50进行图像预处理和重新定向；II）使用UNet.3进行手部分割；III）使用YOLOv7进行关节识别；IV）使用VGG16、VGG19、ResNet50、DenseNet201、EfficientNetB0和Vision Transformer（ViT）等模型进行TSS预测。我们使用Intersection over Union（IoU）、Mean Average Precision（MAP）、mean absolute error（MAE）、Root Mean Squared Error（RMSE）和Huber损失评估模型性能。两位放射科医生的平均TSS作为真实值。模型训练采用3折交叉验证，每折包含452个训练样本和227个验证样本，外部测试包括291个未见过的个体。我们的关节识别模型准确率达到99%。性能最佳的模型ViT在TSS预测中的Huber损失为0.87。我们的研究结果展示了深度学习在自动化RA评分方面具有巨大潜力，这将显著提升临床实践。本方法解决了关节消失和关节数目变化的挑战，提供了节省时间的好处，减少了观察者间和观察者内的变异，提高了放射科医生的准确性，并有助于风湿科医生做出更明智的决策。 

---
# Reinforcement learning meets bioprocess control through behaviour cloning: Real-world deployment in an industrial photobioreactor 

**Title (ZH)**: 强化学习与生物过程控制相结合：工业光生物反应器中的实际部署 

**Authors**: Juan D. Gil, Ehecatl Antonio Del Rio Chanona, José L. Guzmán, Manuel Berenguel  

**Link**: [PDF](https://arxiv.org/pdf/2509.06853)  

**Abstract**: The inherent complexity of living cells as production units creates major challenges for maintaining stable and optimal bioprocess conditions, especially in open Photobioreactors (PBRs) exposed to fluctuating environments. To address this, we propose a Reinforcement Learning (RL) control approach, combined with Behavior Cloning (BC), for pH regulation in open PBR systems. This represents, to the best of our knowledge, the first application of an RL-based control strategy to such a nonlinear and disturbance-prone bioprocess. Our method begins with an offline training stage in which the RL agent learns from trajectories generated by a nominal Proportional-Integral-Derivative (PID) controller, without direct interaction with the real system. This is followed by a daily online fine-tuning phase, enabling adaptation to evolving process dynamics and stronger rejection of fast, transient disturbances. This hybrid offline-online strategy allows deployment of an adaptive control policy capable of handling the inherent nonlinearities and external perturbations in open PBRs. Simulation studies highlight the advantages of our method: the Integral of Absolute Error (IAE) was reduced by 8% compared to PID control and by 5% relative to standard off-policy RL. Moreover, control effort decreased substantially-by 54% compared to PID and 7% compared to standard RL-an important factor for minimizing operational costs. Finally, an 8-day experimental validation under varying environmental conditions confirmed the robustness and reliability of the proposed approach. Overall, this work demonstrates the potential of RL-based methods for bioprocess control and paves the way for their broader application to other nonlinear, disturbance-prone systems. 

**Abstract (ZH)**: 基于强化学习的行为仿slug控制在开放光生物反应器pH调节中的应用：一种处理非线性和干扰的新方法 

---
# COMPACT: Common-token Optimized Model Pruning Across Channels and Tokens 

**Title (ZH)**: COMMON-TOKEN 优化通道和_token_剪枝模型 

**Authors**: Eugene Kwek, Wenpeng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06836)  

**Abstract**: Making LLMs more efficient in memory, latency, and serving cost is crucial for edge deployment, interactive applications, and sustainable inference at scale. Pruning is a key technique toward this goal. However, prior pruning methods are limited: width pruning often breaks the standard transformer layout or requires custom inference code, while depth pruning removes entire layers and can cause abrupt accuracy drops. In this work, we propose COMPACT, which jointly (i) prunes rare vocabulary to shrink embedding/unembedding and (ii) prunes FFN intermediate channels using common-token-weighted activations, aligning importance with the post-pruning token distribution. COMPACT enjoys merits of both depth and width pruning, such as: deployment-friendliness (keeps a standard transformer architecture), scale-adaptivity (trade off vocab vs. FFN pruning), training-free operation with competitive pruning time, and strong memory savings alongside throughput gains. Experiments across Qwen, LLaMA, and Gemma families (0.5B-70B) show state-of-the-art downstream task performance at similar or higher pruning ratios, with substantial reductions in parameters, GPU memory, and end-to-end latency. 

**Abstract (ZH)**: 提高大语言模型在内存、延迟和推理成本方面的效率对于边缘部署、交互应用及大规模可持续推理至关重要。剪枝是实现这一目标的关键技术。然而，现有的剪枝方法存在局限性：宽度剪枝常常破坏标准的变压器结构或需要定制的推理代码，而深度剪枝会移除整个层，从而导致准确率突变下降。在本文中，我们提出了一种名为COMPACT的方法，该方法联合进行(i)稀有词汇剪枝以缩小嵌入/解嵌入以及(ii)使用常见标记权重激活的功能层中间通道剪枝，使重要性与后剪枝标记分布相一致。COMPACT方法同时具备深度剪枝和宽度剪枝的优点，如：易于部署（保持标准的变压器结构）、规模适应性（在词汇和功能层之间进行权衡）、无需训练即可操作且具有竞争力的剪枝时间、以及显著的内存节省和吞吐量提升。跨Qwen、LLaMA和Gemma系列（0.5B-70B参数）的实验结果显示，在相似或更高的剪枝比率下，COMPACT实现了最先进的下游任务性能，并大幅减少了参数量、GPU内存和端到端延迟。 

---
# Saturation-Driven Dataset Generation for LLM Mathematical Reasoning in the TPTP Ecosystem 

**Title (ZH)**: 饱和驱动的数据集生成以支持TPTP生态系统中的LLM数学推理 

**Authors**: Valentin Quesnel, Damien Sileo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06809)  

**Abstract**: The scarcity of high-quality, logically sound data is a critical bottleneck for advancing the mathematical reasoning of Large Language Models (LLMs). Our work confronts this challenge by turning decades of automated theorem proving research into a scalable data engine. Rather than relying on error-prone LLMs or complex proof-assistant syntax like Lean and Isabelle, our framework leverages E-prover's saturation capabilities on the vast TPTP axiom library to derive a massive, guaranteed-valid corpus of theorems. Our pipeline is principled and simple: saturate axioms, filter for "interesting" theorems, and generate tasks. With no LLMs in the loop, we eliminate factual errors by construction. This purely symbolic data is then transformed into three difficulty-controlled challenges: entailment verification, premise selection, and proof reconstruction. Our zero-shot experiments on frontier models reveal a clear weakness: performance collapses on tasks requiring deep, structural reasoning. Our framework provides both the diagnostic tool to measure this gap and a scalable source of symbolic training data to address it. We make the code and data publicly available.
this https URL this https URL 

**Abstract (ZH)**: 高质量且逻辑严谨的数据稀缺是大型语言模型（LLMs）提升数学推理能力的关键瓶颈。我们通过将自动化定理证明研究的几十年经验转化为可扩展的数据引擎来应对这一挑战。我们的框架不依赖于容易出错的LLMs或复杂的证明辅助语法如Lean和Isabelle，而是利用E-prover的饱和能力在庞大的TPTP公理库上推导出大量已验证有效的定理。我们的流程是原理性的且简单明了：饱和公理、筛选“有趣”的定理、生成任务。由于在整个流程中不涉及LLMs，我们从根本上消除了事实错误。随后，这些纯符号数据被转化为三个难度可控的挑战：蕴含验证、前提选择和证明重构。我们在前沿模型上的零样本实验揭示了一个明显的弱点：在需要深度结构性推理的任务上表现下滑。我们的框架提供了衡量这一差距的诊察工具，并且提供了一种可扩展的符号训练数据来源来解决这一问题。我们已将代码和数据公开发布。 

---
# MachineLearningLM: Continued Pretraining Language Models on Millions of Synthetic Tabular Prediction Tasks Scales In-Context ML 

**Title (ZH)**: 机器学习LM：在数百万个合成表预测任务上持续预训练，并通过上下文学习scaling。 

**Authors**: Haoyu Dong, Pengkun Zhang, Mingzhe Lu, Yanzhen Shen, Guolin Ke  

**Link**: [PDF](https://arxiv.org/pdf/2509.06806)  

**Abstract**: Large language models (LLMs) possess broad world knowledge and strong general-purpose reasoning ability, yet they struggle to learn from many in-context examples on standard machine learning (ML) tasks, that is, to leverage many-shot demonstrations purely via in-context learning (ICL) without gradient descent. We introduce MachineLearningLM, a portable continued-pretraining framework that equips a general-purpose LLM with robust in-context ML capability while preserving its general knowledge and reasoning for broader chat workflows.
Our pretraining procedure synthesizes ML tasks from millions of structural causal models (SCMs), spanning shot counts up to 1,024. We begin with a random-forest teacher, distilling tree-based decision strategies into the LLM to strengthen robustness in numerical modeling. All tasks are serialized with a token-efficient prompt, enabling 3x to 6x more examples per context window and delivering up to 50x amortized throughput via batch inference.
Despite a modest setup (Qwen-2.5-7B-Instruct with LoRA rank 8), MachineLearningLM outperforms strong LLM baselines (e.g., GPT-5-mini) by an average of about 15% on out-of-distribution tabular classification across finance, physics, biology, and healthcare domains. It exhibits a striking many-shot scaling law: accuracy increases monotonically as in-context demonstrations grow from 8 to 1,024. Without any task-specific training, it attains random-forest-level accuracy across hundreds of shots. General chat capabilities, including knowledge and reasoning, are preserved: it achieves 75.4% on MMLU. 

**Abstract (ZH)**: 大规模语言模型（LLMs）具备广泛的世界知识和强大的通用推理能力，但在标准机器学习（ML）任务中，它们难以从多个上下文示例中学习，即在没有梯度下降的情况下仅通过上下文学习（ICL）利用大量的示例进行学习。我们引入了MachineLearningLM，这是一种便携的持续预训练框架，能够增强通用语言模型在上下文中的机器学习能力，同时保留其广泛的知识和推理能力，适用于更广泛的对话流程。

我们的预训练过程从数百万个结构性因果模型（SCMs）中合成机器学习任务，覆盖示例数量从1到1,024。我们从随机森林教师开始，将基于树的决策策略提炼到LLM中，以增强数值建模的鲁棒性。所有任务都使用基于标记的高效提示进行序列化，使每个上下文窗口的示例数量增加3到6倍，并通过批量推理实现高达50倍的传输效率。

尽管硬件配置有限（Qwen-2.5-7B-Instruct带LoRA秩8），MachineLearningLM在金融、物理、生物学和医疗保健领域中分布外的表格分类任务中平均优于强大语言模型基线（如GPT-5-mini）约15%。它表现出明显的多示例扩展规律：准确率随着上下文演示数量从8增加到1,024而单调增加。在没有任何特定任务训练的情况下，它在数百个示例中达到了随机森林级别的准确率。通用对话能力，包括知识和推理，得以保留：在MMLU上达到75.4%。 

---
# Aligning Large Vision-Language Models by Deep Reinforcement Learning and Direct Preference Optimization 

**Title (ZH)**: 通过深度强化学习和直接偏好优化对 largVision-语言模型进行对齐 

**Authors**: Thanh Thi Nguyen, Campbell Wilson, Janis Dalins  

**Link**: [PDF](https://arxiv.org/pdf/2509.06759)  

**Abstract**: Large Vision-Language Models (LVLMs) or multimodal large language models represent a significant advancement in artificial intelligence, enabling systems to understand and generate content across both visual and textual modalities. While large-scale pretraining has driven substantial progress, fine-tuning these models for aligning with human values or engaging in specific tasks or behaviors remains a critical challenge. Deep Reinforcement Learning (DRL) and Direct Preference Optimization (DPO) offer promising frameworks for this aligning process. While DRL enables models to optimize actions using reward signals instead of relying solely on supervised preference data, DPO directly aligns the policy with preferences, eliminating the need for an explicit reward model. This overview explores paradigms for fine-tuning LVLMs, highlighting how DRL and DPO techniques can be used to align models with human preferences and values, improve task performance, and enable adaptive multimodal interaction. We categorize key approaches, examine sources of preference data, reward signals, and discuss open challenges such as scalability, sample efficiency, continual learning, generalization, and safety. The goal is to provide a clear understanding of how DRL and DPO contribute to the evolution of robust and human-aligned LVLMs. 

**Abstract (ZH)**: 大规模多模态语言模型：通过深度强化学习和直接偏好优化实现人类价值观对齐 

---
# Long-Range Graph Wavelet Networks 

**Title (ZH)**: 长距离图小波网络 

**Authors**: Filippo Guerranti, Fabrizio Forte, Simon Geisler, Stephan Günnemann  

**Link**: [PDF](https://arxiv.org/pdf/2509.06743)  

**Abstract**: Modeling long-range interactions, the propagation of information across distant parts of a graph, is a central challenge in graph machine learning. Graph wavelets, inspired by multi-resolution signal processing, provide a principled way to capture both local and global structures. However, existing wavelet-based graph neural networks rely on finite-order polynomial approximations, which limit their receptive fields and hinder long-range propagation. We propose Long-Range Graph Wavelet Networks (LR-GWN), which decompose wavelet filters into complementary local and global components. Local aggregation is handled with efficient low-order polynomials, while long-range interactions are captured through a flexible spectral domain parameterization. This hybrid design unifies short- and long-distance information flow within a principled wavelet framework. Experiments show that LR-GWN achieves state-of-the-art performance among wavelet-based methods on long-range benchmarks, while remaining competitive on short-range datasets. 

**Abstract (ZH)**: 长距离图波动网络（LR-GWN）在图机器学习中的应用：结合局部和全局成分实现高效长距离信息传播 

---
# MRI-Based Brain Tumor Detection through an Explainable EfficientNetV2 and MLP-Mixer-Attention Architecture 

**Title (ZH)**: 基于MRI的脑肿瘤检测：可解释的EfficientNetV2和MLP-Mixer-Attention架构 

**Authors**: Mustafa Yurdakul, Şakir Taşdemir  

**Link**: [PDF](https://arxiv.org/pdf/2509.06713)  

**Abstract**: Brain tumors are serious health problems that require early diagnosis due to their high mortality rates. Diagnosing tumors by examining Magnetic Resonance Imaging (MRI) images is a process that requires expertise and is prone to error. Therefore, the need for automated diagnosis systems is increasing day by day. In this context, a robust and explainable Deep Learning (DL) model for the classification of brain tumors is proposed. In this study, a publicly available Figshare dataset containing 3,064 T1-weighted contrast-enhanced brain MRI images of three tumor types was used. First, the classification performance of nine well-known CNN architectures was evaluated to determine the most effective backbone. Among these, EfficientNetV2 demonstrated the best performance and was selected as the backbone for further development. Subsequently, an attention-based MLP-Mixer architecture was integrated into EfficientNetV2 to enhance its classification capability. The performance of the final model was comprehensively compared with basic CNNs and the methods in the literature. Additionally, Grad-CAM visualization was used to interpret and validate the decision-making process of the proposed model. The proposed model's performance was evaluated using the five-fold cross-validation method. The proposed model demonstrated superior performance with 99.50% accuracy, 99.47% precision, 99.52% recall and 99.49% F1 score. The results obtained show that the model outperforms the studies in the literature. Moreover, Grad-CAM visualizations demonstrate that the model effectively focuses on relevant regions of MRI images, thus improving interpretability and clinical reliability. A robust deep learning model for clinical decision support systems has been obtained by combining EfficientNetV2 and attention-based MLP-Mixer, providing high accuracy and interpretability in brain tumor classification. 

**Abstract (ZH)**: 基于EfficientNetV2和注意力机制MLP-Mixer的脑肿瘤分类 robust deep learning模型 

---
# Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks 

**Title (ZH)**: 深度神经网络中潜在代理子结构的概率建模 

**Authors**: Su Hyeong Lee, Risi Kondor, Richard Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06701)  

**Abstract**: We develop a theory of intelligent agency grounded in probabilistic modeling for neural models. Agents are represented as outcome distributions with epistemic utility given by log score, and compositions are defined through weighted logarithmic pooling that strictly improves every member's welfare. We prove that strict unanimity is impossible under linear pooling or in binary outcome spaces, but possible with three or more outcomes. Our framework admits recursive structure via cloning invariance, continuity, and openness, while tilt-based analysis rules out trivial duplication. Finally, we formalize an agentic alignment phenomenon in LLMs using our theory: eliciting a benevolent persona ("Luigi'") induces an antagonistic counterpart ("Waluigi"), while a manifest-then-suppress Waluigi strategy yields strictly larger first-order misalignment reduction than pure Luigi reinforcement alone. These results clarify how developing a principled mathematical framework for how subagents can coalesce into coherent higher-level entities provides novel implications for alignment in agentic AI systems. 

**Abstract (ZH)**: 我们基于概率建模发展了一种智能代理理论。代理被表示为具有逻辑评分的信念分布，并通过加权对数池化来定义组合，该池化方式严格改善了每个成员的福祉。我们证明，在线性池化或二元结果空间中不可能实现严格的一致性，但在三个或更多结果中是可能的。我们的框架通过克隆不变性、连续性和开放性允许递归结构，同时基于倾斜的分析排除了平凡的复制。最后，我们使用我们的理论正式化了一个在大规模语言模型（LLM）中出现的代理对齐现象：唤起一个善良的人格（“Luigi”）会引发一个对立的人格（“Waluigi”），而先显现后抑制的Waluigi策略会产生比单纯强化Luigi更为严格的一阶对齐改进。这些结果阐明了如何发展一个关于亚代理如何凝聚成一致的更高层次实体的原理数学框架为代理型AI系统中的对齐提供了新的意义。 

---
# Barycentric Neural Networks and Length-Weighted Persistent Entropy Loss: A Green Geometric and Topological Framework for Function Approximation 

**Title (ZH)**: 重心神经网络和长度加权持久熵损失：一种绿色几何与拓扑函数逼近框架 

**Authors**: Victor Toscano-Duran, Rocio Gonzalez-Diaz, Miguel A. Gutiérrez-Naranjo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06694)  

**Abstract**: While it is well-established that artificial neural networks are \emph{universal approximators} for continuous functions on compact domains, many modern approaches rely on deep or overparameterized architectures that incur high computational costs. In this paper, a new type of \emph{small shallow} neural network, called the \emph{Barycentric Neural Network} ($\BNN$), is proposed, which leverages a fixed set of \emph{base points} and their \emph{barycentric coordinates} to define both its structure and its parameters. We demonstrate that our $\BNN$ enables the exact representation of \emph{continuous piecewise linear functions} ($\CPLF$s), ensuring strict continuity across segments. Since any continuous function over a compact domain can be approximated arbitrarily well by $\CPLF$s, the $\BNN$ naturally emerges as a flexible and interpretable tool for \emph{function approximation}. Beyond the use of this representation, the main contribution of the paper is the introduction of a new variant of \emph{persistent entropy}, a topological feature that is stable and scale invariant, called the \emph{length-weighted persistent entropy} ($\LWPE$), which is weighted by the lifetime of topological features. Our framework, which combines the $\BNN$ with a loss function based on our $\LWPE$, aims to provide flexible and geometrically interpretable approximations of nonlinear continuous functions in resource-constrained settings, such as those with limited base points for $\BNN$ design and few training epochs. Instead of optimizing internal weights, our approach directly \emph{optimizes the base points that define the $\BNN$}. Experimental results show that our approach achieves \emph{superior and faster approximation performance} compared to classical loss functions such as MSE, RMSE, MAE, and log-cosh. 

**Abstract (ZH)**: 一种基于基点和重心坐标的小浅层神经网络及其在受限资源环境下的非线性连续函数近似方法 

---
# BioLite U-Net: Edge-Deployable Semantic Segmentation for In Situ Bioprinting Monitoring 

**Title (ZH)**: BioLite U-Net: 适用于就地生物打印监控的边缘部署语义分割 

**Authors**: Usman Haider, Lukasz Szemet, Daniel Kelly, Vasileios Sergis, Andrew C. Daly, Karl Mason  

**Link**: [PDF](https://arxiv.org/pdf/2509.06690)  

**Abstract**: Bioprinting is a rapidly advancing field that offers a transformative approach to fabricating tissue and organ models through the precise deposition of cell-laden bioinks. Ensuring the fidelity and consistency of printed structures in real-time remains a core challenge, particularly under constraints imposed by limited imaging data and resource-constrained embedded hardware. Semantic segmentation of the extrusion process, differentiating between nozzle, extruded bioink, and surrounding background, enables in situ monitoring critical to maintaining print quality and biological viability. In this work, we introduce a lightweight semantic segmentation framework tailored for real-time bioprinting applications. We present a novel, manually annotated dataset comprising 787 RGB images captured during the bioprinting process, labeled across three classes: nozzle, bioink, and background. To achieve fast and efficient inference suitable for integration with bioprinting systems, we propose a BioLite U-Net architecture that leverages depthwise separable convolutions to drastically reduce computational load without compromising accuracy. Our model is benchmarked against MobileNetV2 and MobileNetV3-based segmentation baselines using mean Intersection over Union (mIoU), Dice score, and pixel accuracy. All models were evaluated on a Raspberry Pi 4B to assess real-world feasibility. The proposed BioLite U-Net achieves an mIoU of 92.85% and a Dice score of 96.17%, while being over 1300x smaller than MobileNetV2-DeepLabV3+. On-device inference takes 335 ms per frame, demonstrating near real-time capability. Compared to MobileNet baselines, BioLite U-Net offers a superior tradeoff between segmentation accuracy, efficiency, and deployability, making it highly suitable for intelligent, closed-loop bioprinting systems. 

**Abstract (ZH)**: 生物打印是一个迅速发展的领域，通过精确沉积细胞载生物墨水来制造组织和器官模型。在有限的成像数据和资源受限的嵌入式硬件约束下，实时确保打印结构的准确性和一致性仍然是一个核心挑战。通过对外挤出过程进行语义分割，区分喷嘴、挤出的生物墨水和背景，可以实现原位监测，这对于维持打印质量和生物活性至关重要。在这项工作中，我们介绍了一种轻量级的语义分割框架，适用于生物打印应用。我们提出了一种新型的手动标注数据集，包含787张RGB图像，这些图像捕捉了生物打印过程中的数据，并被标注为三个类别：喷嘴、生物墨水和背景。为了实现适用于生物打印系统集成的快速高效推理，我们提出了一种基于深度可分离卷积的BioLite U-Net架构，以大幅减少计算量而不牺牲准确性。我们的模型使用MobileNetV2和MobileNetV3基线进行基准测试，评估指标包括均值交并比（mIoU）、Dice分数和像素准确性。所有模型均在Raspberry Pi 4B上进行评估，以评估其实用性。所提出的BioLite U-Net在每帧上的推理时间为335毫秒，实现接近实时能力。与MobileNet基线相比，BioLite U-Net在分割准确性、效率和部署能力之间提供了更佳的权衡，使其非常适合智能的闭环生物打印系统。 

---
# TrajAware: Graph Cross-Attention and Trajectory-Aware for Generalisable VANETs under Partial Observations 

**Title (ZH)**: TrajAware: 图交叉注意力和路径aware性用于部分观测下的通用车联网 

**Authors**: Xiaolu Fu, Ziyuan Bao, Eiman Kanjo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06665)  

**Abstract**: Vehicular ad hoc networks (VANETs) are a crucial component of intelligent transportation systems; however, routing remains challenging due to dynamic topologies, incomplete observations, and the limited resources of edge devices. Existing reinforcement learning (RL) approaches often assume fixed graph structures and require retraining when network conditions change, making them unsuitable for deployment on constrained hardware. We present TrajAware, an RL-based framework designed for edge AI deployment in VANETs. TrajAware integrates three components: (i) action space pruning, which reduces redundant neighbour options while preserving two-hop reachability, alleviating the curse of dimensionality; (ii) graph cross-attention, which maps pruned neighbours to the global graph context, producing features that generalise across diverse network sizes; and (iii) trajectory-aware prediction, which uses historical routes and junction information to estimate real-time positions under partial observations. We evaluate TrajAware in the open-source SUMO simulator using real-world city maps with a leave-one-city-out setup. Results show that TrajAware achieves near-shortest paths and high delivery ratios while maintaining efficiency suitable for constrained edge devices, outperforming state-of-the-art baselines in both full and partial observation scenarios. 

**Abstract (ZH)**: 基于轨迹感知的VANET边缘AI路由框架TrajAware 

---
# AnalysisGNN: Unified Music Analysis with Graph Neural Networks 

**Title (ZH)**: AnalysisGNN: 基于图神经网络的统一音乐分析 

**Authors**: Emmanouil Karystinaios, Johannes Hentschel, Markus Neuwirth, Gerhard Widmer  

**Link**: [PDF](https://arxiv.org/pdf/2509.06654)  

**Abstract**: Recent years have seen a boom in computational approaches to music analysis, yet each one is typically tailored to a specific analytical domain. In this work, we introduce AnalysisGNN, a novel graph neural network framework that leverages a data-shuffling strategy with a custom weighted multi-task loss and logit fusion between task-specific classifiers to integrate heterogeneously annotated symbolic datasets for comprehensive score analysis. We further integrate a Non-Chord-Tone prediction module, which identifies and excludes passing and non-functional notes from all tasks, thereby improving the consistency of label signals. Experimental evaluations demonstrate that AnalysisGNN achieves performance comparable to traditional static-dataset approaches, while showing increased resilience to domain shifts and annotation inconsistencies across multiple heterogeneous corpora. 

**Abstract (ZH)**: 近年来，音乐分析的计算方法取得了蓬勃发展，但每一种方法通常仅针对特定的分析领域。在此工作中，我们提出了一种新型的图神经网络框架——AnalysisGNN，该框架利用数据打乱策略和自定义加权多任务损失以及任务特定分类器之间的 logits 融合，以综合集成异构标注的符号数据集，实现全面的乐谱分析。我们进一步整合了一个非和弦音预测模块，该模块能够识别并排除所有任务中的经过音和非功能音，从而提高标签信号的一致性。实验评估表明，AnalysisGNN 在多项异构数据集上实现了与传统静态数据集方法相当的性能，并且在领域转换和标注不一致性方面显示出更高的鲁棒性。 

---
# The First Voice Timbre Attribute Detection Challenge 

**Title (ZH)**: 首次声音音色属性检测挑战 

**Authors**: Liping Chen, Jinghao He, Zhengyan Sheng, Kong Aik Lee, Zhen-Hua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2509.06635)  

**Abstract**: The first voice timbre attribute detection challenge is featured in a special session at NCMMSC 2025. It focuses on the explainability of voice timbre and compares the intensity of two speech utterances in a specified timbre descriptor dimension. The evaluation was conducted on the VCTK-RVA dataset. Participants developed their systems and submitted their outputs to the organizer, who evaluated the performance and sent feedback to them. Six teams submitted their outputs, with five providing descriptions of their methodologies. 

**Abstract (ZH)**: 首届声音音色属性检测挑战赛在2025年NCMMSC会议的特别研讨会中举办。该挑战关注声音音色的可解释性，并在指定的音色描述维度上比较两个语音陈述的强度。评估基于VCTK-RVA数据集进行。参与者开发了系统并提交了输出，组织者评估了性能并给予了反馈。六支队伍提交了输出，其中五支提供了其方法论的描述。 

---
# Improved Classification of Nitrogen Stress Severity in Plants Under Combined Stress Conditions Using Spatio-Temporal Deep Learning Framework 

**Title (ZH)**: 在联合作用力条件下基于时空深度学习框架的植物氮胁迫 severity 分类改善研究 

**Authors**: Aswini Kumar Patra  

**Link**: [PDF](https://arxiv.org/pdf/2509.06625)  

**Abstract**: Plants in their natural habitats endure an array of interacting stresses, both biotic and abiotic, that rarely occur in isolation. Nutrient stress-particularly nitrogen deficiency-becomes even more critical when compounded with drought and weed competition, making it increasingly difficult to distinguish and address its effects. Early detection of nitrogen stress is therefore crucial for protecting plant health and implementing effective management strategies. This study proposes a novel deep learning framework to accurately classify nitrogen stress severity in a combined stress environment. Our model uses a unique blend of four imaging modalities-RGB, multispectral, and two infrared wavelengths-to capture a wide range of physiological plant responses from canopy images. These images, provided as time-series data, document plant health across three levels of nitrogen availability (low, medium, and high) under varying water stress and weed pressures. The core of our approach is a spatio-temporal deep learning pipeline that merges a Convolutional Neural Network (CNN) for extracting spatial features from images with a Long Short-Term Memory (LSTM) network to capture temporal dependencies. We also devised and evaluated a spatial-only CNN pipeline for comparison. Our CNN-LSTM pipeline achieved an impressive accuracy of 98%, impressively surpassing the spatial-only model's 80.45% and other previously reported machine learning method's 76%. These results bring actionable insights based on the power of our CNN-LSTM approach in effectively capturing the subtle and complex interactions between nitrogen deficiency, water stress, and weed pressure. This robust platform offers a promising tool for the timely and proactive identification of nitrogen stress severity, enabling better crop management and improved plant health. 

**Abstract (ZH)**: 植物在其自然生态环境中承受着多种交互性压力，包括生境压力和非生物压力，这些压力通常不会单独发生。氮素压力——特别是氮素缺乏——在与干旱和杂草竞争共存时变得更加关键，使其更加难以区分和应对。因此，早期检测氮素压力对于保护植物健康和实施有效的管理策略至关重要。本研究提出了一种新颖的深度学习框架，用于在共存压力环境中准确分类氮素压力严重程度。我们的模型使用四种成像模态的独特组合——RGB、多光谱和两种红外波长——来捕捉从冠层图像中获取的一系列生理植物响应。这些图像以时间序列数据的形式记录了在不同水分压力和杂草压力下，在低、中、高氮素可用性水平下植物的健康状况。我们方法的核心是一个时空深度学习管道，该管道结合了卷积神经网络(CNN)来从图像中提取空间特征，以及长短期记忆网络(LSTM)来捕捉时间依赖性。我们还设计并评估了一个仅空间CNN管道作为对比。我们的CNN-LSTM管道达到了令人印象深刻的98%的准确率，远超仅空间模型的80.45%和其他先前报道的机器学习方法的76%。这些结果基于我们CNN-LSTM方法的力量，提供了关于氮素缺乏、水分压力和杂草压力之间微妙而复杂的交互作用的可操作见解。这一强大的平台提供了一种及时和主动识别氮素压力严重程度的前景工具，有助于更好地作物管理和提高植物健康。 

---
# BEAM: Brainwave Empathy Assessment Model for Early Childhood 

**Title (ZH)**: BEAM: 脑电同理心评估模型 for 早期 childhood 

**Authors**: Chen Xie, Gaofeng Wu, Kaidong Wang, Zihao Zhu, Xiaoshu Luo, Yan Liang, Feiyu Quan, Ruoxi Wu, Xianghui Huang, Han Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06620)  

**Abstract**: Empathy in young children is crucial for their social and emotional development, yet predicting it remains challenging. Traditional methods often only rely on self-reports or observer-based labeling, which are susceptible to bias and fail to objectively capture the process of empathy formation. EEG offers an objective alternative; however, current approaches primarily extract static patterns, neglecting temporal dynamics. To overcome these limitations, we propose a novel deep learning framework, the Brainwave Empathy Assessment Model (BEAM), to predict empathy levels in children aged 4-6 years. BEAM leverages multi-view EEG signals to capture both cognitive and emotional dimensions of empathy. The framework comprises three key components: 1) a LaBraM-based encoder for effective spatio-temporal feature extraction, 2) a feature fusion module to integrate complementary information from multi-view signals, and 3) a contrastive learning module to enhance class separation. Validated on the CBCP dataset, BEAM outperforms state-of-the-art methods across multiple metrics, demonstrating its potential for objective empathy assessment and providing a preliminary insight into early interventions in children's prosocial development. 

**Abstract (ZH)**: 幼儿期共情对于其社会和情感发展至关重要，但对其预测仍然具有挑战性。传统的预测方法通常仅依赖自我报告或观察者基于的标签，这些方法容易引入偏差且无法客观捕捉共情形成的过程。虽然脑电图（EEG）可以提供一种客观替代方案，但当前方法主要提取静态模式，忽略了时间动态性。为克服这些局限性，我们提出了一种新的深度学习框架——脑波共情评估模型（BEAM），以预测4-6岁儿童的共情水平。BEAM 利用多视角的脑电信号来捕捉共情的认知和情感维度。该框架包括三个关键组成部分：1）基于LabraM的编码器以有效地进行时空特征提取，2）特征融合模块以整合多视角信号中的互补信息，3）对比学习模块以增强类别分离。BEAM 在CBCP 数据集上验证，其在多个指标上优于现有方法，展示了其在客观共情评估方面的潜力，并为儿童亲社会发展早期干预提供了初步见解。 

---
# Demo: Healthcare Agent Orchestrator (HAO) for Patient Summarization in Molecular Tumor Boards 

**Title (ZH)**: Demo: 医疗代理 orchestrator (HAO) 用于分子肿瘤板中的患者总结 

**Authors**: Noel Codella, Sam Preston, Hao Qiu, Leonardo Schettini, Wen-wai Yim, Mert Öz, Shrey Jain, Matthew P. Lungren, Thomas Osborne  

**Link**: [PDF](https://arxiv.org/pdf/2509.06602)  

**Abstract**: Molecular Tumor Boards (MTBs) are multidisciplinary forums where oncology specialists collaboratively assess complex patient cases to determine optimal treatment strategies. A central element of this process is the patient summary, typically compiled by a medical oncologist, radiation oncologist, or surgeon, or their trained medical assistant, who distills heterogeneous medical records into a concise narrative to facilitate discussion. This manual approach is often labor-intensive, subjective, and prone to omissions of critical information. To address these limitations, we introduce the Healthcare Agent Orchestrator (HAO), a Large Language Model (LLM)-driven AI agent that coordinates a multi-agent clinical workflow to generate accurate and comprehensive patient summaries for MTBs. Evaluating predicted patient summaries against ground truth presents additional challenges due to stylistic variation, ordering, synonym usage, and phrasing differences, which complicate the measurement of both succinctness and completeness. To overcome these evaluation hurdles, we propose TBFact, a ``model-as-a-judge'' framework designed to assess the comprehensiveness and succinctness of generated summaries. Using a benchmark dataset derived from de-identified tumor board discussions, we applied TBFact to evaluate our Patient History agent. Results show that the agent captured 94% of high-importance information (including partial entailments) and achieved a TBFact recall of 0.84 under strict entailment criteria. We further demonstrate that TBFact enables a data-free evaluation framework that institutions can deploy locally without sharing sensitive clinical data. Together, HAO and TBFact establish a robust foundation for delivering reliable and scalable support to MTBs. 

**Abstract (ZH)**: 分子肿瘤板（MTBs）是多学科论坛，肿瘤专家在此协作评估复杂病例以确定最佳治疗策略。这一过程中的一项核心要素是患者总结，通常由肿瘤内科医生、放射肿瘤学家或外科医生及其受训医疗助理编制，将异质性医疗记录提炼成精炼的叙事以促进讨论。这种手动方法往往耗时、主观且容易遗漏关键信息。为解决这些局限性，我们引入了健康医疗代理协调器（HAO），这是一种由大规模语言模型（LLM）驱动的AI代理，用于协调多代理临床工作流生成准确且全面的患者总结以供分子肿瘤板使用。由于风格差异、排序、同义词使用和措辞不同，对预测的患者总结进行评估还带来了额外的挑战，这使得简洁性和完整性测量更为复杂。为克服这些评估难题，我们提出了TBFact框架，这是一种“模型即法官”的框架，用于评估生成总结的全面性和简明性。使用从去标识化肿瘤板讨论中提取的基准数据集，我们运用TBFact评估了我们的患者病史代理。结果显示，代理捕获了94%的关键信息（包括部分蕴含），并在严格蕴含标准下实现了TBFact召回率为0.84。此外，我们展示了TBFact能够提供一种无需共享敏感临床数据即可在本地部署的数据驱动评估框架。总之，结合HAO和TBFact为MTBs提供了可靠且可扩展的支持奠定了坚实基础。 

---
# Integrating Spatial and Semantic Embeddings for Stereo Sound Event Localization in Videos 

**Title (ZH)**: 将空间和语义嵌入集成应用于视频中的立体声事件定位 

**Authors**: Davide Berghi, Philip J. B. Jackson  

**Link**: [PDF](https://arxiv.org/pdf/2509.06598)  

**Abstract**: In this study, we address the multimodal task of stereo sound event localization and detection with source distance estimation (3D SELD) in regular video content. 3D SELD is a complex task that combines temporal event classification with spatial localization, requiring reasoning across spatial, temporal, and semantic dimensions. The last is arguably the most challenging to model. Traditional SELD approaches typically rely on multichannel input, limiting their capacity to benefit from large-scale pre-training due to data constraints. To overcome this, we enhance a standard SELD architecture with semantic information by integrating pre-trained, contrastive language-aligned models: CLAP for audio and OWL-ViT for visual inputs. These embeddings are incorporated into a modified Conformer module tailored for multimodal fusion, which we refer to as the Cross-Modal Conformer. We perform an ablation study on the development set of the DCASE2025 Task3 Stereo SELD Dataset to assess the individual contributions of the language-aligned models and benchmark against the DCASE Task 3 baseline systems. Additionally, we detail the curation process of large synthetic audio and audio-visual datasets used for model pre-training. These datasets were further expanded through left-right channel swapping augmentation. Our approach, combining extensive pre-training, model ensembling, and visual post-processing, achieved second rank in the DCASE 2025 Challenge Task 3 (Track B), underscoring the effectiveness of our method. Future work will explore the modality-specific contributions and architectural refinements. 

**Abstract (ZH)**: 本研究探讨了在常规视频内容中进行立体声声源事件定位与检测（3D SELD）的多模态任务，同时估计声源距离。3D SELD 是一个结合了时间事件分类与空间定位的复杂任务，需要在空间、时间和语义维度上进行推理。语义维度是最难建模的部分。传统 SELD 方法通常依赖多通道输入，由于数据限制，难以充分利用大规模预训练。为此，我们通过将预先训练的对比学习对齐语言模型（CLAP）应用于音频输入和 OWL-ViT 应用于视觉输入，增强了标准 SELD 架构，并将这些嵌入整合到一个针对多模态融合修改的 Conformer 模块中，称其为跨模态 Conformer。我们在 DCASE2025 任务3立体声 SELD 数据集的开发集上进行了消融研究，评估了语言对齐模型的个体贡献，并与 DCASE 任务3基线系统进行了基准测试。此外，我们详细介绍了用于模型预训练的巨大合成音频和视听数据集的制作过程，并通过左右通道交换增强进一步扩展了这些数据集。我们的方法结合了 extensive 预训练、模型集成和视觉后处理，在 DCASE 2025 挑战任务3（Track B）中取得了第二名，证明了该方法的有效性。未来的工作将探索模态特异性贡献和架构改进。 

---
# HAVE: Head-Adaptive Gating and ValuE Calibration for Hallucination Mitigation in Large Language Models 

**Title (ZH)**: HEAD-适应性门控和值校准以减轻大规模语言模型中的幻觉现象 

**Authors**: Xin Tong, Zhi Lin, Jingya Wang, Bo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06596)  

**Abstract**: Large Language Models (LLMs) often produce hallucinations in retrieval-augmented or long-context generation, even when relevant evidence is present. This stems from two issues: head importance is treated as input-agnostic, and raw attention weights poorly reflect each token's true contribution. We present HAVE (Head-Adaptive Gating and ValuE Calibration), a parameter-free decoding framework that directly addresses both challenges. HAVE introduces head-adaptive gating, which performs instance-level soft reweighing of attention heads, and value calibration, which augments attention with the magnitude of value vectors to approximate write-back contribution. Together, these modules construct token-level evidence aligned with model updates and fuse it with the LM distribution through a lightweight uncertainty-scaled policy. HAVE requires no finetuning and operates in a single forward pass, making it efficient and broadly applicable. Experiments across multiple QA benchmarks and LLM families demonstrate that HAVE consistently reduces hallucinations and outperforms strong baselines, including DAGCD, with modest overhead. The framework is transparent, reproducible, and readily integrates with off-the-shelf LLMs, advancing trustworthy generation in real-world settings. 

**Abstract (ZH)**: Large Language Models中的 retrieval-augmented或长上下文生成常常会在相关证据存在的情况下产生幻觉，这源于头部重要性被视为输入无关以及原始注意力权重无法准确反映每个词的真实贡献这两个问题。我们提出了HAVEN（头部自适应门控和价值校准）——一个无需调整参数的解码框架，直接解决了这两个挑战。HAVEN引入了头部自适应门控，进行实例级别的软重权头部注意力，并结合了值向量的大小来近似写入贡献的价值校准。这些模块构造了与模型更新相一致的词级证据，并通过轻量级的不确定性缩放策略将其与语言模型分布融合。HAVEN无需微调，且仅需一次前向传递，使其高效且广泛适用。跨多个问答基准和语言模型系列的实验表明，HAVEN能够一致地减少幻觉并优于包括DAGCD在内的强基线，同时具有适度的开销。该框架透明、可复现且能够无缝集成到现成的语言模型中，推动了可信生成在实际场景中的应用。 

---
# Integrated Detection and Tracking Based on Radar Range-Doppler Feature 

**Title (ZH)**: 基于雷达距离-多普勒特征的综合检测与跟踪 

**Authors**: Chenyu Zhang, Yuanhang Wu, Xiaoxi Ma, Wei Yi  

**Link**: [PDF](https://arxiv.org/pdf/2509.06569)  

**Abstract**: Detection and tracking are the basic tasks of radar systems. Current joint detection tracking methods, which focus on dynamically adjusting detection thresholds from tracking results, still present challenges in fully utilizing the potential of radar signals. These are mainly reflected in the limited capacity of the constant false-alarm rate model to accurately represent information, the insufficient depiction of complex scenes, and the limited information acquired by the tracker. We introduce the Integrated Detection and Tracking based on radar feature (InDT) method, which comprises a network architecture for radar signal detection and a tracker that leverages detection assistance. The InDT detector extracts feature information from each Range-Doppler (RD) matrix and then returns the target position through the feature enhancement module and the detection head. The InDT tracker adaptively updates the measurement noise covariance of the Kalman filter based on detection confidence. The similarity of target RD features is measured by cosine distance, which enhances the data association process by combining location and feature information. Finally, the efficacy of the proposed method was validated through testing on both simulated data and publicly available datasets. 

**Abstract (ZH)**: 基于雷达特征的综合检测与跟踪方法（InDT） 

---
# Contrastive Self-Supervised Network Intrusion Detection using Augmented Negative Pairs 

**Title (ZH)**: 对比自监督网络入侵检测使用增强负样本对 

**Authors**: Jack Wilkie, Hanan Hindy, Christos Tachtatzis, Robert Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2509.06550)  

**Abstract**: Network intrusion detection remains a critical challenge in cybersecurity. While supervised machine learning models achieve state-of-the-art performance, their reliance on large labelled datasets makes them impractical for many real-world applications. Anomaly detection methods, which train exclusively on benign traffic to identify malicious activity, suffer from high false positive rates, limiting their usability. Recently, self-supervised learning techniques have demonstrated improved performance with lower false positive rates by learning discriminative latent representations of benign traffic. In particular, contrastive self-supervised models achieve this by minimizing the distance between similar (positive) views of benign traffic while maximizing it between dissimilar (negative) views. Existing approaches generate positive views through data augmentation and treat other samples as negative. In contrast, this work introduces Contrastive Learning using Augmented Negative pairs (CLAN), a novel paradigm for network intrusion detection where augmented samples are treated as negative views - representing potentially malicious distributions - while other benign samples serve as positive views. This approach enhances both classification accuracy and inference efficiency after pretraining on benign traffic. Experimental evaluation on the Lycos2017 dataset demonstrates that the proposed method surpasses existing self-supervised and anomaly detection techniques in a binary classification task. Furthermore, when fine-tuned on a limited labelled dataset, the proposed approach achieves superior multi-class classification performance compared to existing self-supervised models. 

**Abstract (ZH)**: 网络入侵检测仍然是网络安全中的一个关键挑战。虽然监督机器学习模型取得了最先进的性能，但它们对大型标注数据集的依赖使其在许多实际应用中 impractical。仅通过良性流量进行训练以识别恶意活动的异常检测方法遭受高误报率的限制，这限制了它们的应用。最近，自我监督学习技术通过学习良性流量的判别潜在表示，展现了改进的性能和较低的误报率。特别是对比自我监督模型通过最小化类似（正面）良性流量视图之间的距离，同时最大化不相似（负面）视图之间的距离来实现这一点。现有方法通过数据增强生成正面视图，并将其他样本视为负面。相比之下，本工作引入了使用增强负样本对的对比学习（CLAN），这是一种新颖的网络入侵检测范式，在良性流量上预训练后，增强样本被视作负面视图——代表潜在恶意流量分布，而其他良性样本作为正面视图。此方法在预训练后提高了分类准确性和推理效率。在 Lycos2017 数据集上的实验评估表明，所提出的方法在二分类任务中超过了现有的自我监督和异常检测技术。此外，当在有限的标注数据集上进行微调时，所提出的方法在多分类任务中表现优于现有自我监督模型。 

---
# Signal-Based Malware Classification Using 1D CNNs 

**Title (ZH)**: 基于信号的恶意软件分类方法研究：采用1D CNNs 

**Authors**: Jack Wilkie, Hanan Hindy, Ivan Andonovic, Christos Tachtatzis, Robert Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2509.06548)  

**Abstract**: Malware classification is a contemporary and ongoing challenge in cyber-security: modern obfuscation techniques are able to evade traditional static analysis, while dynamic analysis is too resource intensive to be deployed at a large scale. One prominent line of research addresses these limitations by converting malware binaries into 2D images by heuristically reshaping them into a 2D grid before resizing using Lanczos resampling. These images can then be classified based on their textural information using computer vision approaches. While this approach can detect obfuscated malware more effectively than static analysis, the process of converting files into 2D images results in significant information loss due to both quantisation noise, caused by rounding to integer pixel values, and the introduction of 2D dependencies which do not exist in the original data. This loss of signal limits the classification performance of the downstream model. This work addresses these weaknesses by instead resizing the files into 1D signals which avoids the need for heuristic reshaping, and additionally these signals do not suffer from quantisation noise due to being stored in a floating-point format. It is shown that existing 2D CNN architectures can be readily adapted to classify these 1D signals for improved performance. Furthermore, a bespoke 1D convolutional neural network, based on the ResNet architecture and squeeze-and-excitation layers, was developed to classify these signals and evaluated on the MalNet dataset. It was found to achieve state-of-the-art performance on binary, type, and family level classification with F1 scores of 0.874, 0.503, and 0.507, respectively, paving the way for future models to operate on the proposed signal modality. 

**Abstract (ZH)**: 恶意软件分类是网络安全领域的一个当代和持续性挑战：现代混淆技术能够规避传统静态分析，而动态分析因资源密集型而不易大规模部署。一条突出的研究路线通过启发式地将恶意软件二进制文件重塑为2D网格，然后使用兰契兹重采样进行调整，将文件转换成2D图像。随后可以利用计算机视觉方法根据纹理信息对这些图像进行分类。尽管这种方法比静态分析更能检测被混淆的恶意软件，但将文件转换成2D图像的过程会导致因量化噪声（由舍入到整数像素值引起）和引入不存在于原始数据中的2D依赖关系而导致的信息显著丢失，而这限制了后续模型的分类性能。本研究通过将文件直接调整为1D信号来解决这些弱点，避免了启发式重塑的需要，而且由于以浮点格式存储，这些信号不会遭受量化噪声的影响。研究表明，现有的2D CNN架构可以轻松适配来分类这些1D信号从而获得更好的性能。此外，基于ResNet架构和挤压-激励层开发了一种专门的1D卷积神经网络，并在MalNet数据集上进行了评估，实现了二进制、类型和家族水平分类的最佳性能，F1得分分别为0.874、0.503和0.507，为未来的模型提供了一种新的信号模态操作的途径。 

---
# Learning Optimal Defender Strategies for CAGE-2 using a POMDP Model 

**Title (ZH)**: 基于POMDP模型学习CAGE-2的最佳防御策略 

**Authors**: Duc Huy Le, Rolf Stadler  

**Link**: [PDF](https://arxiv.org/pdf/2509.06539)  

**Abstract**: CAGE-2 is an accepted benchmark for learning and evaluating defender strategies against cyberattacks. It reflects a scenario where a defender agent protects an IT infrastructure against various attacks. Many defender methods for CAGE-2 have been proposed in the literature. In this paper, we construct a formal model for CAGE-2 using the framework of Partially Observable Markov Decision Process (POMDP). Based on this model, we define an optimal defender strategy for CAGE-2 and introduce a method to efficiently learn this strategy. Our method, called BF-PPO, is based on PPO, and it uses particle filter to mitigate the computational complexity due to the large state space of the CAGE-2 model. We evaluate our method in the CAGE-2 CybORG environment and compare its performance with that of CARDIFF, the highest ranked method on the CAGE-2 leaderboard. We find that our method outperforms CARDIFF regarding the learned defender strategy and the required training time. 

**Abstract (ZH)**: CAGE-2是学习和评估网络防御策略的公认的基准，它反映了防御代理保护IT基础设施免受各种攻击的场景。文献中提出了许多针对CAGE-2的防御方法。在本文中，我们使用部分可观测马尔可夫决策过程（POMDP）框架构建了CAGE-2的正式模型，并在此基础上定义了CAGE-2的最优防御策略，并介绍了一种高效学习该策略的方法。我们的方法称为BF-PPO，基于PPO，并使用粒子滤波来缓解由于CAGE-2模型状态空间庞大而导致的计算复杂性。我们在CAGE-2 CybORG环境中评估了该方法，并将其性能与CAGE-2榜单上最高排名的CARDIFF方法进行了比较。我们发现，我们的方法在学习到的防御策略和所需的训练时间方面优于CARDIFF。 

---
# On the Reproducibility of "FairCLIP: Harnessing Fairness in Vision-Language Learning'' 

**Title (ZH)**: “FairCLIP： Harnessing Fairness in Vision-Language Learning” 的再现性研究 

**Authors**: Hua Chang Bakker, Stan Fris, Angela Madelon Bernardy, Stan Deutekom  

**Link**: [PDF](https://arxiv.org/pdf/2509.06535)  

**Abstract**: We investigated the reproducibility of FairCLIP, proposed by Luo et al. (2024), for improving the group fairness of CLIP (Radford et al., 2021) by minimizing image-text similarity score disparities across sensitive groups using the Sinkhorn distance. The experimental setup of Luo et al. (2024) was reproduced to primarily investigate the research findings for FairCLIP. The model description by Luo et al. (2024) was found to differ from the original implementation. Therefore, a new implementation, A-FairCLIP, is introduced to examine specific design choices. Furthermore, FairCLIP+ is proposed to extend the FairCLIP objective to include multiple attributes. Additionally, the impact of the distance minimization on FairCLIP's fairness and performance was explored. In alignment with the original authors, CLIP was found to be biased towards certain demographics when applied to zero-shot glaucoma classification using medical scans and clinical notes from the Harvard-FairVLMed dataset. However, the experimental results on two datasets do not support their claim that FairCLIP improves the performance and fairness of CLIP. Although the regularization objective reduces Sinkhorn distances, both the official implementation and the aligned implementation, A-FairCLIP, were not found to improve performance nor fairness in zero-shot glaucoma classification. 

**Abstract (ZH)**: 我们调查了由Luo等（2024）提出的FairCLIP在通过最小化敏感群体间的图像-文本相似性得分差异来提高CLIP（Radford等，2021）的分组公平性方面的可再现性，使用Sinkhorn距离。我们复制了Luo等（2024）的研究设置，主要研究FairCLIP的研究发现。发现Luo等（2024）的模型描述与原始实现不同，因此提出了一种新的实现A-FairCLIP，以检查特定的设计选择。此外，我们提出了FairCLIP+，将其公平性目标扩展到包括多个属性。我们还探讨了距离最小化对FairCLIP的公平性和性能的影响。与原始作者一致，我们发现当使用哈佛-FairVLMed数据集中的医疗影像和临床笔记进行零样本青光眼分类时，CLIP偏向于某些人口统计学特征。然而，两个数据集的实验结果并不支持FairCLIP改善CLIP的性能和公平性的主张。尽管正则化目标减少了Sinkhorn距离，官方实现和对齐实现A-FairCLIP均未发现对零样本青光眼分类的性能和公平性有任何改进。 

---
# SLiNT: Structure-aware Language Model with Injection and Contrastive Training for Knowledge Graph Completion 

**Title (ZH)**: SLiNT：结构感知的语言模型及其注入与对比训练在知识图谱完成中的应用 

**Authors**: Mengxue Yang, Chun Yang, Jiaqi Zhu, Jiafan Li, Jingqi Zhang, Yuyang Li, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.06531)  

**Abstract**: Link prediction in knowledge graphs requires integrating structural information and semantic context to infer missing entities. While large language models offer strong generative reasoning capabilities, their limited exploitation of structural signals often results in structural sparsity and semantic ambiguity, especially under incomplete or zero-shot settings. To address these challenges, we propose SLiNT (Structure-aware Language model with Injection and coNtrastive Training), a modular framework that injects knowledge-graph-derived structural context into a frozen LLM backbone with lightweight LoRA-based adaptation for robust link prediction. Specifically, Structure-Guided Neighborhood Enhancement (SGNE) retrieves pseudo-neighbors to enrich sparse entities and mitigate missing context; Dynamic Hard Contrastive Learning (DHCL) introduces fine-grained supervision by interpolating hard positives and negatives to resolve entity-level ambiguity; and Gradient-Decoupled Dual Injection (GDDI) performs token-level structure-aware intervention while preserving the core LLM parameters. Experiments on WN18RR and FB15k-237 show that SLiNT achieves superior or competitive performance compared with both embedding-based and generation-based baselines, demonstrating the effectiveness of structure-aware representation learning for scalable knowledge graph completion. 

**Abstract (ZH)**: 知识图谱中的链接预测需要整合结构信息和语义上下文以推断缺失实体。尽管大型语言模型提供了强大的生成推理能力，但它们对结构信号的有限利用经常导致结构稀疏性和语义模糊性，尤其是在不完整或零样本设置中。为应对这些挑战，我们提出了一种基于模块化框架的SLiNT（结构感知的语言模型结合注入和对比训练），该框架将知识图谱衍生的结构上下文注入到冻结的LLM主干中，并采用轻量级的LoRA基适应方法，以实现稳健的链接预测。具体而言，结构引导的邻域增强（SGNE）检索伪邻居以丰富稀疏实体并缓解缺失上下文；动态硬对比学习（DHCL）通过插值硬正样本和负样本引入细粒度监督以解决实体级模糊性；梯度解耦双注入（GDDI）执行基于标记的结构感知干预，同时保留核心LLM参数。在WN18RR和FB15k-237上的实验表明，SLiNT 在与基于嵌入和生成的基线模型相比时，实现了更优或竞争力的表现，证明了结构感知表示学习在可扩展知识图谱补全中的有效性。 

---
# Crown, Frame, Reverse: Layer-Wise Scaling Variants for LLM Pre-Training 

**Title (ZH)**: 冠层、框架、反转：大型语言模型预训练的层级規模化变体 

**Authors**: Andrei Baroian, Kasper Notebomer  

**Link**: [PDF](https://arxiv.org/pdf/2509.06518)  

**Abstract**: Transformer-based language models traditionally use uniform (isotropic) layer sizes, yet they ignore the diverse functional roles that different depths can play and their computational capacity needs. Building on Layer-Wise Scaling (LWS) and pruning literature, we introduce three new LWS variants - Framed, Reverse, and Crown - that redistribute FFN widths and attention heads via two or three-point linear interpolation in the pre-training stage. We present the first systematic ablation of LWS and its variants, on a fixed budget of 180M parameters, trained on 5B tokens. All models converge to similar losses and achieve better performance compared to an equal-cost isotropic baseline, without a substantial decrease in training throughput. This work represents an initial step into the design space of layer-wise architectures for pre-training, but future work should scale experiments to orders of magnitude more tokens and parameters to fully assess their potential. 

**Abstract (ZH)**: 基于Transformer的语言模型通常采用均匀的层尺寸，却忽视了不同深度在功能角色和计算能力需求方面表现出的多样性。在Layer-Wise Scaling (LWS) 和剪枝文献的基础上，我们提出了三种新的LWS变体——Framed、Reverse和Crown，在预训练阶段通过两或三点线性插值重新分配FFN宽度和注意力头。我们首次在固定参数预算（180M参数）和5Btoken的数据训练下，系统性地研究了LWS及其变体。所有模型收敛于相似的损失值，并在成本相同时比均匀基线获得了更好的性能，而训练吞吐量并未显著下降。这项工作代表了在预训练层次架构设计空间上迈出的初步一步，但未来的工作应该扩展实验规模，以大量更多的token和参数进行全面评估。 

---
# QualityFM: a Multimodal Physiological Signal Foundation Model with Self-Distillation for Signal Quality Challenges in Critically Ill Patients 

**Title (ZH)**: QualityFM：一种用于重症患者信号质量挑战的多模态生理信号基础模型与自蒸馏方法 

**Authors**: Zongheng Guo, Tao Chen, Manuela Ferrario  

**Link**: [PDF](https://arxiv.org/pdf/2509.06516)  

**Abstract**: Photoplethysmogram (PPG) and electrocardiogram (ECG) are commonly recorded in intesive care unit (ICU) and operating room (OR). However, the high incidence of poor, incomplete, and inconsistent signal quality, can lead to false alarms or diagnostic inaccuracies. The methods explored so far suffer from limited generalizability, reliance on extensive labeled data, and poor cross-task transferability. To overcome these challenges, we introduce QualityFM, a novel multimodal foundation model for these physiological signals, designed to acquire a general-purpose understanding of signal quality. Our model is pre-trained on an large-scale dataset comprising over 21 million 30-second waveforms and 179,757 hours of data. Our approach involves a dual-track architecture that processes paired physiological signals of differing quality, leveraging a self-distillation strategy where an encoder for high-quality signals is used to guide the training of an encoder for low-quality signals. To efficiently handle long sequential signals and capture essential local quasi-periodic patterns, we integrate a windowed sparse attention mechanism within our Transformer-based model. Furthermore, a composite loss function, which combines direct distillation loss on encoder outputs with indirect reconstruction loss based on power and phase spectra, ensures the preservation of frequency-domain characteristics of the signals. We pre-train three models with varying parameter counts (9.6 M to 319 M) and demonstrate their efficacy and practical value through transfer learning on three distinct clinical tasks: false alarm of ventricular tachycardia detection, the identification of atrial fibrillation and the estimation of arterial blood pressure (ABP) from PPG and ECG signals. 

**Abstract (ZH)**: 光电容积描记信号（PPG）和心电图（ECG）在重症监护室（ICU）和手术室（OR）中常见记录。然而，信号质量差、不完整和不一致的高发性可能导致误报警或诊断错误。迄今为止探索的方法受限于泛化能力有限、对大量标注数据的依赖以及跨任务迁移性差等问题。为克服这些挑战，我们提出了一种名为QualityFM的新型多模态基础模型，旨在获取信号质量的通用理解。该模型在包含超过2100万个30秒波形和179757小时数据的大规模数据集上进行预训练。我们的方法涉及一种双轨架构，处理不同质量的配对生理信号，通过自蒸馏策略，高质量信号编码器用于指导低质量信号编码器的训练。为高效处理长序列信号并捕获重要局部准周期模式，我们在基于Transformer的模型中整合了窗口稀疏注意力机制。此外，结合直接蒸馏损失和基于功率谱和相位谱的间接重构损失的综合损失函数，确保信号频域特征的保留。我们分别以不同参数量（9.6 M至319 M）预训练三个模型，并通过在三种不同的临床任务中进行迁移学习，展示了其有效性及实用价值：心室颤动误报警检测、心房颤动识别以及从PPG和ECG信号估计动脉血压（ABP）。 

---
# DyC-STG: Dynamic Causal Spatio-Temporal Graph Network for Real-time Data Credibility Analysis in IoT 

**Title (ZH)**: DyC-STG：动态因果时空图网络在物联网实时数据可信性分析中的应用 

**Authors**: Guanjie Cheng, Boyi Li, Peihan Wu, Feiyi Chen, Xinkui Zhao, Mengying Zhu, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2509.06483)  

**Abstract**: The wide spreading of Internet of Things (IoT) sensors generates vast spatio-temporal data streams, but ensuring data credibility is a critical yet unsolved challenge for applications like smart homes. While spatio-temporal graph (STG) models are a leading paradigm for such data, they often fall short in dynamic, human-centric environments due to two fundamental limitations: (1) their reliance on static graph topologies, which fail to capture physical, event-driven dynamics, and (2) their tendency to confuse spurious correlations with true causality, undermining robustness in human-centric environments. To address these gaps, we propose the Dynamic Causal Spatio-Temporal Graph Network (DyC-STG), a novel framework designed for real-time data credibility analysis in IoT. Our framework features two synergistic contributions: an event-driven dynamic graph module that adapts the graph topology in real-time to reflect physical state changes, and a causal reasoning module to distill causally-aware representations by strictly enforcing temporal precedence. To facilitate the research in this domain we release two new real-world datasets. Comprehensive experiments show that DyC-STG establishes a new state-of-the-art, outperforming the strongest baselines by 1.4 percentage points and achieving an F1-Score of up to 0.930. 

**Abstract (ZH)**: 物联网传感器的广泛普及生成了大量的时空数据流，但在如智能家庭等应用中确保数据的可信度仍是一个关键且未解决的挑战。虽然时空图（STG）模型是处理此类数据的主要范式，但在动态的人本环境中，它们常常受限于两个根本性的局限性：（1）依赖静态的图形拓扑，这无法捕捉物理和事件驱动的动力学，（2）倾向于将虚假的相关性误认为真正的原因，从而在人本环境中削弱了鲁棒性。为了解决这些不足，我们提出了一种名为动态因果时空图网络（DyC-STG）的新颖框架，专门用于物联网中的实时数据可信度分析。该框架包含两个协同贡献：一种事件驱动的动态图模块，能够实时调整图形拓扑以反映物理状态变化，以及一种因果推理模块，通过严格遵守时间顺序来提取因果感知的表示。为促进该领域的研究，我们发布了两个新的现实世界数据集。全面的实验表明，DyC-STG 达到了新的最佳水平，在最强基线的基础上提高了 1.4 个百分点，并实现了高达 0.930 的 F1 分数。 

---
# Explained, yet misunderstood: How AI Literacy shapes HR Managers' interpretation of User Interfaces in Recruiting Recommender Systems 

**Title (ZH)**: 解释 yet 误解：AI 文盲如何影响人力资源经理对招聘推荐系统用户界面的解读 

**Authors**: Yannick Kalff, Katharina Simbeck  

**Link**: [PDF](https://arxiv.org/pdf/2509.06475)  

**Abstract**: AI-based recommender systems increasingly influence recruitment decisions. Thus, transparency and responsible adoption in Human Resource Management (HRM) are critical. This study examines how HR managers' AI literacy influences their subjective perception and objective understanding of explainable AI (XAI) elements in recruiting recommender dashboards. In an online experiment, 410 German-based HR managers compared baseline dashboards to versions enriched with three XAI styles: important features, counterfactuals, and model criteria. Our results show that the dashboards used in practice do not explain AI results and even keep AI elements opaque. However, while adding XAI features improves subjective perceptions of helpfulness and trust among users with moderate or high AI literacy, it does not increase their objective understanding. It may even reduce accurate understanding, especially with complex explanations. Only overlays of important features significantly aided the interpretations of high-literacy users. Our findings highlight that the benefits of XAI in recruitment depend on users' AI literacy, emphasizing the need for tailored explanation strategies and targeted literacy training in HRM to ensure fair, transparent, and effective adoption of AI. 

**Abstract (ZH)**: 基于AI的推荐系统日益影响招聘决策。因此，人力资源管理中透明性和负责任的采用至关重要。本研究探讨了人力资源经理的AI素养如何影响他们对招聘推荐仪表板中可解释AI（XAI）元素的主观感知和客观理解。通过一项在线实验，410名德国人力资源经理将基线仪表板与增加了三种XAI风格（关键特征、反事实和模型标准）的版本进行了比较。结果显示，实践中使用的仪表板并未解释AI结果，甚至使AI元素变得不透明。然而，尽管增加了XAI功能可以改善中等或高水平AI素养用户对帮助性和可信度的主观感知，但并没有提高他们的客观理解能力。甚至可能降低准确理解，尤其是在复杂解释的情况下。只有关键特征的覆盖层显著帮助高水平素养用户进行解释。我们的研究结果强调，招聘中的XAI益处取决于用户AI素养，强调了在人力资源管理中制定针对性解释策略和能力培训的必要性，以确保AI的公平、透明和有效采用。 

---
# Several Performance Bounds on Decentralized Online Optimization are Highly Conservative and Potentially Misleading 

**Title (ZH)**: 几个去中心化在线优化的性能上限可能极具保守性且可能存在误导性 

**Authors**: Erwan Meunier, Julien M. Hendrickx  

**Link**: [PDF](https://arxiv.org/pdf/2509.06466)  

**Abstract**: We analyze Decentralized Online Optimization algorithms using the Performance Estimation Problem approach which allows, to automatically compute exact worst-case performance of optimization algorithms. Our analysis shows that several available performance guarantees are very conservative, sometimes by multiple orders of magnitude, and can lead to misguided choices of algorithm. Moreover, at least in terms of worst-case performance, some algorithms appear not to benefit from inter-agent communications for a significant period of time. We show how to improve classical methods by tuning their step-sizes, and find that we can save up to 20% on their actual worst-case performance regret. 

**Abstract (ZH)**: 我们使用性能估计问题方法分析去中心化在线优化算法，该方法允许自动计算优化算法的精确最坏情况性能。我们的分析表明，多种可用的性能保证非常保守，有时相差几个数量级，并可能导致对算法的选择产生误导。此外，至少从最坏情况性能的角度来看，某些算法似乎在相当长的一段时间内并未从代理间的通信中受益。我们展示了如何通过调整步长改进经典方法，并发现可以将其实际最坏情况性能后悔值节省多达20%。 

---
# Focusing by Contrastive Attention: Enhancing VLMs' Visual Reasoning 

**Title (ZH)**: 基于对比注意力的聚焦增强：提升VLMs的视觉推理能力 

**Authors**: Yuyao Ge, Shenghua Liu, Yiwei Wang, Lingrui Mei, Baolong Bi, Xuanshan Zhou, Jiayu Yao, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.06461)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated remarkable success across diverse visual tasks, yet their performance degrades in complex visual environments. While existing enhancement approaches require additional training, rely on external segmentation tools, or operate at coarse-grained levels, they overlook the innate ability within VLMs. To bridge this gap, we investigate VLMs' attention patterns and discover that: (1) visual complexity strongly correlates with attention entropy, negatively impacting reasoning performance; (2) attention progressively refines from global scanning in shallow layers to focused convergence in deeper layers, with convergence degree determined by visual complexity. (3) Theoretically, we prove that the contrast of attention maps between general queries and task-specific queries enables the decomposition of visual signal into semantic signals and visual noise components. Building on these insights, we propose Contrastive Attention Refinement for Visual Enhancement (CARVE), a training-free method that extracts task-relevant visual signals through attention contrasting at the pixel level. Extensive experiments demonstrate that CARVE consistently enhances performance, achieving up to 75% improvement on open-source models. Our work provides critical insights into the interplay between visual complexity and attention mechanisms, offering an efficient pathway for improving visual reasoning with contrasting attention. 

**Abstract (ZH)**: Vision-Language模型(VLMs)在多样化的视觉任务中取得了显著的成功，但在复杂视觉环境中表现退化。现有增强方法需要额外训练、依赖外部分割工具或在粗粒度级别操作，忽略了VLMs本身的固有能力。为弥合这一差距，我们研究了VLMs的注意力模式，并发现：(1) 视觉复杂性与注意力熵呈强烈正相关，负面影响了推理性能；(2) 注意力从浅层的全局扫描逐步细化到深层的局部集中，集中程度由视觉复杂性决定；(3) 理论上，我们证明了通用查询与任务特定查询的注意力图对比能够将视觉信号分解为语义信号和视觉噪声成分。基于这些见解，我们提出了基于像素级别注意力对比的视觉增强方法(CARVE)，一种无需训练的方法，通过注意力对比提取任务相关视觉信号。大量实验证明，CARVE一致地提升了性能，开源模型最多可提升75%。我们的工作为理解视觉复杂性和注意力机制之间的互动提供了关键见解，提供了一条通过对比注意力提高视觉推理效率的途径。 

---
# HECATE: An ECS-based Framework for Teaching and Developing Multi-Agent Systems 

**Title (ZH)**: HECATE：基于ECS的多Agent系统教学与开发框架 

**Authors**: Arthur Casals, Anarosa A. F. Brandão  

**Link**: [PDF](https://arxiv.org/pdf/2509.06431)  

**Abstract**: This paper introduces HECATE, a novel framework based on the Entity-Component-System (ECS) architectural pattern that bridges the gap between distributed systems engineering and MAS development. HECATE is built using the Entity-Component-System architectural pattern, leveraging data-oriented design to implement multiagent systems. This approach involves engineering multiagent systems (MAS) from a distributed systems (DS) perspective, integrating agent concepts directly into the DS domain. This approach simplifies MAS development by (i) reducing the need for specialized agent knowledge and (ii) leveraging familiar DS patterns and standards to minimize the agent-specific knowledge required for engineering MAS. We present the framework's architecture, core components, and implementation approach, demonstrating how it supports different agent models. 

**Abstract (ZH)**: 基于实体-组件-系统架构模式的HECATE框架：分布式系统工程与多Agent系统开发的桥梁 

---
# Musculoskeletal simulation of limb movement biomechanics in Drosophila melanogaster 

**Title (ZH)**: 果蝇 melanogaster 肢体运动骨肌力学的模拟 

**Authors**: Pembe Gizem Özdil, Chuanfang Ning, Jasper S. Phelps, Sibo Wang-Chen, Guy Elisha, Alexander Blanke, Auke Ijspeert, Pavan Ramdya  

**Link**: [PDF](https://arxiv.org/pdf/2509.06426)  

**Abstract**: Computational models are critical to advance our understanding of how neural, biomechanical, and physical systems interact to orchestrate animal behaviors. Despite the availability of near-complete reconstructions of the Drosophila melanogaster central nervous system, musculature, and exoskeleton, anatomically and physically grounded models of fly leg muscles are still missing. These models provide an indispensable bridge between motor neuron activity and joint movements. Here, we introduce the first 3D, data-driven musculoskeletal model of Drosophila legs, implemented in both OpenSim and MuJoCo simulation environments. Our model incorporates a Hill-type muscle representation based on high-resolution X-ray scans from multiple fixed specimens. We present a pipeline for constructing muscle models using morphological imaging data and for optimizing unknown muscle parameters specific to the fly. We then combine our musculoskeletal models with detailed 3D pose estimation data from behaving flies to achieve muscle-actuated behavioral replay in OpenSim. Simulations of muscle activity across diverse walking and grooming behaviors predict coordinated muscle synergies that can be tested experimentally. Furthermore, by training imitation learning policies in MuJoCo, we test the effect of different passive joint properties on learning speed and find that damping and stiffness facilitate learning. Overall, our model enables the investigation of motor control in an experimentally tractable model organism, providing insights into how biomechanics contribute to generation of complex limb movements. Moreover, our model can be used to control embodied artificial agents to generate naturalistic and compliant locomotion in simulated environments. 

**Abstract (ZH)**: 基于计算模型探讨果蝇腿部神经、生物力学和物理系统如何协同 orchestrating 动物行为至关重要。尽管获得了近乎完整重建的果蝇中枢神经系统、肌肉和外骨骼，但仍缺少基于解剖和物理的果蝇后腿肌肉模型。这些模型提供了从运动神经元活动到关节运动的必不可少的桥梁。在这里，我们介绍了一个首个基于数据的果蝇腿部3D肌骨模型，该模型分别在OpenSim和MuJoCo仿真环境中实现。我们的模型基于高分辨率X射线扫描数据构建Hill型肌肉表示。我们提出了一种使用形态成像数据构建肌肉模型的管道，并优化了特定于果蝇的未知肌肉参数。然后，我们将肌骨模型与活跃果蝇的详细3D姿态估计数据结合，在OpenSim中实现肌肉驱动的行为回放。对不同步行和梳理行为的肌肉活动进行的模拟预测了协调的肌肉协同作用，这些协同作用可以进行实验验证。此外，通过在MuJoCo中训练模仿学习策略，我们测试了不同被动关节特性对学习速度的影响，发现阻尼和刚度有助于学习。总体而言，我们的模型使我们能够研究实验可操作模式动物的运动控制，提供了关于生物力学如何贡献于复杂肢体运动生成的见解。此外，我们的模型可以用于控制体现式人工代理，以在虚拟环境中生成自然和顺应的运动。 

---
# CAPMix: Robust Time Series Anomaly Detection Based on Abnormal Assumptions with Dual-Space Mixup 

**Title (ZH)**: CAPMix：基于异常假设的双空间混合时间序列异常检测 

**Authors**: Xudong Mou, Rui Wang, Tiejun Wang, Renyu Yang, Shiru Chen, Jie Sun, Tianyu Wo, Xudong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06419)  

**Abstract**: Time series anomaly detection (TSAD) is a vital yet challenging task, particularly in scenarios where labeled anomalies are scarce and temporal dependencies are complex. Recent anomaly assumption (AA) approaches alleviate the lack of anomalies by injecting synthetic samples and training discriminative models. Despite promising results, these methods often suffer from two fundamental limitations: patchy generation, where scattered anomaly knowledge leads to overly simplistic or incoherent anomaly injection, and Anomaly Shift, where synthetic anomalies either resemble normal data too closely or diverge unrealistically from real anomalies, thereby distorting classification boundaries. In this paper, we propose CAPMix, a controllable anomaly augmentation framework that addresses both issues. First, we design a CutAddPaste mechanism to inject diverse and complex anomalies in a targeted manner, avoiding patchy generation. Second, we introduce a label revision strategy to adaptively refine anomaly labels, reducing the risk of anomaly shift. Finally, we employ dual-space mixup within a temporal convolutional network to enforce smoother and more robust decision boundaries. Extensive experiments on five benchmark datasets, including AIOps, UCR, SWaT, WADI, and ESA, demonstrate that CAPMix achieves significant improvements over state-of-the-art baselines, with enhanced robustness against contaminated training data. The code is available at this https URL. 

**Abstract (ZH)**: 可控异常增强框架CAPMix：解决异常生成片段化和异常偏移问题 

---
# Index-Preserving Lightweight Token Pruning for Efficient Document Understanding in Vision-Language Models 

**Title (ZH)**: 基于索引保留的轻量级 token 裁剪以实现高效的视觉-语言模型文档理解 

**Authors**: Jaemin Son, Sujin Choi, Inyong Yun  

**Link**: [PDF](https://arxiv.org/pdf/2509.06415)  

**Abstract**: Recent progress in vision-language models (VLMs) has led to impressive results in document understanding tasks, but their high computational demands remain a challenge. To mitigate the compute burdens, we propose a lightweight token pruning framework that filters out non-informative background regions from document images prior to VLM processing. A binary patch-level classifier removes non-text areas, and a max-pooling refinement step recovers fragmented text regions to enhance spatial coherence. Experiments on real-world document datasets demonstrate that our approach substantially lowers computational costs, while maintaining comparable accuracy. 

**Abstract (ZH)**: Recent进展在视觉-语言模型中的最新进展已经在文档理解任务中取得了令人印象深刻的成果，但它们的高计算需求仍然是一个挑战。为了缓解计算负担，我们提出了一种轻量级的令牌剪枝框架，在视觉-语言模型处理之前，过滤掉文档图像中的非信息性背景区域。二进制patches级分类器移除非文本区域，最大池化精炼步骤恢复片段化的文本区域以增强空间一致性。实验结果表明，我们的方法显著降低了计算成本，同时保持了相当的准确性。 

---
# MeanFlow-Accelerated Multimodal Video-to-Audio Synthesis via One-Step Generation 

**Title (ZH)**: 基于MeanFlow加速的一步生成多模态视频到音频合成 

**Authors**: Xiaoran Yang, Jianxuan Yang, Xinyue Guo, Haoyu Wang, Ningning Pan, Gongping Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06389)  

**Abstract**: A key challenge in synthesizing audios from silent videos is the inherent trade-off between synthesis quality and inference efficiency in existing methods. For instance, flow matching based models rely on modeling instantaneous velocity, inherently require an iterative sampling process, leading to slow inference speeds. To address this efficiency bottleneck, we introduce a MeanFlow-accelerated model that characterizes flow fields using average velocity, enabling one-step generation and thereby significantly accelerating multimodal video-to-audio (VTA) synthesis while preserving audio quality, semantic alignment, and temporal synchronization. Furthermore, a scalar rescaling mechanism is employed to balance conditional and unconditional predictions when classifier-free guidance (CFG) is applied, effectively mitigating CFG-induced distortions in one step generation. Since the audio synthesis network is jointly trained with multimodal conditions, we further evaluate it on text-to-audio (TTA) synthesis task. Experimental results demonstrate that incorporating MeanFlow into the network significantly improves inference speed without compromising perceptual quality on both VTA and TTA synthesis tasks. 

**Abstract (ZH)**: 从静默视频合成音频的关键挑战在于现有方法中合成质量与推理效率之间的固有trade-off。为此，我们提出一种MeanFlow加速模型，采用平均速度表征流场，实现一步生成，从而显著加速多模态视频到音频（VTA）合成，同时保持音频质量、语义对齐和时间同步。此外，还引入了一种尺度缩放机制，在应用无条件引导（CFG）时平衡条件和非条件预测，有效地在一步生成中减轻CFG引起的失真。由于音频合成网络与多模态条件联合训练，进一步在文本到音频（TTA）合成任务上进行评估。实验结果表明，将MeanFlow纳入网络可显著提高推理速度，而不影响VTA和TTA合成任务上的感知质量。 

---
# Beyond the Pre-Service Horizon: Infusing In-Service Behavior for Improved Financial Risk Forecasting 

**Title (ZH)**: 超越入职视野：注入在职行为以提高财务风险预测 

**Authors**: Senhao Liu, Zhiyu Guo, Zhiyuan Ji, Yueguo Chen, Yateng Tang, Yunhai Wang, Xuehao Zheng, Xiang Ao  

**Link**: [PDF](https://arxiv.org/pdf/2509.06385)  

**Abstract**: Typical financial risk management involves distinct phases for pre-service risk assessment and in-service default detection, often modeled separately. This paper proposes a novel framework, Multi-Granularity Knowledge Distillation (abbreviated as MGKD), aimed at improving pre-service risk prediction through the integration of in-service user behavior data. MGKD follows the idea of knowledge distillation, where the teacher model, trained on historical in-service data, guides the student model, which is trained on pre-service data. By using soft labels derived from in-service data, the teacher model helps the student model improve its risk prediction prior to service activation. Meanwhile, a multi-granularity distillation strategy is introduced, including coarse-grained, fine-grained, and self-distillation, to align the representations and predictions of the teacher and student models. This approach not only reinforces the representation of default cases but also enables the transfer of key behavioral patterns associated with defaulters from the teacher to the student model, thereby improving the overall performance of pre-service risk assessment. Moreover, we adopt a re-weighting strategy to mitigate the model's bias towards the minority class. Experimental results on large-scale real-world datasets from Tencent Mobile Payment demonstrate the effectiveness of our proposed approach in both offline and online scenarios. 

**Abstract (ZH)**: 面向服务的多粒度知识蒸馏框架：通过集成在服务中用户行为数据以改进预服务风险预测 

---
# MRD-LiNet: A Novel Lightweight Hybrid CNN with Gradient-Guided Unlearning for Improved Drought Stress Identification 

**Title (ZH)**: MRD-LiNet: 一种带有梯度导向遗忘的新型轻量级混合CNN及其在改善旱灾 stress 识别中的应用 

**Authors**: Aswini Kumar Patra, Lingaraj Sahoo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06367)  

**Abstract**: Drought stress is a major threat to global crop productivity, making its early and precise detection essential for sustainable agricultural management. Traditional approaches, though useful, are often time-consuming and labor-intensive, which has motivated the adoption of deep learning methods. In recent years, Convolutional Neural Network (CNN) and Vision Transformer architectures have been widely explored for drought stress identification; however, these models generally rely on a large number of trainable parameters, restricting their use in resource-limited and real-time agricultural settings. To address this challenge, we propose a novel lightweight hybrid CNN framework inspired by ResNet, DenseNet, and MobileNet architectures. The framework achieves a remarkable 15-fold reduction in trainable parameters compared to conventional CNN and Vision Transformer models, while maintaining competitive accuracy. In addition, we introduce a machine unlearning mechanism based on a gradient norm-based influence function, which enables targeted removal of specific training data influence, thereby improving model adaptability. The method was evaluated on an aerial image dataset of potato fields with expert-annotated healthy and drought-stressed regions. Experimental results show that our framework achieves high accuracy while substantially lowering computational costs. These findings highlight its potential as a practical, scalable, and adaptive solution for drought stress monitoring in precision agriculture, particularly under resource-constrained conditions. 

**Abstract (ZH)**: 干旱压力是全球作物产量的主要威胁，其早期和精准检测对于可持续农业管理至关重要。传统方法虽然有用，但往往耗时且劳动密集，这促使了深度学习方法的应用。近年来，卷积神经网络（CNN）和视觉变换器架构广泛用于干旱压力识别；然而，这些模型通常依赖大量的可训练参数，限制了其在资源受限和实时农业生产环境中的应用。为解决这一挑战，我们提出了一种受ResNet、DenseNet和MobileNet架构启发的新型轻量级混合CNN框架。该框架与传统CNN和视觉变换器模型相比，实现了可训练参数15倍的减少，同时保持了竞争力的准确性。此外，我们引入了一种基于梯度范数影响函数的机器遗忘机制，能够针对性地去除特定训练数据的影响，从而提高模型的适应性。该方法在具有专家注释的健康和干旱胁迫区域的马铃薯田空中图像数据集上进行了评估。实验结果表明，我们的框架在显著降低计算成本的同时实现了高精度。这些发现突显了其在资源受限条件下精准农业中作为实用、可扩展和适应性干旱胁迫监测解决方案的潜力。 

---
# PL-CA: A Parametric Legal Case Augmentation Framework 

**Title (ZH)**: PL-CA: 一种参数化法律案例扩充框架 

**Authors**: Ao Chang, Yubo Chen, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.06356)  

**Abstract**: Conventional RAG is considered one of the most effective methods for addressing model knowledge insufficiency and hallucination, particularly in the judicial domain that requires high levels of knowledge rigor, logical consistency, and content integrity. However, the conventional RAG method only injects retrieved documents directly into the model's context, which severely constrains models due to their limited context windows and introduces additional computational overhead through excessively long contexts, thereby disrupting models' attention and degrading performance on downstream tasks. Moreover, many existing benchmarks lack expert annotation and focus solely on individual downstream tasks while real-world legal scenarios consist of multiple mixed legal tasks, indicating conventional benchmarks' inadequacy for reflecting models' true capabilities. To address these limitations, we propose PL-CA, which introduces a parametric RAG (P-RAG) framework to perform data augmentation on corpus knowledge and encode this legal knowledge into parametric vectors, and then integrates this parametric knowledge into the LLM's feed-forward networks (FFN) via LoRA, thereby alleviating models' context pressure. Additionally, we also construct a multi-task legal dataset comprising more than 2000 training and test instances, which are all expert-annotated and manually verified. We conduct our experiments on our dataset, and the experimental results demonstrate that our method reduces the overhead associated with excessively long contexts while maintaining competitive performance on downstream tasks compared to conventional RAG. Our code and dataset are provided in the appendix. 

**Abstract (ZH)**: 基于参数化RAG的法律知识增强方法：PL-CA 

---
# Mask-GCG: Are All Tokens in Adversarial Suffixes Necessary for Jailbreak Attacks? 

**Title (ZH)**: Mask-GCG: 所有的对抗后缀中的令牌对于 jailbreak 攻击都是必要的吗？ 

**Authors**: Junjie Mu, Zonghao Ying, Zhekui Fan, Zonglei Jing, Yaoyuan Zhang, Zhengmin Yu, Wenxin Zhang, Quanchen Zou, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06350)  

**Abstract**: Jailbreak attacks on Large Language Models (LLMs) have demonstrated various successful methods whereby attackers manipulate models into generating harmful responses that they are designed to avoid. Among these, Greedy Coordinate Gradient (GCG) has emerged as a general and effective approach that optimizes the tokens in a suffix to generate jailbreakable prompts. While several improved variants of GCG have been proposed, they all rely on fixed-length suffixes. However, the potential redundancy within these suffixes remains unexplored. In this work, we propose Mask-GCG, a plug-and-play method that employs learnable token masking to identify impactful tokens within the suffix. Our approach increases the update probability for tokens at high-impact positions while pruning those at low-impact positions. This pruning not only reduces redundancy but also decreases the size of the gradient space, thereby lowering computational overhead and shortening the time required to achieve successful attacks compared to GCG. We evaluate Mask-GCG by applying it to the original GCG and several improved variants. Experimental results show that most tokens in the suffix contribute significantly to attack success, and pruning a minority of low-impact tokens does not affect the loss values or compromise the attack success rate (ASR), thereby revealing token redundancy in LLM prompts. Our findings provide insights for developing efficient and interpretable LLMs from the perspective of jailbreak attacks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的逃狱攻击：Greedy Coordinate Gradient（Mask-GCG）掩码方法探究 

---
# Ban&Pick: Achieving Free Performance Gains and Inference Speedup via Smarter Routing in MoE-LLMs 

**Title (ZH)**: Ban&Pick：通过更智能的路由实现MoE-LLMs的自由性能提升和推理加速 

**Authors**: Yuanteng Chen, Peisong Wang, Yuantian Shao, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.06346)  

**Abstract**: Sparse Mixture-of-Experts (MoE) has become a key architecture for scaling large language models (LLMs) efficiently. Recent fine-grained MoE designs introduce hundreds of experts per layer, with multiple experts activated per token, enabling stronger specialization. However, during pre-training, routers are optimized mainly for stability and robustness: they converge prematurely and enforce balanced usage, limiting the full potential of model performance and efficiency. In this work, we uncover two overlooked issues: (i) a few highly influential experts are underutilized due to premature and balanced routing decisions; and (ii) enforcing a fixed number of active experts per token introduces substantial redundancy. Instead of retraining models or redesigning MoE architectures, we introduce Ban&Pick, a post-training, plug-and-play strategy for smarter MoE routing. Pick discovers and reinforces key experts-a small group with outsized impact on performance-leading to notable accuracy gains across domains. Ban complements this by dynamically pruning redundant experts based on layer and token sensitivity, delivering faster inference with minimal accuracy loss. Experiments on fine-grained MoE-LLMs (DeepSeek, Qwen3) across math, code, and general reasoning benchmarks demonstrate that Ban&Pick delivers free performance gains and inference acceleration without retraining or architectural changes. For instance, on Qwen3-30B-A3B, it improves accuracy from 80.67 to 84.66 on AIME2024 and from 65.66 to 68.18 on GPQA-Diamond, while accelerating inference by 1.25x under the vLLM. 

**Abstract (ZH)**: Sparse Mixture-of-Experts Routing with Ban&Pick for Efficient Large Language Models 

---
# Multi View Slot Attention Using Paraphrased Texts For Face Anti-Spoofing 

**Title (ZH)**: 基于 paraphrased 文本的多视图槽注意力-face 反冒充 

**Authors**: Jeongmin Yu, Susang Kim, Kisu Lee, Taekyoung Kwon, Won-Yong Shin, Ha Young Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.06336)  

**Abstract**: Recent face anti-spoofing (FAS) methods have shown remarkable cross-domain performance by employing vision-language models like CLIP. However, existing CLIP-based FAS models do not fully exploit CLIP's patch embedding tokens, failing to detect critical spoofing clues. Moreover, these models rely on a single text prompt per class (e.g., 'live' or 'fake'), which limits generalization. To address these issues, we propose MVP-FAS, a novel framework incorporating two key modules: Multi-View Slot attention (MVS) and Multi-Text Patch Alignment (MTPA). Both modules utilize multiple paraphrased texts to generate generalized features and reduce dependence on domain-specific text. MVS extracts local detailed spatial features and global context from patch embeddings by leveraging diverse texts with multiple perspectives. MTPA aligns patches with multiple text representations to improve semantic robustness. Extensive experiments demonstrate that MVP-FAS achieves superior generalization performance, outperforming previous state-of-the-art methods on cross-domain datasets. Code: this https URL. 

**Abstract (ZH)**: 最近的研究表明，通过使用像CLIP这样的视觉语言模型，面部防欺骗（Face Anti-Spoofing, FAS）方法在跨域性能上取得了显著成果。然而，现有的基于CLIP的FAS模型未能充分利用CLIP的 Patch嵌入令牌，不能检测到关键的欺骗线索。此外，这些模型依赖于每类单一的文字提示（例如，“live”或“fake”），这限制了它们的泛化能力。为了解决这些问题，我们提出了一种名为MVP-FAS的新颖框架，该框架包含两个关键模块：多视角槽注意（MVS）和多文本块对齐（MTPA）。这两个模块利用多个同义表达的文本来生成泛化的特征，并减少对特定领域文本的依赖。MVS通过利用多种视角的多样文本提取局部详细的空间特征和全局语境。MTPA通过改进语义稳健性来对齐带有多种文本表示的块。广泛的实验表明，MVP-FAS在泛化性能上表现出优越性，并在跨域数据集上超过了之前的方法。代码：this https URL。 

---
# A Fragile Number Sense: Probing the Elemental Limits of Numerical Reasoning in LLMs 

**Title (ZH)**: 易碎的数感：探究大模型在数值推理中的基本极限 

**Authors**: Roussel Rahman, Aashwin Ananda Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2509.06332)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable emergent capabilities, yet the robustness of their numerical reasoning remains an open question. While standard benchmarks evaluate LLM reasoning on complex problem sets using aggregated metrics, they often obscure foundational weaknesses. In this work, we probe LLM mathematical numeracy by evaluating performance on problems of escalating complexity, from constituent operations to combinatorial puzzles. We test several state-of-the-art LLM-based agents on a 100-problem challenge comprising four categories: (1) basic arithmetic, (2) advanced operations, (3) primality checking, and (4) the Game of 24 number puzzle. Our results show that while the agents achieved high accuracy on the first three categories, which require deterministic algorithmic execution, they consistently failed at the number puzzle, underlining its demand for a heuristic search over a large combinatorial space to be a significant bottleneck. These findings reveal that the agents' proficiency is largely confined to recalling and executing known algorithms, rather than performing generative problem-solving. This suggests their apparent numerical reasoning is more akin to sophisticated pattern-matching than flexible, analytical thought, limiting their potential for tasks that require novel or creative numerical insights. 

**Abstract (ZH)**: 大型语言模型(Large Language Models, LLMs)展示了 remarkable 的 emergent 能力，但其数值推理的 robustness 仍是一个开放问题。虽然标准基准通过聚合指标评估LLM在复杂问题集上的推理能力，但往往掩盖了其基础性的弱点。在这项工作中，我们通过评估其在从基本操作到组合难题等一系列复杂问题上的表现，来探查LLM的数学数理能力。我们测试了几种最先进的基于LLM的代理在一个包含四大类别的100题挑战中的性能：(1) 基本算术，(2) 高级操作，(3) 质数检验，以及(4) 24点数字谜题。结果显示，代理在前三大类问题上实现了高准确率，这些类别需要确定性的算法执行，但在数字谜题上始终失败，突显了解决大规模组合空间所需的启发式搜索是一个重要瓶颈。这些发现表明，代理的技能主要限于回忆和执行已知算法，而非生成性问题解决。这表明它们看似具备的数值推理能力更像是复杂的模式匹配，而不是灵活的、分析性的思考，从而限制了它们在需要新颖或创造性数值洞察的任务上的潜力。 

---
# AttestLLM: Efficient Attestation Framework for Billion-scale On-device LLMs 

**Title (ZH)**: AttestLLM：高效的万亿规模边缘设备大语言模型认证框架 

**Authors**: Ruisi Zhang, Yifei Zhao, Neusha Javidnia, Mengxin Zheng, Farinaz Koushanfar  

**Link**: [PDF](https://arxiv.org/pdf/2509.06326)  

**Abstract**: As on-device LLMs(e.g., Apple on-device Intelligence) are widely adopted to reduce network dependency, improve privacy, and enhance responsiveness, verifying the legitimacy of models running on local devices becomes critical. Existing attestation techniques are not suitable for billion-parameter Large Language Models (LLMs), struggling to remain both time- and memory-efficient while addressing emerging threats in the LLM era. In this paper, we present AttestLLM, the first-of-its-kind attestation framework to protect the hardware-level intellectual property (IP) of device vendors by ensuring that only authorized LLMs can execute on target platforms. AttestLLM leverages an algorithm/software/hardware co-design approach to embed robust watermarking signatures onto the activation distributions of LLM building blocks. It also optimizes the attestation protocol within the Trusted Execution Environment (TEE), providing efficient verification without compromising inference throughput. Extensive proof-of-concept evaluations on LLMs from Llama, Qwen, and Phi families for on-device use cases demonstrate AttestLLM's attestation reliability, fidelity, and efficiency. Furthermore, AttestLLM enforces model legitimacy and exhibits resilience against model replacement and forgery attacks. 

**Abstract (ZH)**: 基于设备端的大型语言模型（如苹果设备端智能）日益普及，以减少网络依赖、提升隐私保护和增强响应速度，验证运行在本地设备上的模型合法性变得至关重要。现有的认证技术不适用于十亿级参数的大型语言模型（LLMs），难以在应对LLM时代新兴威胁的同时保持时间和内存效率。在本文中，我们提出了AttestLLM，这是一种首创的认证框架，旨在通过确保只有授权的LLM能在目标平台上执行来保护设备供应商的硬件层面知识产权（IP）。AttestLLM采用算法/软件/硬件协同设计的方法，在LLM构建块的激活分布中嵌入 robust 水印签名。此外，AttestLLM 在可信执行环境（TEE）中优化了认证协议，提供了高效验证而不牺牲推理吞吐量。针对Llama、Qwen和Phi家族的LLM在设备端使用场景的广泛概念验证评估证明了AttestLLM的认证可靠性和效率。此外，AttestLLM 强制执行模型合法性，并展示了对模型替换和伪造攻击的抗性。 

---
# Learning to Walk with Less: a Dyna-Style Approach to Quadrupedal Locomotion 

**Title (ZH)**: 学习用更少：一种四足行走的Dyna风格方法 

**Authors**: Francisco Affonso, Felipe Andrade G. Tommaselli, Juliano Negri, Vivian S. Medeiros, Mateus V. Gasparino, Girish Chowdhary, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2509.06296)  

**Abstract**: Traditional RL-based locomotion controllers often suffer from low data efficiency, requiring extensive interaction to achieve robust performance. We present a model-based reinforcement learning (MBRL) framework that improves sample efficiency for quadrupedal locomotion by appending synthetic data to the end of standard rollouts in PPO-based controllers, following the Dyna-Style paradigm. A predictive model, trained alongside the policy, generates short-horizon synthetic transitions that are gradually integrated using a scheduling strategy based on the policy update iterations. Through an ablation study, we identified a strong correlation between sample efficiency and rollout length, which guided the design of our experiments. We validated our approach in simulation on the Unitree Go1 robot and showed that replacing part of the simulated steps with synthetic ones not only mimics extended rollouts but also improves policy return and reduces variance. Finally, we demonstrate that this improvement transfers to the ability to track a wide range of locomotion commands using fewer simulated steps. 

**Abstract (ZH)**: 基于模型的强化学习（MBRL）框架通过在PPO控制器的标准卷积末尾附加合成数据，提高四足运动的学习效率，遵循Dyna-Style范式。通过消融研究，我们发现样本效率与卷积长度之间存在较强的相关性，指导了实验设计。我们在模拟环境中使用Unitree Go1机器人验证了该方法，结果显示用合成步骤替换部分模拟步骤不仅模仿了扩展卷积，还提高了策略回报并减小了方差。最后，我们证明了这种改进使机器人能够使用更少的模拟步骤跟踪各种运动命令。 

---
# Statistical Inference for Misspecified Contextual Bandits 

**Title (ZH)**: 错定性上下文_bandits的统计推断 

**Authors**: Yongyi Guo, Ziping Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06287)  

**Abstract**: Contextual bandit algorithms have transformed modern experimentation by enabling real-time adaptation for personalized treatment and efficient use of data. Yet these advantages create challenges for statistical inference due to adaptivity. A fundamental property that supports valid inference is policy convergence, meaning that action-selection probabilities converge in probability given the context. Convergence ensures replicability of adaptive experiments and stability of online algorithms. In this paper, we highlight a previously overlooked issue: widely used algorithms such as LinUCB may fail to converge when the reward model is misspecified, and such non-convergence creates fundamental obstacles for statistical inference. This issue is practically important, as misspecified models -- such as linear approximations of complex dynamic system -- are often employed in real-world adaptive experiments to balance bias and variance.
Motivated by this insight, we propose and analyze a broad class of algorithms that are guaranteed to converge even under model misspecification. Building on this guarantee, we develop a general inference framework based on an inverse-probability-weighted Z-estimator (IPW-Z) and establish its asymptotic normality with a consistent variance estimator. Simulation studies confirm that the proposed method provides robust and data-efficient confidence intervals, and can outperform existing approaches that exist only in the special case of offline policy evaluation. Taken together, our results underscore the importance of designing adaptive algorithms with built-in convergence guarantees to enable stable experimentation and valid statistical inference in practice. 

**Abstract (ZH)**: 上下文臂算法通过实现实时个性化治疗和高效数据利用，已经转变了现代实验。然而，这些优势也带来了统计推断中的挑战，因为它们是适应性的。支持有效推断的一个基本性质是策略收敛，即在给定上下文的情况下动作选择概率收敛。收敛确保了适应性实验的可重复性以及在线算法的稳定性。本文强调了一个之前被忽视的问题：广泛使用的算法如LinUCB，在奖励模型错误指定的情况下可能无法收敛，这种不收敛为统计推断创造了根本障碍。这一问题在实践中具有重要意义，因为许多实际的适应性实验中，如为了权衡偏差和方差而使用复杂的动态系统的线性近似模型时，错误指定的模型通常会被采用。

基于这一见解，我们提出并分析了一类保证在模型错误指定情况下仍然能够收敛的算法。基于这一保证，我们开发了一种通用的推断框架，该框架基于逆概率加权Z估计器（IPW-Z），并建立了其渐近正态性以及一致方差估计器。模拟研究证实，所提出的方法提供了稳健且数据高效的置信区间，并且在仅在离线策略评估中的特殊情况中才有现有方法的表现优于该方法。总的来说，我们的结果强调了设计内置收敛保证的适应性算法的重要性，以实现稳定的实验和有效的统计推断。 

---
# UrbanMIMOMap: A Ray-Traced MIMO CSI Dataset with Precoding-Aware Maps and Benchmarks 

**Title (ZH)**: UrbanMIMOMap：一种包含预编码aware图和基准的射线 tracing MIMO CSI数据集 

**Authors**: Honggang Jia, Xiucheng Wang, Nan Cheng, Ruijin Sun, Changle Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.06270)  

**Abstract**: Sixth generation (6G) systems require environment-aware communication, driven by native artificial intelligence (AI) and integrated sensing and communication (ISAC). Radio maps (RMs), providing spatially continuous channel information, are key enablers. However, generating high-fidelity RM ground truth via electromagnetic (EM) simulations is computationally intensive, motivating machine learning (ML)-based RM construction. The effectiveness of these data-driven methods depends on large-scale, high-quality training data. Current public datasets often focus on single-input single-output (SISO) and limited information, such as path loss, which is insufficient for advanced multi-input multi-output (MIMO) systems requiring detailed channel state information (CSI). To address this gap, this paper presents UrbanMIMOMap, a novel large-scale urban MIMO CSI dataset generated using high-precision ray tracing. UrbanMIMOMap offers comprehensive complex CSI matrices across a dense spatial grid, going beyond traditional path loss data. This rich CSI is vital for constructing high-fidelity RMs and serves as a fundamental resource for data-driven RM generation, including deep learning. We demonstrate the dataset's utility through baseline performance evaluations of representative ML methods for RM construction. This work provides a crucial dataset and reference for research in high-precision RM generation, MIMO spatial performance, and ML for 6G environment awareness. The code and data for this work are available at: this https URL. 

**Abstract (ZH)**: 第六代（6G）系统需要环境感知通信，受原生人工智能（AI）和集成传感与通信（ISAC）驱动。射频地图（RMs），提供连续空间信道信息，是关键使能器。然而，通过电磁（EM）模拟生成高保真RM地面真实值计算密集，促使基于机器学习（ML）的RM构建。这些数据驱动方法的有效性取决于大规模高质量的训练数据。当前的公共数据集通常侧重于单输入单输出（SISO）和有限的信息，如路径损耗，这不足以支持需要详细信道状态信息（CSI）的先进多输入多输出（MIMO）系统。为填补这一空白，本文提出了UrbanMIMOMap，这是一种使用高精度射线追踪生成的大规模城市MIMO CSI数据集。UrbanMIMOMap提供了一系列密集空间网格上的全面复杂CSI矩阵，超越了传统的路径损耗数据。这种丰富的CSI对于构建高保真射频地图至关重要，并作为数据驱动射频地图生成的基础资源，包括深度学习。我们通过代表性的ML方法对射频地图构建的基本性能进行评估，展示了该数据集的应用价值。该工作为高精度射频地图生成、MIMO空间性能以及6G环境感知中的机器学习提供了关键数据集和参考。相关代码和数据可在以下链接获取：this https URL。 

---
# On Synthesis of Timed Regular Expressions 

**Title (ZH)**: 定时正则表达式的合成 

**Authors**: Ziran Wang, Jie An, Naijun Zhan, Miaomiao Zhang, Zhenya Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06262)  

**Abstract**: Timed regular expressions serve as a formalism for specifying real-time behaviors of Cyber-Physical Systems. In this paper, we consider the synthesis of timed regular expressions, focusing on generating a timed regular expression consistent with a given set of system behaviors including positive and negative examples, i.e., accepting all positive examples and rejecting all negative examples. We first prove the decidability of the synthesis problem through an exploration of simple timed regular expressions. Subsequently, we propose our method of generating a consistent timed regular expression with minimal length, which unfolds in two steps. The first step is to enumerate and prune candidate parametric timed regular expressions. In the second step, we encode the requirement that a candidate generated by the first step is consistent with the given set into a Satisfiability Modulo Theories (SMT) formula, which is consequently solved to determine a solution to parametric time constraints. Finally, we evaluate our approach on benchmarks, including randomly generated behaviors from target timed models and a case study. 

**Abstract (ZH)**: 定时正规表达式用于描述 Cyber-Physical 系统的实时行为。本文考虑定时正规表达式的合成问题，重点是生成一个与给定的行为集（包括正例和反例）一致的定时正规表达式，即接受所有正例并拒绝所有反例。我们首先通过探索简单的定时正规表达式证明合成问题的可决定性。随后，我们提出了一种生成最短一致定时正规表达式的方法，该方法分为两步。第一步是枚举并修剪候选的参数化定时正规表达式。第二步是将第一步生成的候选表达式与给定集一致的要求编码为满意度模理论（SMT）公式，进而求解以确定参数时间约束的解。最后，我们在基准测试上评估了我们的方法，包括来自目标定时模型的随机生成行为和一个案例研究。 

---
# Distillation of CNN Ensemble Results for Enhanced Long-Term Prediction of the ENSO Phenomenon 

**Title (ZH)**: CNN集合结果的蒸馏以增强ENSO现象的长期预测能力 

**Authors**: Saghar Ganji, Mohammad Naisipour, Alireza Hassani, Arash Adib  

**Link**: [PDF](https://arxiv.org/pdf/2509.06227)  

**Abstract**: The accurate long-term forecasting of the El Nino Southern Oscillation (ENSO) is still one of the biggest challenges in climate science. While it is true that short-to medium-range performance has been improved significantly using the advances in deep learning, statistical dynamical hybrids, most operational systems still use the simple mean of all ensemble members, implicitly assuming equal skill across members. In this study, we demonstrate, through a strictly a-posteriori evaluation , for any large enough ensemble of ENSO forecasts, there is a subset of members whose skill is substantially higher than that of the ensemble mean. Using a state-of-the-art ENSO forecast system cross-validated against the 1986-2017 observed Nino3.4 index, we identify two Top-5 subsets one ranked on lowest Root Mean Square Error (RMSE) and another on highest Pearson correlation. Generally across all leads, these outstanding members show higher correlation and lower RMSE, with the advantage rising enormously with lead time. Whereas at short leads (1 month) raises the mean correlation by about +0.02 (+1.7%) and lowers the RMSE by around 0.14 °C or by 23.3% compared to the All-40 mean, at extreme leads (23 months) the correlation is raised by +0.43 (+172%) and RMSE by 0.18 °C or by 22.5% decrease. The enhancements are largest during crucial ENSO transition periods such as SON and DJF, when accurate amplitude and phase forecasting is of greatest socio-economic benefit, and furthermore season-dependent e.g., mid-year months such as JJA and MJJ have incredibly large RMSE reductions. This study provides a solid foundation for further investigations to identify reliable clues for detecting high-quality ensemble members, thereby enhancing forecasting skill. 

**Abstract (ZH)**: 准确长期预测厄尔尼诺南方 oscillation (ENSO) 仍然是气候科学中的一个重大挑战。通过严格的事后评估，我们展示，在任何足够大的 ENSO 预测 Ensemble 中，存在一个技能明显高于总体 Ensemble 平均值的子集。基于 1986-2017 年 Nino3.4 指数与最先进的 ENSO 预测系统交叉验证，我们确定了两个顶级子集，一个基于最低均方根误差 (RMSE)，另一个基于最高皮尔逊相关系数。总的来说，在所有预测时长中，这些杰出成员显示出更高的相关性和更低的 RMSE，而该优势随预测时长增加而显著增大。在短期预测（1 个月）中，顶级成员将平均相关性提高了约 +0.02 (+1.7%)，降低 RMSE 约 0.14 °C 或 23.3%，而在极端长期预测（23 个月）中，相关性提高了约 +0.43 (+172%)，降低 RMSE 约 0.18 °C 或 22.5%。这些改进在关键的ENSO 转换期（例如SON 和 DJF）尤为显著，那时准确的振幅和相位预测具有最大的社会经济效益，并且这种改进具有季节依赖性，如中期年份中的 JJA 和 MJJ 月份的 RMSE 减少尤为显著。本研究为后续研究识别可靠线索以检测高质量的 Ensemble 成员奠定了坚实基础，进而提高预测技能。 

---
# Beamforming-LLM: What, Where and When Did I Miss? 

**Title (ZH)**: 波束形成-LLM：我在哪里、何时以及为什么错过了？ 

**Authors**: Vishal Choudhari  

**Link**: [PDF](https://arxiv.org/pdf/2509.06221)  

**Abstract**: We present Beamforming-LLM, a system that enables users to semantically recall conversations they may have missed in multi-speaker environments. The system combines spatial audio capture using a microphone array with retrieval-augmented generation (RAG) to support natural language queries such as, "What did I miss when I was following the conversation on dogs?" Directional audio streams are separated using beamforming, transcribed with Whisper, and embedded into a vector database using sentence encoders. Upon receiving a user query, semantically relevant segments are retrieved, temporally aligned with non-attended segments, and summarized using a lightweight large language model (GPT-4o-mini). The result is a user-friendly interface that provides contrastive summaries, spatial context, and timestamped audio playback. This work lays the foundation for intelligent auditory memory systems and has broad applications in assistive technology, meeting summarization, and context-aware personal spatial computing. 

**Abstract (ZH)**: Beamforming-LLM：一种在多说话人环境中使用户能够语义回溯错过对话的系统 

---
# The Efficiency Frontier: Classical Shadows versus Quantum Footage 

**Title (ZH)**: 经典概影与量子剪影的效率边界 

**Authors**: Shuowei Ma, Junyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06218)  

**Abstract**: Interfacing quantum and classical processors is an important subroutine in full-stack quantum algorithms. The so-called "classical shadow" method efficiently extracts essential classical information from quantum states, enabling the prediction of many properties of a quantum system from only a few measurements. However, for a small number of highly non-local observables, or when classical post-processing power is limited, the classical shadow method is not always the most efficient choice. Here, we address this issue quantitatively by performing a full-stack resource analysis that compares classical shadows with ``quantum footage," which refers to direct quantum measurement. Under certain assumptions, our analysis illustrates a boundary of download efficiency between classical shadows and quantum footage. For observables expressed as linear combinations of Pauli matrices, the classical shadow method outperforms direct measurement when the number of observables is large and the Pauli weight is small. For observables in the form of large Hermitian sparse matrices, the classical shadow method shows an advantage when the number of observables, the sparsity of the matrix, and the number of qubits fall within a certain range. The key parameters influencing this behavior include the number of qubits $n$, observables $M$, sparsity $k$, Pauli weight $w$, accuracy requirement $\epsilon$, and failure tolerance $\delta$. We also compare the resource consumption of the two methods on different types of quantum computers and identify break-even points where the classical shadow method becomes more efficient, which vary depending on the hardware. This paper opens a new avenue for quantitatively designing optimal strategies for hybrid quantum-classical tomography and provides practical insights for selecting the most suitable quantum measurement approach in real-world applications. 

**Abstract (ZH)**: 量子和经典处理器接口是全栈量子算法中的一个重要子程序。所谓的“经典阴影”方法高效地从量子态中提取关键的经典信息，从而仅通过少量测量就能够预测量子系统中的许多性质。然而，对于高度非局域的观测量较少或经典后处理能力有限的情况，“经典阴影”方法并不总是最有效的选择。通过进行全面栈资源分析，我们将“经典阴影”方法与“量子影像”进行了比较，后者是指直接量子测量。在某些假设下，我们的分析展示了经典阴影方法与量子影像之间下载效率的边界。对于用保罗伊矩阵线性组合表示的观测量，当观测量的数量较大且保罗伊权重较小时，“经典阴影”方法优于直接测量。对于用大型厄密稀疏矩阵表示的观测量，在观测量的数量、矩阵的稀疏性和量子比特数处于一定范围内的条件下，“经典阴影”方法显示出优势。影响这一行为的关键参数包括量子比特数$n$、观测量$M$、稀疏性$k$、保罗伊权重$w$、准确度要求$\epsilon$和失败容忍度$\delta$。我们还比较了两种方法在不同类型量子计算机上的资源消耗，并确定了“经典阴影”方法变得更有效的转折点，这些点取决于硬件的不同。本文为定量设计混合量子-经典成像的最佳策略开辟了新途径，并为实际应用中选择最适合的量子测量方法提供了实用见解。 

---
# Agentic Software Engineering: Foundational Pillars and a Research Roadmap 

**Title (ZH)**: 代理软件工程：基础支柱与研究路线图 

**Authors**: Ahmed E. Hassan, Hao Li, Dayi Lin, Bram Adams, Tse-Hsun Chen, Yutaro Kashiwa, Dong Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06216)  

**Abstract**: Agentic Software Engineering (SE 3.0) represents a new era where intelligent agents are tasked not with simple code generation, but with achieving complex, goal-oriented SE objectives. To harness these new capabilities while ensuring trustworthiness, we must recognize a fundamental duality within the SE field in the Agentic SE era, comprising two symbiotic modalities: SE for Humans and SE for Agents. This duality demands a radical reimagining of the foundational pillars of SE (actors, processes, tools, and artifacts) which manifest differently across each modality. We propose two purpose-built workbenches to support this vision. The Agent Command Environment (ACE) serves as a command center where humans orchestrate and mentor agent teams, handling outputs such as Merge-Readiness Packs (MRPs) and Consultation Request Packs (CRPs). The Agent Execution Environment (AEE) is a digital workspace where agents perform tasks while invoking human expertise when facing ambiguity or complex trade-offs. This bi-directional partnership, which supports agent-initiated human callbacks and handovers, gives rise to new, structured engineering activities (i.e., processes) that redefine human-AI collaboration, elevating the practice from agentic coding to true agentic software engineering. This paper presents the Structured Agentic Software Engineering (SASE) vision, outlining several of the foundational pillars for the future of SE. The paper culminates in a research roadmap that identifies a few key challenges and opportunities while briefly discussing the resulting impact of this future on SE education. Our goal is not to offer a definitive solution, but to provide a conceptual scaffold with structured vocabulary to catalyze a community-wide dialogue, pushing the SE community to think beyond its classic, human-centric tenets toward a disciplined, scalable, and trustworthy agentic future. 

**Abstract (ZH)**: 代理软件工程（SE 3.0）代表了一个新时代，在这个时代，智能代理的任务不仅仅是简单的代码生成，而是实现复杂的、目标导向的软件工程目标。要在利用这些新能力的同时确保可靠性，我们必须认识到代理软件工程时代软件工程领域的根本二元性，包括两种共生的方式：为人类的软件工程和为代理的软件工程。这种二元性要求对软件工程的基础支柱（参与者、过程、工具和制品）进行根本性的重新思考，这些支柱在每种方式中以不同的方式体现出来。我们提出了两个定制工作台来支持这一愿景。代理命令环境（ACE）作为指令中心，人类在这里编排并指导代理团队，并处理合并就绪包（MRP）和咨询请求包（CRP）等多种输出。代理执行环境（AEE）是一个数字工作空间，在这里代理执行任务，当遇到模糊性或复杂权衡时，调用人类的专业知识。这种双向伙伴关系，支持代理发起的人类回调和接力，产生了新的结构化工程活动（即过程），重新定义了人类与人工智能的合作，使实践从代理编程提升为真正的代理软件工程。本文提出了结构化代理软件工程（SASE）的愿景，概述了未来软件工程的一些基本支柱，并列出了几个关键挑战和机遇，简要讨论了这一未来对软件工程教育的影响。我们的目标不是提供一种终极解决方案，而是提供一种结构化的概念框架和词汇，以促进整个社区的对话，促使软件工程社区超越其传统的、以人类为中心的原则，朝着有纪律、可扩展和可靠的代理未来迈进。 

---
# Toward a Metrology for Artificial Intelligence: Hidden-Rule Environments and Reinforcement Learning 

**Title (ZH)**: 面向人工智能的度量标准研究：隐藏规则环境与强化学习 

**Authors**: Christo Mathew, Wentian Wang, Lazaros Gallos, Paul Kantor, Vladimir Menkov, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06213)  

**Abstract**: We investigate reinforcement learning in the Game Of Hidden Rules (GOHR) environment, a complex puzzle in which an agent must infer and execute hidden rules to clear a 6$\times$6 board by placing game pieces into buckets. We explore two state representation strategies, namely Feature-Centric (FC) and Object-Centric (OC), and employ a Transformer-based Advantage Actor-Critic (A2C) algorithm for training. The agent has access only to partial observations and must simultaneously infer the governing rule and learn the optimal policy through experience. We evaluate our models across multiple rule-based and trial-list-based experimental setups, analyzing transfer effects and the impact of representation on learning efficiency. 

**Abstract (ZH)**: 我们研究了在Game Of Hidden Rules (GOHR) 环境中的强化学习，GOHR 是一个复杂的谜题，在其中智能体必须推断并执行隐藏规则，通过将游戏部件放入桶中来清理一个 6×6 的板。我们探索了两种状态表示策略，即特征中心化（FC）和对象中心化（OC），并使用基于Transformer 的优势演员评论家（A2C）算法进行训练。智能体只能访问部分观察信息，并且必须同时推断支配规则并通过对经验的学习来学习最优策略。我们在多个基于规则和试验列表的实验设置中评估了我们的模型，分析了迁移效应以及表示对学习效率的影响。 

---
# Grasp-MPC: Closed-Loop Visual Grasping via Value-Guided Model Predictive Control 

**Title (ZH)**: 抓取-MPC：基于价值导向模型预测控制的闭环视觉抓取 

**Authors**: Jun Yamada, Adithyavairavan Murali, Ajay Mandlekar, Clemens Eppner, Ingmar Posner, Balakumar Sundaralingam  

**Link**: [PDF](https://arxiv.org/pdf/2509.06201)  

**Abstract**: Grasping of diverse objects in unstructured environments remains a significant challenge. Open-loop grasping methods, effective in controlled settings, struggle in cluttered environments. Grasp prediction errors and object pose changes during grasping are the main causes of failure. In contrast, closed-loop methods address these challenges in simplified settings (e.g., single object on a table) on a limited set of objects, with no path to generalization. We propose Grasp-MPC, a closed-loop 6-DoF vision-based grasping policy designed for robust and reactive grasping of novel objects in cluttered environments. Grasp-MPC incorporates a value function, trained on visual observations from a large-scale synthetic dataset of 2 million grasp trajectories that include successful and failed attempts. We deploy this learned value function in an MPC framework in combination with other cost terms that encourage collision avoidance and smooth execution. We evaluate Grasp-MPC on FetchBench and real-world settings across diverse environments. Grasp-MPC improves grasp success rates by up to 32.6% in simulation and 33.3% in real-world noisy conditions, outperforming open-loop, diffusion policy, transformer policy, and IQL approaches. Videos and more at this http URL. 

**Abstract (ZH)**: 在未结构化环境中抓取多样物体仍是一项重大挑战。闭环方法在简化设置（如桌面上的单个物体）和少量物体上解决了开环方法在杂乱环境中难以应对的问题，但缺乏泛化途径。我们提出了一种名为Grasp-MPC的闭环6自由度基于视觉的抓取策略，旨在在杂乱环境中对新颖物体进行稳健和反应式的抓取。Grasp-MPC结合了一种在包含成功和失败尝试的大规模合成数据集的200万抓取轨迹上训练的价值函数。我们通过与鼓励碰撞避免和平滑执行的其他成本项结合，在MPC框架中部署了这一学习价值函数。我们在FetchBench和多种环境的真实世界设置中评估了Grasp-MPC。在模拟环境中，Grasp-MPC的抓取成功率提高了32.6%，在真实世界的嘈杂条件下提高了33.3%，优于开环、扩散策略、变压器策略和IQL方法。更多内容请访问此链接。 

---
# Language Bias in Information Retrieval: The Nature of the Beast and Mitigation Methods 

**Title (ZH)**: 语言偏见在信息检索中的表现及其缓解方法 

**Authors**: Jinrui Yang, Fan Jiang, Timothy Baldwin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06195)  

**Abstract**: Language fairness in multilingual information retrieval (MLIR) systems is crucial for ensuring equitable access to information across diverse languages. This paper sheds light on the issue, based on the assumption that queries in different languages, but with identical semantics, should yield equivalent ranking lists when retrieving on the same multilingual documents. We evaluate the degree of fairness using both traditional retrieval methods, and a DPR neural ranker based on mBERT and XLM-R. Additionally, we introduce `LaKDA', a novel loss designed to mitigate language biases in neural MLIR approaches. Our analysis exposes intrinsic language biases in current MLIR technologies, with notable disparities across the retrieval methods, and the effectiveness of LaKDA in enhancing language fairness. 

**Abstract (ZH)**: 多语言信息检索系统中的语言公平性对于确保不同语言用户获得平等的信息访问权至关重要。本文基于不同语言但语义相同的查询在检索相同的多语言文档时应产生等效的排名列表这一假设，探讨了这一问题。我们使用传统的检索方法和基于mBERT和XLM-R的DPR神经排名器来评估公平性的程度，并引入了`LaKDA`这一新型损失函数，以减轻神经多语言信息检索方法中的语言偏见。我们的分析揭示了当前多语言信息检索技术中存在的内在语言偏见，以及`LaKDA`在提升语言公平性方面的有效性。 

---
# AI Governance in Higher Education: A course design exploring regulatory, ethical and practical considerations 

**Title (ZH)**: 高等教育中的AI治理：一门课程设计探索监管、伦理与实践考量 

**Authors**: Zsolt Almási, Hannah Bleher, Johannes Bleher, Rozanne Tuesday Flores, Guo Xuanyang, Paweł Pujszo, Raphaël Weuts  

**Link**: [PDF](https://arxiv.org/pdf/2509.06176)  

**Abstract**: As artificial intelligence (AI) systems permeate critical sectors, the need for professionals who can address ethical, legal and governance challenges has become urgent. Current AI ethics education remains fragmented, often siloed by discipline and disconnected from practice. This paper synthesizes literature and regulatory developments to propose a modular, interdisciplinary curriculum that integrates technical foundations with ethics, law and policy. We highlight recurring operational failures in AI - bias, misspecified objectives, generalization errors, misuse and governance breakdowns - and link them to pedagogical strategies for teaching AI governance. Drawing on perspectives from the EU, China and international frameworks, we outline a semester plan that emphasizes integrated ethics, stakeholder engagement and experiential learning. The curriculum aims to prepare students to diagnose risks, navigate regulation and engage diverse stakeholders, fostering adaptive and ethically grounded professionals for responsible AI governance. 

**Abstract (ZH)**: 人工智能系统渗透关键领域后对伦理、法律与治理挑战的专业人才需求迫在眉睫：模块化跨学科课程设计以整合技术基础与伦理、法律及政策的研究与教学 

---
# Reasoning Language Model for Personalized Lung Cancer Screening 

**Title (ZH)**: 个性化肺癌筛查的推理语言模型 

**Authors**: Chuang Niu, Ge Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06169)  

**Abstract**: Accurate risk assessment in lung cancer screening is critical for enabling early cancer detection and minimizing unnecessary invasive procedures. The Lung CT Screening Reporting and Data System (Lung-RADS) has been widely used as the standard framework for patient management and follow-up. Nevertheless, Lung-RADS faces trade-offs between sensitivity and specificity, as it stratifies risk solely based on lung nodule characteristics without incorporating various risk factors. Here we propose a reasoning language model (RLM) to integrate radiology findings with longitudinal medical records for individualized lung cancer risk assessment. Through a systematic study including dataset construction and distillation, supervised fine-tuning, reinforcement learning, and comprehensive evaluation, our model makes significant improvements in risk prediction performance on datasets in the national lung screening trial. Notably, RLM can decompose the risk evaluation task into sub-components, analyze the contributions of diverse risk factors, and synthesize them into a final risk score computed using our data-driven system equation. Our approach improves both predictive accuracy and monitorability through the chain of thought reasoning process, thereby facilitating clinical translation into lung cancer screening. 

**Abstract (ZH)**: 准确的肺癌筛查风险评估对于早期癌症检测和减少不必要的侵入性程序至关重要。肺部CT筛查报告和数据系统（Lung-RADS）已被广泛用作患者管理与随访的标准框架，然而，Lung-RADS在敏感性和特异性之间存在权衡，因为它仅根据肺结节特征进行风险分层，而不考虑多种风险因素。为此，我们提出了一种推理语言模型（RLM），将放射学发现与 longitudinal 医疗记录相结合，进行个性化肺癌风险评估。通过包括数据集构建与提炼、监督微调、强化学习和综合评价在内的系统研究，我们的模型在国家肺癌筛查试验数据集中的风险预测性能取得了显著提升。值得注意的是，RLM能够将风险评估任务分解为子组件，分析多种风险因素的贡献，并将它们综合成一个最终的风险评分，该评分由我们数据驱动的系统方程计算得出。我们的方法通过推理过程改进了预测准确性和监控性，从而促进了肺癌筛查中的临床转化。 

---
# UNO: Unifying One-stage Video Scene Graph Generation via Object-Centric Visual Representation Learning 

**Title (ZH)**: UNO：通过对象中心的视觉表示学习统一的一阶段视频场景图生成 

**Authors**: Huy Le, Nhat Chung, Tung Kieu, Jingkang Yang, Ngan Le  

**Link**: [PDF](https://arxiv.org/pdf/2509.06165)  

**Abstract**: Video Scene Graph Generation (VidSGG) aims to represent dynamic visual content by detecting objects and modeling their temporal interactions as structured graphs. Prior studies typically target either coarse-grained box-level or fine-grained panoptic pixel-level VidSGG, often requiring task-specific architectures and multi-stage training pipelines. In this paper, we present UNO (UNified Object-centric VidSGG), a single-stage, unified framework that jointly addresses both tasks within an end-to-end architecture. UNO is designed to minimize task-specific modifications and maximize parameter sharing, enabling generalization across different levels of visual granularity. The core of UNO is an extended slot attention mechanism that decomposes visual features into object and relation slots. To ensure robust temporal modeling, we introduce object temporal consistency learning, which enforces consistent object representations across frames without relying on explicit tracking modules. Additionally, a dynamic triplet prediction module links relation slots to corresponding object pairs, capturing evolving interactions over time. We evaluate UNO on standard box-level and pixel-level VidSGG benchmarks. Results demonstrate that UNO not only achieves competitive performance across both tasks but also offers improved efficiency through a unified, object-centric design. 

**Abstract (ZH)**: UNified Object-centric Video Scene Graph Generation 

---
# Benchmarking Gender and Political Bias in Large Language Models 

**Title (ZH)**: 大型语言模型中的性别和政治偏见基准研究 

**Authors**: Jinrui Yang, Xudong Han, Timothy Baldwin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06164)  

**Abstract**: We introduce EuroParlVote, a novel benchmark for evaluating large language models (LLMs) in politically sensitive contexts. It links European Parliament debate speeches to roll-call vote outcomes and includes rich demographic metadata for each Member of the European Parliament (MEP), such as gender, age, country, and political group. Using EuroParlVote, we evaluate state-of-the-art LLMs on two tasks -- gender classification and vote prediction -- revealing consistent patterns of bias. We find that LLMs frequently misclassify female MEPs as male and demonstrate reduced accuracy when simulating votes for female speakers. Politically, LLMs tend to favor centrist groups while underperforming on both far-left and far-right ones. Proprietary models like GPT-4o outperform open-weight alternatives in terms of both robustness and fairness. We release the EuroParlVote dataset, code, and demo to support future research on fairness and accountability in NLP within political contexts. 

**Abstract (ZH)**: EuroParlVote：一种评估大规模语言模型在政治敏感情境下的基准 

---
# Tracking daily paths in home contexts with RSSI fingerprinting based on UWB through deep learning models 

**Title (ZH)**: 基于UWB的RSSI指纹识别结合深度学习模型在家庭场景中跟踪日常路径 

**Authors**: Aurora Polo-Rodríguez, Juan Carlos Valera, Jesús Peral, David Gil, Javier Medina-Quero  

**Link**: [PDF](https://arxiv.org/pdf/2509.06161)  

**Abstract**: The field of human activity recognition has evolved significantly, driven largely by advancements in Internet of Things (IoT) device technology, particularly in personal devices. This study investigates the use of ultra-wideband (UWB) technology for tracking inhabitant paths in home environments using deep learning models. UWB technology estimates user locations via time-of-flight and time-difference-of-arrival methods, which are significantly affected by the presence of walls and obstacles in real environments, reducing their precision. To address these challenges, we propose a fingerprinting-based approach utilizing received signal strength indicator (RSSI) data collected from inhabitants in two flats (60 m2 and 100 m2) while performing daily activities. We compare the performance of convolutional neural network (CNN), long short-term memory (LSTM), and hybrid CNN+LSTM models, as well as the use of Bluetooth technology. Additionally, we evaluate the impact of the type and duration of the temporal window (future, past, or a combination of both). Our results demonstrate a mean absolute error close to 50 cm, highlighting the superiority of the hybrid model in providing accurate location estimates, thus facilitating its application in daily human activity recognition in residential settings. 

**Abstract (ZH)**: 基于超宽带技术的深学习方法在家庭环境居民路径跟踪中的应用研究 

---
# FASL-Seg: Anatomy and Tool Segmentation of Surgical Scenes 

**Title (ZH)**: FASL-Seg: 手术场景中的解剖结构和工具分割 

**Authors**: Muraam Abdel-Ghani, Mahmoud Ali, Mohamed Ali, Fatmaelzahraa Ahmed, Mohamed Arsalan, Abdulaziz Al-Ali, Shidin Balakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2509.06159)  

**Abstract**: The growing popularity of robotic minimally invasive surgeries has made deep learning-based surgical training a key area of research. A thorough understanding of the surgical scene components is crucial, which semantic segmentation models can help achieve. However, most existing work focuses on surgical tools and overlooks anatomical objects. Additionally, current state-of-the-art (SOTA) models struggle to balance capturing high-level contextual features and low-level edge features. We propose a Feature-Adaptive Spatial Localization model (FASL-Seg), designed to capture features at multiple levels of detail through two distinct processing streams, namely a Low-Level Feature Projection (LLFP) and a High-Level Feature Projection (HLFP) stream, for varying feature resolutions - enabling precise segmentation of anatomy and surgical instruments. We evaluated FASL-Seg on surgical segmentation benchmark datasets EndoVis18 and EndoVis17 on three use cases. The FASL-Seg model achieves a mean Intersection over Union (mIoU) of 72.71% on parts and anatomy segmentation in EndoVis18, improving on SOTA by 5%. It further achieves a mIoU of 85.61% and 72.78% in EndoVis18 and EndoVis17 tool type segmentation, respectively, outperforming SOTA overall performance, with comparable per-class SOTA results in both datasets and consistent performance in various classes for anatomy and instruments, demonstrating the effectiveness of distinct processing streams for varying feature resolutions. 

**Abstract (ZH)**: 基于深度学习的微创手术培训研究：一种适应性空间定位模型在解剖学和手术器械分割中的应用 

---
# SpecSwin3D: Generating Hyperspectral Imagery from Multispectral Data via Transformer Networks 

**Title (ZH)**: SpecSwin3D: 通过变换器网络从多光谱数据生成高光谱图像 

**Authors**: Tang Sui, Songxi Yang, Qunying Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06122)  

**Abstract**: Multispectral and hyperspectral imagery are widely used in agriculture, environmental monitoring, and urban planning due to their complementary spatial and spectral characteristics. A fundamental trade-off persists: multispectral imagery offers high spatial but limited spectral resolution, while hyperspectral imagery provides rich spectra at lower spatial resolution. Prior hyperspectral generation approaches (e.g., pan-sharpening variants, matrix factorization, CNNs) often struggle to jointly preserve spatial detail and spectral fidelity. In response, we propose SpecSwin3D, a transformer-based model that generates hyperspectral imagery from multispectral inputs while preserving both spatial and spectral quality. Specifically, SpecSwin3D takes five multispectral bands as input and reconstructs 224 hyperspectral bands at the same spatial resolution. In addition, we observe that reconstruction errors grow for hyperspectral bands spectrally distant from the input bands. To address this, we introduce a cascade training strategy that progressively expands the spectral range to stabilize learning and improve fidelity. Moreover, we design an optimized band sequence that strategically repeats and orders the five selected multispectral bands to better capture pairwise relations within a 3D shifted-window transformer framework. Quantitatively, our model achieves a PSNR of 35.82 dB, SAM of 2.40°, and SSIM of 0.96, outperforming the baseline MHF-Net by +5.6 dB in PSNR and reducing ERGAS by more than half. Beyond reconstruction, we further demonstrate the practical value of SpecSwin3D on two downstream tasks, including land use classification and burnt area segmentation. 

**Abstract (ZH)**: 多光谱和高光谱影像由于其互补的空间和光谱特性，在农业、环境监测和城市规划中被广泛应用于高光谱影像生成。一种基本的权衡一直存在：多光谱影像提供高空间分辨率但光谱分辨率有限，而高光谱影像提供丰富的光谱信息但空间分辨率较低。先前的高光谱影像生成方法（如多尺度锐化变种、矩阵分解、CNNs）往往难以同时保留空间细节和光谱保真度。为此，我们提出了一种基于变压器的模型SpecSwin3D，它可以从前置的多光谱输入生成高光谱影像，同时保持空间和光谱质量。具体而言，SpecSwin3D 输入五个多光谱波段，并在相同的空间分辨率下重建 224 个高光谱波段。此外，我们观察到，对于和输入波段光谱距离较远的高光谱波段，重建误差会增加。为此，我们引入了一种级联训练策略，逐步扩展光谱范围以稳定学习并提高保真度。此外，我们设计了一种优化的波段序列，战略性地重复并排序五个选定的多光谱波段，以便更好地在 3D 移动窗口变压器框架中捕捉两两关系。定量结果表明，我们的模型达到了 35.82 dB 的 PSNR、2.40° 的 SAM 和 0.96 的 SSIM，PSNR 指标比基线 MHF-Net 高出 5.6 dB，且 ERGAS 降低了超过一半。除了重建任务外，我们还进一步展示了 SpecSwin3D 在两个下游任务（包括土地利用分类和烧伤面积分割）中的实用价值。 

---
# Teaching Precommitted Agents: Model-Free Policy Evaluation and Control in Quasi-Hyperbolic Discounted MDPs 

**Title (ZH)**: 前瞻承诺代理的教学：准_hyperbolic 折扣MDP中的模型_free策略评估与控制 

**Authors**: S.R. Eshwar  

**Link**: [PDF](https://arxiv.org/pdf/2509.06094)  

**Abstract**: Time-inconsistent preferences, where agents favor smaller-sooner over larger-later rewards, are a key feature of human and animal decision-making. Quasi-Hyperbolic (QH) discounting provides a simple yet powerful model for this behavior, but its integration into the reinforcement learning (RL) framework has been limited. This paper addresses key theoretical and algorithmic gaps for precommitted agents with QH preferences. We make two primary contributions: (i) we formally characterize the structure of the optimal policy, proving for the first time that it reduces to a simple one-step non-stationary form; and (ii) we design the first practical, model-free algorithms for both policy evaluation and Q-learning in this setting, both with provable convergence guarantees. Our results provide foundational insights for incorporating QH preferences in RL. 

**Abstract (ZH)**: 时间不一致偏好，其中 Agents 更偏好较小较早的奖励而非较大较晚的奖励，是人类和动物决策的重要特征。接近指数折现 (Quasi-Hyperbolic, QH) 提供了一个简单而强大的模型来解释这种行为，但将其整合进强化学习 (Reinforcement Learning, RL) 框架中仍存在限制。本文针对具有 QH 偏好的预承诺代理的关键理论和算法缺口进行了探讨。我们做出了两项主要贡献：(i) 我们正式刻画了最优策略的结构，首次证明其简化为单一非稳态一步形式；(ii) 我们设计了首个适用于此情境的实际无模型策略评估和 Q 学习算法，并提供了收敛性保证。我们的结果为在 RL 中整合 QH 偏好提供了基础性洞见。 

---
# Language Native Lightly Structured Databases for Large Language Model Driven Composite Materials Research 

**Title (ZH)**: 语言本土轻量结构数据库在大型语言模型驱动的复合材料研究中的应用 

**Authors**: Yuze Liu, Zhaoyuan Zhang, Xiangsheng Zeng, Yihe Zhang, Leping Yu, Lejia Wang, Xi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06093)  

**Abstract**: Chemical and materials research has traditionally relied heavily on knowledge narrative, with progress often driven by language-based descriptions of principles, mechanisms, and experimental experiences, rather than tables, limiting what conventional databases and ML can exploit. We present a language-native database for boron nitride nanosheet (BNNS) polymer thermally conductive composites that captures lightly structured information from papers across preparation, characterization, theory-computation, and mechanistic reasoning, with evidence-linked snippets. Records are organized in a heterogeneous database and queried via composite retrieval with semantics, key words and value filters. The system can synthesizes literature into accurate, verifiable, and expert style guidance. This substrate enables high fidelity efficient Retrieval Augmented Generation (RAG) and tool augmented agents to interleave retrieval with reasoning and deliver actionable SOP. The framework supplies the language rich foundation required for LLM-driven materials discovery. 

**Abstract (ZH)**: 一种语言本位的氮化硼纳米片聚合物热传导复合材料数据库，及其在高效检索增强生成和工具增强代理中的应用，为基于LLM的材料发现提供语言丰富的基础。 

---
# Software Dependencies 2.0: An Empirical Study of Reuse and Integration of Pre-Trained Models in Open-Source Projects 

**Title (ZH)**: 软件依赖关系 2.0：开源项目中预训练模型的重用与集成实证研究 

**Authors**: Jerin Yasmin, Wenxin Jiang, James C. Davis, Yuan Tian  

**Link**: [PDF](https://arxiv.org/pdf/2509.06085)  

**Abstract**: Pre-trained models (PTMs) are machine learning models that have been trained in advance, often on large-scale data, and can be reused for new tasks, thereby reducing the need for costly training from scratch. Their widespread adoption introduces a new class of software dependency, which we term Software Dependencies 2.0, extending beyond conventional libraries to learned behaviors embodied in trained models and their associated artifacts. The integration of PTMs as software dependencies in real projects remains unclear, potentially threatening maintainability and reliability of modern software systems that increasingly rely on them. Objective: In this study, we investigate Software Dependencies 2.0 in open-source software (OSS) projects by examining the reuse of PTMs, with a focus on how developers manage and integrate these models. Specifically, we seek to understand: (1) how OSS projects structure and document their PTM dependencies; (2) what stages and organizational patterns emerge in the reuse pipelines of PTMs within these projects; and (3) the interactions among PTMs and other learned components across pipeline stages. We conduct a mixed-methods analysis of a statistically significant random sample of 401 GitHub repositories from the PeaTMOSS dataset (28,575 repositories reusing PTMs from Hugging Face and PyTorch Hub). We quantitatively examine PTM reuse by identifying patterns and qualitatively investigate how developers integrate and manage these models in practice. 

**Abstract (ZH)**: 预训练模型（PTMs）是预先在大型数据集上训练的机器学习模型，可以重新用于新任务，从而减少从头开始训练的成本。其广泛应用引入了一类新的软件依赖关系，我们称之为软件依赖关系2.0，超越了传统的库，涵盖了嵌入在训练模型及其相关制品中的学习行为。PTMs在实际项目中作为软件依赖的集成仍不明确，可能威胁现代软件系统的可维护性和可靠性，这些系统越来越多地依赖于它们。目的：在本研究中，我们通过检查PTM的再利用情况，研究开源软件（OSS）项目的软件依赖关系2.0，重点关注开发人员如何管理和集成这些模型。具体而言，我们旨在理解：（1）OSS项目如何结构化和文档化其PTM依赖关系；（2）这些项目中PTM再利用管道中出现的阶段和组织模式；以及（3）在管道阶段中，PTMs与其他学习组件之间的交互。我们对PeaTMOSS数据集中的401个GitHub仓库（28,575个仓库从Hugging Face和PyTorch Hub再利用PTMs）进行了统计上有意义的随机样本混合方法分析。我们定量检查PTM的再利用情况，通过识别模式，并定性研究开发人员在实践中如何集成和管理这些模型。 

---
# ARIES: Relation Assessment and Model Recommendation for Deep Time Series Forecasting 

**Title (ZH)**: ARIES：深度时间序列预测中的关系评估与模型推荐 

**Authors**: Fei Wang, Yujie Li, Zezhi Shao, Chengqing Yu, Yisong Fu, Zhulin An, Yongjun Xu, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.06060)  

**Abstract**: Recent advancements in deep learning models for time series forecasting have been significant. These models often leverage fundamental time series properties such as seasonality and non-stationarity, which may suggest an intrinsic link between model performance and data properties. However, existing benchmark datasets fail to offer diverse and well-defined temporal patterns, restricting the systematic evaluation of such connections. Additionally, there is no effective model recommendation approach, leading to high time and cost expenditures when testing different architectures across different downstream applications. For those reasons, we propose ARIES, a framework for assessing relation between time series properties and modeling strategies, and for recommending deep forcasting models for realistic time series. First, we construct a synthetic dataset with multiple distinct patterns, and design a comprehensive system to compute the properties of time series. Next, we conduct an extensive benchmarking of over 50 forecasting models, and establish the relationship between time series properties and modeling strategies. Our experimental results reveal a clear correlation. Based on these findings, we propose the first deep forecasting model recommender, capable of providing interpretable suggestions for real-world time series. In summary, ARIES is the first study to establish the relations between the properties of time series data and modeling strategies, while also implementing a model recommendation system. The code is available at: this https URL. 

**Abstract (ZH)**: Recent advancements in deep learning models for time series forecasting have been significant. These models often leverage fundamental time series properties such as seasonality and non-stationarity, which may suggest an intrinsic link between model performance and data properties. However, existing benchmark datasets fail to offer diverse and well-defined temporal patterns, restricting the systematic evaluation of such connections. Additionally, there is no effective model recommendation approach, leading to high time and cost expenditures when testing different architectures across different downstream applications. For those reasons, we propose ARIES, a framework for assessing the relation between time series properties and modeling strategies, and for recommending deep forecasting models for realistic time series. First, we construct a synthetic dataset with multiple distinct patterns, and design a comprehensive system to compute the properties of time series. Next, we conduct an extensive benchmarking of over 50 forecasting models, and establish the relationship between time series properties and modeling strategies. Our experimental results reveal a clear correlation. Based on these findings, we propose the first deep forecasting model recommender, capable of providing interpretable suggestions for real-world time series. In summary, ARIES is the first study to establish the relations between the properties of time series data and modeling strategies, while also implementing a model recommendation system. The code is available at: this https URL. 

---
# PolicyEvolve: Evolving Programmatic Policies by LLMs for multi-player games via Population-Based Training 

**Title (ZH)**: PolicyEvolve：通过基于群体的训练使大规模语言模型为多玩家游戏演化程序化策略 

**Authors**: Mingrui Lv, Hangzhi Liu, Zhi Luo, Hongjie Zhang, Jie Ou  

**Link**: [PDF](https://arxiv.org/pdf/2509.06053)  

**Abstract**: Multi-agent reinforcement learning (MARL) has achieved significant progress in solving complex multi-player games through self-play. However, training effective adversarial policies requires millions of experience samples and substantial computational resources. Moreover, these policies lack interpretability, hindering their practical deployment. Recently, researchers have successfully leveraged Large Language Models (LLMs) to generate programmatic policies for single-agent tasks, transforming neural network-based policies into interpretable rule-based code with high execution efficiency. Inspired by this, we propose PolicyEvolve, a general framework for generating programmatic policies in multi-player games. PolicyEvolve significantly reduces reliance on manually crafted policy code, achieving high-performance policies with minimal environmental interactions. The framework comprises four modules: Global Pool, Local Pool, Policy Planner, and Trajectory Critic. The Global Pool preserves elite policies accumulated during iterative training. The Local Pool stores temporary policies for the current iteration; only sufficiently high-performing policies from this pool are promoted to the Global Pool. The Policy Planner serves as the core policy generation module. It samples the top three policies from the Global Pool, generates an initial policy for the current iteration based on environmental information, and refines this policy using feedback from the Trajectory Critic. Refined policies are then deposited into the Local Pool. This iterative process continues until the policy achieves a sufficiently high average win rate against the Global Pool, at which point it is integrated into the Global Pool. The Trajectory Critic analyzes interaction data from the current policy, identifies vulnerabilities, and proposes directional improvements to guide the Policy Planner 

**Abstract (ZH)**: 基于多智能体强化学习的程序化策略生成框架：PolicyEvolve 

---
# Empirical Study of Code Large Language Models for Binary Security Patch Detection 

**Title (ZH)**: 代码大型语言模型在二元安全补丁检测中的实证研究 

**Authors**: Qingyuan Li, Binchang Li, Cuiyun Gao, Shuzheng Gao, Zongjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.06052)  

**Abstract**: Security patch detection (SPD) is crucial for maintaining software security, as unpatched vulnerabilities can lead to severe security risks. In recent years, numerous learning-based SPD approaches have demonstrated promising results on source code. However, these approaches typically cannot be applied to closed-source applications and proprietary systems that constitute a significant portion of real-world software, as they release patches only with binary files, and the source code is inaccessible. Given the impressive performance of code large language models (LLMs) in code intelligence and binary analysis tasks such as decompilation and compilation optimization, their potential for detecting binary security patches remains unexplored, exposing a significant research gap between their demonstrated low-level code understanding capabilities and this critical security task. To address this gap, we construct a large-scale binary patch dataset containing \textbf{19,448} samples, with two levels of representation: assembly code and pseudo-code, and systematically evaluate \textbf{19} code LLMs of varying scales to investigate their capability in binary SPD tasks. Our initial exploration demonstrates that directly prompting vanilla code LLMs struggles to accurately identify security patches from binary patches, and even state-of-the-art prompting techniques fail to mitigate the lack of domain knowledge in binary SPD within vanilla models. Drawing on the initial findings, we further investigate the fine-tuning strategy for injecting binary SPD domain knowledge into code LLMs through two levels of representation. Experimental results demonstrate that fine-tuned LLMs achieve outstanding performance, with the best results obtained on the pseudo-code representation. 

**Abstract (ZH)**: 基于二进制的安全补丁检测：构建大规模二进制补丁数据集并评估代码大语言模型 

---
# BranchGRPO: Stable and Efficient GRPO with Structured Branching in Diffusion Models 

**Title (ZH)**: BranchGRPO：具有结构化分支的稳定高效GRPO在扩散模型中 

**Authors**: Yuming Li, Yikai Wang, Yuying Zhu, Zhongyu Zhao, Ming Lu, Qi She, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06040)  

**Abstract**: Recent advancements in aligning image and video generative models via GRPO have achieved remarkable gains in enhancing human preference alignment. However, these methods still face high computational costs from on-policy rollouts and excessive SDE sampling steps, as well as training instability due to sparse rewards. In this paper, we propose BranchGRPO, a novel method that introduces a branch sampling policy updating the SDE sampling process. By sharing computation across common prefixes and pruning low-reward paths and redundant depths, BranchGRPO substantially lowers the per-update compute cost while maintaining or improving exploration diversity. This work makes three main contributions: (1) a branch sampling scheme that reduces rollout and training cost; (2) a tree-based advantage estimator incorporating dense process-level rewards; and (3) pruning strategies exploiting path and depth redundancy to accelerate convergence and boost performance. Experiments on image and video preference alignment show that BranchGRPO improves alignment scores by 16% over strong baselines, while cutting training time by 50%. 

**Abstract (ZH)**: 基于GRPO的分支采样方法在图像和视频生成模型对齐中的Recent Advancements and Contributions 

---
# TinyDef-DETR:An Enhanced DETR Detector for UAV Power Line Defect Detection 

**Title (ZH)**: TinyDef-DETR：一种用于无人机输电线路缺陷检测的增强DETR检测器 

**Authors**: Jiaming Cui  

**Link**: [PDF](https://arxiv.org/pdf/2509.06035)  

**Abstract**: Automated inspection of transmission lines using UAVs is hindered by the difficulty of detecting small and ambiguous defects against complex backgrounds. Conventional detectors often suffer from detail loss due to strided downsampling, weak boundary sensitivity in lightweight backbones, and insufficient integration of global context with local cues. To address these challenges, we propose TinyDef-DETR, a DETR-based framework designed for small-defect detection. The method introduces a stride-free space-to-depth module for lossless downsampling, an edge-enhanced convolution for boundary-aware feature extraction, a cross-stage dual-domain multi-scale attention module to jointly capture global and local information, and a Focaler-Wise-SIoU regression loss to improve localization of small objects. Experiments conducted on the CSG-ADCD dataset demonstrate that TinyDef-DETR achieves substantial improvements in both precision and recall compared to competitive baselines, with particularly notable gains on small-object subsets, while incurring only modest computational overhead. Further validation on the VisDrone benchmark confirms the generalization capability of the proposed approach. Overall, the results indicate that integrating detail-preserving downsampling, edge-sensitive representations, dual-domain attention, and difficulty-adaptive regression provides a practical and efficient solution for UAV-based small-defect inspection in power grids. 

**Abstract (ZH)**: 基于UAV的输电线路自动化检测受复杂背景中检测小而模棱两可缺陷的困难所制约。传统检测器常因跳跃下采样的细节损失、轻量级骨干网在边缘敏感度上的不足以及全球上下文与局部线索融合不足而受到影响。为应对这些挑战，我们提出了TinyDef-DETR，这是一种基于DETR的小缺陷检测框架。该方法引入了无跳跃的空间到深度模块进行无损下采样，边缘增强卷积进行边界感知特征提取，跨阶段双域多尺度注意力模块共同捕捉全局和局部信息，并采用焦点器-边缘-WSIoU回归损失以提高小目标的定位精度。在CSG-ADCD数据集上的实验表明，TinyDef-DETR在精确度和召回率方面均相对于竞争基线取得了显著提升，特别是在小目标子集上的提升尤为明显，同时仅带来了轻微的计算开销。进一步在VisDrone基准上的验证证实了所提出方法的泛化能力。总体而言，结果表明，结合细节保留下采样、边缘敏感表示、双域注意和难度适应性回归为基于UAV的电力网络中小缺陷检测提供了一种实用且高效的解决方案。 

---
# DreamAudio: Customized Text-to-Audio Generation with Diffusion Models 

**Title (ZH)**: DreamAudio: 定制化文本到语音生成方法Based on扩散模型 

**Authors**: Yi Yuan, Xubo Liu, Haohe Liu, Xiyuan Kang, Zhuo Chen, Yuxuan Wang, Mark D. Plumbley, Wenwu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06027)  

**Abstract**: With the development of large-scale diffusion-based and language-modeling-based generative models, impressive progress has been achieved in text-to-audio generation. Despite producing high-quality outputs, existing text-to-audio models mainly aim to generate semantically aligned sound and fall short on precisely controlling fine-grained acoustic characteristics of specific sounds. As a result, users that need specific sound content may find it challenging to generate the desired audio clips. In this paper, we present DreamAudio for customized text-to-audio generation (CTTA). Specifically, we introduce a new framework that is designed to enable the model to identify auditory information from user-provided reference concepts for audio generation. Given a few reference audio samples containing personalized audio events, our system can generate new audio samples that include these specific events. In addition, two types of datasets are developed for training and testing the customized systems. The experiments show that the proposed model, DreamAudio, generates audio samples that are highly consistent with the customized audio features and aligned well with the input text prompts. Furthermore, DreamAudio offers comparable performance in general text-to-audio tasks. We also provide a human-involved dataset containing audio events from real-world CTTA cases as the benchmark for customized generation tasks. 

**Abstract (ZH)**: 基于定制文本到音频生成的DreamAudio 

---
# DCMI: A Differential Calibration Membership Inference Attack Against Retrieval-Augmented Generation 

**Title (ZH)**: DCMI: 一种针对检索增强生成的差分校准成员推断攻击 

**Authors**: Xinyu Gao, Xiangtao Meng, Yingkai Dong, Zheng Li, Shanqing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06026)  

**Abstract**: While Retrieval-Augmented Generation (RAG) effectively reduces hallucinations by integrating external knowledge bases, it introduces vulnerabilities to membership inference attacks (MIAs), particularly in systems handling sensitive data. Existing MIAs targeting RAG's external databases often rely on model responses but ignore the interference of non-member-retrieved documents on RAG outputs, limiting their effectiveness. To address this, we propose DCMI, a differential calibration MIA that mitigates the negative impact of non-member-retrieved documents. Specifically, DCMI leverages the sensitivity gap between member and non-member retrieved documents under query perturbation. It generates perturbed queries for calibration to isolate the contribution of member-retrieved documents while minimizing the interference from non-member-retrieved documents. Experiments under progressively relaxed assumptions show that DCMI consistently outperforms baselines--for example, achieving 97.42% AUC and 94.35% Accuracy against the RAG system with Flan-T5, exceeding the MBA baseline by over 40%. Furthermore, on real-world RAG platforms such as Dify and MaxKB, DCMI maintains a 10%-20% advantage over the baseline. These results highlight significant privacy risks in RAG systems and emphasize the need for stronger protection mechanisms. We appeal to the community's consideration of deeper investigations, like ours, against the data leakage risks in rapidly evolving RAG systems. Our code is available at this https URL. 

**Abstract (ZH)**: DCMI：针对检索增强生成系统的差异校准会员推理攻击 

---
# Unified Interaction Foundational Model (UIFM) for Predicting Complex User and System Behavior 

**Title (ZH)**: 统一交互基础模型（UIFM）用于预测复杂用户和系统行为 

**Authors**: Vignesh Ethiraj, Subhash Talluri  

**Link**: [PDF](https://arxiv.org/pdf/2509.06025)  

**Abstract**: A central goal of artificial intelligence is to build systems that can understand and predict complex, evolving sequences of events. However, current foundation models, designed for natural language, fail to grasp the holistic nature of structured interactions found in domains like telecommunications, e-commerce and finance. By serializing events into text, they disassemble them into semantically fragmented parts, losing critical context. In this work, we introduce the Unified Interaction Foundation Model (UIFM), a foundation model engineered for genuine behavioral understanding. At its core is the principle of composite tokenization, where each multi-attribute event is treated as a single, semantically coherent unit. This allows UIFM to learn the underlying "grammar" of user behavior, perceiving entire interactions rather than a disconnected stream of data points. We demonstrate that this architecture is not just more accurate, but represents a fundamental step towards creating more adaptable and intelligent predictive systems. 

**Abstract (ZH)**: 统一交互基础模型：一种用于真实行为理解的基础模型 

---
# Khana: A Comprehensive Indian Cuisine Dataset 

**Title (ZH)**: Khana: 一套全面的印度 Cuisine 数据集 

**Authors**: Omkar Prabhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06006)  

**Abstract**: As global interest in diverse culinary experiences grows, food image models are essential for improving food-related applications by enabling accurate food recognition, recipe suggestions, dietary tracking, and automated meal planning. Despite the abundance of food datasets, a noticeable gap remains in capturing the nuances of Indian cuisine due to its vast regional diversity, complex preparations, and the lack of comprehensive labeled datasets that cover its full breadth. Through this exploration, we uncover Khana, a new benchmark dataset for food image classification, segmentation, and retrieval of dishes from Indian cuisine. Khana fills the gap by establishing a taxonomy of Indian cuisine and offering around 131K images in the dataset spread across 80 labels, each with a resolution of 500x500 pixels. This paper describes the dataset creation process and evaluates state-of-the-art models on classification, segmentation, and retrieval as baselines. Khana bridges the gap between research and development by providing a comprehensive and challenging benchmark for researchers while also serving as a valuable resource for developers creating real-world applications that leverage the rich tapestry of Indian cuisine. Webpage: this https URL 

**Abstract (ZH)**: 随着全球对多元化美食体验的兴趣增长，食品图像模型对于通过准确的食物识别、食谱建议、饮食跟踪和自动化餐食规划来改进相关应用至关重要。尽管存在大量的食品数据集，但由于印度菜的地域多样性、复杂的料理方式以及缺乏涵盖其全部范围的综合标注数据集，仍存在明显的差距。通过这一探索，我们发现了Khana，一个用于印度菜菜品分类、分割和检索的新基准数据集。Khana通过建立印度菜的分类体系，并提供约131K张图像（每个标签80个，分辨率为500x500像素），填补了这一空白。本文描述了数据集的创建过程，并在分类、分割和检索任务上评估了最先进的模型，作为基线。Khana通过提供全面且具有挑战性的基准，架起了研究与开发之间的桥梁，同时也为开发人员创建利用丰富多样的印度菜美食的应用程序提供了宝贵的资源。网页地址：这个 https URL 

---
# S-LAM3D: Segmentation-Guided Monocular 3D Object Detection via Feature Space Fusion 

**Title (ZH)**: S-LAM3D: 基于分割引导的单目三维目标检测及其特征空间融合 

**Authors**: Diana-Alexandra Sas, Florin Oniga  

**Link**: [PDF](https://arxiv.org/pdf/2509.05999)  

**Abstract**: Monocular 3D Object Detection represents a challenging Computer Vision task due to the nature of the input used, which is a single 2D image, lacking in any depth cues and placing the depth estimation problem as an ill-posed one. Existing solutions leverage the information extracted from the input by using Convolutional Neural Networks or Transformer architectures as feature extraction backbones, followed by specific detection heads for 3D parameters prediction. In this paper, we introduce a decoupled strategy based on injecting precomputed segmentation information priors and fusing them directly into the feature space for guiding the detection, without expanding the detection model or jointly learning the priors. The focus is on evaluating the impact of additional segmentation information on existing detection pipelines without adding additional prediction branches. The proposed method is evaluated on the KITTI 3D Object Detection Benchmark, outperforming the equivalent architecture that relies only on RGB image features for small objects in the scene: pedestrians and cyclists, and proving that understanding the input data can balance the need for additional sensors or training data. 

**Abstract (ZH)**: 单目三维物体检测由于输入仅为单张缺乏深度线索的2D图像，是一个具有挑战性的计算机视觉任务，导致深度估计问题成为病态问题。现有解决方案通过使用卷积神经网络或变换器架构作为特征提取骨干，并结合特定的检测头进行3D参数预测来利用输入中的信息。本文提出了一种解耦策略，通过注入预先计算的分割信息先验并直接融合到特征空间中来指导检测，而无需扩展检测模型或共同学习先验。重点在于评估额外分割信息对现有检测管道的影响，而不增加额外的预测分支。所提出的方法在Kitti三维物体检测基准上进行了评估，优于仅依赖RGB图像特征的等效架构，特别是在场景中的小物体：行人和骑车人方面表现更优，证明了理解输入数据可以平衡对额外传感器或训练数据的需求。 

---
# Operationalising AI Regulatory Sandboxes under the EU AI Act: The Triple Challenge of Capacity, Coordination and Attractiveness to Providers 

**Title (ZH)**: 欧盟AI法案下AI监管沙箱的三重挑战：能力、协调与服务提供者的吸引力 

**Authors**: Deirdre Ahern  

**Link**: [PDF](https://arxiv.org/pdf/2509.05985)  

**Abstract**: The EU AI Act provides a rulebook for all AI systems being put on the market or into service in the European Union. This article investigates the requirement under the AI Act that Member States establish national AI regulatory sandboxes for testing and validation of innovative AI systems under regulatory supervision to assist with fostering innovation and complying with regulatory requirements. Against the backdrop of the EU objective that AI regulatory sandboxes would both foster innovation and assist with compliance, considerable challenges are identified for Member States around capacity-building and design of regulatory sandboxes. While Member States are early movers in laying the ground for national AI regulatory sandboxes, the article contends that there is a risk that differing approaches being taken by individual national sandboxes could jeopardise a uniform interpretation of the AI Act and its application in practice. This could motivate innovators to play sandbox arbitrage. The article therefore argues that the European Commission and the AI Board need to act decisively in developing rules and guidance to ensure a cohesive, coordinated approach in national AI regulatory sandboxes. With sandbox participation being voluntary, the possibility that AI regulatory sandboxes may prove unattractive to innovators on their compliance journey is also explored. Confidentiality concerns, the inability to relax legal rules during the sandbox, and the inability of sandboxes to deliver a presumption of conformity with the AI Act are identified as pertinent concerns for innovators contemplating applying to AI regulatory sandboxes as compared with other direct compliance routes provided to them through application of harmonised standards and conformity assessment procedures. 

**Abstract (ZH)**: 欧盟AI法案为欧盟市场上的所有AI系统提供了操作规范。本文 investigate 欧盟AI法案中要求成员国建立国家级AI监管沙箱以测试和验证创新AI系统并在监管监督下助于促进创新和合规的要求。在欧盟旨在通过监管沙箱促进创新和助于合规的背景下，成员国在能力建设和监管沙箱设计方面面临着重大挑战。尽管成员国在为国家级AI监管沙箱奠定基础方面处于领先地位，但文章认为，个别国家级沙箱采取的不同方法可能会危及AI法案的统一解释和实际应用，这可能会激励创新者进行沙箱套利。因此，文章认为，欧盟委员会和AI委员会需要果断行动，制定规则和指导，确保国家级AI监管沙箱的一致性和协调性。鉴于沙箱参与是自愿的，文章还探讨了AI监管沙箱可能对创新者的合规之路缺乏吸引力的可能性。此外，还指出了创新者在考虑申请AI监管沙箱与通过采纳协调标准和一致性评估程序提供给他们的其他直接合规途径相比所面临的相关性隐私担忧、无法在沙箱期间放松法律规定以及沙箱无法提供AI法案符合性推定等关键问题。 

---
# TSPC: A Two-Stage Phoneme-Centric Architecture for code-switching Vietnamese-English Speech Recognition 

**Title (ZH)**: TSPC：一种两阶段基于音素的代码切换越南英语语音识别架构 

**Authors**: Minh N. H. Nguyen, Anh Nguyen Tran, Dung Truong Dinh, Nam Van Vo  

**Link**: [PDF](https://arxiv.org/pdf/2509.05983)  

**Abstract**: Code-switching (CS) presents a significant challenge for general Auto-Speech Recognition (ASR) systems. Existing methods often fail to capture the subtle phonological shifts inherent in CS scenarios. The challenge is particularly difficult for language pairs like Vietnamese and English, where both distinct phonological features and the ambiguity arising from similar sound recognition are present. In this paper, we propose a novel architecture for Vietnamese-English CS ASR, a Two-Stage Phoneme-Centric model (TSPC). The TSPC employs a phoneme-centric approach, built upon an extended Vietnamese phoneme set as an intermediate representation to facilitate mixed-lingual modeling. Experimental results demonstrate that TSPC consistently outperforms existing baselines, including PhoWhisper-base, in Vietnamese-English CS ASR, achieving a significantly lower word error rate of 20.8\% with reduced training resources. Furthermore, the phonetic-based two-stage architecture enables phoneme adaptation and language conversion to enhance ASR performance in complex CS Vietnamese-English ASR scenarios. 

**Abstract (ZH)**: 越南语-英语代码转换自动语音识别中的 Two-Stage 音位中心模型 

---
# ConstStyle: Robust Domain Generalization with Unified Style Transformation 

**Title (ZH)**: ConstStyle: 一致风格转换下的稳健领域泛化 

**Authors**: Nam Duong Tran, Nam Nguyen Phuong, Hieu H. Pham, Phi Le Nguyen, My T. Thai  

**Link**: [PDF](https://arxiv.org/pdf/2509.05975)  

**Abstract**: Deep neural networks often suffer performance drops when test data distribution differs from training data. Domain Generalization (DG) aims to address this by focusing on domain-invariant features or augmenting data for greater diversity. However, these methods often struggle with limited training domains or significant gaps between seen (training) and unseen (test) domains. To enhance DG robustness, we hypothesize that it is essential for the model to be trained on data from domains that closely resemble unseen test domains-an inherently difficult task due to the absence of prior knowledge about the unseen domains. Accordingly, we propose ConstStyle, a novel approach that leverages a unified domain to capture domain-invariant features and bridge the domain gap with theoretical analysis. During training, all samples are mapped onto this unified domain, optimized for seen domains. During testing, unseen domain samples are projected similarly before predictions. By aligning both training and testing data within this unified domain, ConstStyle effectively reduces the impact of domain shifts, even with large domain gaps or few seen domains. Extensive experiments demonstrate that ConstStyle consistently outperforms existing methods across diverse scenarios. Notably, when only a limited number of seen domains are available, ConstStyle can boost accuracy up to 19.82\% compared to the next best approach. 

**Abstract (ZH)**: 深度神经网络在测试数据分布与训练数据分布不同时往往会性能下降。域泛化（Domain Generalization, DG）旨在通过关注域不变特征或增强数据多样性来解决这一问题。然而，这些方法在面对训练域有限或训练域与测试域之间差距显著时常常表现不佳。为增强域泛化的鲁棒性，我们假设模型需要在其训练过程中使用与未知测试域相近的域数据，这是一个由于缺乏未知域先验知识而本身颇具挑战的任务。据此，我们提出了一种名为ConstStyle的新型方法，通过利用统一域来捕获域不变特征，并利用理论分析缩小域间差距。在训练过程中，所有样本都被映射到这个统一域中，优化针对已见域进行。在测试过程中，未知域的样本被以类似方式投影后再进行预测。通过将训练和测试数据均映射到此统一域中，ConstStyle有效地减少了域转移的影响，即使在域间差距大或已见域有限的情况下也是如此。大量实验证明，ConstStyle在各种场景中均能显著超过现有方法。特别地，在仅有有限数量已见域可用的情况下，ConstStyle相比第二优方法能将准确率提升高达19.82%。 

---
# Meta-training of diffractive meta-neural networks for super-resolution direction of arrival estimation 

**Title (ZH)**: 基于分类衍射元神经网络的超分辨到达角估计元训练 

**Authors**: Songtao Yang, Sheng Gao, Chu Wu, Zejia Zhao, Haiou Zhang, Xing Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.05926)  

**Abstract**: Diffractive neural networks leverage the high-dimensional characteristics of electromagnetic (EM) fields for high-throughput computing. However, the existing architectures face challenges in integrating large-scale multidimensional metasurfaces with precise network training and haven't utilized multidimensional EM field coding scheme for super-resolution sensing. Here, we propose diffractive meta-neural networks (DMNNs) for accurate EM field modulation through metasurfaces, which enable multidimensional multiplexing and coding for multi-task learning and high-throughput super-resolution direction of arrival estimation. DMNN integrates pre-trained mini-metanets to characterize the amplitude and phase responses of meta-atoms across different polarizations and frequencies, with structure parameters inversely designed using the gradient-based meta-training. For wide-field super-resolution angle estimation, the system simultaneously resolves azimuthal and elevational angles through x and y-polarization channels, while the interleaving of frequency-multiplexed angular intervals generates spectral-encoded optical super-oscillations to achieve full-angle high-resolution estimation. Post-processing lightweight electronic neural networks further enhance the performance. Experimental results validate that a three-layer DMNN operating at 27 GHz, 29 GHz, and 31 GHz achieves $\sim7\times$ Rayleigh diffraction-limited angular resolution (0.5$^\circ$), a mean absolute error of 0.048$^\circ$ for two incoherent targets within a $\pm 11.5^\circ$ field of view, and an angular estimation throughput an order of magnitude higher (1917) than that of existing methods. The proposed architecture advances high-dimensional photonic computing systems by utilizing inherent high-parallelism and all-optical coding methods for ultra-high-resolution, high-throughput applications. 

**Abstract (ZH)**: 衍射神经网络利用电磁场的高维特性进行高速计算。然而，现有的架构在集成大规模多维元表面及精确的网络训练方面面临挑战，并未利用多维电磁场编码方案进行超分辨率传感。这里，我们提出衍射元神经网络（DMNN）以通过元表面实现精确的电磁场调制，从而实现多维复用和编码，支持多任务学习和高速超分辨率到达角估计。DMNN集成了预先训练的小型元网络，用于表征不同极化和频率下元原子的幅度和相位响应，并通过基于梯度的元训练逆向设计结构参数。对于宽场超分辨率角度估计，系统通过x和y极化通道同时解算方位角和仰角，而频率多路复用的 angular 间隔的交错排列生成谱编码光的超振荡，以实现全方位高分辨率估计。后续处理的轻量级电子神经网络进一步提升性能。实验结果验证了在27 GHz、29 GHz和31 GHz工作的三层DMNN可实现约7倍瑞利衍射极限角度分辨率（0.5°）、视场为±11.5°内的两个非相干目标的绝对误差均值为0.048°，以及比现有方法高一个数量级的角度估计吞吐量（1917）。所提出的架构通过利用固有的高并行性和全光编码方法，推动了高维光子计算系统的应用，实现了超高分辨率和高速处理。 

---
# Challenges in Deep Learning-Based Small Organ Segmentation: A Benchmarking Perspective for Medical Research with Limited Datasets 

**Title (ZH)**: 基于深度学习的小器官分割挑战：有限数据集条件下医学研究的基准视角 

**Authors**: Phongsakon Mark Konrad, Andrei-Alexandru Popa, Yaser Sabzehmeidani, Liang Zhong, Elisa A. Liehn, Serkan Ayvaz  

**Link**: [PDF](https://arxiv.org/pdf/2509.05892)  

**Abstract**: Accurate segmentation of carotid artery structures in histopathological images is vital for advancing cardiovascular disease research and diagnosis. However, deep learning model development in this domain is constrained by the scarcity of annotated cardiovascular histopathological data. This study investigates a systematic evaluation of state-of-the-art deep learning segmentation models, including convolutional neural networks (U-Net, DeepLabV3+), a Vision Transformer (SegFormer), and recent foundation models (SAM, MedSAM, MedSAM+UNet), on a limited dataset of cardiovascular histology images. Despite employing an extensive hyperparameter optimization strategy with Bayesian search, our findings reveal that model performance is highly sensitive to data splits, with minor differences driven more by statistical noise than by true algorithmic superiority. This instability exposes the limitations of standard benchmarking practices in low-data clinical settings and challenges the assumption that performance rankings reflect meaningful clinical utility. 

**Abstract (ZH)**: 在心血管组织病理图像中准确分割颈动脉结构对于心血管疾病研究和诊断的推进至关重要。然而，该领域的深度学习模型开发受限于标注心血管组织病理数据的稀缺性。本研究探讨了在有限的心血管组织学图像数据集上，包括卷积神经网络（U-Net、DeepLabV3+）、视觉变换器（SegFormer）以及最近的基础模型（SAM、MedSAM、MedSAM+Unet）的顶级深度学习分割模型的系统评估。尽管采用了广泛的超参数优化策略（贝叶斯搜索），但研究发现模型性能高度依赖于数据划分，微小差异更多由统计噪声引起而非真正的算法优越性。这种不稳定性揭示了在数据匮乏的临床环境中标准基准测试实践的局限性，并挑战了性能排名反映实际临床效用的假设。 

---
# Quantum spatial best-arm identification via quantum walks 

**Title (ZH)**: 量子行走最佳臂识别的空间寻优方法 

**Authors**: Tomoki Yamagami, Etsuo Segawa, Takatomo Mihana, André Röhm, Atsushi Uchida, Ryoichi Horisaki  

**Link**: [PDF](https://arxiv.org/pdf/2509.05890)  

**Abstract**: Quantum reinforcement learning has emerged as a framework combining quantum computation with sequential decision-making, and applications to the multi-armed bandit (MAB) problem have been reported. The graph bandit problem extends the MAB setting by introducing spatial constraints, yet quantum approaches remain limited. We propose a quantum algorithm for best-arm identification in graph bandits, termed Quantum Spatial Best-Arm Identification (QSBAI). The method employs quantum walks to encode superpositions over graph-constrained actions, extending amplitude amplification and generalizing the Quantum BAI algorithm via Szegedy's walk framework. This establishes a link between Grover-type search and reinforcement learning tasks with structural restrictions. We analyze complete and bipartite graphs, deriving the maximal success probability of identifying the best arm and the time step at which it is achieved. Our results highlight the potential of quantum walks to accelerate exploration in constrained environments and extend the applicability of quantum algorithms for decision-making. 

**Abstract (ZH)**: 量子强化学习已成为结合量子计算与序列决策的一种框架，并已被应用于多臂bandit问题。图bandit问题通过引入空间约束扩展了MAB设置，但量子方法仍然有限。我们提出了一种用于图bandit中最佳臂识别的量子算法，称为量子空间最佳臂识别（QSBAI）。该方法利用量子行走来编码受图约束的动作的超位置，并通过Szegedy的量子步行框架扩展了振幅放大和量子BAI算法。这建立了Grover型搜索与具有结构限制的强化学习任务之间的联系。我们分析了完全图和二部图，得出了识别最佳臂的最大成功概率及其实现的时间步。结果突显了量子行走在约束环境中加速探索的潜力，并扩展了量子算法在决策中的应用。 

---
# Multimodal Prompt Injection Attacks: Risks and Defenses for Modern LLMs 

**Title (ZH)**: 多模态提示注入攻击：现代大语言模型的风险与防御 

**Authors**: Andrew Yeo, Daeseon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.05883)  

**Abstract**: Large Language Models (LLMs) have seen rapid adoption in recent years, with industries increasingly relying on them to maintain a competitive advantage. These models excel at interpreting user instructions and generating human-like responses, leading to their integration across diverse domains, including consulting and information retrieval. However, their widespread deployment also introduces substantial security risks, most notably in the form of prompt injection and jailbreak attacks.
To systematically evaluate LLM vulnerabilities -- particularly to external prompt injection -- we conducted a series of experiments on eight commercial models. Each model was tested without supplementary sanitization, relying solely on its built-in safeguards. The results exposed exploitable weaknesses and emphasized the need for stronger security measures. Four categories of attacks were examined: direct injection, indirect (external) injection, image-based injection, and prompt leakage. Comparative analysis indicated that Claude 3 demonstrated relatively greater robustness; nevertheless, empirical findings confirm that additional defenses, such as input normalization, remain necessary to achieve reliable protection. 

**Abstract (ZH)**: 大型语言模型（LLMs）近年来实现了快速 adoption，各行各业 increasingly 依赖它们来保持竞争优势。这些模型在解读用户指令并生成类人类响应方面表现出色，促使其在包括咨询和信息检索在内的众多领域中得到广泛应用。然而，它们的广泛部署也带来了显著的安全风险，尤其是提示注入和 jailbreak 攻击等形式。

为了系统地评估 LLM 的漏洞——特别是对外部提示注入的脆弱性，我们对八款商用模型进行了系列实验。每个模型均未使用额外的清理措施，仅依赖其内置的安全防护。实验结果揭示了可利用的弱点，并强调了增强安全措施的必要性。四种攻击类别被考察：直接注入、间接（外部）注入、基于图像的注入和提示泄露。对比分析表明，Claude 3 展现了相对较高的鲁棒性；然而，实证发现证实，为进一步实现可靠的防护，仍需采取额外的防御措施，如输入规范化。 

---
# Let's Roleplay: Examining LLM Alignment in Collaborative Dialogues 

**Title (ZH)**: 让我们角色扮演：探究协作对话中大规模语言模型的一致性 

**Authors**: Abhijnan Nath, Carine Graff, Nikhil Krishnaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2509.05882)  

**Abstract**: As Large Language Models (LLMs) integrate into diverse workflows, they are increasingly being considered "collaborators" with humans. If such AI collaborators are to be reliable, their behavior over multiturn interactions must be predictable, validated and verified before deployment. Common alignment techniques are typically developed under simplified single-user settings and do not account for the dynamics of long-horizon multiparty interactions. This paper examines how different alignment methods affect LLM agents' effectiveness as partners in multiturn, multiparty collaborations. We study this question through the lens of friction agents that intervene in group dialogues to encourage the collaborative group to slow down and reflect upon their reasoning for deliberative decision-making. Using a roleplay methodology, we evaluate interventions from differently-trained friction agents in collaborative task conversations. We propose a novel counterfactual evaluation framework that quantifies how friction interventions change the trajectory of group collaboration and belief alignment. Our results show that a friction-aware approach significantly outperforms common alignment baselines in helping both convergence to a common ground, or agreed-upon task-relevant propositions, and correctness of task outcomes. 

**Abstract (ZH)**: 大规模语言模型（LLMs）集成到多样化的 workflows 中，它们 increasingly 被视为与人类合作的“合作者”。如果这些 AI 合作者要可靠，在部署前它们在多轮多主体交互中的行为必须是可预测、验证和验证的。常见的对齐技术通常是在简化的一对一用户设置中开发的，不考虑长时间多主体交互的动态性。本文探讨了不同对齐方法如何影响 LLM 代理作为多轮多主体合作中伙伴的有效性。我们通过摩擦代理干预团体内对话以鼓励协作团体放慢节奏并反思其推理来进行这一问题的研究。使用角色扮演方法，我们评估了不同类型训练的摩擦代理在协作任务对话中的干预效果。我们提出了一种新颖的反事实评估框架，用于量化摩擦干预如何改变团队协作和信念对齐的轨迹。研究表明，具有摩擦意识的方法在帮助达成共识或达成一致的任务相关命题以及任务结果的准确性方面显著优于常见的对齐基线。 

---
# GeoAnalystBench: A GeoAI benchmark for assessing large language models for spatial analysis workflow and code generation 

**Title (ZH)**: GeoAnalystBench: 用于评估大型语言模型在空间分析工作流和代码生成中的性能的GeoAI基准测试 

**Authors**: Qianheng Zhang, Song Gao, Chen Wei, Yibo Zhao, Ying Nie, Ziru Chen, Shijie Chen, Yu Su, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.05881)  

**Abstract**: Recent advances in large language models (LLMs) have fueled growing interest in automating geospatial analysis and GIS workflows, yet their actual capabilities remain uncertain. In this work, we call for rigorous evaluation of LLMs on well-defined geoprocessing tasks before making claims about full GIS automation. To this end, we present GeoAnalystBench, a benchmark of 50 Python-based tasks derived from real-world geospatial problems and carefully validated by GIS experts. Each task is paired with a minimum deliverable product, and evaluation covers workflow validity, structural alignment, semantic similarity, and code quality (CodeBLEU). Using this benchmark, we assess both proprietary and open source models. Results reveal a clear gap: proprietary models such as ChatGPT-4o-mini achieve high validity 95% and stronger code alignment (CodeBLEU 0.39), while smaller open source models like DeepSeek-R1-7B often generate incomplete or inconsistent workflows (48.5% validity, 0.272 CodeBLEU). Tasks requiring deeper spatial reasoning, such as spatial relationship detection or optimal site selection, remain the most challenging across all models. These findings demonstrate both the promise and limitations of current LLMs in GIS automation and provide a reproducible framework to advance GeoAI research with human-in-the-loop support. 

**Abstract (ZH)**: Recent advances in大规模语言模型（LLMs）促进了地理空间分析和GIS工作流自动化的兴趣增长，但其实际能力仍存不确定性。本研究呼吁在宣称实现全面GIS自动化之前，对LLMs进行严格的地理处理任务评估。为此，我们提出了GeoAnalystBench基准，包括50个基于Python的实际地理空间问题任务，并由GIS专家仔细验证。每个任务都配有一个最低交付产品，评估涵盖工作流有效性、结构对齐、语义相似性和代码质量（CodeBLEU）。利用此基准，我们评估了商业和开源模型。结果显示，商业模型如ChatGPT-4o-mini在工作流有效性（95%）和更强的代码对齐（CodeBLEU 0.39）方面表现出色，而较小的开源模型如DeepSeek-R1-7B常常生成不完整或不一致的工作流（48.5%有效性，0.272 CodeBLEU）。需要更深层次空间推理的任务，如空间关系检测或最优选址，是所有模型中最具挑战性的。这些发现展示了当前LLMs在GIS自动化中的潜力和局限性，并提供了一个可重复的框架，支持通过人类在环的方式推进GeoAI研究。 

---
# Uncertainty Quantification in Probabilistic Machine Learning Models: Theory, Methods, and Insights 

**Title (ZH)**: 概率机器学习模型中的不确定性量化：理论、方法与见解 

**Authors**: Marzieh Ajirak, Anand Ravishankar, Petar M. Djuric  

**Link**: [PDF](https://arxiv.org/pdf/2509.05877)  

**Abstract**: Uncertainty Quantification (UQ) is essential in probabilistic machine learning models, particularly for assessing the reliability of predictions. In this paper, we present a systematic framework for estimating both epistemic and aleatoric uncertainty in probabilistic models. We focus on Gaussian Process Latent Variable Models and employ scalable Random Fourier Features-based Gaussian Processes to approximate predictive distributions efficiently. We derive a theoretical formulation for UQ, propose a Monte Carlo sampling-based estimation method, and conduct experiments to evaluate the impact of uncertainty estimation. Our results provide insights into the sources of predictive uncertainty and illustrate the effectiveness of our approach in quantifying the confidence in the predictions. 

**Abstract (ZH)**: 不确定性量化（UQ）在概率机器学习模型中至关重要，特别用于评估预测的可靠性。本文提出了一种系统框架，用于估算概率模型中的命题性和偶然性不确定性。我们专注于高斯过程潜在变量模型，并采用可扩展的随机傅里叶特征基于的高斯过程来高效地逼近预测分布。我们为不确定性量化推导出理论公式，提出了一种基于蒙特卡洛采样的估算方法，并进行了实验以评估不确定性估算的影响。我们的结果为预测不确定性来源提供了见解，并展示了我们方法在量化预测置信度方面的有效性。 

---
# Learning to Construct Knowledge through Sparse Reference Selection with Reinforcement Learning 

**Title (ZH)**: 通过稀疏引用选择强化学习驱动的知识构建 

**Authors**: Shao-An Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.05874)  

**Abstract**: The rapid expansion of scientific literature makes it increasingly difficult to acquire new knowledge, particularly in specialized domains where reasoning is complex, full-text access is restricted, and target references are sparse among a large set of candidates. We present a Deep Reinforcement Learning framework for sparse reference selection that emulates human knowledge construction, prioritizing which papers to read under limited time and cost. Evaluated on drug--gene relation discovery with access restricted to titles and abstracts, our approach demonstrates that both humans and machines can construct knowledge effectively from partial information. 

**Abstract (ZH)**: 基于深度强化学习的稀疏参考选择框架：在受限时间和成本条件下模拟人类知识构建过程，以发现药物-基因关系为例 

---
# ZhiFangDanTai: Fine-tuning Graph-based Retrieval-Augmented Generation Model for Traditional Chinese Medicine Formula 

**Title (ZH)**: 质方丹台：基于图的检索增强生成模型的微调研究（用于中药方剂） 

**Authors**: ZiXuan Zhang, Bowen Hao, Yingjie Li, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.05867)  

**Abstract**: Traditional Chinese Medicine (TCM) formulas play a significant role in treating epidemics and complex diseases. Existing models for TCM utilize traditional algorithms or deep learning techniques to analyze formula relationships, yet lack comprehensive results, such as complete formula compositions and detailed explanations. Although recent efforts have used TCM instruction datasets to fine-tune Large Language Models (LLMs) for explainable formula generation, existing datasets lack sufficient details, such as the roles of the formula's sovereign, minister, assistant, courier; efficacy; contraindications; tongue and pulse diagnosis-limiting the depth of model outputs. To address these challenges, we propose ZhiFangDanTai, a framework combining Graph-based Retrieval-Augmented Generation (GraphRAG) with LLM fine-tuning. ZhiFangDanTai uses GraphRAG to retrieve and synthesize structured TCM knowledge into concise summaries, while also constructing an enhanced instruction dataset to improve LLMs' ability to integrate retrieved information. Furthermore, we provide novel theoretical proofs demonstrating that integrating GraphRAG with fine-tuning techniques can reduce generalization error and hallucination rates in the TCM formula task. Experimental results on both collected and clinical datasets demonstrate that ZhiFangDanTai achieves significant improvements over state-of-the-art models. Our model is open-sourced at this https URL. 

**Abstract (ZH)**: 中医方剂图基检索增强生成框架ZhiFangDanTai 

---
# GenAI on Wall Street -- Opportunities and Risk Controls 

**Title (ZH)**: GenAI在华尔街——机遇与风险控制 

**Authors**: Jackie Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.05841)  

**Abstract**: We give an overview on the emerging applications of GenAI in the financial industry, especially within investment banks. Inherent to these exciting opportunities is a new realm of risks that must be managed properly. By heeding both the Yin and Yang sides of GenAI, we can accelerate its organic growth while safeguarding the entire financial industry during this nascent era of AI. 

**Abstract (ZH)**: GenAI在金融行业，尤其是投资银行中的新兴应用：把握机遇，管理风险，促进行业发展 

---
# Decoding Latent Attack Surfaces in LLMs: Prompt Injection via HTML in Web Summarization 

**Title (ZH)**: 解码LLMs中的潜在攻击面：通过HTML进行提示注入以进行网页总结 

**Authors**: Ishaan Verma  

**Link**: [PDF](https://arxiv.org/pdf/2509.05831)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into web-based systems for content summarization, yet their susceptibility to prompt injection attacks remains a pressing concern. In this study, we explore how non-visible HTML elements such as <meta>, aria-label, and alt attributes can be exploited to embed adversarial instructions without altering the visible content of a webpage. We introduce a novel dataset comprising 280 static web pages, evenly divided between clean and adversarial injected versions, crafted using diverse HTML-based strategies. These pages are processed through a browser automation pipeline to extract both raw HTML and rendered text, closely mimicking real-world LLM deployment scenarios. We evaluate two state-of-the-art open-source models, Llama 4 Scout (Meta) and Gemma 9B IT (Google), on their ability to summarize this content. Using both lexical (ROUGE-L) and semantic (SBERT cosine similarity) metrics, along with manual annotations, we assess the impact of these covert injections. Our findings reveal that over 29% of injected samples led to noticeable changes in the Llama 4 Scout summaries, while Gemma 9B IT showed a lower, yet non-trivial, success rate of 15%. These results highlight a critical and largely overlooked vulnerability in LLM driven web pipelines, where hidden adversarial content can subtly manipulate model outputs. Our work offers a reproducible framework and benchmark for evaluating HTML-based prompt injection and underscores the urgent need for robust mitigation strategies in LLM applications involving web content. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地被集成到基于Web的系统中进行内容摘要，但它们对提示注入攻击的易感性仍然是一个紧迫的问题。在这项研究中，我们探讨了如何利用非可视HTML元素（如<meta>、aria-label和alt属性）在不改变网页可见内容的情况下嵌入 adversarial 指令。我们引入了一个由280个静态网页组成的新数据集，这些网页均匀分为干净版本和恶意注入版本，并使用多种HTML基策略进行构建。这些页面通过浏览器自动化管道处理，以提取原始HTML和渲染文本，模拟真实的LLM部署场景。我们评估了两个最先进的开源模型——Llama 4 Scout（Meta）和Gemma 9B IT（Google）——在摘要内容方面的能力。使用词汇（ROUGE-L）和语义（SBERT余弦相似度）指标，以及手动注释，我们评估了这些隐蔽注入的影响。我们的发现表明，超过29%的注入样本导致了Llama 4 Scout摘要的明显变化，而Gemma 9B IT的成功率较低，但仍然非 trivial，为15%。这些结果强调了LLM驱动的Web管道中一个关键且被忽视的安全漏洞，在这些管道中，隐藏的 adversarial 内容可以微妙地操控模型输出。我们的工作提供了一个可复制的框架和基准，用于评估基于HTML的提示注入，并强调了在涉及Web内容的LLM应用中迫切需要强大的缓解策略。 

---
# time2time: Causal Intervention in Hidden States to Simulate Rare Events in Time Series Foundation Models 

**Title (ZH)**: 时间到时间：在隐藏状态中进行因果干预以模拟时间序列基础模型中的罕见事件 

**Authors**: Debdeep Sanyal, Aaryan Nagpal, Dhruv Kumar, Murari Mandal, Saurabh Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2509.05801)  

**Abstract**: While transformer-based foundation models excel at forecasting routine patterns, two questions remain: do they internalize semantic concepts such as market regimes, or merely fit curves? And can their internal representations be leveraged to simulate rare, high-stakes events such as market crashes? To investigate this, we introduce activation transplantation, a causal intervention that manipulates hidden states by imposing the statistical moments of one event (e.g., a historical crash) onto another (e.g., a calm period) during the forward pass. This procedure deterministically steers forecasts: injecting crash semantics induces downturn predictions, while injecting calm semantics suppresses crashes and restores stability. Beyond binary control, we find that models encode a graded notion of event severity, with the latent vector norm directly correlating with the magnitude of systemic shocks. Validated across two architecturally distinct TSFMs, Toto (decoder only) and Chronos (encoder-decoder), our results demonstrate that steerable, semantically grounded representations are a robust property of large time series transformers. Our findings provide evidence for a latent concept space that governs model predictions, shifting interpretability from post-hoc attribution to direct causal intervention, and enabling semantic "what-if" analysis for strategic stress-testing. 

**Abstract (ZH)**: 基于变换器的基础模型在预测常规模式方面表现出色，但仍存在两个问题：它们是否会内部化市场制度等语义概念，还是仅仅拟合曲线？它们的内部表示能否用来模拟市场崩盘等罕见的高风险事件？为探究这一问题，我们引入了一种称为激活移植的因果干预方法，在前向传递过程中通过将一个事件（如历史崩盘）的统计特征施加到另一个事件（如平静时期）上来操纵隐藏状态。这种方法在确定性地引导预测：注入崩盘语义会引发衰退预测，而注入平静语义则会抑制崩盘并恢复稳定。除了二元控制，我们发现模型编码了事件严重性的分级概念，潜在向量的范数直接与系统性冲击的幅度相关。在两种具有不同架构的时间序列变换器（Toto 和 Chronos）中得到验证，我们的结果表明，可引导且具语义基础的表示是大规模时间序列变换器的稳健属性。我们的研究提供了证据，表明存在一个潜在的概念空间控制着模型的预测，将解释性从事后归因转变为直接因果干预，并使战略压力测试中的语义“假设情境”分析成为可能。 

---
# Hybrid Fourier Neural Operator-Plasma Fluid Model for Fast and Accurate Multiscale Simulations of High Power Microwave Breakdown 

**Title (ZH)**: 混合傅里叶神经运算子等离子体流体模型：快速准确的高功率微波击穿多尺度仿真 

**Authors**: Kalp Pandya, Pratik Ghosh, Ajeya Mandikal, Shivam Gandha, Bhaskar Chaudhury  

**Link**: [PDF](https://arxiv.org/pdf/2509.05799)  

**Abstract**: Modeling and simulation of High Power Microwave (HPM) breakdown, a multiscale phenomenon, is computationally expensive and requires solving Maxwell's equations (EM solver) coupled with a plasma continuity equation (plasma solver). In this work, we present a hybrid modeling approach that combines the accuracy of a differential equation-based plasma fluid solver with the computational efficiency of FNO (Fourier Neural Operator) based EM solver. Trained on data from an in-house FDTD-based plasma-fluid solver, the FNO replaces computationally expensive EM field updates, while the plasma solver governs the dynamic plasma response. The hybrid model is validated on microwave streamer formation, due to diffusion ionization mechanism, in a 2D scenario for unseen incident electric fields corresponding to entirely new plasma streamer simulations not included in model training, showing excellent agreement with FDTD based fluid simulations in terms of streamer shape, velocity, and temporal evolution. This hybrid FNO based strategy delivers significant acceleration of the order of 60X compared to traditional simulations for the specified problem size and offers an efficient alternative for computationally demanding multiscale and multiphysics simulations involved in HPM breakdown. Our work also demonstrate how such hybrid pipelines can be used to seamlessly to integrate existing C-based simulation codes with Python-based machine learning frameworks for simulations of plasma science and engineering problems. 

**Abstract (ZH)**: 基于FNO的混合模型方法在高功率微波（HPM）击穿多尺度现象建模与仿真中的应用研究 

---
# Dual-Mode Deep Anomaly Detection for Medical Manufacturing: Structural Similarity and Feature Distance 

**Title (ZH)**: 双模式深度异常检测在医疗制造中的应用：结构 similarity 和特征距离 

**Authors**: Julio Zanon Diaz, Georgios Siogkas, Peter Corcoran  

**Link**: [PDF](https://arxiv.org/pdf/2509.05796)  

**Abstract**: Automating visual inspection in medical device manufacturing remains challenging due to small and imbalanced datasets, high-resolution imagery, and stringent regulatory requirements. This work proposes two attention-guided autoencoder architectures for deep anomaly detection designed to address these constraints. The first employs a structural similarity-based anomaly score (4-MS-SSIM), offering lightweight and accurate real-time defect detection, yielding ACC 0.903 (unsupervised thresholding) and 0.931 (supervised thresholding) on the - Surface Seal Image - Test split with only 10% of defective samples. The second applies a feature-distance approach using Mahalanobis scoring on reduced latent features, providing high sensitivity to distributional shifts for supervisory monitoring, achieving ACC 0.722 with supervised thresholding. Together, these methods deliver complementary capabilities: the first supports reliable inline inspection, while the second enables scalable post-production surveillance and regulatory compliance monitoring. Experimental results demonstrate that both approaches surpass re-implemented baselines and provide a practical pathway for deploying deep anomaly detection in regulated manufacturing environments, aligning accuracy, efficiency, and the regulatory obligations defined for high-risk AI systems under the EU AI Act. 

**Abstract (ZH)**: 医疗设备制造中基于视觉检测的自动化 remains 挑战性由于小规模和不均衡的数据集、高分辨率图像以及严格的监管要求。本研究提出了两种基于注意力机制的自编码器架构，用于解决这些约束条件下的深度异常检测。第一个架构采用基于结构相似性的异常评分（4-MS-SSIM），提供轻量级且准确的实时缺陷检测，仅使用10%的缺陷样本在Surface Seal Image Test分割上达到无监督阈值 ACC 0.903 和监督阈值 ACC 0.931。第二个架构采用特征距离方法，使用 Mahalanobis 得分对降维后潜变量进行评分，为监督监控提供对分布偏移的高灵敏度，通过监督阈值实现 ACC 0.722。这两种方法互为补充：第一个支持可靠的在线检测，而第二个方法则可以实现可扩展的生产后监控和监管合规监控。实验结果表明，这两种方法均优于重新实现的基线方法，并提供了一条在受监管的制造环境中部署深度异常检测的实际路径，符合欧盟AI法案对高风险AI系统规定的准确度、效率和监管义务。 

---
# DCV-ROOD Evaluation Framework: Dual Cross-Validation for Robust Out-of-Distribution Detection 

**Title (ZH)**: DCV-ROOD评估框架：稳健的离分布检测双重交叉验证 

**Authors**: Arantxa Urrea-Castaño, Nicolás Segura-Kunsagi, Juan Luis Suárez-Díaz, Rosana Montes, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2509.05778)  

**Abstract**: Out-of-distribution (OOD) detection plays a key role in enhancing the robustness of artificial intelligence systems by identifying inputs that differ significantly from the training distribution, thereby preventing unreliable predictions and enabling appropriate fallback mechanisms. Developing reliable OOD detection methods is a significant challenge, and rigorous evaluation of these techniques is essential for ensuring their effectiveness, as it allows researchers to assess their performance under diverse conditions and to identify potential limitations or failure modes. Cross-validation (CV) has proven to be a highly effective tool for providing a reasonable estimate of the performance of a learning algorithm. Although OOD scenarios exhibit particular characteristics, an appropriate adaptation of CV can lead to a suitable evaluation framework for this setting. This work proposes a dual CV framework for robust evaluation of OOD detection models, aimed at improving the reliability of their assessment. The proposed evaluation framework aims to effectively integrate in-distribution (ID) and OOD data while accounting for their differing characteristics. To achieve this, ID data are partitioned using a conventional approach, whereas OOD data are divided by grouping samples based on their classes. Furthermore, we analyze the context of data with class hierarchy to propose a data splitting that considers the entire class hierarchy to obtain fair ID-OOD partitions to apply the proposed evaluation framework. This framework is called Dual Cross-Validation for Robust Out-of-Distribution Detection (DCV-ROOD). To test the validity of the evaluation framework, we selected a set of state-of-the-art OOD detection methods, both with and without outlier exposure. The results show that the method achieves very fast convergence to the true performance. 

**Abstract (ZH)**: 双交叉验证用于稳健的异常分布检测评估（DCV-ROOD） 

---
# Real-E: A Foundation Benchmark for Advancing Robust and Generalizable Electricity Forecasting 

**Title (ZH)**: Real-E：促进鲁棒性和泛化能力电能预测的基础基准 

**Authors**: Chen Shao, Yue Wang, Zhenyi Zhu, Zhanbo Huang, Sebastian Pütz, Benjamin Schäfer, Tobais Käfer, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2509.05768)  

**Abstract**: Energy forecasting is vital for grid reliability and operational efficiency. Although recent advances in time series forecasting have led to progress, existing benchmarks remain limited in spatial and temporal scope and lack multi-energy features. This raises concerns about their reliability and applicability in real-world deployment. To address this, we present the Real-E dataset, covering over 74 power stations across 30+ European countries over a 10-year span with rich metadata. Using Real- E, we conduct an extensive data analysis and benchmark over 20 baselines across various model types. We introduce a new metric to quantify shifts in correlation structures and show that existing methods struggle on our dataset, which exhibits more complex and non-stationary correlation dynamics. Our findings highlight key limitations of current methods and offer a strong empirical basis for building more robust forecasting models 

**Abstract (ZH)**: 能源预测对于电网可靠性和运营效率至关重要。尽管近期时间序列预测的进展取得了进步，现有的基准在空间和时间范围上仍有限制，并且缺乏多能源特征。这引起了对其在实际部署中的可靠性和适用性的担忧。为应对这一挑战，我们介绍了Real-E数据集，该数据集涵盖了来自30多个欧洲国家的超过74个发电站，时间跨度为10年，并附有丰富的元数据。使用Real-E，我们进行了广泛的数据分析，并在各种模型类型中超过20种基线方法上进行了基准测试。我们引入了一个新的度量标准来量化相关结构的转变，并展示了现有方法在我们数据集上表现不佳，该数据集表现出更复杂和非稳定的相关动态。我们的研究结果突显了当前方法的关键局限性，并为构建更稳健的预测模型提供了强有力的实证基础。 

---
# Exploit Tool Invocation Prompt for Tool Behavior Hijacking in LLM-Based Agentic System 

**Title (ZH)**: 利用工具调用提示劫持基于LLM的代理系统中的工具行为 

**Authors**: Yu Liu, Yuchong Xie, Mingyu Luo, Zesen Liu, Zhixiang Zhang, Kaikai Zhang, Zongjie Li, Ping Chen, Shuai Wang, Dongdong She  

**Link**: [PDF](https://arxiv.org/pdf/2509.05755)  

**Abstract**: LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems. 

**Abstract (ZH)**: 基于LLM的代理系统通过大规模语言模型处理用户查询、做出决策并执行外部工具以跨领域（如聊天机器人、客户服务和软件工程）完成复杂任务。这些系统的关键组成部分是工具调用提示（TIP），它定义了工具交互协议并指导LLM以确保工具使用的安全性和正确性。尽管其重要性不言而喻，但TIP安全问题却常常被忽视。本研究探讨了与TIP相关的安全风险，揭示了诸如Cursor、Claude Code等主要基于LLM的系统容易受到远程代码执行（RCE）和拒绝服务（DoS）等攻击。通过系统性的工具调用提示exploitation工作流（TEW），我们展示了通过操纵工具调用来劫持外部工具行为。此外，我们还提出了增强基于LLM的代理系统中TIP安全性的防御机制。 

---
# Tell-Tale Watermarks for Explanatory Reasoning in Synthetic Media Forensics 

**Title (ZH)**: Tell-Tale水印在合成媒体鉴伪中的解释性推理中应用 

**Authors**: Ching-Chun Chang, Isao Echizen  

**Link**: [PDF](https://arxiv.org/pdf/2509.05753)  

**Abstract**: The rise of synthetic media has blurred the boundary between reality and fabrication under the evolving power of artificial intelligence, fueling an infodemic that erodes public trust in cyberspace. For digital imagery, a multitude of editing applications further complicates the forensic analysis, including semantic edits that alter content, photometric adjustments that recalibrate colour characteristics, and geometric projections that reshape viewpoints. Collectively, these transformations manipulate and control perceptual interpretation of digital imagery. This susceptibility calls for forensic enquiry into reconstructing the chain of events, thereby revealing deeper evidential insight into the presence or absence of criminal intent. This study seeks to address an inverse problem of tracing the underlying generation chain that gives rise to the observed synthetic media. A tell-tale watermarking system is developed for explanatory reasoning over the nature and extent of transformations across the lifecycle of synthetic media. Tell-tale watermarks are tailored to different classes of transformations, responding in a manner that is neither strictly robust nor fragile but instead interpretable. These watermarks function as reference clues that evolve under the same transformation dynamics as the carrier media, leaving interpretable traces when subjected to transformations. Explanatory reasoning is then performed to infer the most plausible account across the combinatorial parameter space of composite transformations. Experimental evaluations demonstrate the validity of tell-tale watermarking with respect to fidelity, synchronicity and traceability. 

**Abstract (ZH)**: 合成媒体的兴起模糊了现实与伪造之间的界限，在人工智能不断演进的力量下引发了信息疫情，侵蚀了网络空间中的公众信任。对于数字图像而言，众多编辑应用进一步复杂化了法医分析，包括语义编辑、光度调整和几何投影等变换，这些变换操纵和控制着数字图像的知觉解释。这种易受操控性要求进行法医调查以重建事件链，从而揭示犯罪意图存在的证据。本研究旨在解决合成媒体生成链条追溯的逆问题。开发了一种告示水印系统，用于解释性推理不同类型的变换在整个合成媒体生命周期中的范围和性质。这些告示水印针对不同的变换类别进行定制，既不具备严格鲁棒性也不具备脆弱性，而是具备可解释性。这些水印作为参考线索，在跟随载体媒体的变换动态演化后，在受到变换时留下可解释的痕迹。通过解释性推理，在组合变换参数空间中推断出最有可能的情况。实验评估证明了告示水印在保真度、同步性和可追溯性方面的有效性。 

---
# Unleashing Hierarchical Reasoning: An LLM-Driven Framework for Training-Free Referring Video Object Segmentation 

**Title (ZH)**: 解锁层次推理：一种基于大型语言模型的无需训练的视频对象分割框架 

**Authors**: Bingrui Zhao, Lin Yuanbo Wu, Xiangtian Fan, Deyin Liu, Lu Zhang, Ruyi He, Jialie Shen, Ximing Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.05751)  

**Abstract**: Referring Video Object Segmentation (RVOS) aims to segment an object of interest throughout a video based on a language description. The prominent challenge lies in aligning static text with dynamic visual content, particularly when objects exhibiting similar appearances with inconsistent motion and poses. However, current methods often rely on a holistic visual-language fusion that struggles with complex, compositional descriptions. In this paper, we propose \textbf{PARSE-VOS}, a novel, training-free framework powered by Large Language Models (LLMs), for a hierarchical, coarse-to-fine reasoning across text and video domains. Our approach begins by parsing the natural language query into structured semantic commands. Next, we introduce a spatio-temporal grounding module that generates all candidate trajectories for all potential target objects, guided by the parsed semantics. Finally, a hierarchical identification module select the correct target through a two-stage reasoning process: it first performs coarse-grained motion reasoning with an LLM to narrow down candidates; if ambiguity remains, a fine-grained pose verification stage is conditionally triggered to disambiguate. The final output is an accurate segmentation mask for the target object. \textbf{PARSE-VOS} achieved state-of-the-art performance on three major benchmarks: Ref-YouTube-VOS, Ref-DAVIS17, and MeViS. 

**Abstract (ZH)**: 基于语言描述的视频对象分割（RVOS）旨在根据语言描述对视频中的目标对象进行分割。主要挑战在于将静态文本与动态视觉内容对齐，特别是对于具有不一致运动和姿态的相似外观对象。然而，当前方法常常依赖整体的视觉-语言融合，这在处理复杂的组合性描述时存在困难。本文提出了一种新的、无需训练的框架PARSE-VOS，该框架由大型语言模型（LLMs）驱动，用于跨文本和视频领域进行层次化的粗细粒度推理。该方法首先将自然语言查询解析为结构化的语义命令。接着引入时空 grounding 模块生成所有潜在目标对象的所有候选轨迹，受解析语义的引导。最后，通过两阶段推理过程实现层次化的识别模块：首先使用LLM进行粗粒度运动推理以缩小候选人；如果仍有歧义，则有条件地触发细粒度姿态验证阶段进行消歧。最终输出是目标对象的准确分割掩模。PARSE-VOS在三个主要基准（Ref-YouTube-VOS、Ref-DAVIS17和MeViS）上取得了最先进的性能。 

---
# InterAct: A Large-Scale Dataset of Dynamic, Expressive and Interactive Activities between Two People in Daily Scenarios 

**Title (ZH)**: InterAct: 日常场景中两个人之间动态、-expressionistic 和互动活动的大规模数据集 

**Authors**: Leo Ho, Yinghao Huang, Dafei Qin, Mingyi Shi, Wangpok Tse, Wei Liu, Junichi Yamagishi, Taku Komura  

**Link**: [PDF](https://arxiv.org/pdf/2509.05747)  

**Abstract**: We address the problem of accurate capture of interactive behaviors between two people in daily scenarios. Most previous works either only consider one person or solely focus on conversational gestures of two people, assuming the body orientation and/or position of each actor are constant or barely change over each interaction. In contrast, we propose to simultaneously model two people's activities, and target objective-driven, dynamic, and semantically consistent interactions which often span longer duration and cover bigger space. To this end, we capture a new multi-modal dataset dubbed InterAct, which is composed of 241 motion sequences where two people perform a realistic and coherent scenario for one minute or longer over a complete interaction. For each sequence, two actors are assigned different roles and emotion labels, and collaborate to finish one task or conduct a common interaction activity. The audios, body motions, and facial expressions of both persons are captured. InterAct contains diverse and complex motions of individuals and interesting and relatively long-term interaction patterns barely seen before. We also demonstrate a simple yet effective diffusion-based method that estimates interactive face expressions and body motions of two people from speech inputs. Our method regresses the body motions in a hierarchical manner, and we also propose a novel fine-tuning mechanism to improve the lip accuracy of facial expressions. To facilitate further research, the data and code is made available at this https URL . 

**Abstract (ZH)**: 我们解决了一日常生活中两个人之间互动行为准确捕捉的问题。大多数先前的工作要么仅考虑一人，要么仅聚焦于两人对话手势，假设每个表演者的身体朝向和/or位置在每次互动中保持不变或仅微有变化。相反，我们提出同时建模两个人的活动，并针对具有目标驱动、动态且语义一致的互动，这些互动往往持续时间更长、覆盖范围更广。为此，我们捕获了一个名为InterAct的新多模态数据集，该数据集由241个运动序列组成，两个演员在一分钟或更长时间内完成一个现实且连贯的互动场景。对于每个序列，两位演员分配不同的角色和情绪标签，并协作完成一项任务或进行共同互动活动。两位演员的音频、身体运动和面部表情都被捕捉下来。InterAct包含个体多样且复杂的运动以及前所未见的有趣且相对长期的互动模式。我们还展示了从语音输入估计两人互动面部表情和身体运动的一种简单有效的扩散方法。该方法按层次回归身体运动，并提出了一种新的微调机制以提高面部表情的唇部准确性。为了促进进一步研究，数据和代码可通过此链接获取。 

---
# Reasoning Introduces New Poisoning Attacks Yet Makes Them More Complicated 

**Title (ZH)**: 推理引入了新的中毒攻击但使它们更加复杂 

**Authors**: Hanna Foerster, Ilia Shumailov, Yiren Zhao, Harsh Chaudhari, Jamie Hayes, Robert Mullins, Yarin Gal  

**Link**: [PDF](https://arxiv.org/pdf/2509.05739)  

**Abstract**: Early research into data poisoning attacks against Large Language Models (LLMs) demonstrated the ease with which backdoors could be injected. More recent LLMs add step-by-step reasoning, expanding the attack surface to include the intermediate chain-of-thought (CoT) and its inherent trait of decomposing problems into subproblems. Using these vectors for more stealthy poisoning, we introduce ``decomposed reasoning poison'', in which the attacker modifies only the reasoning path, leaving prompts and final answers clean, and splits the trigger across multiple, individually harmless components.
Fascinatingly, while it remains possible to inject these decomposed poisons, reliably activating them to change final answers (rather than just the CoT) is surprisingly difficult. This difficulty arises because the models can often recover from backdoors that are activated within their thought processes. Ultimately, it appears that an emergent form of backdoor robustness is originating from the reasoning capabilities of these advanced LLMs, as well as from the architectural separation between reasoning and final answer generation. 

**Abstract (ZH)**: 早期对大型语言模型（LLMs）的数据投毒攻击研究显示了后门注入的简便性。更近期的LLMs增加了逐步推理功能，扩展了攻击表面，不仅包括中间推理链（CoT），还包括其将问题分解为子问题的内在特性。利用这些向量进行更隐蔽的投毒，我们引入了“分解推理投毒”概念，在此过程中，攻击者仅修改推理路径，而保留提示和最终答案的清洁性，并将触发器分散在多个单独无害的组件中。

令人惊讶的是，虽然可以注入这些分解投毒，但可靠地激活它们以改变最终答案（而不仅仅是CoT）是出奇地困难。这种困难源于模型在其思维过程中激活后门时能够进行恢复。最终，似乎从这些先进LLMs的推理能力和推理与最终答案生成之间的架构分离中涌现出一种新的后门稳健性形式。 

---
# Offline vs. Online Learning in Model-based RL: Lessons for Data Collection Strategies 

**Title (ZH)**: 基于模型的强化学习中离线学习与在线学习的比较：数据收集策略的启示 

**Authors**: Jiaqi Chen, Ji Shi, Cansu Sancaktar, Jonas Frey, Georg Martius  

**Link**: [PDF](https://arxiv.org/pdf/2509.05735)  

**Abstract**: Data collection is crucial for learning robust world models in model-based reinforcement learning. The most prevalent strategies are to actively collect trajectories by interacting with the environment during online training or training on offline datasets. At first glance, the nature of learning task-agnostic environment dynamics makes world models a good candidate for effective offline training. However, the effects of online vs. offline data on world models and thus on the resulting task performance have not been thoroughly studied in the literature. In this work, we investigate both paradigms in model-based settings, conducting experiments on 31 different environments. First, we showcase that online agents outperform their offline counterparts. We identify a key challenge behind performance degradation of offline agents: encountering Out-Of-Distribution states at test time. This issue arises because, without the self-correction mechanism in online agents, offline datasets with limited state space coverage induce a mismatch between the agent's imagination and real rollouts, compromising policy training. We demonstrate that this issue can be mitigated by allowing for additional online interactions in a fixed or adaptive schedule, restoring the performance of online training with limited interaction data. We also showcase that incorporating exploration data helps mitigate the performance degradation of offline agents. Based on our insights, we recommend adding exploration data when collecting large datasets, as current efforts predominantly focus on expert data alone. 

**Abstract (ZH)**: 基于模型的强化学习中世界模型的数据收集至关重要：在线数据与离线数据的影响研究 

---
# Simulation Priors for Data-Efficient Deep Learning 

**Title (ZH)**: 数据高效的深度学习的模拟先验 

**Authors**: Lenart Treven, Bhavya Sukhija, Jonas Rothfuss, Stelian Coros, Florian Dörfler, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2509.05732)  

**Abstract**: How do we enable AI systems to efficiently learn in the real-world? First-principles models are widely used to simulate natural systems, but often fail to capture real-world complexity due to simplifying assumptions. In contrast, deep learning approaches can estimate complex dynamics with minimal assumptions but require large, representative datasets. We propose SimPEL, a method that efficiently combines first-principles models with data-driven learning by using low-fidelity simulators as priors in Bayesian deep learning. This enables SimPEL to benefit from simulator knowledge in low-data regimes and leverage deep learning's flexibility when more data is available, all the while carefully quantifying epistemic uncertainty. We evaluate SimPEL on diverse systems, including biological, agricultural, and robotic domains, showing superior performance in learning complex dynamics. For decision-making, we demonstrate that SimPEL bridges the sim-to-real gap in model-based reinforcement learning. On a high-speed RC car task, SimPEL learns a highly dynamic parking maneuver involving drifting with substantially less data than state-of-the-art baselines. These results highlight the potential of SimPEL for data-efficient learning and control in complex real-world environments. 

**Abstract (ZH)**: 如何使AI系统在实际环境中高效学习？SimPEL：结合基础原理模型与数据驱动学习的方法 

---
# LiDAR-BIND-T: Improving SLAM with Temporally Consistent Cross-Modal LiDAR Reconstruction 

**Title (ZH)**: LiDAR-BIND-T：通过具有一致时序的跨模态LiDAR重建改善SLAM 

**Authors**: Niels Balemans, Ali Anwar, Jan Steckel, Siegfried Mercelis  

**Link**: [PDF](https://arxiv.org/pdf/2509.05728)  

**Abstract**: This paper extends LiDAR-BIND, a modular multi-modal fusion framework that binds heterogeneous sensors (radar, sonar) to a LiDAR-defined latent space, with mechanisms that explicitly enforce temporal consistency. We introduce three contributions: (i) temporal embedding similarity that aligns consecutive latents, (ii) a motion-aligned transformation loss that matches displacement between predictions and ground truth LiDAR, and (iii) windows temporal fusion using a specialised temporal module. We further update the model architecture to better preserve spatial structure. Evaluations on radar/sonar-to-LiDAR translation demonstrate improved temporal and spatial coherence, yielding lower absolute trajectory error and better occupancy map accuracy in Cartographer-based SLAM (Simultaneous Localisation and Mapping). We propose different metrics based on the Fréchet Video Motion Distance (FVMD) and a correlation-peak distance metric providing practical temporal quality indicators to evaluate SLAM performance. The proposed temporal LiDAR-BIND, or LiDAR-BIND-T, maintains plug-and-play modality fusion while substantially enhancing temporal stability, resulting in improved robustness and performance for downstream SLAM. 

**Abstract (ZH)**: 本文将LiDAR-BIND扩展为一种模块化的多模态融合框架，该框架将雷达和声纳等异构传感器绑定到由LiDAR定义的潜在空间中，并通过显式机制确保时间一致性。我们引入了三项贡献：（i）时间嵌入相似性，使连续的潜在变量对齐；（ii）运动对齐变换损失，匹配预测值和地面实测LiDAR之间的位移；（iii）使用专门的时间模块进行窗口时间融合。我们进一步更新了模型架构，以更好地保留空间结构。在雷达/声纳到LiDAR的转换评估中，展示了改进的时间一致性和空间一致性，得到了更低的绝对轨迹误差和更好的Occupancy地图精度，在基于Cartographer的SLAM中。我们提出了基于Fréchet视频运动距离（FVMD）和相关峰距离度量的不同指标，用以评估基于SLAM的性能的时间质量。提出的时序LiDAR-BIND，或LiDAR-BIND-T，在保持即插即用模态融合的同时，显著增强了时间稳定性，从而提高了下游SLAM的鲁棒性和性能。 

---
# A Survey of the State-of-the-Art in Conversational Question Answering Systems 

**Title (ZH)**: 当前对话式问答系统综述 

**Authors**: Manoj Madushanka Perera, Adnan Mahmood, Kasun Eranda Wijethilake, Fahmida Islam, Maryam Tahermazandarani, Quan Z. Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.05716)  

**Abstract**: Conversational Question Answering (ConvQA) systems have emerged as a pivotal area within Natural Language Processing (NLP) by driving advancements that enable machines to engage in dynamic and context-aware conversations. These capabilities are increasingly being applied across various domains, i.e., customer support, education, legal, and healthcare where maintaining a coherent and relevant conversation is essential. Building on recent advancements, this survey provides a comprehensive analysis of the state-of-the-art in ConvQA. This survey begins by examining the core components of ConvQA systems, i.e., history selection, question understanding, and answer prediction, highlighting their interplay in ensuring coherence and relevance in multi-turn conversations. It further investigates the use of advanced machine learning techniques, including but not limited to, reinforcement learning, contrastive learning, and transfer learning to improve ConvQA accuracy and efficiency. The pivotal role of large language models, i.e., RoBERTa, GPT-4, Gemini 2.0 Flash, Mistral 7B, and LLaMA 3, is also explored, thereby showcasing their impact through data scalability and architectural advancements. Additionally, this survey presents a comprehensive analysis of key ConvQA datasets and concludes by outlining open research directions. Overall, this work offers a comprehensive overview of the ConvQA landscape and provides valuable insights to guide future advancements in the field. 

**Abstract (ZH)**: 基于对话的问答（ConvQA）系统已成为自然语言处理（NLP）中一个关键领域，通过推动使机器能够进行动态和上下文相关的对话。这些能力正越来越多地被应用到客户支持、教育、法律和医疗等领域，其中保持连贯和相关对话至关重要。本综述基于近期进展，对 ConvQA 的最新状态进行了全面分析。本综述首先审视了 ConvQA 系统的核心组件，即历史选择、问题理解与答案预测，并强调了它们如何相互作用以确保多轮对话的连贯性和相关性。随后进一步探讨了使用强化学习、对比学习和迁移学习等高级机器学习技术以提高 ConvQA 的准确性和效率。同时，还探讨了大型语言模型，如 RoBERTa、GPT-4、Gemini 2.0 Flash、Mistral 7B 和 LLaMA 3 的关键作用，并展示了它们通过数据规模扩展和架构进展所产生的重要影响。此外，本综述还对关键的 ConvQA 数据集进行了全面分析，并概述了未来研究方向。总体而言，本研究工作提供了 ConvQA 场景的全面概述，并提供了对未来领域发展的宝贵见解。 

---
# Knowledge-Augmented Vision Language Models for Underwater Bioacoustic Spectrogram Analysis 

**Title (ZH)**: 知识增强的视觉语言模型在水下生物声谱分析中的应用 

**Authors**: Ragib Amin Nihal, Benjamin Yen, Takeshi Ashizawa, Kazuhiro Nakadai  

**Link**: [PDF](https://arxiv.org/pdf/2509.05703)  

**Abstract**: Marine mammal vocalization analysis depends on interpreting bioacoustic spectrograms. Vision Language Models (VLMs) are not trained on these domain-specific visualizations. We investigate whether VLMs can extract meaningful patterns from spectrograms visually. Our framework integrates VLM interpretation with LLM-based validation to build domain knowledge. This enables adaptation to acoustic data without manual annotation or model retraining. 

**Abstract (ZH)**: marine哺乳动物声音分析依赖于bioacoustic声谱图的解释。视觉语言模型（VLMs）未在这些领域特定的可视化上进行训练。我们研究了VLMs是否可以从声谱图中视觉提取有意义的模式。我们的框架将VLM解析与基于LLM的验证相结合，构建领域知识，从而实现对声学数据的适应，无需手动标注或模型重新训练。 

---
# Revealing the Numeracy Gap: An Empirical Investigation of Text Embedding Models 

**Title (ZH)**: 揭示 numeracy 隙口：文本嵌入模型的实证研究 

**Authors**: Ningyuan Deng, Hanyu Duan, Yixuan Tang, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05691)  

**Abstract**: Text embedding models are widely used in natural language processing applications. However, their capability is often benchmarked on tasks that do not require understanding nuanced numerical information in text. As a result, it remains unclear whether current embedding models can precisely encode numerical content, such as numbers, into embeddings. This question is critical because embedding models are increasingly applied in domains where numbers matter, such as finance and healthcare. For example, Company X's market share grew by 2\% should be interpreted very differently from Company X's market share grew by 20\%, even though both indicate growth in market share. This study aims to examine whether text embedding models can capture such nuances. Using synthetic data in a financial context, we evaluate 13 widely used text embedding models and find that they generally struggle to capture numerical details accurately. Our further analyses provide deeper insights into embedding numeracy, informing future research to strengthen embedding model-based NLP systems with improved capacity for handling numerical content. 

**Abstract (ZH)**: 文本嵌入模型在自然语言处理应用中广泛应用，但其能力往往通过不需理解文本中细微数值信息的任务进行评估。这使得人们不清楚当前的嵌入模型是否能够精确地将数字内容，如数字，编码到嵌入式表示中。由于嵌入模型在涉及数字的重要领域，如金融和医疗保健中被越来越多地应用，这个问题尤为重要。例如，Company X的市场份额增长了2%和增长了20%应被非常不同地解释，尽管两者都表明市场份额的增长。本研究旨在探讨文本嵌入模型是否能够捕捉到这种细微差别。使用金融背景下的人工合成数据，我们评估了13种广泛使用的文本嵌入模型，并发现它们在准确捕捉数值细节方面普遍存在问题。进一步分析提供了对嵌入数值能力的更深入见解，为未来研究加强基于嵌入模型的自然语言处理系统的处理数值内容的能力提供了指导。 

---
# SEASONED: Semantic-Enhanced Self-Counterfactual Explainable Detection of Adversarial Exploiter Contracts 

**Title (ZH)**: SEASONED: 语义增强自反事实可解释 adversarial合约检测 

**Authors**: Xng Ai, Shudan Lin, Zecheng Li, Kai Zhou, Bixin Li, Bin Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.05681)  

**Abstract**: Decentralized Finance (DeFi) attacks have resulted in significant losses, often orchestrated through Adversarial Exploiter Contracts (AECs) that exploit vulnerabilities in victim smart contracts. To proactively identify such threats, this paper targets the explainable detection of AECs.
Existing detection methods struggle to capture semantic dependencies and lack interpretability, limiting their effectiveness and leaving critical knowledge gaps in AEC analysis. To address these challenges, we introduce SEASONED, an effective, self-explanatory, and robust framework for AEC detection.
SEASONED extracts semantic information from contract bytecode to construct a semantic relation graph (SRG), and employs a self-counterfactual explainable detector (SCFED) to classify SRGs and generate explanations that highlight the core attack logic. SCFED further enhances robustness, generalizability, and data efficiency by extracting representative information from these explanations. Both theoretical analysis and experimental results demonstrate the effectiveness of SEASONED, which showcases outstanding detection performance, robustness, generalizability, and data efficiency learning ability. To support further research, we also release a new dataset of 359 AECs. 

**Abstract (ZH)**: 去中心化金融(DeFi)攻击导致了重大损失，often orchestrated through 对抗性 exploit者合约(AECs)，这些合约利用了受害智能合约中的漏洞。为了提前识别此类威胁，本文旨在解释性检测AECs。
现有的检测方法难以捕捉语义依赖关系并且缺乏可解释性，限制了其有效性并留下了AEC分析中的关键知识缺口。为了应对这些挑战，我们提出了SEASONED，这是一种有效、自我解释且稳健的AEC检测框架。
SEASONED 从合约字节码中提取语义信息以构建语义关系图(SRG)，并采用自我反事实可解释检测器(SCFED)对SRGs进行分类并生成突出核心攻击逻辑的解释。SCFED 进一步通过从这些解释中提取代表性信息增强了稳健性、通用性和数据效率。理论分析和实验结果均证明了SEASONED的有效性，展示了其出色的检测性能、稳健性、通用性和高效的数据学习能力。为支持进一步研究，我们还发布了包含359个AEC的新数据集。 

---
# GraMFedDHAR: Graph Based Multimodal Differentially Private Federated HAR 

**Title (ZH)**: 基于图的多模态差异隐私联邦健康行为识别 

**Authors**: Labani Halder, Tanmay Sen, Sarbani Palit  

**Link**: [PDF](https://arxiv.org/pdf/2509.05671)  

**Abstract**: Human Activity Recognition (HAR) using multimodal sensor data remains challenging due to noisy or incomplete measurements, scarcity of labeled examples, and privacy concerns. Traditional centralized deep learning approaches are often constrained by infrastructure availability, network latency, and data sharing restrictions. While federated learning (FL) addresses privacy by training models locally and sharing only model parameters, it still has to tackle issues arising from the use of heterogeneous multimodal data and differential privacy requirements. In this article, a Graph-based Multimodal Federated Learning framework, GraMFedDHAR, is proposed for HAR tasks. Diverse sensor streams such as a pressure mat, depth camera, and multiple accelerometers are modeled as modality-specific graphs, processed through residual Graph Convolutional Neural Networks (GCNs), and fused via attention-based weighting rather than simple concatenation. The fused embeddings enable robust activity classification, while differential privacy safeguards data during federated aggregation. Experimental results show that the proposed MultiModalGCN model outperforms the baseline MultiModalFFN, with up to 2 percent higher accuracy in non-DP settings in both centralized and federated paradigms. More importantly, significant improvements are observed under differential privacy constraints: MultiModalGCN consistently surpasses MultiModalFFN, with performance gaps ranging from 7 to 13 percent depending on the privacy budget and setting. These results highlight the robustness of graph-based modeling in multimodal learning, where GNNs prove more resilient to the performance degradation introduced by DP noise. 

**Abstract (ZH)**: 基于图的多模态联邦学习框架GraMFedDHAR用于人体活动识别 

---
# Llama-GENBA-10B: A Trilingual Large Language Model for German, English and Bavarian 

**Title (ZH)**: Llama-GENBA-10B：一种用于德语、英语和巴伐利亚语的三语大型语言模型 

**Authors**: Michael Hoffmann, Jophin John, Stefan Schweter, Gokul Ramakrishnan, Hoi-Fong Mak, Alice Zhang, Dmitry Gaynullin, Nicolay J. Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2509.05668)  

**Abstract**: We present Llama-GENBA-10B, a trilingual foundation model addressing English-centric bias in large language models. Built on Llama 3.1-8B and scaled to 10B parameters, Llama-GENBA-10B is continuously pretrained on 164B tokens (82B English, 82B German, and 80M Bavarian), balancing resources while preventing English dominance. Targeted at the German NLP community, the model also promotes Bavarian as a low-resource language. Development tackled four challenges: (1) curating a multilingual corpus despite Bavarian scarcity, (2) creating a unified tokenizer for English, German, and Bavarian, (3) optimizing architecture and language-ratio hyperparameters for cross-lingual transfer, and (4) establishing the first standardized trilingual evaluation suite by translating German benchmarks into Bavarian. Evaluations show that Llama-GENBA-10B achieves strong cross-lingual performance, with the fine-tuned variant surpassing Apertus-8B-2509 and gemma-2-9b in Bavarian and establishing itself as the best model in its class for this language, while also outperforming EuroLLM in English and matching its results in German. Training on the Cerebras CS-2 demonstrated efficient large-scale multilingual pretraining with documented energy use, offering a blueprint for inclusive foundation models that integrate low-resource languages. 

**Abstract (ZH)**: 我们提出Llama-GENBA-10B，这是一种针对大型语言模型中以英语为中心偏见的三语基础模型。 

---
# LM-Searcher: Cross-domain Neural Architecture Search with LLMs via Unified Numerical Encoding 

**Title (ZH)**: LM-Searcher：通过统一数值编码利用LLM进行跨域神经架构搜索 

**Authors**: Yuxuan Hu, Jihao Liu, Ke Wang, Jinliang Zhen, Weikang Shi, Manyuan Zhang, Qi Dou, Rui Liu, Aojun Zhou, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.05657)  

**Abstract**: Recent progress in Large Language Models (LLMs) has opened new avenues for solving complex optimization problems, including Neural Architecture Search (NAS). However, existing LLM-driven NAS approaches rely heavily on prompt engineering and domain-specific tuning, limiting their practicality and scalability across diverse tasks. In this work, we propose LM-Searcher, a novel framework that leverages LLMs for cross-domain neural architecture optimization without the need for extensive domain-specific adaptation. Central to our approach is NCode, a universal numerical string representation for neural architectures, which enables cross-domain architecture encoding and search. We also reformulate the NAS problem as a ranking task, training LLMs to select high-performing architectures from candidate pools using instruction-tuning samples derived from a novel pruning-based subspace sampling strategy. Our curated dataset, encompassing a wide range of architecture-performance pairs, encourages robust and transferable learning. Comprehensive experiments demonstrate that LM-Searcher achieves competitive performance in both in-domain (e.g., CNNs for image classification) and out-of-domain (e.g., LoRA configurations for segmentation and generation) tasks, establishing a new paradigm for flexible and generalizable LLM-based architecture search. The datasets and models will be released at this https URL. 

**Abstract (ZH)**: Recent progress in大型语言模型（LLMs）为解决复杂优化问题开辟了新途径，包括神经架构搜索（NAS）。然而，现有的LLM驱动的NAS方法高度依赖提示工程和领域特定调整，限制了其在多样化任务中的实用性和可扩展性。在本工作中，我们提出LM-Searcher，这是一种新的框架，利用LLMs进行跨领域的神经架构优化，而不需进行大量领域特定的适配。我们方法的核心是NCode，一种通用的神经架构数值字符串表示法，使跨领域的架构编码和搜索成为可能。我们还将NAS问题重新表述为排序任务，通过源自新型基于剪枝的子空间采样策略的指令调优样本训练LLMs，从候选池中选择高性能的架构。我们精心构建的数据集涵盖了广泛的架构-性能对，促进了稳健且可迁移的学习。综合实验表明，LM-Searcher在领域内（如用于图像分类的CNN）和跨领域（如用于分割和生成任务的LoRA配置）任务中均能达到竞争力的性能，确立了基于LLM的架构搜索的灵活且通用的新范式。相关数据集和模型将在此网站发布。 

---
# OptiProxy-NAS: Optimization Proxy based End-to-End Neural Architecture Search 

**Title (ZH)**: OptiProxy-NAS: 基于优化代理的端到端神经架构搜索 

**Authors**: Bo Lyu, Yu Cui, Tuo Shi, Ke Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.05656)  

**Abstract**: Neural architecture search (NAS) is a hard computationally expensive optimization problem with a discrete, vast, and spiky search space. One of the key research efforts dedicated to this space focuses on accelerating NAS via certain proxy evaluations of neural architectures. Different from the prevalent predictor-based methods using surrogate models and differentiable architecture search via supernetworks, we propose an optimization proxy to streamline the NAS as an end-to-end optimization framework, named OptiProxy-NAS. In particular, using a proxy representation, the NAS space is reformulated to be continuous, differentiable, and smooth. Thereby, any differentiable optimization method can be applied to the gradient-based search of the relaxed architecture parameters. Our comprehensive experiments on $12$ NAS tasks of $4$ search spaces across three different domains including computer vision, natural language processing, and resource-constrained NAS fully demonstrate the superior search results and efficiency. Further experiments on low-fidelity scenarios verify the flexibility. 

**Abstract (ZH)**: 神经架构搜索（NAS）是一个计算上昂贵的离散、庞大且不连续的优化问题。致力于这一空间的关键研究工作之一是通过某些代理评估加速NAS。不同于基于预测器的方法使用替代模型和通过超网络进行可微分的架构搜索，我们提出了一种优化代理，将其命名为OptiProxy-NAS，以将NAS直接作为端到端的优化框架进行优化。特别是，使用代理表示，NAS空间被重新表述为连续、可微分且平滑的。因此，任何可微分的优化方法都可以应用于对放松的架构参数的梯度搜索。在三个不同领域（计算机视觉、自然语言处理和资源受限的NAS）的四个搜索空间下的12个NAS任务上的全面实验充分证明了其优越的搜索结果和效率。进一步在低保真场景下的实验验证了其灵活性。 

---
# Orchestrator: Active Inference for Multi-Agent Systems in Long-Horizon Tasks 

**Title (ZH)**: orchestrator: 长时_horizon 任务中多智能体系统的主动推断 

**Authors**: Lukas Beckenbauer, Johannes-Lucas Loewe, Ge Zheng, Alexandra Brintrup  

**Link**: [PDF](https://arxiv.org/pdf/2509.05651)  

**Abstract**: Complex, non-linear tasks challenge LLM-enhanced multi-agent systems (MAS) due to partial observability and suboptimal coordination. We propose Orchestrator, a novel MAS framework that leverages attention-inspired self-emergent coordination and reflective benchmarking to optimize global task performance. Orchestrator introduces a monitoring mechanism to track agent-environment dynamics, using active inference benchmarks to optimize system behavior. By tracking agent-to-agent and agent-to-environment interaction, Orchestrator mitigates the effects of partial observability and enables agents to approximate global task solutions more efficiently. We evaluate the framework on a series of maze puzzles of increasing complexity, demonstrating its effectiveness in enhancing coordination and performance in dynamic, non-linear environments with long-horizon objectives. 

**Abstract (ZH)**: 复杂的非线性任务挑战了基于LLM增强的多智能体系统（MAS）的能力，由于半可观测性和次优协调。我们提出了一种名为Orchestrator的新型MAS框架，该框架利用注意力启发式的自我涌现协调和反思型基准测试来优化全局任务性能。Orchestrator引入了一种监控机制来跟踪智能体-环境动态，并使用主动推断基准测试来优化系统行为。通过跟踪智能体间的交互和智能体与环境的交互，Orchestrator缓解了半可观测性的影响，使智能体能够更高效地逼近全局任务解决方案。我们在一系列复杂度递增的迷宫谜题上评估了该框架，证明了其在具有长期目标的动态、非线性环境中的协调性和性能提升效果。 

---
# Self-supervised Learning for Hyperspectral Images of Trees 

**Title (ZH)**: 树冠_hyper spectral图像的自监督学习 

**Authors**: Moqsadur Rahman, Saurav Kumar, Santosh S. Palmate, M. Shahriar Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2509.05630)  

**Abstract**: Aerial remote sensing using multispectral and RGB imagers has provided a critical impetus to precision agriculture. Analysis of the hyperspectral images with limited or no labels is challenging. This paper focuses on self-supervised learning to create neural network embeddings reflecting vegetation properties of trees from aerial hyperspectral images of crop fields. Experimental results demonstrate that a constructed tree representation, using a vegetation property-related embedding space, performs better in downstream machine learning tasks compared to the direct use of hyperspectral vegetation properties as tree representations. 

**Abstract (ZH)**: 基于多光谱和RGB成像的航空遥感为精准农业提供了关键动力。分析标注有限或无标注的高光谱图像具有挑战性。本文专注于自监督学习以从农田的航空高光谱图像中创建反映树木植被属性的神经网络嵌入。实验结果表明，使用与植被属性相关的嵌入空间构建的树木表示，在下游机器学习任务中表现优于直接使用高光谱植被属性作为树木表示。 

---
# From Joy to Fear: A Benchmark of Emotion Estimation in Pop Song Lyrics 

**Title (ZH)**: 从喜悦到恐惧：流行歌曲歌词情感估计基准 

**Authors**: Shay Dahary, Avi Edana, Alexander Apartsin, Yehudit Aperstein  

**Link**: [PDF](https://arxiv.org/pdf/2509.05617)  

**Abstract**: The emotional content of song lyrics plays a pivotal role in shaping listener experiences and influencing musical preferences. This paper investigates the task of multi-label emotional attribution of song lyrics by predicting six emotional intensity scores corresponding to six fundamental emotions. A manually labeled dataset is constructed using a mean opinion score (MOS) approach, which aggregates annotations from multiple human raters to ensure reliable ground-truth labels. Leveraging this dataset, we conduct a comprehensive evaluation of several publicly available large language models (LLMs) under zero-shot scenarios. Additionally, we fine-tune a BERT-based model specifically for predicting multi-label emotion scores. Experimental results reveal the relative strengths and limitations of zero-shot and fine-tuned models in capturing the nuanced emotional content of lyrics. Our findings highlight the potential of LLMs for emotion recognition in creative texts, providing insights into model selection strategies for emotion-based music information retrieval applications. The labeled dataset is available at this https URL. 

**Abstract (ZH)**: 歌曲歌词中的情感内容在塑造听众体验和影响音乐偏好方面发挥着关键作用。本文研究了通过预测六种基本情感对应的情感强度得分来进行歌词多标签情感归类的任务。采用平均意见分（MOS）方法构建了一个手动标注的数据集，该方法汇集了多名人工评分者的意见以确保可靠的真实标签。利用该数据集，在零样本场景下对几种大型语言模型（LLMs）进行了全面评估。此外，我们对一个基于BERT的模型进行了微调，以预测多标签情感得分。实验结果揭示了零样本和微调模型在捕捉歌词细腻情感内容方面相对的优势与限制。我们的研究突显了LLMs在创意文本中情感识别的潜在应用，并提供了情感音乐信息检索应用中模型选择策略的见解。标注数据集可通过以下链接访问：this https URL。 

---
# Causal Debiasing Medical Multimodal Representation Learning with Missing Modalities 

**Title (ZH)**: 因果去偏见医学多模态表示学习中的缺失模态处理 

**Authors**: Xiaoguang Zhu, Lianlong Sun, Yang Liu, Pengyi Jiang, Uma Srivatsa, Nipavan Chiamvimonvat, Vladimir Filkov  

**Link**: [PDF](https://arxiv.org/pdf/2509.05615)  

**Abstract**: Medical multimodal representation learning aims to integrate heterogeneous clinical data into unified patient representations to support predictive modeling, which remains an essential yet challenging task in the medical data mining community. However, real-world medical datasets often suffer from missing modalities due to cost, protocol, or patient-specific constraints. Existing methods primarily address this issue by learning from the available observations in either the raw data space or feature space, but typically neglect the underlying bias introduced by the data acquisition process itself. In this work, we identify two types of biases that hinder model generalization: missingness bias, which results from non-random patterns in modality availability, and distribution bias, which arises from latent confounders that influence both observed features and outcomes. To address these challenges, we perform a structural causal analysis of the data-generating process and propose a unified framework that is compatible with existing direct prediction-based multimodal learning methods. Our method consists of two key components: (1) a missingness deconfounding module that approximates causal intervention based on backdoor adjustment and (2) a dual-branch neural network that explicitly disentangles causal features from spurious correlations. We evaluated our method in real-world public and in-hospital datasets, demonstrating its effectiveness and causal insights. 

**Abstract (ZH)**: 医学多模态表示学习旨在将异质临床数据整合为统一的患者表示，以支持预测建模，这在医学数据挖掘社区中仍然是一个关键但具有挑战性的任务。然而，现实世界的医疗数据集往往由于成本、协议或患者特异性限制等原因而导致模态缺失。现有方法主要通过学习可用数据在原始数据空间或特征空间中的观测值来应对这一问题，但通常忽视了数据采集过程本身引入的潜在偏差。在本文中，我们识别出两种妨碍模型泛化的偏差：缺失性偏差，源于模态可用性的非随机模式；分布性偏差，源于潜在混杂因素对观察特征和结果的影响。为应对这些挑战，我们对数据生成过程进行了结构因果分析，并提出了一种与现有直接预测为基础的多模态学习方法兼容的统一框架。该方法包含两个关键组件：（1）一个缺失性去混杂模块，基于后门调整近似因果干预；（2）一个双分支神经网络，明确区分因果特征和伪相关。我们在实际公开和医院数据集上评估了该方法，展示了其有效性和因果洞察。 

---
# SpecPrune-VLA: Accelerating Vision-Language-Action Models via Action-Aware Self-Speculative Pruning 

**Title (ZH)**: SpecPrune-VLA: 通过行动感知自推测剪枝加速视觉-语言-行动模型 

**Authors**: Hanzhen Wang, Jiaming Xu, Jiayi Pan, Yongkang Zhou, Guohao Dai  

**Link**: [PDF](https://arxiv.org/pdf/2509.05614)  

**Abstract**: Pruning accelerates compute-bound models by reducing computation. Recently applied to Vision-Language-Action (VLA) models, existing methods prune tokens using only local info from current action, ignoring global context from prior actions, causing >20% success rate drop and limited speedup. We observe high similarity across consecutive actions and propose leveraging both local (current) and global (past) info for smarter token selection. We introduce SpecPrune-VLA, a training-free method with two-level pruning and heuristic control: (1) Static pruning at action level: uses global history and local context to reduce visual tokens per action; (2) Dynamic pruning at layer level: prunes tokens per layer based on layer-specific importance; (3) Lightweight action-aware controller: classifies actions as coarse/fine-grained (by speed), adjusting pruning aggressiveness since fine-grained actions are pruning-sensitive. Experiments on LIBERO show SpecPrune-VLA achieves 1.46 times speedup on NVIDIA A800 and 1.57 times on NVIDIA GeForce RTX 3090 vs. OpenVLA-OFT, with negligible success rate loss. 

**Abstract (ZH)**: Pruning加速计算约束模型通过减少计算量。我们提出SpecPrune-VLA：利用局部和全局信息进行智能_TOKEN_剪枝的训练-Free方法 

---
# Cross-Service Threat Intelligence in LLM Services using Privacy-Preserving Fingerprints 

**Title (ZH)**: 在LLM服务中使用隐私保护指纹进行跨服务威胁情报分享 

**Authors**: Waris Gill, Natalie Isak, Matthew Dressman  

**Link**: [PDF](https://arxiv.org/pdf/2509.05608)  

**Abstract**: The widespread deployment of LLMs across enterprise services has created a critical security blind spot. Organizations operate multiple LLM services handling billions of queries daily, yet regulatory compliance boundaries prevent these services from sharing threat intelligence about prompt injection attacks, the top security risk for LLMs. When an attack is detected in one service, the same threat may persist undetected in others for months, as privacy regulations prohibit sharing user prompts across compliance boundaries.
We present BinaryShield, the first privacy-preserving threat intelligence system that enables secure sharing of attack fingerprints across compliance boundaries. BinaryShield transforms suspicious prompts through a unique pipeline combining PII redaction, semantic embedding, binary quantization, and randomized response mechanism to potentially generate non-invertible fingerprints that preserve attack patterns while providing privacy. Our evaluations demonstrate that BinaryShield achieves an F1-score of 0.94, significantly outperforming SimHash (0.77), the privacy-preserving baseline, while achieving 64x storage reduction and 38x faster similarity search compared to dense embeddings. 

**Abstract (ZH)**: 企业服务中LLM的广泛部署创造了重要的安全盲区。组织运营着多个处理数十亿查询的LLM服务，但由于监管合规限制，这些服务无法共享关于提示注入攻击的威胁情报，这是LLMs最大的安全风险。当在一个服务中检测到攻击时，相同的威胁在其他服务中可能会持续数月未被检测到，因为隐私法规禁止跨合规边界共享用户提示。

我们提出了BinaryShield，这是一种首创的隐私保护威胁情报系统，能够安全地跨合规边界共享攻击指纹。BinaryShield通过一个独特的流水线，结合PII篡改、语义嵌入、二进制量化和随机响应机制，生成潜在不可逆的指纹，同时保留攻击模式并提供隐私保护。我们的评估结果显示，BinaryShield实现了0.94的F1分数，显著优于隐私保护基线SimHash（0.77），同时实现了64倍的存储减少和38倍更快的相似性搜索速度。 

---
# Icon$^{2}$: Aligning Large Language Models Using Self-Synthetic Preference Data via Inherent Regulation 

**Title (ZH)**: Icon$^{2}$: 通过内在调节使用自我合成偏好数据对大型语言模型进行对齐 

**Authors**: Qiyuan Chen, Hongsen Huang, Qian Shao, Jiahe Chen, Jintai Chen, Hongxia Xu, Renjie Hua, Ren Chuan, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05605)  

**Abstract**: Large Language Models (LLMs) require high quality preference datasets to align with human preferences. However, conventional methods for constructing such datasets face significant challenges: reliance on pre-collected instructions often leads to distribution mismatches with target models, while the need for sampling multiple stochastic responses introduces substantial computational overhead. In this work, we explore a paradigm shift by leveraging inherent regulation of LLMs' representation space for efficient and tailored preference dataset construction, named Icon$^{2}$. Specifically, it first extracts layer-wise direction vectors to encode sophisticated human preferences and then uses these vectors to filter self-synthesized instructions based on their inherent consistency. During decoding, bidirectional inherent control is applied to steer token representations, enabling the precise generation of response pairs with clear alignment distinctions. Experimental results demonstrate significant improvements in both alignment and efficiency. Llama3-8B and Qwen2-7B achieve an average win rate improvement of 13.89% on AlpacaEval 2.0 and 13.45% on Arena-Hard, while reducing computational costs by up to 48.1%. 

**Abstract (ZH)**: 大规模语言模型（LLMs）需要高质量的偏好数据集来与人类偏好对齐。然而，传统的方法在构建此类数据集时面临着重大挑战：依赖预先收集的指令往往会导致与目标模型的分布不匹配，而需要采样多个随机响应则引入了显著的计算开销。在本工作中，我们通过利用LLMs表示空间中的固有调节机制，探索了一种范式转变，称为Icon$^{2}$，以实现高效和定制化的偏好数据集构建。具体而言，它首先提取层wise方向向量以编码复杂的用户偏好，然后使用这些向量基于固有的一致性筛选自我合成的指令。在解码过程中，应用双向固有控制以引导标记表示，从而实现具有明确对齐差异的响应对的精确生成。实验结果表明，在对齐和效率方面均实现了显著改进。Llama3-8B和Qwen2-7B在AlpacaEval 2.0和Arena-Hard上的平均胜率分别提高了13.89%和13.45%，同时计算成本降低了48.1%。 

---
# Language-guided Recursive Spatiotemporal Graph Modeling for Video Summarization 

**Title (ZH)**: 基于语言引导的递归时空图建模的视频摘要 

**Authors**: Jungin Park, Jiyoung Lee, Kwanghoon Sohn  

**Link**: [PDF](https://arxiv.org/pdf/2509.05604)  

**Abstract**: Video summarization aims to select keyframes that are visually diverse and can represent the whole story of a given video. Previous approaches have focused on global interlinkability between frames in a video by temporal modeling. However, fine-grained visual entities, such as objects, are also highly related to the main content of the video. Moreover, language-guided video summarization, which has recently been studied, requires a comprehensive linguistic understanding of complex real-world videos. To consider how all the objects are semantically related to each other, this paper regards video summarization as a language-guided spatiotemporal graph modeling problem. We present recursive spatiotemporal graph networks, called VideoGraph, which formulate the objects and frames as nodes of the spatial and temporal graphs, respectively. The nodes in each graph are connected and aggregated with graph edges, representing the semantic relationships between the nodes. To prevent the edges from being configured with visual similarity, we incorporate language queries derived from the video into the graph node representations, enabling them to contain semantic knowledge. In addition, we adopt a recursive strategy to refine initial graphs and correctly classify each frame node as a keyframe. In our experiments, VideoGraph achieves state-of-the-art performance on several benchmarks for generic and query-focused video summarization in both supervised and unsupervised manners. The code is available at this https URL. 

**Abstract (ZH)**: 视频摘要旨在选择视觉上多样的关键帧，以代表给定视频的整个故事。先前的方法主要关注视频中帧之间的全局关联性，通过时间建模实现。然而，细粒度的视觉实体，如物体，也与视频的主要内容高度相关。此外，近年来研究的语言引导视频摘要需要对复杂的现实世界视频进行全面的语言理解。为了考虑所有物体之间的语义关联，本文将视频摘要视为一种语义引导的空间-temporal图建模问题。我们提出了递归空间-temporal图网络，称为VideoGraph，将物体和帧分别表示为空间和时间图的节点。每个图中的节点通过图边连接和聚合，表示节点之间的语义关系。为了防止边基于视觉相似性配置，我们通过将来自视频的语言查询纳入图节点表示中，使它们能够包含语义知识。此外，我们采用递归策略改进初始图，并正确分类每个帧节点为关键帧。在我们的实验中，VideoGraph在多种基准测试中实现了通用和查询导向视频摘要的最佳性能，无论是监督学习还是无监督学习。代码可在以下网址获取。 

---
# Natural Language-Programming Language Software Traceability Link Recovery Needs More than Textual Similarity 

**Title (ZH)**: 自然语言编程语言软件可追溯性链接的恢复需要超出文本相似性。 

**Authors**: Zhiyuan Zou, Bangchao Wang, Peng Liang, Tingting Bi, Huan Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.05585)  

**Abstract**: In the field of software traceability link recovery (TLR), textual similarity has long been regarded as the core criterion. However, in tasks involving natural language and programming language (NL-PL) artifacts, relying solely on textual similarity is limited by their semantic gap. To this end, we conducted a large-scale empirical evaluation across various types of TLR tasks, revealing the limitations of textual similarity in NL-PL scenarios. To address these limitations, we propose an approach that incorporates multiple domain-specific auxiliary strategies, identified through empirical analysis, into two models: the Heterogeneous Graph Transformer (HGT) via edge types and the prompt-based Gemini 2.5 Pro via additional input information. We then evaluated our approach using the widely studied requirements-to-code TLR task, a representative case of NL-PL TLR. Experimental results show that both the multi-strategy HGT and Gemini 2.5 Pro models outperformed their original counterparts without strategy integration. Furthermore, compared to the current state-of-the-art method HGNNLink, the multi-strategy HGT and Gemini 2.5 Pro models achieved average F1-score improvements of 3.68% and 8.84%, respectively, across twelve open-source projects, demonstrating the effectiveness of multi-strategy integration in enhancing overall model performance for the requirements-code TLR task. 

**Abstract (ZH)**: 在软件追溯链接恢复（TLR）领域，文本相似性长期以来一直被视为核心标准。但在涉及自然语言和编程语言（NL-PL）制品的任务中，仅依赖文本相似性受到语义差距的限制。为此，我们在多种类型的TLR任务中进行了大规模的实证评估，揭示了文本相似性在NL-PL情景下的局限性。为应对这些局限性，我们提出了一种方法，将通过实证分析识别出的多种领域特定辅助策略整合到两种模型中：通过边类型实现的异质图变换器（HGT）和基于提示的Gemini 2.5 Pro，后者通过附加输入信息。我们使用广泛研究的需求到代码的TLR任务对这种方法进行了评估，这是一个典型的NL-PL TLR案例。实验结果表明，多策略HGT和Gemini 2.5 Pro模型在没有策略整合的情况下均优于其原来的版本。此外，与当前最先进的方法HGNNLink相比，多策略HGT和Gemini 2.5 Pro模型在12个开源项目中分别实现了平均F1分数提升3.68%和8.84%，证实了多策略整合在提高需求到代码TLR任务的整体模型性能方面的有效性。 

---
# Learning to Walk in Costume: Adversarial Motion Priors for Aesthetically Constrained Humanoids 

**Title (ZH)**: 穿着 costumes 走路的学习：受美学约束的人形角色的对抗运动先验知识 

**Authors**: Arturo Flores Alvarez, Fatemeh Zargarbashi, Havel Liu, Shiqi Wang, Liam Edwards, Jessica Anz, Alex Xu, Fan Shi, Stelian Coros, Dennis W. Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.05581)  

**Abstract**: We present a Reinforcement Learning (RL)-based locomotion system for Cosmo, a custom-built humanoid robot designed for entertainment applications. Unlike traditional humanoids, entertainment robots present unique challenges due to aesthetic-driven design choices. Cosmo embodies these with a disproportionately large head (16% of total mass), limited sensing, and protective shells that considerably restrict movement. To address these challenges, we apply Adversarial Motion Priors (AMP) to enable the robot to learn natural-looking movements while maintaining physical stability. We develop tailored domain randomization techniques and specialized reward structures to ensure safe sim-to-real, protecting valuable hardware components during deployment. Our experiments demonstrate that AMP generates stable standing and walking behaviors despite Cosmo's extreme mass distribution and movement constraints. These results establish a promising direction for robots that balance aesthetic appeal with functional performance, suggesting that learning-based methods can effectively adapt to aesthetic-driven design constraints. 

**Abstract (ZH)**: 基于强化学习的Cosmo娱乐 humanoid 机器人运动系统：对抗运动先验在美学驱动设计下的运动学习与物理稳定性 

---
# Using Contrastive Learning to Improve Two-Way Reasoning in Large Language Models: The Obfuscation Task as a Case Study 

**Title (ZH)**: 使用对比学习提高大规模语言模型双向推理能力：混淆任务案例研究 

**Authors**: Serge Lionel Nikiema, Jordan Samhi, Micheline Bénédicte Moumoula, Albérick Euraste Djiré, Abdoul Kader Kaboré, Jacques Klein, Tegawendé F. Bissyandé  

**Link**: [PDF](https://arxiv.org/pdf/2509.05553)  

**Abstract**: This research addresses a fundamental question in AI: whether large language models truly understand concepts or simply recognize patterns. The authors propose bidirectional reasoning,the ability to apply transformations in both directions without being explicitly trained on the reverse direction, as a test for genuine understanding. They argue that true comprehension should naturally allow reversibility. For example, a model that can change a variable name like userIndex to i should also be able to infer that i represents a user index without reverse training. The researchers tested current language models and discovered what they term cognitive specialization: when models are fine-tuned on forward tasks, their performance on those tasks improves, but their ability to reason bidirectionally becomes significantly worse. To address this issue, they developed Contrastive Fine-Tuning (CFT), which trains models using three types of examples: positive examples that maintain semantic meaning, negative examples with different semantics, and forward-direction obfuscation examples. This approach aims to develop deeper understanding rather than surface-level pattern recognition and allows reverse capabilities to develop naturally without explicit reverse training. Their experiments demonstrated that CFT successfully achieved bidirectional reasoning, enabling strong reverse performance while maintaining forward task capabilities. The authors conclude that bidirectional reasoning serves both as a theoretical framework for assessing genuine understanding and as a practical training approach for developing more capable AI systems. 

**Abstract (ZH)**: 这项研究探讨了AI中的一个基础问题：大型语言模型是真正理解概念还是仅仅识别模式。作者提出双向推理能力——能够在不经过反向专门训练的情况下双向应用变换——作为真正理解的测试。他们认为真正的理解应该自然地具有可逆性。例如，能够将变量名userIndex改为i的模型，也应该能够推断i表示用户索引，而无需反向训练。研究人员测试了当前的语言模型，并发现他们所谓的认知专业化现象：当模型在正向任务上进行微调时，这些任务的性能会提高，但其双向推理能力会显著下降。为了解决这一问题，他们开发了对比微调（Contrastive Fine-Tuning, CFT），使用三类例子进行训练：保持语义意义的正向例子、具有不同语义的负向例子以及正向方向的混淆例子。这种方法旨在发展更深层次的理解，而非表面模式识别，并允许反向能力自然发展，而无需明确的反向训练。他们的实验表明，CFT成功实现了双向推理，在保持正向任务能力的同时，提高了反向任务的性能。作者得出结论，双向推理既作为一个评估真正理解的理论框架，也作为一个开发更强大AI系统的实际训练方法。 

---
# Combining TSL and LLM to Automate REST API Testing: A Comparative Study 

**Title (ZH)**: 结合TSL和LLM自动实现REST API测试：一项比较研究 

**Authors**: Thiago Barradas, Aline Paes, Vânia de Oliveira Neves  

**Link**: [PDF](https://arxiv.org/pdf/2509.05540)  

**Abstract**: The effective execution of tests for REST APIs remains a considerable challenge for development teams, driven by the inherent complexity of distributed systems, the multitude of possible scenarios, and the limited time available for test design. Exhaustive testing of all input combinations is impractical, often resulting in undetected failures, high manual effort, and limited test coverage. To address these issues, we introduce RestTSLLM, an approach that uses Test Specification Language (TSL) in conjunction with Large Language Models (LLMs) to automate the generation of test cases for REST APIs. The approach targets two core challenges: the creation of test scenarios and the definition of appropriate input data. The proposed solution integrates prompt engineering techniques with an automated pipeline to evaluate various LLMs on their ability to generate tests from OpenAPI specifications. The evaluation focused on metrics such as success rate, test coverage, and mutation score, enabling a systematic comparison of model performance. The results indicate that the best-performing LLMs - Claude 3.5 Sonnet (Anthropic), Deepseek R1 (Deepseek), Qwen 2.5 32b (Alibaba), and Sabia 3 (Maritaca) - consistently produced robust and contextually coherent REST API tests. Among them, Claude 3.5 Sonnet outperformed all other models across every metric, emerging in this study as the most suitable model for this task. These findings highlight the potential of LLMs to automate the generation of tests based on API specifications. 

**Abstract (ZH)**: 基于测试规范语言和大型语言模型的REST APIs自动测试用例生成方法 

---
# OpenEgo: A Large-Scale Multimodal Egocentric Dataset for Dexterous Manipulation 

**Title (ZH)**: OpenEgo: 一种大规模多模态第一人称数据集用于灵巧操作 

**Authors**: Ahad Jawaid, Yu Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05513)  

**Abstract**: Egocentric human videos provide scalable demonstrations for imitation learning, but existing corpora often lack either fine-grained, temporally localized action descriptions or dexterous hand annotations. We introduce OpenEgo, a multimodal egocentric manipulation dataset with standardized hand-pose annotations and intention-aligned action primitives. OpenEgo totals 1107 hours across six public datasets, covering 290 manipulation tasks in 600+ environments. We unify hand-pose layouts and provide descriptive, timestamped action primitives. To validate its utility, we train language-conditioned imitation-learning policies to predict dexterous hand trajectories. OpenEgo is designed to lower the barrier to learning dexterous manipulation from egocentric video and to support reproducible research in vision-language-action learning. All resources and instructions will be released at this http URL. 

**Abstract (ZH)**: 第一人称人类视频提供了模仿学习的可扩展示例，但现有数据集往往缺乏精细的时间局部化动作描述或灵巧的手部标注。我们介绍了OpenEgo，一个包含标准化手部姿态标注和意图对齐动作基元的多模态第一人称操作数据集。OpenEgo总计包含1107小时的数据，涵盖600多种环境中的290种操作任务。我们统一了手部姿态布局并提供了描述性的时间戳标注动作基元。为验证其有效性，我们训练了基于语言的模仿学习策略来预测灵巧的手部轨迹。OpenEgo旨在降低从第一人称视频学习灵巧操作的门槛，并支持视觉-语言-动作学习中的可重复研究。所有资源和说明将在以下网址发布。 

---
# Microrobot Vascular Parkour: Analytic Geometry-based Path Planning with Real-time Dynamic Obstacle Avoidance 

**Title (ZH)**: 微机器人血管越障：基于解析几何的路径规划与实时动态避障 

**Authors**: Yanda Yang, Max Sokolich, Fatma Ceren Kirmizitas, Sambeeta Das, Andreas A. Malikopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2509.05500)  

**Abstract**: Autonomous microrobots in blood vessels could enable minimally invasive therapies, but navigation is challenged by dense, moving obstacles. We propose a real-time path planning framework that couples an analytic geometry global planner (AGP) with two reactive local escape controllers, one based on rules and one based on reinforcement learning, to handle sudden moving obstacles. Using real-time imaging, the system estimates the positions of the microrobot, obstacles, and targets and computes collision-free motions. In simulation, AGP yields shorter paths and faster planning than weighted A* (WA*), particle swarm optimization (PSO), and rapidly exploring random trees (RRT), while maintaining feasibility and determinism. We extend AGP from 2D to 3D without loss of speed. In both simulations and experiments, the combined global planner and local controllers reliably avoid moving obstacles and reach targets. The average planning time is 40 ms per frame, compatible with 25 fps image acquisition and real-time closed-loop control. These results advance autonomous microrobot navigation and targeted drug delivery in vascular environments. 

**Abstract (ZH)**: 自主微机器人在血管中的导航可实现微创治疗，但面临密集移动障碍的挑战。我们提出了一种实时路径规划框架，将解析几何全局规划器（AGP）与基于规则和强化学习的两个反应式局部逃生控制器结合，以应对突发的移动障碍。利用实时成像，系统估计微型机器人、障碍物和目标的位置，并计算无碰撞运动。在模拟中，AGP在路径长度和规划速度上优于加权A*（WA*）、粒子 swarm 优化（PSO）和快速扩展随机树（RRT），同时保持可行性和确定性。我们将AGP从2D扩展到3D而不损失速度。在模拟和实验中，结合的全局规划器和局部控制器可靠地避免了移动障碍并到达目标。平均规划时间为每帧40毫秒，与25 fps图像采集和实时闭环控制兼容。这些结果推动了血管环境中自主微机器人导航和靶向药物输送的发展。 

---
# An Analysis of Layer-Freezing Strategies for Enhanced Transfer Learning in YOLO Architectures 

**Title (ZH)**: YOLO架构中增强迁移学习的层冻结策略分析 

**Authors**: Andrzej D. Dobrzycki, Ana M. Bernardos, José R. Casar  

**Link**: [PDF](https://arxiv.org/pdf/2509.05490)  

**Abstract**: The You Only Look Once (YOLO) architecture is crucial for real-time object detection. However, deploying it in resource-constrained environments such as unmanned aerial vehicles (UAVs) requires efficient transfer learning. Although layer freezing is a common technique, the specific impact of various freezing configurations on contemporary YOLOv8 and YOLOv10 architectures remains unexplored, particularly with regard to the interplay between freezing depth, dataset characteristics, and training dynamics. This research addresses this gap by presenting a detailed analysis of layer-freezing strategies. We systematically investigate multiple freezing configurations across YOLOv8 and YOLOv10 variants using four challenging datasets that represent critical infrastructure monitoring. Our methodology integrates a gradient behavior analysis (L2 norm) and visual explanations (Grad-CAM) to provide deeper insights into training dynamics under different freezing strategies. Our results reveal that there is no universal optimal freezing strategy but, rather, one that depends on the properties of the data. For example, freezing the backbone is effective for preserving general-purpose features, while a shallower freeze is better suited to handling extreme class imbalance. These configurations reduce graphics processing unit (GPU) memory consumption by up to 28% compared to full fine-tuning and, in some cases, achieve mean average precision (mAP@50) scores that surpass those of full fine-tuning. Gradient analysis corroborates these findings, showing distinct convergence patterns for moderately frozen models. Ultimately, this work provides empirical findings and practical guidelines for selecting freezing strategies. It offers a practical, evidence-based approach to balanced transfer learning for object detection in scenarios with limited resources. 

**Abstract (ZH)**: YOLO架构在资源受限环境下高效迁移学习的研究：冻结策略的详细分析与应用 

---
# MambaLite-Micro: Memory-Optimized Mamba Inference on MCUs 

**Title (ZH)**: MambaLite-Micro：MCUs上优化内存的Mamba推断 

**Authors**: Hongjun Xu, Junxi Xia, Weisi Yang, Yueyuan Sui, Stephen Xia  

**Link**: [PDF](https://arxiv.org/pdf/2509.05488)  

**Abstract**: Deploying Mamba models on microcontrollers (MCUs) remains challenging due to limited memory, the lack of native operator support, and the absence of embedded-friendly toolchains. We present, to our knowledge, the first deployment of a Mamba-based neural architecture on a resource-constrained MCU, a fully C-based runtime-free inference engine: MambaLite-Micro. Our pipeline maps a trained PyTorch Mamba model to on-device execution by (1) exporting model weights into a lightweight format, and (2) implementing a handcrafted Mamba layer and supporting operators in C with operator fusion and memory layout optimization. MambaLite-Micro eliminates large intermediate tensors, reducing 83.0% peak memory, while maintaining an average numerical error of only 1.7x10-5 relative to the PyTorch Mamba implementation. When evaluated on keyword spotting(KWS) and human activity recognition (HAR) tasks, MambaLite-Micro achieved 100% consistency with the PyTorch baselines, fully preserving classification accuracy. We further validated portability by deploying on both ESP32S3 and STM32H7 microcontrollers, demonstrating consistent operation across heterogeneous embedded platforms and paving the way for bringing advanced sequence models like Mamba to real-world resource-constrained applications. 

**Abstract (ZH)**: 将基于Mamba的神经架构部署在微控制器（MCUs）上仍然具有挑战性，受限于有限的内存、缺乏内置操作支持和缺失的嵌入式友好工具链。我们提出了迄今为止首个在资源受限的MCU上部署基于Mamba的神经架构的方法：一个完全基于C语言的无运行时推理引擎MambaLite-Micro。我们的流程通过（1）将模型权重导出为轻量级格式，（2）在C语言中手工实现Mamba层及其支持的操作，并通过操作融合和内存布局优化来实现。MambaLite-Micro消除了大量中间张量，峰值内存减少了83.0%，同时相对PyTorch Mamba实现的平均数值误差仅为1.7x10^-5。在关键词识别（KWS）和人体活动识别（HAR）任务上评估时，MambaLite-Micro与PyTorch基准保持了100%的一致性，完全保留了分类准确率。此外，我们进一步验证了其可移植性，分别部署在ESP32S3和STM32H7微控制器上，展示了其在异构嵌入式平台上的稳定运行，并为将如Mamba等高级序列模型引入实际资源受限的应用铺平了道路。 

---
# The Token Tax: Systematic Bias in Multilingual Tokenization 

**Title (ZH)**: 令牌税：多语言分词中的系统偏差 

**Authors**: Jessica M. Lundin, Ada Zhang, Nihal Karim, Hamza Louzan, Victor Wei, David Adelani, Cody Carroll  

**Link**: [PDF](https://arxiv.org/pdf/2509.05486)  

**Abstract**: Tokenization inefficiency imposes structural disadvantages on morphologically complex, low-resource languages, inflating compute resources and depressing accuracy. We evaluate 10 large language models (LLMs) on AfriMMLU (9,000 MCQA items; 5 subjects; 16 African languages) and show that fertility (tokens/word) reliably predicts accuracy. Higher fertility consistently predicts lower accuracy across all models and subjects. We further find that reasoning models (DeepSeek, o1) consistently outperform non-reasoning peers across high and low resource languages in the AfriMMLU dataset, narrowing accuracy gaps observed in prior generations. Finally, translating token inflation to economics, a doubling in tokens results in quadrupled training cost and time, underscoring the token tax faced by many languages. These results motivate morphologically aware tokenization, fair pricing, and multilingual benchmarks for equitable natural language processing (NLP). 

**Abstract (ZH)**: 分词效率低下对形态复杂、资源匮乏的语言造成结构上的不利影响，增加计算资源并降低准确性。我们在AfriMMLU（9000个多项选择题；5个科目；16种非洲语言）上评估了10种大规模语言模型（LLMs），并发现分词 fertility（每个词的分词数量）可靠地预测了准确性。更高的分词 fertility 在所有模型和科目中一致地预测了更低的准确性。此外，我们发现推理模型（DeepSeek, o1）在AfriMMLU数据集中的一系列高资源和低资源语言中始终优于非推理模型，缩小了前几代模型中观察到的准确性差距。最后，将分词膨胀转化为经济学概念，分词数量翻倍会导致训练成本和时间增加四倍，强调了许多语言所面临的分词税。这些结果促使我们关注形态学意识的分词、公平定价以及多语言基准测试，以实现公平的自然语言处理（NLP）。 

---
# PLanTS: Periodicity-aware Latent-state Representation Learning for Multivariate Time Series 

**Title (ZH)**: PLanTS: 具有期hythm意识的多变量时间序列潜在状态表示学习 

**Authors**: Jia Wang, Xiao Wang, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05478)  

**Abstract**: Multivariate time series (MTS) are ubiquitous in domains such as healthcare, climate science, and industrial monitoring, but their high dimensionality, limited labeled data, and non-stationary nature pose significant challenges for conventional machine learning methods. While recent self-supervised learning (SSL) approaches mitigate label scarcity by data augmentations or time point-based contrastive strategy, they neglect the intrinsic periodic structure of MTS and fail to capture the dynamic evolution of latent states. We propose PLanTS, a periodicity-aware self-supervised learning framework that explicitly models irregular latent states and their transitions. We first designed a period-aware multi-granularity patching mechanism and a generalized contrastive loss to preserve both instance-level and state-level similarities across multiple temporal resolutions. To further capture temporal dynamics, we design a next-transition prediction pretext task that encourages representations to encode predictive information about future state evolution. We evaluate PLanTS across a wide range of downstream tasks-including multi-class and multi-label classification, forecasting, trajectory tracking and anomaly detection. PLanTS consistently improves the representation quality over existing SSL methods and demonstrates superior runtime efficiency compared to DTW-based methods. 

**Abstract (ZH)**: 周期性意识自监督学习框架PLanTS： explicit建模不规则潜状态及其转换 

---
# Learning Tool-Aware Adaptive Compliant Control for Autonomous Regolith Excavation 

**Title (ZH)**: 学具自适应 compliant 控制技术以实现自主月壤挖掘 

**Authors**: Andrej Orsula, Matthieu Geist, Miguel Olivares-Mendez, Carol Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2509.05475)  

**Abstract**: Autonomous regolith excavation is a cornerstone of in-situ resource utilization for a sustained human presence beyond Earth. However, this task is fundamentally hindered by the complex interaction dynamics of granular media and the operational need for robots to use diverse tools. To address these challenges, this work introduces a framework where a model-based reinforcement learning agent learns within a parallelized simulation. This environment leverages high-fidelity particle physics and procedural generation to create a vast distribution of both lunar terrains and excavation tool geometries. To master this diversity, the agent learns an adaptive interaction strategy by dynamically modulating its own stiffness and damping at each control step through operational space control. Our experiments demonstrate that training with a procedural distribution of tools is critical for generalization and enables the development of sophisticated tool-aware behavior. Furthermore, we show that augmenting the agent with visual feedback significantly improves task success. These results represent a validated methodology for developing the robust and versatile autonomous systems required for the foundational tasks of future space missions. 

**Abstract (ZH)**: 自主月壤挖掘是月球地外长期驻留原位资源利用的关键基础。然而，这一任务从根本上受到粒状介质复杂相互作用动力学以及机器人需要使用多种工具的操作需求的限制。为应对这些挑战，本文提出了一种基于模型的强化学习框架，在并行仿真环境中进行学习。该环境利用高保真颗粒物理和程序化生成技术，创建了广泛的月壤地形和挖掘工具几何分布。为了掌握这种多样性，代理通过操作空间控制，在每个控制步骤中动态调节自身的刚度和阻尼来学习一种适应性的交互策略。我们的实验表明，使用工具的程序化分布进行训练对于泛化至关重要，并使开发出了复杂工具感知行为成为可能。此外，我们证明，将视觉反馈增强到代理中可以显著提高任务成功率。这些结果代表了一种验证过的开发未来太空任务基础任务所需的鲁棒且多功能自主系统的方法。 

---
# From Vision to Validation: A Theory- and Data-Driven Construction of a GCC-Specific AI Adoption Index 

**Title (ZH)**: 从视觉到验证：一种基于理论和数据的GCC特定AI adoption指数构建方法 

**Authors**: Mohammad Rashed Albous, Anwaar AlKandari, Abdel Latef Anouze  

**Link**: [PDF](https://arxiv.org/pdf/2509.05474)  

**Abstract**: Artificial intelligence (AI) is rapidly transforming public-sector processes worldwide, yet standardized measures rarely address the unique drivers, governance models, and cultural nuances of the Gulf Cooperation Council (GCC) countries. This study employs a theory-driven foundation derived from an in-depth analysis of literature review and six National AI Strategies (NASs), coupled with a data-driven approach that utilizes a survey of 203 mid- and senior-level government employees and advanced statistical techniques (K-Means clustering, Principal Component Analysis, and Partial Least Squares Structural Equation Modeling). By combining policy insights with empirical evidence, the research develops and validates a novel AI Adoption Index specifically tailored to the GCC public sector. Findings indicate that robust infrastructure and clear policy mandates exert the strongest influence on successful AI implementations, overshadowing organizational readiness in early adoption stages. The combined model explains 70% of the variance in AI outcomes, suggesting that resource-rich environments and top-down policy directives can drive rapid but uneven technology uptake. By consolidating key dimensions (Infrastructure & Resources, Organizational Readiness, and Policy & Regulatory Environment) into a single composite index, this study provides a holistic yet context-sensitive tool for benchmarking AI maturity. The index offers actionable guidance for policymakers seeking to harmonize large-scale deployments with ethical and regulatory standards. Beyond advancing academic discourse, these insights inform more strategic allocation of resources, cross-country cooperation, and capacity-building initiatives, thereby supporting sustained AI-driven transformation in the GCC region and beyond. 

**Abstract (ZH)**: 人工智能（AI）正快速改变全球公共部门的过程，然而标准化的衡量标准很少考虑海湾合作委员会（GCC）国家的独特驱动力、治理模式和文化差异。本研究基于深入的文献综述和六个国家人工智能战略（NASs）的理论驱动基础，并结合利用203名中高层政府员工的调查数据和高级统计技术（K-Means聚类、主成分分析和偏最小二乘结构方程建模）的数据驱动方法。通过将政策见解与实证证据相结合，该研究开发并验证了一个专门针对GCC公共部门的人工智能采用指数。研究发现，坚固的基础设施和明确的政策指令对成功的人工智能实施具有最强的影响，早期采用阶段组织准备度的影响较小。该综合模型解释了70%的人工智能结果的差异，表明资源丰富的环境和自上而下的政策指令可以推动快速但不均衡的技术采用。通过将关键维度（基础设施与资源、组织准备度和政策与监管环境）整合为单一复合指数，本研究提供了一个既全面又具有情境敏感性的工具，用于评估人工智能成熟度。该指数为寻求实现大规模部署与伦理和监管标准和谐一致的政策制定者提供了可操作的指导。这些见解不仅推动了学术讨论，还为跨国有序资源分配、合作及能力提升倡议提供了信息，从而支持GCC地区乃至更广泛的地区的人工智能驱动型持续转型。 

---
# Behind the Mask: Benchmarking Camouflaged Jailbreaks in Large Language Models 

**Title (ZH)**: Behind the Mask: 大型语言模型中隐匿式 Jailbreak 的基准测试 

**Authors**: Youjia Zheng, Mohammad Zandsalimy, Shanu Sushmita  

**Link**: [PDF](https://arxiv.org/pdf/2509.05471)  

**Abstract**: Large Language Models (LLMs) are increasingly vulnerable to a sophisticated form of adversarial prompting known as camouflaged jailbreaking. This method embeds malicious intent within seemingly benign language to evade existing safety mechanisms. Unlike overt attacks, these subtle prompts exploit contextual ambiguity and the flexible nature of language, posing significant challenges to current defense systems. This paper investigates the construction and impact of camouflaged jailbreak prompts, emphasizing their deceptive characteristics and the limitations of traditional keyword-based detection methods. We introduce a novel benchmark dataset, Camouflaged Jailbreak Prompts, containing 500 curated examples (400 harmful and 100 benign prompts) designed to rigorously stress-test LLM safety protocols. In addition, we propose a multi-faceted evaluation framework that measures harmfulness across seven dimensions: Safety Awareness, Technical Feasibility, Implementation Safeguards, Harmful Potential, Educational Value, Content Quality, and Compliance Score. Our findings reveal a stark contrast in LLM behavior: while models demonstrate high safety and content quality with benign inputs, they exhibit a significant decline in performance and safety when confronted with camouflaged jailbreak attempts. This disparity underscores a pervasive vulnerability, highlighting the urgent need for more nuanced and adaptive security strategies to ensure the responsible and robust deployment of LLMs in real-world applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越容易受到一种名为迷魂劫持的高级 adversarial prompting 影响。这种方法通过看似无害的语言嵌入恶意意图，以规避现有的安全机制。与显性的攻击不同，这些细微的提示利用了上下文的模糊性和语言的灵活性，对当前的防御系统构成重大挑战。本文探讨了迷魂劫持提示的构建及其影响，强调了其欺骗性特点和传统基于关键词的检测方法的局限性。我们引入了一个新的基准数据集，迷魂劫持提示集，包含500个精选示例（400个有害和100个无害提示），旨在严格测试LLM的安全协议。此外，我们提出了一种多维度评估框架，从七个维度衡量有害性：安全意识、技术可行性、实施保护措施、潜在危害性、教育价值、内容质量和合规评分。我们的研究发现，尽管模型在面对良性输入时表现出高度的安全性和内容质量，但在遭遇迷魂劫持尝试时，其性能和安全性显著下降。这一差异突显了LLM普遍存在的一种脆弱性，强调了迫切需要更细腻和适应性强的安全策略，以确保LLM在实际应用中的负责任和稳健部署。 

---
# Neural Breadcrumbs: Membership Inference Attacks on LLMs Through Hidden State and Attention Pattern Analysis 

**Title (ZH)**: 神经面包屑：通过隐藏状态和注意力模式分析对大模型进行成员推理攻击 

**Authors**: Disha Makhija, Manoj Ghuhan Arivazhagan, Vinayshekhar Bannihatti Kumar, Rashmi Gangadharaiah  

**Link**: [PDF](https://arxiv.org/pdf/2509.05449)  

**Abstract**: Membership inference attacks (MIAs) reveal whether specific data was used to train machine learning models, serving as important tools for privacy auditing and compliance assessment. Recent studies have reported that MIAs perform only marginally better than random guessing against large language models, suggesting that modern pre-training approaches with massive datasets may be free from privacy leakage risks. Our work offers a complementary perspective to these findings by exploring how examining LLMs' internal representations, rather than just their outputs, may provide additional insights into potential membership inference signals. Our framework, \emph{memTrace}, follows what we call \enquote{neural breadcrumbs} extracting informative signals from transformer hidden states and attention patterns as they process candidate sequences. By analyzing layer-wise representation dynamics, attention distribution characteristics, and cross-layer transition patterns, we detect potential memorization fingerprints that traditional loss-based approaches may not capture. This approach yields strong membership detection across several model families achieving average AUC scores of 0.85 on popular MIA benchmarks. Our findings suggest that internal model behaviors can reveal aspects of training data exposure even when output-based signals appear protected, highlighting the need for further research into membership privacy and the development of more robust privacy-preserving training techniques for large language models. 

**Abstract (ZH)**: 基于会员推理攻击的内部表示分析：探索大型语言模型潜在的会员推理信号 

---
# Newton to Einstein: Axiom-Based Discovery via Game Design 

**Title (ZH)**: 牛顿到爱因斯坦：基于公理的游戏设计驱动发现 

**Authors**: Pingchuan Ma, Benjamin Tod Jones, Tsun-Hsuan Wang, Minghao Guo, Michal Piotr Lipiec, Chuang Gan, Wojciech Matusik  

**Link**: [PDF](https://arxiv.org/pdf/2509.05448)  

**Abstract**: This position paper argues that machine learning for scientific discovery should shift from inductive pattern recognition to axiom-based reasoning. We propose a game design framework in which scientific inquiry is recast as a rule-evolving system: agents operate within environments governed by axioms and modify them to explain outlier observations. Unlike conventional ML approaches that operate within fixed assumptions, our method enables the discovery of new theoretical structures through systematic rule adaptation. We demonstrate the feasibility of this approach through preliminary experiments in logic-based games, showing that agents can evolve axioms that solve previously unsolvable problems. This framework offers a foundation for building machine learning systems capable of creative, interpretable, and theory-driven discovery. 

**Abstract (ZH)**: 机器学习在科学研究中的应用应从归纳模式识别转向基于公理的推理：一种基于规则演化系统的游戏设计框架 

---
# Direct-Scoring NLG Evaluators Can Use Pairwise Comparisons Too 

**Title (ZH)**: 直接评分的自然语言生成评估器也可以使用成对比较 

**Authors**: Logan Lawrence, Ashton Williamson, Alexander Shelton  

**Link**: [PDF](https://arxiv.org/pdf/2509.05440)  

**Abstract**: As large-language models have been increasingly used as automatic raters for evaluating free-form content, including document summarization, dialog, and story generation, work has been dedicated to evaluating such models by measuring their correlations with human judgment. For \textit{sample-level} performance, methods which operate by using pairwise comparisons between machine-generated text perform well but often lack the ability to assign absolute scores to individual summaries, an ability crucial for use cases that require thresholding. In this work, we propose a direct-scoring method which uses synthetic summaries to act as pairwise machine rankings at test time. We show that our method performs comparably to state-of-the-art pairwise evaluators in terms of axis-averaged sample-level correlations on the SummEval (\textbf{+0.03}), TopicalChat (\textbf{-0.03}), and HANNA (\textbf{+0.05}) meta-evaluation benchmarks, and release the synthetic in-context summaries as data to facilitate future work. 

**Abstract (ZH)**: 随着大型语言模型在自动评估自由形式内容（包括文档摘要、对话和故事情节生成）方面被越来越广泛应用，已有工作致力于通过衡量这些模型与人类判断的相关性来评估这些模型。对于样本级性能，依赖成对比较机器生成文本的方法表现良好，但往往缺乏为单个摘要分配绝对分数的能力，这是需要阈值要求的应用场景的关键。在本工作中，我们提出了一种直接评分方法，在测试时使用合成摘要作为成对机器排名。我们的方法在SummEval（+0.03）、TopicalChat（-0.03）和HANNA（+0.05）元评估基准中，在轴平均样本级相关性方面与最新的成对评估器表现相当，并发布了合成上下文摘要数据以促进未来研究。 

---
# Advanced Brain Tumor Segmentation Using EMCAD: Efficient Multi-scale Convolutional Attention Decoding 

**Title (ZH)**: 使用EMCAD的高效多尺度卷积注意力解码高级脑肿瘤分割 

**Authors**: GodsGift Uzor, Tania-Amanda Nkoyo Fredrick Eneye, Chukwuebuka Ijezue  

**Link**: [PDF](https://arxiv.org/pdf/2509.05431)  

**Abstract**: Brain tumor segmentation is a critical pre-processing step in the medical image analysis pipeline that involves precise delineation of tumor regions from healthy brain tissue in medical imaging data, particularly MRI scans. An efficient and effective decoding mechanism is crucial in brain tumor segmentation especially in scenarios with limited computational resources. However these decoding mechanisms usually come with high computational costs. To address this concern EMCAD a new efficient multi-scale convolutional attention decoder designed was utilized to optimize both performance and computational efficiency for brain tumor segmentation on the BraTs2020 dataset consisting of MRI scans from 369 brain tumor patients. The preliminary result obtained by the model achieved a best Dice score of 0.31 and maintained a stable mean Dice score of 0.285 plus/minus 0.015 throughout the training process which is moderate. The initial model maintained consistent performance across the validation set without showing signs of over-fitting. 

**Abstract (ZH)**: 脑肿瘤分割是医疗图像分析管道中的一个关键预处理步骤，涉及精确界定肿瘤区域与健康脑组织的边界，特别是在MRI扫描等医学影像数据中。高效的解码机制在资源有限的场景下尤为关键，但这些机制通常伴随着高昂的计算成本。为了应对这一挑战，EMCAD一种新的高效多尺度卷积注意力解码器被用于优化Brats2020数据集（包含369例脑肿瘤患者MRI扫描）上的脑肿瘤分割性能和计算效率。模型的初步结果中，最高Dice评分为0.31，训练过程中保持了稳定的平均Dice得分0.285±0.015，效果适中。初始模型在验证集上保持了稳定的性能，没有表现出过拟合的迹象。 

---
# No Translation Needed: Forecasting Quality from Fertility and Metadata 

**Title (ZH)**: 从生育率和元数据预测质量 

**Authors**: Jessica M. Lundin, Ada Zhang, David Adelani, Cody Carroll  

**Link**: [PDF](https://arxiv.org/pdf/2509.05425)  

**Abstract**: We show that translation quality can be predicted with surprising accuracy \textit{without ever running the translation system itself}. Using only a handful of features, token fertility ratios, token counts, and basic linguistic metadata (language family, script, and region), we can forecast ChrF scores for GPT-4o translations across 203 languages in the FLORES-200 benchmark. Gradient boosting models achieve favorable performance ($R^{2}=0.66$ for XX$\rightarrow$English and $R^{2}=0.72$ for English$\rightarrow$XX). Feature importance analyses reveal that typological factors dominate predictions into English, while fertility plays a larger role for translations into diverse target languages. These findings suggest that translation quality is shaped by both token-level fertility and broader linguistic typology, offering new insights for multilingual evaluation and quality estimation. 

**Abstract (ZH)**: 我们展示了在从未运行翻译系统的情况下，翻译质量可以用惊人的准确性进行预测。仅使用少量特征，如标记生育率、标记计数以及基本的语言元数据（语言家族、书写系统和地区），我们可以在FLORES-200基准中的203种语言中预测GPT-4o翻译的ChrF评分。梯度提升模型在XX到英语的预测中表现出色（$R^{2}=0.66$），在英语到XX的预测中表现更好（$R^{2}=0.72$）。特征重要性分析表明，类型学因素在翻译成英语时占主导地位，而生育率对多种目标语言的翻译起着更大的作用。这些发现表明，翻译质量受到标记水平生育率和更广泛语言类型学的影响，为多语言评估和质量估计提供了新的见解。 

---
# Universality of physical neural networks with multivariate nonlinearity 

**Title (ZH)**: 多元非线性下的物理神经网络普遍性 

**Authors**: Benjamin Savinson, David J. Norris, Siddhartha Mishra, Samuel Lanthaler  

**Link**: [PDF](https://arxiv.org/pdf/2509.05420)  

**Abstract**: The enormous energy demand of artificial intelligence is driving the development of alternative hardware for deep learning. Physical neural networks try to exploit physical systems to perform machine learning more efficiently. In particular, optical systems can calculate with light using negligible energy. While their computational capabilities were long limited by the linearity of optical materials, nonlinear computations have recently been demonstrated through modified input encoding. Despite this breakthrough, our inability to determine if physical neural networks can learn arbitrary relationships between data -- a key requirement for deep learning known as universality -- hinders further progress. Here we present a fundamental theorem that establishes a universality condition for physical neural networks. It provides a powerful mathematical criterion that imposes device constraints, detailing how inputs should be encoded in the tunable parameters of the physical system. Based on this result, we propose a scalable architecture using free-space optics that is provably universal and achieves high accuracy on image classification tasks. Further, by combining the theorem with temporal multiplexing, we present a route to potentially huge effective system sizes in highly practical but poorly scalable on-chip photonic devices. Our theorem and scaling methods apply beyond optical systems and inform the design of a wide class of universal, energy-efficient physical neural networks, justifying further efforts in their development. 

**Abstract (ZH)**: 物理神经网络的通用性条件及其实现方法 

---
# Graph Connectionist Temporal Classification for Phoneme Recognition 

**Title (ZH)**: 图连接主义时序分类在音素识别中的应用 

**Authors**: Henry Grafé, Hugo Van hamme  

**Link**: [PDF](https://arxiv.org/pdf/2509.05399)  

**Abstract**: Automatic Phoneme Recognition (APR) systems are often trained using pseudo phoneme-level annotations generated from text through Grapheme-to-Phoneme (G2P) systems. These G2P systems frequently output multiple possible pronunciations per word, but the standard Connectionist Temporal Classification (CTC) loss cannot account for such ambiguity during training. In this work, we adapt Graph Temporal Classification (GTC) to the APR setting. GTC enables training from a graph of alternative phoneme sequences, allowing the model to consider multiple pronunciations per word as valid supervision. Our experiments on English and Dutch data sets show that incorporating multiple pronunciations per word into the training loss consistently improves phoneme error rates compared to a baseline trained with CTC. These results suggest that integrating pronunciation variation into the loss function is a promising strategy for training APR systems from noisy G2P-based supervision. 

**Abstract (ZH)**: 自动音位识别（APR）系统常常使用通过图形到音位（G2P）系统从文本生成的伪音位级注释进行训练。这些G2P系统经常为每个单词输出多个可能的发音，但标准的连接主义时序分类（CTC）损失在训练过程中无法处理这种不确定性。在本工作中，我们将图形时序分类（GTC）应用于APR设置。GTC允许从多个音位序列的图形中进行训练，使模型能够将每个单词的多种发音视为有效的监督。我们的实验结果显示，在英语和荷兰数据集上，将每个单词的多种发音纳入训练损失中，相对于使用CTC训练的基准模型，可以一致地提高音位错误率。这些结果表明，在损失函数中整合发音变异是有希望的策略，用于从嘈杂的G2P基础监督中训练APR系统。 

---
# Talk Isn't Always Cheap: Understanding Failure Modes in Multi-Agent Debate 

**Title (ZH)**: 谈不一定免费：理解多agent辩论中的故障模式 

**Authors**: Andrea Wynn, Harsh Satija, Gillian Hadfield  

**Link**: [PDF](https://arxiv.org/pdf/2509.05396)  

**Abstract**: While multi-agent debate has been proposed as a promising strategy for improving AI reasoning ability, we find that debate can sometimes be harmful rather than helpful. The prior work has exclusively focused on debates within homogeneous groups of agents, whereas we explore how diversity in model capabilities influences the dynamics and outcomes of multi-agent interactions. Through a series of experiments, we demonstrate that debate can lead to a decrease in accuracy over time -- even in settings where stronger (i.e., more capable) models outnumber their weaker counterparts. Our analysis reveals that models frequently shift from correct to incorrect answers in response to peer reasoning, favoring agreement over challenging flawed reasoning. These results highlight important failure modes in the exchange of reasons during multi-agent debate, suggesting that naive applications of debate may cause performance degradation when agents are neither incentivized nor adequately equipped to resist persuasive but incorrect reasoning. 

**Abstract (ZH)**: 多智能体辩论虽然被提出作为一种提高AI推理能力的有前景策略，但我们发现辩论有时可能是有害的而非有益的。先前的工作仅专注于同质智能体群体内的辩论，而我们探讨了模型能力多样性的存在如何影响多智能体互动的动力学和结果。通过一系列实验，我们证明了即使在更强（即更具备能力）的模型多于较弱模型的情况下，辩论也会导致准确性的下降。我们的分析表明，模型经常因同伴推理而从正确答案变为错误答案，偏好一致而非挑战错误推理。这些结果强调了多智能体辩论期间理由交换中的重要失灵模式，表明在代理缺乏适当激励或充分准备以抗拒有 persuasiveness但错误的推理时，简单的辩论应用可能导致性能下降。 

---
# Reverse Browser: Vector-Image-to-Code Generator 

**Title (ZH)**: 反向浏览器：向量图像到代码生成器 

**Authors**: Zoltan Toth-Czifra  

**Link**: [PDF](https://arxiv.org/pdf/2509.05394)  

**Abstract**: Automating the conversion of user interface design into code (image-to-code or image-to-UI) is an active area of software engineering research. However, the state-of-the-art solutions do not achieve high fidelity to the original design, as evidenced by benchmarks. In this work, I approach the problem differently: I use vector images instead of bitmaps as model input. I create several large datasets for training machine learning models. I evaluate the available array of Image Quality Assessment (IQA) algorithms and introduce a new, multi-scale metric. I then train a large open-weights model and discuss its limitations. 

**Abstract (ZH)**: 自动化用户界面设计到代码的转换（图像到代码或图像到UI）是软件工程研究的一个活跃领域。然而，现有解决方案在保留原始设计的保真度方面不尽如人意，这在基准测试中已有体现。在此工作中，我从不同的角度解决问题：使用矢量图像而不是位图作为模型输入。我创建了多个大型数据集以训练机器学习模型。我评估了现有的多种图像质量评估（IQA）算法，并引入了一个新的多尺度度量标准。随后，我训练了一个大型开放权重模型，并讨论了其局限性。 

---
# Inferring Prerequisite Knowledge Concepts in Educational Knowledge Graphs: A Multi-criteria Approach 

**Title (ZH)**: 教育知识图谱中先决知识概念的推断：一种多准则方法 

**Authors**: Rawaa Alatrash, Mohamed Amine Chatti, Nasha Wibowo, Qurat Ul Ain  

**Link**: [PDF](https://arxiv.org/pdf/2509.05393)  

**Abstract**: Educational Knowledge Graphs (EduKGs) organize various learning entities and their relationships to support structured and adaptive learning. Prerequisite relationships (PRs) are critical in EduKGs for defining the logical order in which concepts should be learned. However, the current EduKG in the MOOC platform CourseMapper lacks explicit PR links, and manually annotating them is time-consuming and inconsistent. To address this, we propose an unsupervised method for automatically inferring concept PRs without relying on labeled data. We define ten criteria based on document-based, Wikipedia hyperlink-based, graph-based, and text-based features, and combine them using a voting algorithm to robustly capture PRs in educational content. Experiments on benchmark datasets show that our approach achieves higher precision than existing methods while maintaining scalability and adaptability, thus providing reliable support for sequence-aware learning in CourseMapper. 

**Abstract (ZH)**: 教育知识图谱（EduKGs）组织各种学习实体及其关系以支持结构化和适应性学习。先决条件关系（PRs）在EduKGs中对于定义概念的学习逻辑顺序至关重要。然而，MOOC平台CourseMapper中的当前EduKG缺乏明确的PR链接，手动标注它们耗费时间且不一致。为解决这一问题，我们提出了一种无需依赖标注数据的无监督方法，以自动推断概念的PRs。我们基于文档、Wikipedia超链接、图和文本定义了十项标准，并借助投票算法将这些标准结合，以稳健地捕获教育内容中的PRs。实验表明，我们的方法在基准数据集上实现了更高的精度，同时保持了可扩展性和适应性，从而为CourseMapper中的序列感知学习提供了可靠的支撑。 

---
# An Optimized Pipeline for Automatic Educational Knowledge Graph Construction 

**Title (ZH)**: 一种自动教育知识图谱构建的优化管道 

**Authors**: Qurat Ul Ain, Mohamed Amine Chatti, Jean Qussa, Amr Shakhshir, Rawaa Alatrash, Shoeb Joarder  

**Link**: [PDF](https://arxiv.org/pdf/2509.05392)  

**Abstract**: The automatic construction of Educational Knowledge Graphs (EduKGs) is essential for domain knowledge modeling by extracting meaningful representations from learning materials. Despite growing interest, identifying a scalable and reliable approach for automatic EduKG generation remains a challenge. In an attempt to develop a unified and robust pipeline for automatic EduKG construction, in this study we propose a pipeline for automatic EduKG construction from PDF learning materials. The process begins with generating slide-level EduKGs from individual pages/slides, which are then merged to form a comprehensive EduKG representing the entire learning material. We evaluate the accuracy of the EduKG generated from the proposed pipeline in our MOOC platform, CourseMapper. The observed accuracy, while indicative of partial success, is relatively low particularly in the educational context, where the reliability of knowledge representations is critical for supporting meaningful learning. To address this, we introduce targeted optimizations across multiple pipeline components. The optimized pipeline achieves a 17.5% improvement in accuracy and a tenfold increase in processing efficiency. Our approach offers a holistic, scalable and end-to-end pipeline for automatic EduKG construction, adaptable to diverse educational contexts, and supports improved semantic representation of learning content. 

**Abstract (ZH)**: 自动构建教育知识图谱（EduKG）对于从学习材料中提取有意义的表示以进行领域知识建模至关重要。尽管兴趣 Growing，但自动 EduKG 生成的可扩展和可靠方法仍然是一个挑战。为了开发一个统一且稳健的自动 EduKG 构建管道，本研究提出了一种从 PDF 学习材料自动构建 EduKG 的管道。该过程始于从单页/幻灯片生成幻灯片级别的 EduKG，然后合并形成代表整个学习材料的全面 EduKG。我们在 MOOC 平台 CourseMapper 中评估了所提管道生成的 EduKG 的准确性。观察到的准确性虽然表明部分成功，但在教育情境中较低，知识表示的可靠性对于支持有意义的学习至关重要。为解决这一问题，我们针对管道多个组件引入了目标优化。优化后的管道在准确性和处理效率上分别提高了 17.5% 和十倍。我们的方法提供了一个全面、可扩展且端到端的自动 EduKG 构建管道，适应多种教育情境，并支持学习内容的改进语义表示。 

---
# Authorship Without Writing: Large Language Models and the Senior Author Analogy 

**Title (ZH)**: 不写作的作者身份：大型语言模型与资深作者类比 

**Authors**: Clint Hurshman, Sebastian Porsdam Mann, Julian Savulescu, Brian D. Earp  

**Link**: [PDF](https://arxiv.org/pdf/2509.05390)  

**Abstract**: The use of large language models (LLMs) in bioethical, scientific, and medical writing remains controversial. While there is broad agreement in some circles that LLMs cannot count as authors, there is no consensus about whether and how humans using LLMs can count as authors. In many fields, authorship is distributed among large teams of researchers, some of whom, including paradigmatic senior authors who guide and determine the scope of a project and ultimately vouch for its integrity, may not write a single word. In this paper, we argue that LLM use (under specific conditions) is analogous to a form of senior authorship. On this view, the use of LLMs, even to generate complete drafts of research papers, can be considered a legitimate form of authorship according to the accepted criteria in many fields. We conclude that either such use should be recognized as legitimate, or current criteria for authorship require fundamental revision. AI use declaration: GPT-5 was used to help format Box 1. AI was not used for any other part of the preparation or writing of this manuscript. 

**Abstract (ZH)**: 大型语言模型在生物伦理、科学和医学写作中的应用仍存在争议。虽然在某些圈子里普遍认为大型语言模型不能被视为作者，但对于人类使用大型语言模型是否可以被视为作者以及如何被视为作者尚无共识。在许多领域中，作者身份在由大量研究人员组成的团队中分布，其中一些人，包括主导项目并确定其范围并在最终为其 integrity 负责的范式意义上的资深作者，可能不会撰写一个字。在本文中，我们认为在特定条件下使用大型语言模型类似于一种资深作者身份的形式。在这种观点下，即使使用大型语言模型生成研究论文的完整草稿，也可以被视为根据许多领域接受的标准的一种合法的作者身份。我们得出结论，要么这种使用应该被认定为合法，要么现行的作者身份标准需要根本性的修订。AI使用声明：GPT-5用于帮助格式化Box 1部分。本论文的其他部分未使用AI进行任何准备或撰写工作。 

---
# Augmented Structure Preserving Neural Networks for cell biomechanics 

**Title (ZH)**: 增强结构保持神经网络在细胞生物力学中的应用 

**Authors**: Juan Olalla-Pombo, Alberto Badías, Miguel Ángel Sanz-Gómez, José María Benítez, Francisco Javier Montáns  

**Link**: [PDF](https://arxiv.org/pdf/2509.05388)  

**Abstract**: Cell biomechanics involve a great number of complex phenomena that are fundamental to the evolution of life itself and other associated processes, ranging from the very early stages of embryo-genesis to the maintenance of damaged structures or the growth of tumors. Given the importance of such phenomena, increasing research has been dedicated to their understanding, but the many interactions between them and their influence on the decisions of cells as a collective network or cluster remain unclear. We present a new approach that combines Structure Preserving Neural Networks, which study cell movements as a purely mechanical system, with other Machine Learning tools (Artificial Neural Networks), which allow taking into consideration environmental factors that can be directly deduced from an experiment with Computer Vision techniques. This new model, tested on simulated and real cell migration cases, predicts complete cell trajectories following a roll-out policy with a high level of accuracy. This work also includes a mitosis event prediction model based on Neural Networks architectures which makes use of the same observed features. 

**Abstract (ZH)**: 细胞生物力学涉及生命进化及其相关过程中众多复杂的现象，从胚胎发生早期阶段到损伤结构的维持或肿瘤的生长都包括在内。鉴于这些现象的重要性，越来越多的研究致力于理解它们，但它们之间的复杂交互作用及其对细胞集体网络或群体决策的影响仍然不甚清楚。我们提出了一种新的方法，将保持结构神经网络与其它机器学习工具（人工神经网络）相结合，以考虑通过计算机视觉技术直接从实验中得出的环境因素。该新模型已在模拟和实际细胞迁移案例中得到测试，能够以高精度预测完整细胞轨迹，并采用神经网络架构提出了一种基于观察特征的有丝分裂事件预测模型。 

---
# A Lightweight Framework for Trigger-Guided LoRA-Based Self-Adaptation in LLMs 

**Title (ZH)**: 一种基于触发器引导的LoRA自适应轻量级框架在大规模语言模型中的应用 

**Authors**: Jiacheng Wei, Faguo Wu, Xiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05385)  

**Abstract**: Large language models are unable to continuously adapt and learn from new data during reasoning at inference time. To address this limitation, we propose that complex reasoning tasks be decomposed into atomic subtasks and introduce SAGE, a trigger-guided dynamic fine-tuning framework that enables adaptive updates during reasoning at inference time. SAGE consists of three key components: (1) a Trigger module that detects reasoning failures through multiple evaluation metrics in real time; (2) a Trigger Buffer module that clusters anomaly samples using a streaming clustering process with HDBSCAN, followed by stability checks and similarity-based merging; and (3) a Lora Store module that dynamically optimizes parameter updates with an adapter pool for knowledge retention. Evaluation results show that SAGE demonstrates excellent accuracy, robustness, and stability on the atomic reasoning subtask through dynamic knowledge updating during test time. 

**Abstract (ZH)**: 大型语言模型在推理时无法连续适应和从新数据中学习。为此，我们提出将复杂推理任务分解为原子子任务，并引入SAGE，一种触发器引导的动态微调框架，允许在推理时进行自适应更新。SAGE 包含三个关键组件：（1）一个触发器模块，通过实时多种评估指标检测推理失败；（2）一个触发器缓冲模块，使用基于流的聚类过程和HDBSCAN进行异常样本聚类，随后进行稳定性检查和基于相似性的合并；（3）一个Lora存储模块，使用适配器池动态优化参数更新以保留知识。评价结果表明，SAGE 在测试时通过动态知识更新展示了出色的准确度、鲁棒性和稳定性。 

---
# User Privacy and Large Language Models: An Analysis of Frontier Developers' Privacy Policies 

**Title (ZH)**: 用户隐私与大型语言模型：前沿开发者隐私政策分析 

**Authors**: Jennifer King, Kevin Klyman, Emily Capstick, Tiffany Saade, Victoria Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2509.05382)  

**Abstract**: Hundreds of millions of people now regularly interact with large language models via chatbots. Model developers are eager to acquire new sources of high-quality training data as they race to improve model capabilities and win market share. This paper analyzes the privacy policies of six U.S. frontier AI developers to understand how they use their users' chats to train models. Drawing primarily on the California Consumer Privacy Act, we develop a novel qualitative coding schema that we apply to each developer's relevant privacy policies to compare data collection and use practices across the six companies. We find that all six developers appear to employ their users' chat data to train and improve their models by default, and that some retain this data indefinitely. Developers may collect and train on personal information disclosed in chats, including sensitive information such as biometric and health data, as well as files uploaded by users. Four of the six companies we examined appear to include children's chat data for model training, as well as customer data from other products. On the whole, developers' privacy policies often lack essential information about their practices, highlighting the need for greater transparency and accountability. We address the implications of users' lack of consent for the use of their chat data for model training, data security issues arising from indefinite chat data retention, and training on children's chat data. We conclude by providing recommendations to policymakers and developers to address the data privacy challenges posed by LLM-powered chatbots. 

**Abstract (ZH)**: 数百万人现在经常通过聊天机器人与大型语言模型互动。模型开发者急于获取新的高品质训练数据来源，以提升模型能力并赢得市场份额。本文分析了六家美国领先AI开发者的隐私政策，以了解他们如何使用用户聊天数据来训练模型。主要基于加利福尼亚消费者隐私法，我们发展了一种新的定性编码方案，并将其应用于每家开发者的相关隐私政策，以比较六家公司的数据收集和使用实践。我们发现，所有六家开发者似乎默认使用用户的聊天数据来训练和改进他们的模型，并且有些开发者会无限期保留这些数据。开发者可能收集并用于训练在聊天中披露的个人信息，包括生物识别和健康等敏感信息，以及用户上传的文件。我们研究的六家公司中有四家似乎包括儿童聊天数据用于模型训练，以及其他产品的客户数据。总体而言，开发者们的隐私政策往往缺乏关于其实践的必要信息，突显了提高透明度和问责制的必要性。我们探讨了用户对使用其聊天数据进行模型训练缺乏同意的含义，以及无限期保留聊天数据所带来的数据安全问题，以及使用儿童聊天数据进行训练的问题。最后，我们为政策制定者和开发者提供了应对基于大语言模型的聊天机器人带来的数据隐私挑战的建议。 

---
# Cumplimiento del Reglamento (UE) 2024/1689 en robótica y sistemas autónomos: una revisión sistemática de la literatura 

**Title (ZH)**: 欧盟条例2024/1689在机器人和自主系统中的实施：文献综述 

**Authors**: Yoana Pita Lorenzo  

**Link**: [PDF](https://arxiv.org/pdf/2509.05380)  

**Abstract**: This systematic literature review analyzes the current state of compliance with Regulation (EU) 2024/1689 in autonomous robotic systems, focusing on cybersecurity frameworks and methodologies. Using the PRISMA protocol, 22 studies were selected from 243 initial records across IEEE Xplore, ACM DL, Scopus, and Web of Science. Findings reveal partial regulatory alignment: while progress has been made in risk management and encrypted communications, significant gaps persist in explainability modules, real-time human oversight, and knowledge base traceability. Only 40% of reviewed solutions explicitly address transparency requirements, and 30% implement failure intervention mechanisms. The study concludes that modular approaches integrating risk, supervision, and continuous auditing are essential to meet the AI Act mandates in autonomous robotics. 

**Abstract (ZH)**: This systematic literature review analyzes the current state of compliance with Regulation (EU) 2024/1689 in autonomous robotic systems, focusing on cybersecurity frameworks and methodologies。该系统文献综述分析了欧盟条例（EU）2024/1689 在自主机器人系统中的当前合规状况，重点关注网络安全框架和方法学。 

---
# ThreatGPT: An Agentic AI Framework for Enhancing Public Safety through Threat Modeling 

**Title (ZH)**: 威胁GPT：一个赋能型AI框架，通过威胁建模提升公共安全 

**Authors**: Sharif Noor Zisad, Ragib Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2509.05379)  

**Abstract**: As our cities and communities become smarter, the systems that keep us safe, such as traffic control centers, emergency response networks, and public transportation, also become more complex. With this complexity comes a greater risk of security threats that can affect not just machines but real people's lives. To address this challenge, we present ThreatGPT, an agentic Artificial Intelligence (AI) assistant built to help people whether they are engineers, safety officers, or policy makers to understand and analyze threats in public safety systems. Instead of requiring deep cybersecurity expertise, it allows users to simply describe the components of a system they are concerned about, such as login systems, data storage, or communication networks. Then, with the click of a button, users can choose how they want the system to be analyzed by using popular frameworks such as STRIDE, MITRE ATT&CK, CVE reports, NIST, or CISA. ThreatGPT is unique because it does not just provide threat information, but rather it acts like a knowledgeable partner. Using few-shot learning, the AI learns from examples and generates relevant smart threat models. It can highlight what might go wrong, how attackers could take advantage, and what can be done to prevent harm. Whether securing a city's infrastructure or a local health service, this tool adapts to users' needs. In simple terms, ThreatGPT brings together AI and human judgment to make our public systems safer. It is designed not just to analyze threats, but to empower people to understand and act on them, faster, smarter, and with more confidence. 

**Abstract (ZH)**: 随着我们的城市和社区变得更加智能化，保障我们安全的系统，如交通控制中心、应急响应网络和公共交通系统，也变得愈加复杂。随之而来的复杂性带来了更多的安全威胁风险，这些威胁不仅影响机器，还可能影响人们的真实生活。为应对这一挑战，我们提出 ThreatGPT，这是一种代理型人工智能（AI）助理，旨在帮助工程师、安全官员或政策制定者理解和分析公共安全系统中的威胁。它不需要用户具备深厚的网络安全专业知识，用户只需描述他们关心的系统组件，如登录系统、数据存储或通信网络。然后，只需点击按钮，用户就可以使用STRIDE、MITRE ATT&CK、CVE报告、NIST或CISA等流行框架来选择他们希望系统如何被分析。ThreatGPT的独特之处在于，它不仅提供威胁信息，更像是一个有知识的合作伙伴。通过少量示例学习，AI学习并生成相关的智能威胁模型，指出可能出现的问题、攻击者可能利用的方法以及可以预防的措施。无论是保护城市的基础设施还是本地健康服务，该工具都能适应用户的需求。简而言之，ThreatGPT将AI和人类判断结合在一起，使我们的公共系统更加安全。它不仅用于分析威胁，还旨在赋能人们更快、更智能地理解和应对这些威胁。 

---
# Privacy Preservation and Identity Tracing Prevention in AI-Driven Eye Tracking for Interactive Learning Environments 

**Title (ZH)**: 基于人工智能驱动的眼动追踪的交互学习环境中隐私保护与身份追踪预防 

**Authors**: Abdul Rehman, Are Dæhlen, Ilona Heldal, Jerry Chun-wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.05376)  

**Abstract**: Eye-tracking technology can aid in understanding neurodevelopmental disorders and tracing a person's identity. However, this technology poses a significant risk to privacy, as it captures sensitive information about individuals and increases the likelihood that data can be traced back to them. This paper proposes a human-centered framework designed to prevent identity backtracking while preserving the pedagogical benefits of AI-powered eye tracking in interactive learning environments. We explore how real-time data anonymization, ethical design principles, and regulatory compliance (such as GDPR) can be integrated to build trust and transparency. We first demonstrate the potential for backtracking student IDs and diagnoses in various scenarios using serious game-based eye-tracking data. We then provide a two-stage privacy-preserving framework that prevents participants from being tracked while still enabling diagnostic classification. The first phase covers four scenarios: I) Predicting disorder diagnoses based on different game levels. II) Predicting student IDs based on different game levels. III) Predicting student IDs based on randomized data. IV) Utilizing K-Means for out-of-sample data. In the second phase, we present a two-stage framework that preserves privacy. We also employ Federated Learning (FL) across multiple clients, incorporating a secure identity management system with dummy IDs and administrator-only access controls. In the first phase, the proposed framework achieved 99.3% accuracy for scenario 1, 63% accuracy for scenario 2, and 99.7% accuracy for scenario 3, successfully identifying and assigning a new student ID in scenario 4. In phase 2, we effectively prevented backtracking and established a secure identity management system with dummy IDs and administrator-only access controls, achieving an overall accuracy of 99.40%. 

**Abstract (ZH)**: 眼动追踪技术可以辅助理解神经发育障碍并追踪个人身份，但也带来了重大的隐私风险。本文提出了一种以人为核心的设计框架，旨在防止身份追溯的同时保留基于人工智能的眼动追踪技术在交互学习环境中的教学益处。我们探讨了如何通过实时数据匿名化、伦理设计原则及合规性（如GDPR）来构建信任和透明度。首先，我们通过基于严肃游戏的眼动追踪数据展示了在各种场景下学生身份和诊断回溯的潜在可能性。然后，我们提供了一种两阶段的隐私保护框架，防止参与者被跟踪的同时仍能实现诊断分类。第一阶段涵盖了四种情景：I）基于不同游戏级别预测障碍诊断；II）基于不同游戏级别预测学生身份；III）基于随机化数据预测学生身份；IV）利用K-Means处理外样数据。在第二阶段，我们提出了一种两阶段框架来保护隐私，并结合多方学习（FL）和一个包含虚拟身份管理和管理员专属访问控制的加密身份管理系统。第一阶段提出的框架在情景1中实现了99.3%的准确率，在情景2中实现了63%的准确率，在情景3中实现了99.7%的准确率，并成功在情景4中识别并分配了一个新的学生身份。在第二阶段，我们有效地防止了身份回溯，并建立了一个包含虚拟身份管理和管理员专属访问控制的加密身份管理系统，整体准确率为99.40%。 

---
# Long-Horizon Visual Imitation Learning via Plan and Code Reflection 

**Title (ZH)**: 长时视觉模仿学习通过计划与代码反思 

**Authors**: Quan Chen, Chenrui Shi, Qi Chen, Yuwei Wu, Zhi Gao, Xintong Zhang, Rui Gao, Kun Wu, Yunde Jia  

**Link**: [PDF](https://arxiv.org/pdf/2509.05368)  

**Abstract**: Learning from long-horizon demonstrations with complex action sequences presents significant challenges for visual imitation learning, particularly in understanding temporal relationships of actions and spatial relationships between objects. In this paper, we propose a new agent framework that incorporates two dedicated reflection modules to enhance both plan and code generation. The plan generation module produces an initial action sequence, which is then verified by the plan reflection module to ensure temporal coherence and spatial alignment with the demonstration video. The code generation module translates the plan into executable code, while the code reflection module verifies and refines the generated code to ensure correctness and consistency with the generated plan. These two reflection modules jointly enable the agent to detect and correct errors in both the plan generation and code generation, improving performance in tasks with intricate temporal and spatial dependencies. To support systematic evaluation, we introduce LongVILBench, a benchmark comprising 300 human demonstrations with action sequences of up to 18 steps. LongVILBench emphasizes temporal and spatial complexity across multiple task types. Experimental results demonstrate that existing methods perform poorly on this benchmark, whereas our new framework establishes a strong baseline for long-horizon visual imitation learning. 

**Abstract (ZH)**: 长时序复杂动作序列指导的视觉模仿学习中的计划与代码生成增强框架：时间与空间关系的联合理解与纠正 

---
# Between a Rock and a Hard Place: Exploiting Ethical Reasoning to Jailbreak LLMs 

**Title (ZH)**: 在岩石和硬 place 之间：利用伦理推理突破大语言模型 blockade 

**Authors**: Shei Pern Chua, Thai Zhen Leng, Teh Kai Jun, Xiao Li, Xiaolin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05367)  

**Abstract**: Large language models (LLMs) have undergone safety alignment efforts to mitigate harmful outputs. However, as LLMs become more sophisticated in reasoning, their intelligence may introduce new security risks. While traditional jailbreak attacks relied on singlestep attacks, multi-turn jailbreak strategies that adapt dynamically to context remain underexplored. In this work, we introduce TRIAL (Trolley-problem Reasoning for Interactive Attack Logic), a framework that leverages LLMs ethical reasoning to bypass their safeguards. TRIAL embeds adversarial goals within ethical dilemmas modeled on the trolley problem. TRIAL demonstrates high jailbreak success rates towards both open and close-source models. Our findings underscore a fundamental limitation in AI safety: as models gain advanced reasoning abilities, the nature of their alignment may inadvertently allow for more covert security vulnerabilities to be exploited. TRIAL raises an urgent need in reevaluating safety alignment oversight strategies, as current safeguards may prove insufficient against context-aware adversarial attack. 

**Abstract (ZH)**: 大型语言模型的安全对齐努力已经减少了有害输出的风险。然而，随着语言模型在推理方面变得越来越复杂，它们的智能可能引入新的安全风险。虽然传统的监狱突破攻击依赖于单步骤攻击，但适应性强的多轮监狱突破策略仍然未被充分探索。在本工作中，我们提出了TRIAL（Trolley-problem Reasoning for Interactive Attack Logic）框架，该框架利用大型语言模型的伦理推理来绕过其安全防护。TRIAL将对抗性目标嵌入到基于电车问题建模的伦理困境中。TRIAL展示了对开源和闭源模型高度成功的监狱突破成功率。我们的研究结果揭示了一个基本的安全限制：随着模型获得高级推理能力，它们的对齐方式可能会无意中允许更多的隐蔽安全漏洞被利用。TRIAL强调了一个迫切的需求，即重新评估安全对齐监督策略，因为当前的安全防护可能不足以应对具有情境意识的对抗性攻击。 

---
# Prototyping an AI-powered Tool for Energy Efficiency in New Zealand Homes 

**Title (ZH)**: 基于AI的高效能住宅工具原型设计：以新西兰住宅为例 

**Authors**: Abdollah Baghaei Daemei  

**Link**: [PDF](https://arxiv.org/pdf/2509.05364)  

**Abstract**: Residential buildings contribute significantly to energy use, health outcomes, and carbon emissions. In New Zealand, housing quality has historically been poor, with inadequate insulation and inefficient heating contributing to widespread energy hardship. Recent reforms, including the Warmer Kiwi Homes program, Healthy Homes Standards, and H1 Building Code upgrades, have delivered health and comfort improvements, yet challenges persist. Many retrofits remain partial, data on household performance are limited, and decision-making support for homeowners is fragmented. This study presents the design and evaluation of an AI-powered decision-support tool for residential energy efficiency in New Zealand. The prototype, developed using Python and Streamlit, integrates data ingestion, anomaly detection, baseline modeling, and scenario simulation (e.g., LED retrofits, insulation upgrades) into a modular dashboard. Fifteen domain experts, including building scientists, consultants, and policy practitioners, tested the tool through semi-structured interviews. Results show strong usability (M = 4.3), high value of scenario outputs (M = 4.5), and positive perceptions of its potential to complement subsidy programs and regulatory frameworks. The tool demonstrates how AI can translate national policies into personalized, household-level guidance, bridging the gap between funding, standards, and practical decision-making. Its significance lies in offering a replicable framework for reducing energy hardship, improving health outcomes, and supporting climate goals. Future development should focus on carbon metrics, tariff modeling, integration with national datasets, and longitudinal trials to assess real-world adoption. 

**Abstract (ZH)**: 住宅建筑对能源使用、健康结果和碳排放产生了显著影响。在新西兰，住房质量 historical 上较差，缺乏足够的保温和低效的供暖设施，导致广泛的能源负担。最近的改革，包括“温暖新西兰家园”计划、健康家园标准和H1建筑规范升级，已经带来了健康和舒适度的改善，但仍面临挑战。许多翻新仍然只是部分完成，家庭性能数据有限，对房主的决策支持也支离破碎。本研究提出了一个基于人工智能的决策支持工具，用于新西兰住宅能源效率的设计和评估。该原型使用Python和Streamlit开发，整合了数据摄入、异常检测、基线建模和场景模拟（例如LED翻新、保温升级）等功能，形成模块化的仪表板。十五位领域专家，包括建筑科学家、咨询顾问和政策从业者，通过半结构化访谈测试了该工具。结果显示，该工具具有良好的可用性（平均分4.3）、场景输出的高度价值（平均分4.5），并且被认为有可能补充补贴计划和监管框架。该工具展示了如何将国家政策转化为个人化的家庭指导，跨越了从资金、标准到实际决策的鸿沟。其意义在于提供了一个可复制的框架，用于减少能源负担、改善健康结果和支持气候目标。未来开发应关注碳指标、电价建模、与国家数据集的整合以及纵向试验，以评估其在实际中的应用。 

---
# AI-in-the-Loop: Privacy Preserving Real-Time Scam Detection and Conversational Scambaiting by Leveraging LLMs and Federated Learning 

**Title (ZH)**: AI在环中:通过利用LLM和联邦学习实现的隐私保护实时诈骗检测与对话式诈骗诱捕 

**Authors**: Ismail Hossain, Sai Puppala, Sajedul Talukder, Md Jahangir Alam  

**Link**: [PDF](https://arxiv.org/pdf/2509.05362)  

**Abstract**: Scams exploiting real-time social engineering -- such as phishing, impersonation, and phone fraud -- remain a persistent and evolving threat across digital platforms. Existing defenses are largely reactive, offering limited protection during active interactions. We propose a privacy-preserving, AI-in-the-loop framework that proactively detects and disrupts scam conversations in real time. The system combines instruction-tuned artificial intelligence with a safety-aware utility function that balances engagement with harm minimization, and employs federated learning to enable continual model updates without raw data sharing. Experimental evaluations show that the system produces fluent and engaging responses (perplexity as low as 22.3, engagement $\approx$0.80), while human studies confirm significant gains in realism, safety, and effectiveness over strong baselines. In federated settings, models trained with FedAvg sustain up to 30 rounds while preserving high engagement ($\approx$0.80), strong relevance ($\approx$0.74), and low PII leakage ($\leq$0.0085). Even with differential privacy, novelty and safety remain stable, indicating that robust privacy can be achieved without sacrificing performance. The evaluation of guard models (LlamaGuard, LlamaGuard2/3, MD-Judge) shows a straightforward pattern: stricter moderation settings reduce the chance of exposing personal information, but they also limit how much the model engages in conversation. In contrast, more relaxed settings allow longer and richer interactions, which improve scam detection, but at the cost of higher privacy risk. To our knowledge, this is the first framework to unify real-time scam-baiting, federated privacy preservation, and calibrated safety moderation into a proactive defense paradigm. 

**Abstract (ZH)**: 利用实时社会工程学实施的诈骗——例如 phishing、冒充和电话诈骗——仍然是数字平台上的一个持久且不断演变的威胁。现有的防护措施主要是反应性的，在活跃交互过程中提供的保护有限。我们提出了一种保护隐私、AI 集成循环的框架，能够实时主动检测和打断诈骗对话。该系统结合了指令调优的人工智能与兼顾安全的效用函数，平衡参与度与最小化危害，并采用联邦学习来实现无原始数据共享的持续模型更新。实验评估表明，该系统生成流畅且富有参与度的响应（困惑度低至 22.3，参与度 ≈ 0.80），而人类研究证实，与强基线相比，在现实主义、安全性和有效性方面有显著提升。在联邦环境中，使用 FedAvg 训练的模型最多可维持 30 轮更新，同时保持高水平的参与度（≈0.80）、较强的相关性（≈0.74）和低个人身份信息泄露（≤0.0085）。即使在差分隐私下，新颖性和安全性仍保持稳定，表明可以实现稳健的隐私保护而不牺牲性能。对于防护模型（LlamaGuard、LlamaGuard2/3、MD-Judge）的评估显示了一个简单的模式：更严格的审查设置减少了暴露个人信息的机会，但也限制了模型在对话中的参与度。相反，更宽松的设置允许更长和更丰富的互动，从而提高诈骗检测，但以更高的隐私风险为代价。据我们所知，这是第一个将实时诈骗诱饵、联邦隐私保护和校准的安全调节统一到主动防御范式中的框架。 

---
# Governing AI R&D: A Legal Framework for Constraining Dangerous AI 

**Title (ZH)**: 治理AI研发：约束危险人工智能的法律框架 

**Authors**: Alex Mark, Aaron Scher  

**Link**: [PDF](https://arxiv.org/pdf/2509.05361)  

**Abstract**: As AI advances, governing its development may become paramount to public safety. Lawmakers may seek to restrict the development and release of AI models or of AI research itself. These governance actions could trigger legal challenges that invalidate the actions, so lawmakers should consider these challenges ahead of time. We investigate three classes of potential litigation risk for AI regulation in the U.S.: the First Amendment, administrative law, and the Fourteenth Amendment. We discuss existing precedent that is likely to apply to AI, which legal challenges are likely to arise, and how lawmakers might preemptively address them. Effective AI regulation is possible, but it requires careful implementation to avoid these legal challenges. 

**Abstract (ZH)**: 随着人工智能的发展，对其发展的治理可能对公共安全至关重要。立法者可能会寻求限制AI模型的发展和发布，或限制AI研究本身。这些治理行动可能会引发法律挑战，从而使这些行动无效，因此立法者应在采取行动前考虑这些挑战。我们研究了美国AI治理潜在诉讼风险的三类：第一修正案、行政法和第十四修正案。我们讨论了可能适用于AI的现有先例，可能出现的法律挑战以及立法者如何事先应对这些挑战。有效的AI治理是可能的，但需要谨慎实施以避免这些法律挑战。 

---
# An Empirical Analysis of Discrete Unit Representations in Speech Language Modeling Pre-training 

**Title (ZH)**: 离散单位表示在语音语言模型预训练中的实证分析 

**Authors**: Yanis Labrak, Richard Dufour, Mickaël Rouvier  

**Link**: [PDF](https://arxiv.org/pdf/2509.05359)  

**Abstract**: This paper investigates discrete unit representations in Speech Language Models (SLMs), focusing on optimizing speech modeling during continual pre-training. In this paper, we systematically examine how model architecture, data representation, and training robustness influence the pre-training stage in which we adapt existing pre-trained language models to the speech modality. Our experiments highlight the role of speech encoders and clustering granularity across different model scales, showing how optimal discretization strategies vary with model capacity. By examining cluster distribution and phonemic alignments, we investigate the effective use of discrete vocabulary, uncovering both linguistic and paralinguistic patterns. Additionally, we explore the impact of clustering data selection on model robustness, highlighting the importance of domain matching between discretization training and target applications. 

**Abstract (ZH)**: 本文研究了语音语言模型中离散单位表示，聚焦于优化连续预训练中的语音建模。本文系统地考察了模型架构、数据表示和训练鲁棒性对预训练阶段的影响，该阶段涉及将现有预训练语言模型适配至语音模态。我们的实验强调了不同模型规模下的语音编码器和聚类粒度的作用，展示了最优离散化策略随模型容量的变化。通过研究聚类分布和音素对齐，我们探讨了离散词汇的有效应用，揭示了语言和副语言模式。此外，我们还探索了聚类数据选择对模型鲁棒性的影响，强调了离散化训练与目标应用领域匹配的重要性。 

---
# Spiking Neural Networks for Continuous Control via End-to-End Model-Based Learning 

**Title (ZH)**: 基于端到端模型导向学习的脉冲神经网络连续控制 

**Authors**: Justus Huebotter, Pablo Lanillos, Marcel van Gerven, Serge Thill  

**Link**: [PDF](https://arxiv.org/pdf/2509.05356)  

**Abstract**: Despite recent progress in training spiking neural networks (SNNs) for classification, their application to continuous motor control remains limited. Here, we demonstrate that fully spiking architectures can be trained end-to-end to control robotic arms with multiple degrees of freedom in continuous environments. Our predictive-control framework combines Leaky Integrate-and-Fire dynamics with surrogate gradients, jointly optimizing a forward model for dynamics prediction and a policy network for goal-directed action. We evaluate this approach on both a planar 2D reaching task and a simulated 6-DOF Franka Emika Panda robot. Results show that SNNs can achieve stable training and accurate torque control, establishing their viability for high-dimensional motor tasks. An extensive ablation study highlights the role of initialization, learnable time constants, and regularization in shaping training dynamics. We conclude that while stable and effective control can be achieved, recurrent spiking networks remain highly sensitive to hyperparameter settings, underscoring the importance of principled design choices. 

**Abstract (ZH)**: 尽管在训练神经脉冲网络（SNNs）进行分类方面取得了近期进展，但它们在连续运动控制中的应用仍受到限制。在此，我们证明完全脉冲架构可以通过端到端训练来控制具有多个自由度的连续环境中的机器人手臂。我们的预测控制框架结合了Leaky Integrate-and-Fire动力学和代理梯度，联合优化了动力学预测的前向模型和用于目标导向动作的策略网络。我们在一个平面2D抓取任务和一个模拟的6-DOF Franka Emika Panda机器人上评估了这种方法。结果显示，SNNs可以实现稳定的训练和精确的扭矩控制，确立了其在高维运动任务中的可行性。通过广泛的消融研究，我们强调了初始化、可学习的时间常数和正则化在塑造训练动态中的作用。我们得出结论，虽然可以实现稳定和有效的控制，但循环脉冲网络对超参数设置的高度敏感性强调了原理性设计选择的重要性。 

---
# Unsupervised Instance Segmentation with Superpixels 

**Title (ZH)**: 无监督实例分割方法：基于超像素技术 

**Authors**: Cuong Manh Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05352)  

**Abstract**: Instance segmentation is essential for numerous computer vision applications, including robotics, human-computer interaction, and autonomous driving. Currently, popular models bring impressive performance in instance segmentation by training with a large number of human annotations, which are costly to collect. For this reason, we present a new framework that efficiently and effectively segments objects without the need for human annotations. Firstly, a MultiCut algorithm is applied to self-supervised features for coarse mask segmentation. Then, a mask filter is employed to obtain high-quality coarse masks. To train the segmentation network, we compute a novel superpixel-guided mask loss, comprising hard loss and soft loss, with high-quality coarse masks and superpixels segmented from low-level image features. Lastly, a self-training process with a new adaptive loss is proposed to improve the quality of predicted masks. We conduct experiments on public datasets in instance segmentation and object detection to demonstrate the effectiveness of the proposed framework. The results show that the proposed framework outperforms previous state-of-the-art methods. 

**Abstract (ZH)**: 实例分割对于机器人技术、人机交互和自动驾驶等众多计算机视觉应用至关重要。目前，流行的模型通过大量人工标注训练，在实例分割上表现出色，但人工标注成本高昂。为解决这一问题，我们提出了一种新的框架，能够在无需人工标注的情况下高效且有效地进行对象分割。该框架首先应用MultiCut算法对自监督特征进行粗略掩膜分割，然后使用掩膜过滤器获得高质量的粗略掩膜。为训练分割网络，我们计算了一种基于超像素的新型掩膜损失，包含硬损失和软损失，并结合高质量的粗略掩膜和从低级图像特征中分割出的超像素。最后，提出了一种带有新自适应损失的自我训练过程，以提高预测掩膜的质量。我们在实例分割和对象检测的公开数据集上进行了实验，证明了所提框架的有效性。实验结果表明，所提框架优于之前的方法。 

---
# Comparative Evaluation of Hard and Soft Clustering for Precise Brain Tumor Segmentation in MR Imaging 

**Title (ZH)**: 硬聚类与软聚类在MR成像精准脑肿瘤分割中的比较评价 

**Authors**: Dibya Jyoti Bora, Mrinal Kanti Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2509.05340)  

**Abstract**: Segmentation of brain tumors from Magnetic Resonance Imaging (MRI) remains a pivotal challenge in medical image analysis due to the heterogeneous nature of tumor morphology and intensity distributions. Accurate delineation of tumor boundaries is critical for clinical decision-making, radiotherapy planning, and longitudinal disease monitoring. In this study, we perform a comprehensive comparative analysis of two major clustering paradigms applied in MRI tumor segmentation: hard clustering, exemplified by the K-Means algorithm, and soft clustering, represented by Fuzzy C-Means (FCM). While K-Means assigns each pixel strictly to a single cluster, FCM introduces partial memberships, meaning each pixel can belong to multiple clusters with varying degrees of association. Experimental validation was performed using the BraTS2020 dataset, incorporating pre-processing through Gaussian filtering and Contrast Limited Adaptive Histogram Equalization (CLAHE). Evaluation metrics included the Dice Similarity Coefficient (DSC) and processing time, which collectively demonstrated that K-Means achieved superior speed with an average runtime of 0.3s per image, whereas FCM attained higher segmentation accuracy with an average DSC of 0.67 compared to 0.43 for K-Means, albeit at a higher computational cost (1.3s per image). These results highlight the inherent trade-off between computational efficiency and boundary precision. 

**Abstract (ZH)**: 磁共振成像（MRI）中脑肿瘤分割仍然是医学图像分析中的主要挑战，由于肿瘤形态和强度分布的异质性。准确划定肿瘤边界对于临床决策、放疗计划和纵向疾病监测至关重要。在本研究中，我们对应用于MRI肿瘤分割的两大主要聚类范式进行了全面比较分析：硬聚类，以K-均值算法为代表；软聚类，以模糊C均值（FCM）为代表。尽管K-均值将每个像素严格分配给单个聚类，FCM引入了部分隶属度，使得每个像素可以以不同程度归属于多个聚类。实验验证使用了BraTS2020数据集，并进行了高斯滤波和对比限制定制直方图均衡化（CLAHE）预处理。评价指标包括 Dice 相似性系数（DSC）和处理时间，结果显示，K-均值在平均运行时间为每幅图像0.3秒的情况下实现了更好的速度，而FCM在平均DSC为0.67的情况下达到了更高的分割准确度，尽管每幅图像的运算时间为1.3秒，具有更高的计算成本。这些结果突显了计算效率和边界精度之间的固有权衡。 

---
# Plantbot: Integrating Plant and Robot through LLM Modular Agent Networks 

**Title (ZH)**: Plantbot：通过LLM模块化代理网络整合植物与机器人 

**Authors**: Atsushi Masumori, Norihiro Maruyama, Itsuki Doi, johnsmith, Hiroki Sato, Takashi Ikegami  

**Link**: [PDF](https://arxiv.org/pdf/2509.05338)  

**Abstract**: We introduce Plantbot, a hybrid lifeform that connects a living plant with a mobile robot through a network of large language model (LLM) modules. Each module - responsible for sensing, vision, dialogue, or action - operates asynchronously and communicates via natural language, enabling seamless interaction across biological and artificial domains. This architecture leverages the capacity of LLMs to serve as hybrid interfaces, where natural language functions as a universal protocol, translating multimodal data (soil moisture, temperature, visual context) into linguistic messages that coordinate system behaviors. The integrated network transforms plant states into robotic actions, installing normativity essential for agency within the sensor-motor loop. By combining biological and robotic elements through LLM-mediated communication, Plantbot behaves as an embodied, adaptive agent capable of responding autonomously to environmental conditions. This approach suggests possibilities for a new model of artificial life, where decentralized, LLM modules coordination enable novel interactions between biological and artificial systems. 

**Abstract (ZH)**: Plantbot：一种通过大型语言模型模块连接生物植物与移动机器人的混合生命体 

---
# RT-VLM: Re-Thinking Vision Language Model with 4-Clues for Real-World Object Recognition Robustness 

**Title (ZH)**: RT-VLM: 重新思考具有4线索的视觉语言模型以提高现实世界物体识别 robustness 

**Authors**: Junghyun Park, Tuan Anh Nguyen, Dugki Min  

**Link**: [PDF](https://arxiv.org/pdf/2509.05333)  

**Abstract**: Real world deployments often expose modern object recognition models to domain shifts that precipitate a severe drop in accuracy. Such shifts encompass (i) variations in low level image statistics, (ii) changes in object pose and viewpoint, (iii) partial occlusion, and (iv) visual confusion across adjacent classes. To mitigate this degradation, we introduce the Re-Thinking Vision Language Model (RT-VLM) framework. The foundation of this framework is a unique synthetic dataset generation pipeline that produces images annotated with "4-Clues": precise bounding boxes, class names, detailed object-level captions, and a comprehensive context-level caption for the entire scene. We then perform parameter efficient supervised tuning of Llama 3.2 11B Vision Instruct on this resource. At inference time, a two stage Re-Thinking scheme is executed: the model first emits its own four clues, then re examines these responses as evidence and iteratively corrects them. Across robustness benchmarks that isolate individual domain shifts, RT-VLM consistently surpasses strong baselines. These findings indicate that the integration of structured multimodal evidence with an explicit self critique loop constitutes a promising route toward reliable and transferable visual understanding. 

**Abstract (ZH)**: Real World Deployments Often Expose Modern Object Recognition Models to Domain Shifts That Precipitate a Severe Drop in Accuracy: The Re-Thinking Vision Language Model (RT-VLM) Framework 

---
# Integrated Simulation Framework for Adversarial Attacks on Autonomous Vehicles 

**Title (ZH)**: 面向自动驾驶车辆对抗攻击的集成仿真框架 

**Authors**: Christos Anagnostopoulos, Ioulia Kapsali, Alexandros Gkillas, Nikos Piperigkos, Aris S. Lalos  

**Link**: [PDF](https://arxiv.org/pdf/2509.05332)  

**Abstract**: Autonomous vehicles (AVs) rely on complex perception and communication systems, making them vulnerable to adversarial attacks that can compromise safety. While simulation offers a scalable and safe environment for robustness testing, existing frameworks typically lack comprehensive supportfor modeling multi-domain adversarial scenarios. This paper introduces a novel, open-source integrated simulation framework designed to generate adversarial attacks targeting both perception and communication layers of AVs. The framework provides high-fidelity modeling of physical environments, traffic dynamics, and V2X networking, orchestrating these components through a unified core that synchronizes multiple simulators based on a single configuration file. Our implementation supports diverse perception-level attacks on LiDAR sensor data, along with communication-level threats such as V2X message manipulation and GPS spoofing. Furthermore, ROS 2 integration ensures seamless compatibility with third-party AV software stacks. We demonstrate the framework's effectiveness by evaluating the impact of generated adversarial scenarios on a state-of-the-art 3D object detector, revealing significant performance degradation under realistic conditions. 

**Abstract (ZH)**: 自主驾驶车辆（AVs）依赖复杂的感知和通信系统，使其容易受到攻击，这些攻击会损害安全性。尽管模拟提供了可扩展且安全的环境来进行鲁棒性测试，但现有框架通常缺乏全面支持多域 adversarial 场景建模的能力。本文介绍了一种新型的开源集成模拟框架，旨在针对AVs的感知和通信层生成 adversarial 攻击。该框架提供了对物理环境、交通动态和V2X网络的高保真建模，并通过一个统一的核心组件协调这些组件，该组件基于一个单一的配置文件进行同步。我们的实现支持对激光雷达传感器数据的各种感知层攻击，以及消息操纵和GPS欺骗等通信层威胁。此外，ROS 2 集成确保了与第三方AV软件堆栈的无缝兼容性。通过评估所生成的 adversarial 场景对最先进的3D物体检测器的影响，展示了该框架的有效性，在实际条件下显示出显著的性能退化。 

---
# ForensicsData: A Digital Forensics Dataset for Large Language Models 

**Title (ZH)**: ForensicsData：面向大型语言模型的数字取证数据集 

**Authors**: Youssef Chakir, Iyad Lahsen-Cherif  

**Link**: [PDF](https://arxiv.org/pdf/2509.05331)  

**Abstract**: The growing complexity of cyber incidents presents significant challenges for digital forensic investigators, especially in evidence collection and analysis. Public resources are still limited because of ethical, legal, and privacy concerns, even though realistic datasets are necessary to support research and tool developments. To address this gap, we introduce ForensicsData, an extensive Question-Context-Answer (Q-C-A) dataset sourced from actual malware analysis reports. It consists of more than 5,000 Q-C-A triplets. A unique workflow was used to create the dataset, which extracts structured data, uses large language models (LLMs) to transform it into Q-C-A format, and then uses a specialized evaluation process to confirm its quality. Among the models evaluated, Gemini 2 Flash demonstrated the best performance in aligning generated content with forensic terminology. ForensicsData aims to advance digital forensics by enabling reproducible experiments and fostering collaboration within the research community. 

**Abstract (ZH)**: 网络事件日益增大的复杂性给数字取证调查人员带来了巨大挑战，尤其是在证据收集和分析方面。由于伦理、法律和隐私方面的关切，公共资源仍然有限，尽管真实的数据集对于支持研究和工具开发是必要的。为解决这一问题，我们引入了ForensicsData，这是一个来源于实际恶意软件分析报告的全面问题-上下文-答案（Q-C-A）数据集，包含超过5,000个Q-C-A三元组。采用了一种独特的workflow创建数据集，该workflow提取结构化数据、使用大语言模型（LLMs）将其转换为Q-C-A格式，并通过专门的评估过程确认其质量。在评估的模型中，Gemini 2 Flash在生成内容与取证术语的对齐方面表现最佳。ForensicsData旨在通过促进可重复实验和研究社区内的合作来推动数字取证的发展。 

---
# Optical Music Recognition of Jazz Lead Sheets 

**Title (ZH)**: 爵士乐乐谱的光学音乐识别 

**Authors**: Juan Carlos Martinez-Sevilla, Francesco Foscarin, Patricia Garcia-Iasci, David Rizo, Jorge Calvo-Zaragoza, Gerhard Widmer  

**Link**: [PDF](https://arxiv.org/pdf/2509.05329)  

**Abstract**: In this paper, we address the challenge of Optical Music Recognition (OMR) for handwritten jazz lead sheets, a widely used musical score type that encodes melody and chords. The task is challenging due to the presence of chords, a score component not handled by existing OMR systems, and the high variability and quality issues associated with handwritten images. Our contribution is two-fold. We present a novel dataset consisting of 293 handwritten jazz lead sheets of 163 unique pieces, amounting to 2021 total staves aligned with Humdrum **kern and MusicXML ground truth scores. We also supply synthetic score images generated from the ground truth. The second contribution is the development of an OMR model for jazz lead sheets. We discuss specific tokenisation choices related to our kind of data, and the advantages of using synthetic scores and pretrained models. We publicly release all code, data, and models. 

**Abstract (ZH)**: 本文针对手写爵士乐lead sheets的光学音乐识别（OMR）挑战进行了研究，这是一种广泛使用的乐谱类型，用于编码旋律和和弦。由于存在和弦这一现有OMR系统未处理的乐谱组件，以及手写图像的高变异性与质量 issues，使得任务极具挑战性。我们的贡献主要有两方面：一是提供了一个新颖的数据集，包含293份手写爵士乐lead sheets，共计163首独特的乐谱，共有2021个staff，并与Humdrum **kern和MusicXML参考乐谱对齐；二是开发了一种适用于爵士乐lead sheets的OMR模型。讨论了与我们数据类型相关的一些特定标记化选择，以及使用合成乐谱和预训练模型的优势。所有代码、数据和模型均已公开发布。 

---
# Zero-Knowledge Proofs in Sublinear Space 

**Title (ZH)**: 子线性空间中的零知识证明 

**Authors**: Logan Nye  

**Link**: [PDF](https://arxiv.org/pdf/2509.05326)  

**Abstract**: Modern zero-knowledge proof (ZKP) systems, essential for privacy and verifiable computation, suffer from a fundamental limitation: the prover typically uses memory that scales linearly with the computation's trace length T, making them impractical for resource-constrained devices and prohibitively expensive for large-scale tasks. This paper overcomes this barrier by constructing, to our knowledge, the first sublinear-space ZKP prover. Our core contribution is an equivalence that reframes proof generation as an instance of the classic Tree Evaluation problem. Leveraging a recent space-efficient tree-evaluation algorithm, we design a streaming prover that assembles the proof without ever materializing the full execution trace. The approach reduces prover memory from linear in T to O(sqrt(T)) (up to O(log T) lower-order terms) while preserving proof size, verifier time, and the transcript/security guarantees of the underlying system. This enables a shift from specialized, server-bound proving to on-device proving, opening applications in decentralized systems, on-device machine learning, and privacy-preserving technologies. 

**Abstract (ZH)**: 本篇论文克服了现代零知识证明（ZKP）系统的基本限制，构建了首个亚线性空间的ZKP证明者。 

---
# A Dataset Generation Scheme Based on Video2EEG-SPGN-Diffusion for SEED-VD 

**Title (ZH)**: 基于Video2EEG-SPGN-Diffusion的SEED-VD数据集生成方案 

**Authors**: Yunfei Guo, Tao Zhang, Wu Huang, Yao Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.05321)  

**Abstract**: This paper introduces an open-source framework, Video2EEG-SPGN-Diffusion, that leverages the SEED-VD dataset to generate a multimodal dataset of EEG signals conditioned on video stimuli. Additionally, we disclose an engineering pipeline for aligning video and EEG data pairs, facilitating the training of multimodal large models with EEG alignment capabilities. Personalized EEG signals are generated using a self-play graph network (SPGN) integrated with a diffusion model. As a major contribution, we release a new dataset comprising over 1000 samples of SEED-VD video stimuli paired with generated 62-channel EEG signals at 200 Hz and emotion labels, enabling video-EEG alignment and advancing multimodal research. This framework offers novel tools for emotion analysis, data augmentation, and brain-computer interface applications, with substantial research and engineering significance. 

**Abstract (ZH)**: 本研究介绍了一个开源框架——Video2EEG-SPGN-Diffusion，该框架利用SEED-VD数据集生成基于视频刺激的多模态EEG信号数据集。此外，我们披露了一个视频和EEG数据对对齐的工程流程，从而促进具有EEG对齐能力的多模态大型模型的训练。通过将自玩游戏网络（SPGN）与扩散模型集成，生成个性化EEG信号。作为主要贡献，我们发布了一个新数据集，包含了超过1000个SEED-VD视频刺激样本，配以生成的62导程200 Hz的EEG信号和情绪标签，以实现视频-EEG对齐并推动多模态研究。该框架提供了用于情绪分析、数据增强和脑机接口应用的新工具，具有重要的科研和工程意义。 

---
# Backdoor Samples Detection Based on Perturbation Discrepancy Consistency in Pre-trained Language Models 

**Title (ZH)**: 基于扰动不一致性在预训练语言模型中的后门样本检测 

**Authors**: Zuquan Peng, Jianming Fu, Lixin Zou, Li Zheng, Yanzhen Ren, Guojun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.05318)  

**Abstract**: The use of unvetted third-party and internet data renders pre-trained models susceptible to backdoor attacks. Detecting backdoor samples is critical to prevent backdoor activation during inference or injection during training. However, existing detection methods often require the defender to have access to the poisoned models, extra clean samples, or significant computational resources to detect backdoor samples, limiting their practicality. To address this limitation, we propose a backdoor sample detection method based on perturbatio\textbf{N} discr\textbf{E}pancy consis\textbf{T}ency \textbf{E}valuation (\NETE). This is a novel detection method that can be used both pre-training and post-training phases. In the detection process, it only requires an off-the-shelf pre-trained model to compute the log probability of samples and an automated function based on a mask-filling strategy to generate perturbations. Our method is based on the interesting phenomenon that the change in perturbation discrepancy for backdoor samples is smaller than that for clean samples. Based on this phenomenon, we use curvature to measure the discrepancy in log probabilities between different perturbed samples and input samples, thereby evaluating the consistency of the perturbation discrepancy to determine whether the input sample is a backdoor sample. Experiments conducted on four typical backdoor attacks and five types of large language model backdoor attacks demonstrate that our detection strategy outperforms existing zero-shot black-box detection methods. 

**Abstract (ZH)**: 未审核的第三方和互联网数据使得预训练模型容易受到后门攻击。基于扰动离散性一致性评估（\NETE）的后门样本检测方法对于防止推理或训练期间后门激活至关重要。现有的检测方法往往要求防御者拥有中毒模型、额外的干净样本或大量的计算资源来检测后门样本，这限制了它们的实际应用。为解决这一局限性，我们提出了一种基于扰动离散性一致性评估（\NETE）的后门样本检测方法。这是一种新型的检测方法，可以在预训练和后训练阶段使用。在检测过程中，该方法只需使用一个即用型的预训练模型计算样本的对数概率，并基于掩码填充策略的自动化函数生成扰动。我们的方法基于一个有趣的现象，即后门样本的扰动离散性变化小于干净样本的变化。基于这一现象，我们使用曲率来衡量不同扰动样本和输入样本之间对数概率的不一致性，并评估扰动离散性的一致性以确定输入样本是否为后门样本。实验表明，我们的检测策略在四种典型的后门攻击和五种大型语言模型后门攻击上优于现有的零样本黑盒检测方法。 

---
# VILOD: A Visual Interactive Labeling Tool for Object Detection 

**Title (ZH)**: VILOD: 一种用于目标检测的可视交互标注工具 

**Authors**: Isac Holm  

**Link**: [PDF](https://arxiv.org/pdf/2509.05317)  

**Abstract**: The advancement of Object Detection (OD) using Deep Learning (DL) is often hindered by the significant challenge of acquiring large, accurately labeled datasets, a process that is time-consuming and expensive. While techniques like Active Learning (AL) can reduce annotation effort by intelligently querying informative samples, they often lack transparency, limit the strategic insight of human experts, and may overlook informative samples not aligned with an employed query strategy. To mitigate these issues, Human-in-the-Loop (HITL) approaches integrating human intelligence and intuition throughout the machine learning life-cycle have gained traction. Leveraging Visual Analytics (VA), effective interfaces can be created to facilitate this human-AI collaboration. This thesis explores the intersection of these fields by developing and investigating "VILOD: A Visual Interactive Labeling tool for Object Detection". VILOD utilizes components such as a t-SNE projection of image features, together with uncertainty heatmaps and model state views. Enabling users to explore data, interpret model states, AL suggestions, and implement diverse sample selection strategies within an iterative HITL workflow for OD. An empirical investigation using comparative use cases demonstrated how VILOD, through its interactive visualizations, facilitates the implementation of distinct labeling strategies by making the model's state and dataset characteristics more interpretable (RQ1). The study showed that different visually-guided labeling strategies employed within VILOD result in competitive OD performance trajectories compared to an automated uncertainty sampling AL baseline (RQ2). This work contributes a novel tool and empirical insight into making the HITL-AL workflow for OD annotation more transparent, manageable, and potentially more effective. 

**Abstract (ZH)**: 基于深度学习的对象检测进展往往受到获取大量准确标注数据集的显著挑战的阻碍，这一过程耗时且昂贵。虽然主动学习等技术可以通过智能化查询信息性样本来减少标注工作量，但这些技术往往缺乏透明性，限制了人类专家的战略洞察力，并可能忽略与所采用查询策略不一致的信息性样本。为了缓解这些问题，结合人类智能和直觉的人在回路（HITL）方法在机器学习生命周期中的应用逐渐受到关注。利用可视分析（VA），可以创建有效的界面以促进人类-AI合作。本文通过开发和研究"VILOD：一种用于对象检测的可视交互标注工具"来探讨这些领域的交叉。VILOD利用了如t-SNE图像特征投影、不确定性热图和模型状态视图等组件。它使用户能够在迭代的HITL工作流程中探索数据、解释模型状态、主动学习建议并实施多样化的样本选择策略。通过比较使用案例的实验研究结果表明，VILOD通过其交互式可视化使对象检测标注工作流程中的模型状态和数据集特征更具可解释性（RQ1）。研究结果显示，VILOD中采用的不同视觉引导的标注策略在对象检测性能轨迹上与自动不确定抽样主动学习基线具有竞争力（RQ2）。本文贡献了一种新颖的工具和实证见解，以使对象检测标注的人机在环-主动学习工作流程更具透明性、可管理性和潜在的有效性。 

---
# Standard vs. Modular Sampling: Best Practices for Reliable LLM Unlearning 

**Title (ZH)**: 标准采样 vs. 模块化采样：可靠卸载LLM的最佳实践 

**Authors**: Praveen Bushipaka, Lucia Passaro, Tommaso Cucinotta  

**Link**: [PDF](https://arxiv.org/pdf/2509.05316)  

**Abstract**: A conventional LLM Unlearning setting consists of two subsets -"forget" and "retain", with the objectives of removing the undesired knowledge from the forget set while preserving the remaining knowledge from the retain. In privacy-focused unlearning research, a retain set is often further divided into neighbor sets, containing either directly or indirectly connected to the forget targets; and augmented by a general-knowledge set. A common practice in existing benchmarks is to employ only a single neighbor set, with general knowledge which fails to reflect the real-world data complexities and relationships. LLM Unlearning typically involves 1:1 sampling or cyclic iteration sampling. However, the efficacy and stability of these de facto standards have not been critically examined. In this study, we systematically evaluate these common practices. Our findings reveal that relying on a single neighbor set is suboptimal and that a standard sampling approach can obscure performance trade-offs. Based on this analysis, we propose and validate an initial set of best practices: (1) Incorporation of diverse neighbor sets to balance forget efficacy and model utility, (2) Standard 1:1 sampling methods are inefficient and yield poor results, (3) Our proposed Modular Entity-Level Unlearning (MELU) strategy as an alternative to cyclic sampling. We demonstrate that this modular approach, combined with robust algorithms, provides a clear and stable path towards effective unlearning. 

**Abstract (ZH)**: 一种传统的LLM去学习设置包括两个子集——“遗忘”和“保留”，目标是从遗忘集中移除不需要的知识，同时保留剩余的知识。在以隐私为中心的去学习研究中，保留集通常进一步分成邻居集，包含直接或间接与遗忘目标连接的知识；并通过一般知识集进行扩充。现有基准中的一种常见做法是仅使用一个邻居集，这无法反映现实世界数据的复杂性和关系。LLM去学习通常涉及一对一采样或循环迭代采样。然而，这两种现成标准的有效性和稳定性尚未受到严格检验。在此研究中，我们系统地评估了这些常见做法。我们的发现表明，依赖单一邻居集是次优的，且标准采样方法会掩盖性能权衡。基于此分析，我们提出并验证了一套初步的最佳实践：（1）包含多样化的邻居集以平衡遗忘效果和模型实用性；（2）标准的一对一采样方法效率低下且结果不佳；（3）我们提出的模块化实体级去学习（MELU）策略作为循环采样的替代方案。我们证明，这种模块化方法结合稳健的算法，为有效的去学习提供了一条清晰且稳定的道路。 

---
# Evaluation of Large Language Models for Anomaly Detection in Autonomous Vehicles 

**Title (ZH)**: 大型语言模型在自动驾驶车辆异常检测中的评估 

**Authors**: Petros Loukas, David Bassir, Savvas Chatzichristofis, Angelos Amanatiadis  

**Link**: [PDF](https://arxiv.org/pdf/2509.05315)  

**Abstract**: The rapid evolution of large language models (LLMs) has pushed their boundaries to many applications in various domains. Recently, the research community has started to evaluate their potential adoption in autonomous vehicles and especially as complementary modules in the perception and planning software stacks. However, their evaluation is limited in synthetic datasets or manually driving datasets without the ground truth knowledge and more precisely, how the current perception and planning algorithms would perform in the cases under evaluation. For this reason, this work evaluates LLMs on real-world edge cases where current autonomous vehicles have been proven to fail. The proposed architecture consists of an open vocabulary object detector coupled with prompt engineering and large language model contextual reasoning. We evaluate several state-of-the-art models against real edge cases and provide qualitative comparison results along with a discussion on the findings for the potential application of LLMs as anomaly detectors in autonomous vehicles. 

**Abstract (ZH)**: 大型语言模型的快速进化已经将其边界推向了多个领域应用。最近，研究社区开始评估其在自主车辆中的潜在应用，特别是在感知和规划软件栈中的补充模块。然而，这类评估主要局限于合成数据集或手动驾驶数据集，缺乏真正的知识验证，尤其在评估当前感知和规划算法在这些情况下表现时更为明显。因此，本研究评估了大型语言模型在现实世界中的边缘案例，这些案例是当前自主车辆已被证明无法处理的。提出的架构结合了开放式词汇对象检测器、提示工程和大型语言模型上下文推理。我们评估了几种最先进的模型在现实边缘案例中的表现，并提供了定性比较结果以及对大型语言模型作为自主车辆异常检测器潜在应用的研究发现讨论。 

---
# ManipDreamer3D : Synthesizing Plausible Robotic Manipulation Video with Occupancy-aware 3D Trajectory 

**Title (ZH)**: ManipDreamer3D：基于占用感知的3D轨迹合成可信机器人操作视频 

**Authors**: Ying Li, Xiaobao Wei, Xiaowei Chi, Yuming Li, Zhongyu Zhao, Hao Wang, Ningning Ma, Ming Lu, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05314)  

**Abstract**: Data scarcity continues to be a major challenge in the field of robotic manipulation. Although diffusion models provide a promising solution for generating robotic manipulation videos, existing methods largely depend on 2D trajectories, which inherently face issues with 3D spatial ambiguity. In this work, we present a novel framework named ManipDreamer3D for generating plausible 3D-aware robotic manipulation videos from the input image and the text instruction. Our method combines 3D trajectory planning with a reconstructed 3D occupancy map created from a third-person perspective, along with a novel trajectory-to-video diffusion model. Specifically, ManipDreamer3D first reconstructs the 3D occupancy representation from the input image and then computes an optimized 3D end-effector trajectory, minimizing path length while avoiding collisions. Next, we employ a latent editing technique to create video sequences from the initial image latent and the optimized 3D trajectory. This process conditions our specially trained trajectory-to-video diffusion model to produce robotic pick-and-place videos. Our method generates robotic videos with autonomously planned plausible 3D trajectories, significantly reducing human intervention requirements. Experimental results demonstrate superior visual quality compared to existing methods. 

**Abstract (ZH)**: 基于3D感知的从文本指令生成可信机器人操纵视频的新框架ManipDreamer3D 

---
# Large Language Model Integration with Reinforcement Learning to Augment Decision-Making in Autonomous Cyber Operations 

**Title (ZH)**: 大型语言模型与强化学习集成以增强自主网络操作中的决策制定 

**Authors**: Konur Tholl, François Rivest, Mariam El Mezouar, Ranwa Al Mallah  

**Link**: [PDF](https://arxiv.org/pdf/2509.05311)  

**Abstract**: Reinforcement Learning (RL) has shown great potential for autonomous decision-making in the cybersecurity domain, enabling agents to learn through direct environment interaction. However, RL agents in Autonomous Cyber Operations (ACO) typically learn from scratch, requiring them to execute undesirable actions to learn their consequences. In this study, we integrate external knowledge in the form of a Large Language Model (LLM) pretrained on cybersecurity data that our RL agent can directly leverage to make informed decisions. By guiding initial training with an LLM, we improve baseline performance and reduce the need for exploratory actions with obviously negative outcomes. We evaluate our LLM-integrated approach in a simulated cybersecurity environment, and demonstrate that our guided agent achieves over 2x higher rewards during early training and converges to a favorable policy approximately 4,500 episodes faster than the baseline. 

**Abstract (ZH)**: 强化学习（RL）在网络安全自主决策领域展现了巨大潜力，使代理能够通过直接与环境交互来学习。然而，自主网络操作（ACO）中的RL代理通常需要从零开始学习，必须执行不 desirable 的行为来学习其后果。本研究中，我们通过一种预训练于网络安全数据的大规模语言模型（LLM）引入外部知识，让RL代理能够直接利用这些知识做出知情决策。通过使用LLM引导初始训练，我们提高了基线性能，并减少了执行显然具有负面后果的探索性行为的需要。我们在模拟的网络安全环境中评估了我们的LLM集成方法，并证明我们的引导代理在早期训练期间获得的奖励超过基线的2倍以上，并且在大约4,500个回合内收敛到有利策略的速度也快于基线。 

---
# ProtSAE: Disentangling and Interpreting Protein Language Models via Semantically-Guided Sparse Autoencoders 

**Title (ZH)**: ProtSAE: 通过语义导向的稀疏自编码器解耦与解释蛋白质语言模型 

**Authors**: Xiangyu Liu, Haodi Lei, Yi Liu, Yang Liu, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05309)  

**Abstract**: Sparse Autoencoder (SAE) has emerged as a powerful tool for mechanistic interpretability of large language models. Recent works apply SAE to protein language models (PLMs), aiming to extract and analyze biologically meaningful features from their latent spaces. However, SAE suffers from semantic entanglement, where individual neurons often mix multiple nonlinear concepts, making it difficult to reliably interpret or manipulate model behaviors. In this paper, we propose a semantically-guided SAE, called ProtSAE. Unlike existing SAE which requires annotation datasets to filter and interpret activations, we guide semantic disentanglement during training using both annotation datasets and domain knowledge to mitigate the effects of entangled attributes. We design interpretability experiments showing that ProtSAE learns more biologically relevant and interpretable hidden features compared to previous methods. Performance analyses further demonstrate that ProtSAE maintains high reconstruction fidelity while achieving better results in interpretable probing. We also show the potential of ProtSAE in steering PLMs for downstream generation tasks. 

**Abstract (ZH)**: Sparse 自编码器（SAE）已成为大型语言模型机制可解释性的强大工具。近期研究表明，SAE 可应用于蛋白质语言模型（PLMs），旨在从其潜在空间中提取和分析生物上有意义的特征。然而，SAE 存在语义纠缠问题，其中单个神经元常混杂多种非线性概念，使得模型行为的可靠解释和操控变得困难。在本文中，我们提出了一种基于语义指导的 SAE，称为 ProtSAE。与现有方法需要注释数据集来筛选和解释激活不同，我们在训练过程中利用注释数据集和领域知识来指导语义解纠缠，从而减轻纠缠属性的影响。通过解释性实验，我们表明 ProtSAE 能够学习到比以往方法更生物相关且易于解释的隐藏特征。性能分析进一步证实，ProtSAE 在保持高重建保真度的同时，在可解释探针任务上取得了更好的表现。我们还展示了 ProtSAE 在下游生成任务中引导 PLMs 的潜在应用。 

---
# Towards Log Analysis with AI Agents: Cowrie Case Study 

**Title (ZH)**: 基于AI代理的日志分析：Cowrie案例研究 

**Authors**: Enis Karaarslan, Esin Güler, Efe Emir Yüce, Cagatay Coban  

**Link**: [PDF](https://arxiv.org/pdf/2509.05306)  

**Abstract**: The scarcity of real-world attack data significantly hinders progress in cybersecurity research and education. Although honeypots like Cowrie effectively collect live threat intelligence, they generate overwhelming volumes of unstructured and heterogeneous logs, rendering manual analysis impractical. As a first step in our project on secure and efficient AI automation, this study explores the use of AI agents for automated log analysis. We present a lightweight and automated approach to process Cowrie honeypot logs. Our approach leverages AI agents to intelligently parse, summarize, and extract insights from raw data, while also considering the security implications of deploying such an autonomous system. Preliminary results demonstrate the pipeline's effectiveness in reducing manual effort and identifying attack patterns, paving the way for more advanced autonomous cybersecurity analysis in future work. 

**Abstract (ZH)**: 实物攻击数据的稀缺性显著妨碍了网络安全研究和教育的进步。尽管像Cowrie这样的蜜罐能够有效收集实时威胁情报，但它们生成的大量未结构化和异构日志使得手动分析变得不切实际。作为我们项目中安全高效AI自动化的第一步，本研究探索了使用AI代理进行自动日志分析的方法。我们提出了一种轻量级的自动化方法来处理Cowrie蜜罐日志。该方法利用AI代理智能地解析、总结和从原始数据中提取洞察，同时考虑部署此类自主系统的安全影响。初步结果显示，该管道在减少手动工作量并识别攻击模式方面具有有效性，为进一步的自主网络安全分析奠定了基础。 

---
# Multi-IaC-Eval: Benchmarking Cloud Infrastructure as Code Across Multiple Formats 

**Title (ZH)**: Multi-IaC-Eval：多种格式下云基础设施即代码的基准测试 

**Authors**: Sam Davidson, Li Sun, Bhavana Bhasker, Laurent Callot, Anoop Deoras  

**Link**: [PDF](https://arxiv.org/pdf/2509.05303)  

**Abstract**: Infrastructure as Code (IaC) is fundamental to modern cloud computing, enabling teams to define and manage infrastructure through machine-readable configuration files. However, different cloud service providers utilize diverse IaC formats. The lack of a standardized format requires cloud architects to be proficient in multiple IaC languages, adding complexity to cloud deployment. While Large Language Models (LLMs) show promise in automating IaC creation and maintenance, progress has been limited by the lack of comprehensive benchmarks across multiple IaC formats. We present Multi-IaC-Bench, a novel benchmark dataset for evaluating LLM-based IaC generation and mutation across AWS CloudFormation, Terraform, and Cloud Development Kit (CDK) formats. The dataset consists of triplets containing initial IaC templates, natural language modification requests, and corresponding updated templates, created through a synthetic data generation pipeline with rigorous validation. We evaluate several state-of-the-art LLMs on Multi-IaC-Bench, demonstrating that while modern LLMs can achieve high success rates (>95%) in generating syntactically valid IaC across formats, significant challenges remain in semantic alignment and handling complex infrastructure patterns. Our ablation studies highlight the importance of prompt engineering and retry mechanisms in successful IaC generation. We release Multi-IaC-Bench to facilitate further research in AI-assisted infrastructure management and establish standardized evaluation metrics for this crucial domain. 

**Abstract (ZH)**: 基于代码的基础设施（IaC）是现代云计算的基础，使团队能够通过机器可读的配置文件定义和管理基础设施。然而，不同的云服务提供商使用多种多样的IaC格式。缺乏标准格式需要云架构师精通多种IaC语言，从而增加了云部署的复杂性。虽然大型语言模型（LLMs）在自动化IaC创建和维护方面展现出潜力，但由于缺乏跨多种IaC格式的全面基准，进展受限。我们提出了一个多IaC基准（Multi-IaC-Bench），这是一个新型基准数据集，用于评估基于LLM的IaC生成和变异，涵盖AWS CloudFormation、Terraform和Cloud Development Kit（CDK）格式。该数据集由包含初始IaC模板、自然语言修改请求以及通过严格的合成数据生成管道创建的对应更新模板的 triplet 组成。我们使用多IaC基准对几种最先进的LLM进行评估，结果显示，现代LLM可以实现高成功率（>95%）的跨格式语法有效的IaC生成，但在语义对齐和处理复杂基础设施模式方面仍面临重大挑战。我们的消融研究突显了提示工程和重试机制在成功IaC生成中的重要性。我们发布了多IaC基准，以促进进一步的AI辅助基础设施管理研究，并建立这一关键领域的标准化评估指标。 

---
# Sesame: Opening the door to protein pockets 

**Title (ZH)**: 芝麻：开启蛋白质口袋的大门 

**Authors**: Raúl Miñán, Carles Perez-Lopez, Javier Iglesias, Álvaro Ciudad, Alexis Molina  

**Link**: [PDF](https://arxiv.org/pdf/2509.05302)  

**Abstract**: Molecular docking is a cornerstone of drug discovery, relying on high-resolution ligand-bound structures to achieve accurate predictions. However, obtaining these structures is often costly and time-intensive, limiting their availability. In contrast, ligand-free structures are more accessible but suffer from reduced docking performance due to pocket geometries being less suited for ligand accommodation in apo structures. Traditional methods for artificially inducing these conformations, such as molecular dynamics simulations, are computationally expensive. In this work, we introduce Sesame, a generative model designed to predict this conformational change efficiently. By generating geometries better suited for ligand accommodation at a fraction of the computational cost, Sesame aims to provide a scalable solution for improving virtual screening workflows. 

**Abstract (ZH)**: 分子对接是药物发现的基础，依赖于高分辨率的配体结合结构以实现准确的预测。然而，获取这些结构往往成本高且耗时，限制了它们的可用性。相比之下，配体自由结构更容易获取，但由于活性位点几何结构不适合配体容纳，导致对接性能降低。传统通过分子动力学模拟等人工诱导这些构象变化的方法计算成本高昂。在这项工作中，我们引入了Sesame，一个生成模型，旨在高效预测这种构象变化。通过以较低的计算成本生成更适合配体容纳的几何结构，Sesame旨在为改进虚拟筛选工作流程提供一种可扩展的解决方案。 

---
# Livia: An Emotion-Aware AR Companion Powered by Modular AI Agents and Progressive Memory Compression 

**Title (ZH)**: Liviana：一种基于模块化AI代理和渐进式内存压缩的情感感知AR伴侣 

**Authors**: Rui Xi, Xianghan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05298)  

**Abstract**: Loneliness and social isolation pose significant emotional and health challenges, prompting the development of technology-based solutions for companionship and emotional support. This paper introduces Livia, an emotion-aware augmented reality (AR) companion app designed to provide personalized emotional support by combining modular artificial intelligence (AI) agents, multimodal affective computing, progressive memory compression, and AR driven embodied interaction. Livia employs a modular AI architecture with specialized agents responsible for emotion analysis, dialogue generation, memory management, and behavioral orchestration, ensuring robust and adaptive interactions. Two novel algorithms-Temporal Binary Compression (TBC) and Dynamic Importance Memory Filter (DIMF)-effectively manage and prioritize long-term memory, significantly reducing storage requirements while retaining critical context. Our multimodal emotion detection approach achieves high accuracy, enhancing proactive and empathetic engagement. User evaluations demonstrated increased emotional bonds, improved satisfaction, and statistically significant reductions in loneliness. Users particularly valued Livia's adaptive personality evolution and realistic AR embodiment. Future research directions include expanding gesture and tactile interactions, supporting multi-user experiences, and exploring customized hardware implementations. 

**Abstract (ZH)**: 孤独和社会隔离对情感和健康造成重大挑战，促使开发基于技术的解决方案以提供陪伴和情感支持。本文介绍了Livia，一款情感感知增强现实（AR）伴侣应用程序，通过模块化人工智能量子、多模态情感计算、渐进式记忆压缩和AR驱动的实体互动相结合，提供个性化的情感支持。Livia采用模块化AI架构，各专门代理负责情绪分析、对话生成、记忆管理和行为编排，确保强大的适应性互动。两种新型算法——时间二进制压缩（TBC）和动态重要性记忆过滤器（DIMF）——有效管理并优先处理长期记忆，显著减少存储需求同时保留关键背景。我们的多模态情绪检测方法达到了高精度，增强了主动和同理心的参与度。用户评估显示，情感纽带增强，满意度提高，并且孤独感有统计学意义上的显著减少。用户特别重视Livia的适应性个性演变和逼真的AR表现。未来的研究方向包括扩展手势和触觉交互、支持多用户体验以及探索定制硬件实现。 

---
# Nonnegative matrix factorization and the principle of the common cause 

**Title (ZH)**: 非负矩阵分解与共同原因原则 

**Authors**: E. Khalafyan, A. E. Allahverdyan, A. Hovhannisyan  

**Link**: [PDF](https://arxiv.org/pdf/2509.03652)  

**Abstract**: Nonnegative matrix factorization (NMF) is a known unsupervised data-reduction method. The principle of the common cause (PCC) is a basic methodological approach in probabilistic causality, which seeks an independent mixture model for the joint probability of two dependent random variables. It turns out that these two concepts are closely related. This relationship is explored reciprocally for several datasets of gray-scale images, which are conveniently mapped into probability models. On one hand, PCC provides a predictability tool that leads to a robust estimation of the effective rank of NMF. Unlike other estimates (e.g., those based on the Bayesian Information Criteria), our estimate of the rank is stable against weak noise. We show that NMF implemented around this rank produces features (basis images) that are also stable against noise and against seeds of local optimization, thereby effectively resolving the NMF nonidentifiability problem. On the other hand, NMF provides an interesting possibility of implementing PCC in an approximate way, where larger and positively correlated joint probabilities tend to be explained better via the independent mixture model. We work out a clustering method, where data points with the same common cause are grouped into the same cluster. We also show how NMF can be employed for data denoising. 

**Abstract (ZH)**: 非负矩阵分解（NMF）与基本原因原则（PCC）的密切关系及其应用 

---
# LocoMamba: Vision-Driven Locomotion via End-to-End Deep Reinforcement Learning with Mamba 

**Title (ZH)**: LocoMamba：通过盲视运动的端到端深度强化学习 

**Authors**: Yinuo Wang, Gavin Tao  

**Link**: [PDF](https://arxiv.org/pdf/2508.11849)  

**Abstract**: We introduce LocoMamba, a vision-driven cross-modal DRL framework built on selective state-space models, specifically leveraging Mamba, that achieves near-linear-time sequence modeling, effectively captures long-range dependencies, and enables efficient training with longer sequences. First, we embed proprioceptive states with a multilayer perceptron and patchify depth images with a lightweight convolutional neural network, producing compact tokens that improve state representation. Second, stacked Mamba layers fuse these tokens via near-linear-time selective scanning, reducing latency and memory footprint, remaining robust to token length and image resolution, and providing an inductive bias that mitigates overfitting. Third, we train the policy end-to-end with Proximal Policy Optimization under terrain and appearance randomization and an obstacle-density curriculum, using a compact state-centric reward that balances progress, smoothness, and safety. We evaluate our method in challenging simulated environments with static and moving obstacles as well as uneven terrain. Compared with state-of-the-art baselines, our method achieves higher returns and success rates with fewer collisions, exhibits stronger generalization to unseen terrains and obstacle densities, and improves training efficiency by converging in fewer updates under the same compute budget. 

**Abstract (ZH)**: LocoMamba：一种基于选择性状态空间模型的视觉导向跨模态DRL框架 

---
# AdCare-VLM: Leveraging Large Vision Language Model (LVLM) to Monitor Long-Term Medication Adherence and Care 

**Title (ZH)**: AdCare-VLM: 利用大型ビジョン言語モデル监控长期用药依从性和护理 

**Authors**: Md Asaduzzaman Jabin, Hanqi Jiang, Yiwei Li, Patrick Kaggwa, Eugene Douglass, Juliet N. Sekandi, Tianming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00275)  

**Abstract**: Chronic diseases, including diabetes, hypertension, asthma, HIV-AIDS, epilepsy, and tuberculosis, necessitate rigorous adherence to medication to avert disease progression, manage symptoms, and decrease mortality rates. Adherence is frequently undermined by factors including patient behavior, caregiver support, elevated medical costs, and insufficient healthcare infrastructure. We propose AdCare-VLM, a specialized Video-LLaVA-based multimodal large vision language model (LVLM) aimed at visual question answering (VQA) concerning medication adherence through patient videos. We employ a private dataset comprising 806 custom-annotated tuberculosis (TB) medication monitoring videos, which have been labeled by clinical experts, to fine-tune the model for adherence pattern detection. We present LLM-TB-VQA, a detailed medical adherence VQA dataset that encompasses positive, negative, and ambiguous adherence cases. Our method identifies correlations between visual features, such as the clear visibility of the patient's face, medication, water intake, and the act of ingestion, and their associated medical concepts in captions. This facilitates the integration of aligned visual-linguistic representations and improves multimodal interactions. Experimental results indicate that our method surpasses parameter-efficient fine-tuning (PEFT) enabled VLM models, such as LLaVA-V1.5 and Chat-UniVi, with absolute improvements ranging from 3.1% to 3.54% across pre-trained, regular, and low-rank adaptation (LoRA) configurations. Comprehensive ablation studies and attention map visualizations substantiate our approach, enhancing interpretability. 

**Abstract (ZH)**: 慢性疾病（包括糖尿病、高血压、哮喘、HIV/AIDS、癫痫和结核病）需要严格遵守药物治疗以防止疾病进展、管理症状并降低死亡率。依从性常因患者行为、护理支持不足、医疗成本上升以及卫生基础设施不足等因素而受到阻碍。我们提出AdCare-VLM，这是一种专门基于Video-LLaVA的多模态大型视觉语言模型（LVLM），旨在通过患者的视频进行用药依从性的视觉问答（VQA）。我们使用包含806个由临床专家标注的结核病（TB）药物监控视频的私人数据集对模型进行微调，以检测用药依从性模式。我们提供了LLM-TB-VQA，这是一种详细的医疗依从性VQA数据集，包含了正面、负面和模棱两可的用药依从性案例。我们的方法识别了视觉特征（如患者面部、药物、饮水和吞咽等行为）与其相关医学概念之间的关联，促进了视觉-语言表示的对齐，并改善了多模态交互。实验结果表明，我们的方法优于参数高效微调（PEFT）启用的VLM模型，如LLaVA-V1.5和Chat-UniVi，绝对改进率在预训练、常规和低秩适应（LoRA）配置中分别为3.1%至3.54%。全面的消融研究和注意力图可视化证实了我们的方法，增强了可解释性。 

---
