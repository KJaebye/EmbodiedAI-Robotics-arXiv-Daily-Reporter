# Autonomous Vehicle Lateral Control Using Deep Reinforcement Learning with MPC-PID Demonstration 

**Title (ZH)**: 基于MPC-PID演示的自主车辆横向控制深度强化学习方法 

**Authors**: Chengdong Wu, Sven Kirchner, Nils Purschke, Alois C. Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2506.04040)  

**Abstract**: The controller is one of the most important modules in the autonomous driving pipeline, ensuring the vehicle reaches its desired position. In this work, a reinforcement learning based lateral control approach, despite the imperfections in the vehicle models due to measurement errors and simplifications, is presented. Our approach ensures comfortable, efficient, and robust control performance considering the interface between controlling and other modules. The controller consists of the conventional Model Predictive Control (MPC)-PID part as the basis and the demonstrator, and the Deep Reinforcement Learning (DRL) part which leverages the online information from the MPC-PID part. The controller's performance is evaluated in CARLA using the ground truth of the waypoints as inputs. Experimental results demonstrate the effectiveness of the controller when vehicle information is incomplete, and the training of DRL can be stabilized with the demonstration part. These findings highlight the potential to reduce development and integration efforts for autonomous driving pipelines in the future. 

**Abstract (ZH)**: 基于强化学习的车道控制方法：考虑控制与其他模块接口的舒适、高效和稳健性能 

---
# Optimizing Mesh to Improve the Triangular Expansion Algorithm for Computing Visibility Regions 

**Title (ZH)**: 优化网格以提高计算可视区域的三角扩展算法性能 

**Authors**: Jan Mikula, Miroslav Kulich  

**Link**: [PDF](https://arxiv.org/pdf/2506.04086)  

**Abstract**: This paper addresses the problem of improving the query performance of the triangular expansion algorithm (TEA) for computing visibility regions by finding the most advantageous instance of the triangular mesh, the preprocessing structure. The TEA recursively traverses the mesh while keeping track of the visible region, the set of all points visible from a query point in a polygonal world. We show that the measured query time is approximately proportional to the number of triangle edge expansions during the mesh traversal. We propose a new type of triangular mesh that minimizes the expected number of expansions assuming the query points are drawn from a known probability distribution. We design a heuristic method to approximate the mesh and evaluate the approach on many challenging instances that resemble real-world environments. The proposed mesh improves the mean query times by 12-16% compared to the reference constrained Delaunay triangulation. The approach is suitable to boost offline applications that require computing millions of queries without addressing the preprocessing time. The implementation is publicly available to replicate our experiments and serve the community. 

**Abstract (ZH)**: 本文解决了提高计算可视区域的三角扩展算法（TEA）查询性能的问题，通过找到最有利的三角网实例，即预处理结构。我们展示了测量的查询时间大约与mesh遍历时的三角边扩展次数成正比。我们提出了一种新的三角网类型，该类型在查询点来自已知概率分布的情况下，最小化了预期的扩展次数。我们设计了一种启发式方法来近似三角网，并在许多类似于真实环境的挑战性实例上评估了该方法。与参考的约束Delaunay三角剖分相比，所提出的三角网将平均查询时间改善了12-16%。该方法适用于需要在不考虑预处理时间的情况下计算数百万查询的离线应用。实现已公开，可供复制我们的实验并服务于社区使用。 

---
# Does Thinking More always Help? Understanding Test-Time Scaling in Reasoning Models 

**Title (ZH)**: 思考更多always有助于提高吗？理解推理模型的测试时缩放问题 

**Authors**: Soumya Suvra Ghosal, Souradip Chakraborty, Avinash Reddy, Yifu Lu, Mengdi Wang, Dinesh Manocha, Furong Huang, Mohammad Ghavamzadeh, Amrit Singh Bedi  

**Link**: [PDF](https://arxiv.org/pdf/2506.04210)  

**Abstract**: Recent trends in test-time scaling for reasoning models (e.g., OpenAI o1, DeepSeek R1) have led to a popular belief that extending thinking traces using prompts like "Wait" or "Let me rethink" can improve performance. This raises a natural question: Does thinking more at test-time truly lead to better reasoning? To answer this question, we perform a detailed empirical study across models and benchmarks, which reveals a consistent pattern of initial performance improvements from additional thinking followed by a decline, due to "overthinking". To understand this non-monotonic trend, we consider a simple probabilistic model, which reveals that additional thinking increases output variance-creating an illusion of improved reasoning while ultimately undermining precision. Thus, observed gains from "more thinking" are not true indicators of improved reasoning, but artifacts stemming from the connection between model uncertainty and evaluation metric. This suggests that test-time scaling through extended thinking is not an effective way to utilize the inference thinking budget. Recognizing these limitations, we introduce an alternative test-time scaling approach, parallel thinking, inspired by Best-of-N sampling. Our method generates multiple independent reasoning paths within the same inference budget and selects the most consistent response via majority vote, achieving up to 20% higher accuracy compared to extended thinking. This provides a simple yet effective mechanism for test-time scaling of reasoning models. 

**Abstract (ZH)**: Recent trends in 测试时扩展推理模型（例如OpenAI o1、DeepSeek R1）的最新进展导致一种普遍 belief 认为，使用“等待”或“让我重新思考”等提示扩展思考轨迹可以提高性能。这引发了一个自然问题：测试时更多的思考真的能提高推理能力吗？为了回答这个问题，我们在不同模型和基准上进行了详细的实证研究，揭示了初始性能提高后随后下降的一致模式，原因是“过度思考”。为了理解这一非单调趋势，我们考虑了一个简单的概率模型，揭示了额外思考增加了输出的变异性——创造出改进推理的错觉，实际上却削弱了精度。因此，“更多思考”所观察到的收益并不是改进推理的真实指标，而是源自模型不确定性与评估指标之间的联系。这表明通过扩展思考进行测试时扩展推理模型的推理预算并不是一种有效的方法。认识这些局限性，我们引入了一种替代的测试时扩展方法，即并行思考，灵感来自于Best-of-N采样方法。该方法在相同的推理预算内生成多个独立的推理路径，并通过多数投票选择最一致的响应，相较于扩展思考，可以实现高达20%的更高准确率。这提供了一种简单而有效的测试时扩展推理模型的方法。 

---
# macOSWorld: A Multilingual Interactive Benchmark for GUI Agents 

**Title (ZH)**: macOS世界：多语言交互基准测试用于GUI代理 

**Authors**: Pei Yang, Hai Ci, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2506.04135)  

**Abstract**: Graphical User Interface (GUI) agents show promising capabilities for automating computer-use tasks and facilitating accessibility, but existing interactive benchmarks are mostly English-only, covering web-use or Windows, Linux, and Android environments, but not macOS. macOS is a major OS with distinctive GUI patterns and exclusive applications. To bridge the gaps, we present macOSWorld, the first comprehensive benchmark for evaluating GUI agents on macOS. macOSWorld features 202 multilingual interactive tasks across 30 applications (28 macOS-exclusive), with task instructions and OS interfaces offered in 5 languages (English, Chinese, Arabic, Japanese, and Russian). As GUI agents are shown to be vulnerable to deception attacks, macOSWorld also includes a dedicated safety benchmarking subset. Our evaluation on six GUI agents reveals a dramatic gap: proprietary computer-use agents lead at above 30% success rate, while open-source lightweight research models lag at below 2%, highlighting the need for macOS domain adaptation. Multilingual benchmarks also expose common weaknesses, especially in Arabic, with a 27.5% average degradation compared to English. Results from safety benchmarking also highlight that deception attacks are more general and demand immediate attention. macOSWorld is available at this https URL. 

**Abstract (ZH)**: macOSWorld: 针对 macOS 的首个全面图形用户界面代理基准 

---
# Interpretability by Design for Efficient Multi-Objective Reinforcement Learning 

**Title (ZH)**: 设计中的可解释性以实现高效的多目标强化学习 

**Authors**: Qiyue Xia, J. Michael Herrmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.04022)  

**Abstract**: Multi-objective reinforcement learning (MORL) aims at optimising several, often conflicting goals in order to improve flexibility and reliability of RL in practical tasks. This can be achieved by finding diverse policies that are optimal for some objective preferences and non-dominated by optimal policies for other preferences so that they form a Pareto front in the multi-objective performance space. The relation between the multi-objective performance space and the parameter space that represents the policies is generally non-unique. Using a training scheme that is based on a locally linear map between the parameter space and the performance space, we show that an approximate Pareto front can provide an interpretation of the current parameter vectors in terms of the objectives which enables an effective search within contiguous solution domains. Experiments are conducted with and without retraining across different domains, and the comparison with previous methods demonstrates the efficiency of our approach. 

**Abstract (ZH)**: 多目标强化学习（MORL）旨在优化多个常常互相冲突的目标，以提高强化学习在实际任务中的灵活性和可靠性。这可以通过找到针对某些目标偏好最优且不劣于其他偏好最优策略的多样策略来实现，使得它们在多目标performance空间中形成Pareto前沿。多目标performance空间与表示策略的参数空间之间的关系通常是非唯一的。通过基于参数空间与performance空间之间的局部线性映射的训练方案，我们展示了近似Pareto前沿可以解释当前参数向量在目标方面的意义，并能够在连续的解域内进行有效搜索。在不同领域进行了带有和不带重新训练的实验，与先前方法的比较表明了我们方法的效率。 

---
# A framework for Conditional Reasoning in Answer Set Programming 

**Title (ZH)**: 条件推理在回答集编程中的框架 

**Authors**: Mario Alviano, Laura Giordano, Daniele Theseider Dupré  

**Link**: [PDF](https://arxiv.org/pdf/2506.03997)  

**Abstract**: In this paper we introduce a Conditional Answer Set Programming framework (Conditional ASP) for the definition of conditional extensions of Answer Set Programming (ASP). The approach builds on a conditional logic with typicality, and on the combination of a conditional knowledge base with an ASP program, and allows for conditional reasoning over the answer sets of the program. The formalism relies on a multi-preferential semantics (and on the KLM preferential semantics, as a special case) to provide an interpretation of conditionals. 

**Abstract (ZH)**: 本论文介绍了一种条件回答集编程框架（Conditional ASP）用于回答集编程（ASP）的条件扩展的定义。该方法基于典型性的条件逻辑，并结合条件知识库与ASP程序，允许对程序的回答集进行条件推理。该形式主义依赖多偏好语义（以及作为特殊情况的KLM偏好语义）来解释条件。 

---
# Causal Explanations Over Time: Articulated Reasoning for Interactive Environments 

**Title (ZH)**: 时间维度上的因果解释：交互式环境中的 articulate 推理 

**Authors**: Sebastian Rödling, Matej Zečević, Devendra Singh Dhami, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2506.03915)  

**Abstract**: Structural Causal Explanations (SCEs) can be used to automatically generate explanations in natural language to questions about given data that are grounded in a (possibly learned) causal model. Unfortunately they work for small data only. In turn they are not attractive to offer reasons for events, e.g., tracking causal changes over multiple time steps, or a behavioral component that involves feedback loops through actions of an agent. To this end, we generalize SCEs to a (recursive) formulation of explanation trees to capture the temporal interactions between reasons. We show the benefits of this more general SCE algorithm on synthetic time-series data and a 2D grid game, and further compare it to the base SCE and other existing methods for causal explanations. 

**Abstract (ZH)**: 结构因果解释（SCEs）可以用于基于可能学习到的因果模型自动生成关于给定数据的自然语言解释。然而，它们仅适用于小数据集。因此，它们不适用于提供事件原因，例如，在多个时间步骤中跟踪因果变化，或涉及代理行为的反馈环的行为组件。为此，我们将SCEs推广为解释树的递归形式，以捕捉原因之间的时序交互。我们在合成时间序列数据和一个2D网格游戏中展示了这一更通用的SCE算法的优势，并将其与基础SCE和其他现有的因果解释方法进行了比较。 

---
# AssetOpsBench: Benchmarking AI Agents for Task Automation in Industrial Asset Operations and Maintenance 

**Title (ZH)**: AssetOpsBench: 工业资产运维任务自动化工智能体基准测试 

**Authors**: Dhaval Patel, Shuxin Lin, James Rayfield, Nianjun Zhou, Roman Vaculin, Natalia Martinez, Fearghal O'donncha, Jayant Kalagnanam  

**Link**: [PDF](https://arxiv.org/pdf/2506.03828)  

**Abstract**: AI for Industrial Asset Lifecycle Management aims to automate complex operational workflows -- such as condition monitoring, maintenance planning, and intervention scheduling -- to reduce human workload and minimize system downtime. Traditional AI/ML approaches have primarily tackled these problems in isolation, solving narrow tasks within the broader operational pipeline. In contrast, the emergence of AI agents and large language models (LLMs) introduces a next-generation opportunity: enabling end-to-end automation across the entire asset lifecycle. This paper envisions a future where AI agents autonomously manage tasks that previously required distinct expertise and manual coordination. To this end, we introduce AssetOpsBench -- a unified framework and environment designed to guide the development, orchestration, and evaluation of domain-specific agents tailored for Industry 4.0 applications. We outline the key requirements for such holistic systems and provide actionable insights into building agents that integrate perception, reasoning, and control for real-world industrial operations. The software is available at this https URL. 

**Abstract (ZH)**: 工业资产生命周期管理中的AI旨在自动化复杂的运营工作流——例如状态监测、维护计划和干预调度——以减少人力负担并最小化系统停机时间。传统的AI/ML方法主要在孤立的情况下解决这些问题，专注于运营管道中的窄任务。相比之下，AI代理和大型语言模型（LLMs）的出现为端到端的自动化提供了新一代机会：在整个资产生命周期中实现全流程自动化。本文构想了一个未来，在这个未来中，AI代理能够自主管理需要特定专业知识和手动协调的任务。为此，我们提出了一种统一框架和环境——AssetOpsBench——旨在指导适用于工业4.0应用的领域特定代理的研发、编排和评估。我们概述了这类整体系统的关键要求，并提供了如何构建能够集成感知、推理和控制的代理以适应真实工业运营环境的具体建议。该软件可在以下链接访问：this https URL。 

---
# Joint Beamforming and Resource Allocation for Delay Optimization in RIS-Assisted OFDM Systems: A DRL Approach 

**Title (ZH)**: 基于RIS辅助OFDM系统的延迟优化联合波束forming和资源分配：一种深度强化学习方法 

**Authors**: Yu Ma, Chongtao Guo, Le Liang, Xiao Li, Shi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.03586)  

**Abstract**: This paper investigates a joint phase design and resource allocation problem in downlink reconfigurable intelligent surface (RIS)-assisted orthogonal frequency division multiplexing (OFDM) systems to optimize average delay, where data packets for each user arrive at the base station stochastically. The sequential optimization problem is inherently a Markov decision process (MDP), making it fall within the scope of reinforcement learning. To effectively handle the mixed action space and reduce the state space dimensionality, a hybrid deep reinforcement learning (DRL) approach is proposed. Specifically, proximal policy optimization (PPO)-$\Theta$ is employed to optimize RIS phase shift design, while PPO-N is responsible for subcarrier allocation decisions. To further mitigate the curse of dimensionality associated with subcarrier allocation, a multi-agent strategy is introduced to optimize subcarrier allocation indicater more efficiently. Moreover, to achieve more adaptive resource allocation and accurately capture network dynamics, key factors closely related to average delay, including the number of backlogged packets in buffers and the current packet arrivals, are incorporated into the state space. Furthermore, a transfer learning framework is introduced to enhance training efficiency and accelerate convergence. Simulation results demonstrate that the proposed algorithm significantly reduces average delay, enhances resource allocation efficiency, and achieves superior system robustness and fairness compared to baseline methods. 

**Abstract (ZH)**: 基于重构智能表面辅助OFDM系统下行链路的联合相位设计与资源分配优化算法 

---
# SUMO-MCP: Leveraging the Model Context Protocol for Autonomous Traffic Simulation and Optimization 

**Title (ZH)**: SUMO-MCP：利用模型上下文协议进行自主交通模拟与优化 

**Authors**: Chenglong Ye, Gang Xiong, Junyou Shang, Xingyuan Dai, Xiaoyan Gong, Yisheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2506.03548)  

**Abstract**: Traffic simulation tools, such as SUMO, are essential for urban mobility research. However, such tools remain challenging for users due to complex manual workflows involving network download, demand generation, simulation setup, and result analysis. In this paper, we introduce SUMO-MCP, a novel platform that not only wraps SUMO' s core utilities into a unified tool suite but also provides additional auxiliary utilities for common preprocessing and postprocessing tasks. Using SUMO-MCP, users can issue simple natural-language prompts to generate traffic scenarios from OpenStreetMap data, create demand from origin-destination matrices or random patterns, run batch simulations with multiple signal-control strategies, perform comparative analyses with automated reporting, and detect congestion for signal-timing optimization. Furthermore, the platform allows flexible custom workflows by dynamically combining exposed SUMO tools without additional coding. Experiments demonstrate that SUMO-MCP significantly makes traffic simulation more accessible and reliable for researchers. We will release code for SUMO-MCP at this https URL in the future. 

**Abstract (ZH)**: 基于SUMO-MCP的交通仿真平台：使交通仿真更易于研究人员使用和可靠 

---
# Verification-Guided Falsification for Safe RL via Explainable Abstraction and Risk-Aware Exploration 

**Title (ZH)**: 基于可解释抽象和风险意识探索的指导验证反验证方法以实现安全的RL 

**Authors**: Tuan Le, Risal Shefin, Debashis Gupta, Thai Le, Sarra Alqahtani  

**Link**: [PDF](https://arxiv.org/pdf/2506.03469)  

**Abstract**: Ensuring the safety of reinforcement learning (RL) policies in high-stakes environments requires not only formal verification but also interpretability and targeted falsification. While model checking provides formal guarantees, its effectiveness is limited by abstraction quality and the completeness of the underlying trajectory dataset. We propose a hybrid framework that integrates (1) explainability, (2) model checking, and (3) risk-guided falsification to achieve both rigor and coverage. Our approach begins by constructing a human-interpretable abstraction of the RL policy using Comprehensible Abstract Policy Summarization (CAPS). This abstract graph, derived from offline trajectories, is both verifier-friendly, semantically meaningful, and can be used as input to Storm probabilistic model checker to verify satisfaction of temporal safety specifications. If the model checker identifies a violation, it will return an interpretable counterexample trace by which the policy fails the safety requirement. However, if no violation is detected, we cannot conclude satisfaction due to potential limitation in the abstraction and coverage of the offline dataset. In such cases, we estimate associated risk during model checking to guide a falsification strategy that prioritizes searching in high-risk states and regions underrepresented in the trajectory dataset. We further provide PAC-style guarantees on the likelihood of uncovering undetected violations. Finally, we incorporate a lightweight safety shield that switches to a fallback policy at runtime when such a risk exceeds a threshold, facilitating failure mitigation without retraining. 

**Abstract (ZH)**: 确保高风险环境中强化学习策略的安全性不仅需要形式验证，还需要可解释性和靶向反证。我们提出了一种混合框架，该框架整合了（1）可解释性、（2）模型检查和（3）风险引导下的反证，以实现严谨性和覆盖率的结合。该方法首先使用可解释抽象政策总结（CAPS）构建一个符合人类可解释性的RL策略抽象图，该抽象图来源于离线轨迹数据，并具有语义意义，可作为Storm概率模型检查器的输入，用于验证时间安全规范的满足情况。如果模型检查器检测到违规行为，它将返回一个可解释的反例轨迹，表明政策未能满足安全要求。然而，如果没有检测到违规行为，由于离线数据集的潜在抽象和覆盖限制，我们不能断言其满足情况。在这种情况下，我们在模型检查过程中估计相关风险，以指导一个优先搜索高风险状态和轨迹数据中未充分代表的区域的反证策略。我们还提供了关于未检测违规行为可能性的PAC风格保证。最后，我们引入了一个轻量级的安全防护，当这种风险超过阈值时，运行时切换到备用策略，从而在无需重新训练的情况下进行故障缓解。 

---
# Axiomatics of Restricted Choices by Linear Orders of Sets with Minimum as Fallback 

**Title (ZH)**: 集族中以最小值为后备的线性顺序的限制选择公理 

**Authors**: Kai Sauerwald, Kenneth Skiba, Eduardo Fermé, Thomas Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2506.03315)  

**Abstract**: We study how linear orders can be employed to realise choice functions for which the set of potential choices is restricted, i.e., the possible choice is not possible among the full powerset of all alternatives. In such restricted settings, constructing a choice function via a relation on the alternatives is not always possible. However, we show that one can always construct a choice function via a linear order on sets of alternatives, even when a fallback value is encoded as the minimal element in the linear order. The axiomatics of such choice functions are presented for the general case and the case of union-closed input restrictions. Restricted choice structures have applications in knowledge representation and reasoning, and here we discuss their applications for theory change and abstract argumentation. 

**Abstract (ZH)**: 我们研究线序如何用于实现潜在选择集受限的选择函数，即可能的选择在所有替代方案的全体幂集上未必存在。在这样的受限设置中，通过替代方案上的关系构建选择函数并不总是可能的。然而，我们证明即使将备用值编码为线序中的最小元素，也可以通过替代方案集上的线序总是构建出选择函数。此类选择函数的公理化在一般情况和并封闭输入限制情况下被提出。受限选择结构在知识表示与推理中有应用，并讨论了它们在理论变更和抽象论辩中的应用。 

---
# A Trustworthiness-based Metaphysics of Artificial Intelligence Systems 

**Title (ZH)**: 基于可信性的形而上学人工智能系统 

**Authors**: Andrea Ferrario  

**Link**: [PDF](https://arxiv.org/pdf/2506.03233)  

**Abstract**: Modern AI systems are man-made objects that leverage machine learning to support our lives across a myriad of contexts and applications. Despite extensive epistemological and ethical debates, their metaphysical foundations remain relatively under explored. The orthodox view simply suggests that AI systems, as artifacts, lack well-posed identity and persistence conditions -- their metaphysical kinds are no real kinds. In this work, we challenge this perspective by introducing a theory of metaphysical identity of AI systems. We do so by characterizing their kinds and introducing identity criteria -- formal rules that answer the questions "When are two AI systems the same?" and "When does an AI system persist, despite change?" Building on Carrara and Vermaas' account of fine-grained artifact kinds, we argue that AI trustworthiness provides a lens to understand AI system kinds and formalize the identity of these artifacts by relating their functional requirements to their physical make-ups. The identity criteria of AI systems are determined by their trustworthiness profiles -- the collection of capabilities that the systems must uphold over time throughout their artifact histories, and their effectiveness in maintaining these capabilities. Our approach suggests that the identity and persistence of AI systems is sensitive to the socio-technical context of their design and utilization via their trustworthiness, providing a solid metaphysical foundation to the epistemological, ethical, and legal discussions about these artifacts. 

**Abstract (ZH)**: 现代AI系统的形上学身份理论：基于信任性的本质刻画和身份 criteri 

---
# Efficient Knowledge Editing via Minimal Precomputation 

**Title (ZH)**: 通过最小前置计算实现高效的知识编辑 

**Authors**: Akshat Gupta, Maochuan Lu, Thomas Hartvigsen, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2506.04226)  

**Abstract**: Knowledge editing methods like MEMIT are able to make data and compute efficient updates of factual knowledge by using a single sentence to update facts and their consequences. However, what is often overlooked is a "precomputation step", which requires a one-time but significant computational cost. The authors of MEMIT originally precompute approximately 44 million hidden vectors per edited layer, which requires a forward pass over 44 million tokens. For GPT-J (6B), this precomputation step takes 36 hours on a single GPU, while it takes approximately 40 hours for Llama2-7B. Additionally, this precomputation time grows with model size. In this paper, we show that this excessive computational cost is unnecessary. Knowledge editing using MEMIT and related methods, such as ROME and EMMET, can be performed by pre-computing a very small portion of the 44 million hidden vectors. We first present the theoretical minimum number of hidden vector precomputation required for solutions of these editing methods to exist. We then empirically show that knowledge editing using these methods can be done by pre-computing significantly fewer hidden vectors. Specifically, we show that the precomputation step can be done with less than 0.3% of the originally stipulated number of hidden vectors. This saves a significant amount of precomputation time and allows users to begin editing new models within a few minutes. 

**Abstract (ZH)**: 基于MEMIT的知识编辑方法能够通过单一句子更新事实及其后果，从而实现数据和计算的有效更新。然而，通常被忽视的是“预计算步驟”，这需要一次性的但显著的计算成本。MEMIT的作者最初为每个编辑层预计算约4400万个隐藏向量，这需要对4400万个标记进行前向传播。对于GPT-J（6B），这一预计算步骤在单个GPU上耗时36小时，而对于Llama2-7B，则大约需要40小时。此外，随着模型规模的增加，预计算时间也会增加。本文表明，这种过多的计算成本是不必要的。使用MEMIT及其相关方法（如ROME和EMMET）进行知识编辑可以通过预计算极小部分的4400万个隐藏向量来实现。我们首先给出了这些编辑方法解存在所需的理论最小隐藏向量预计算数量。然后实验证明，可以使用显著更少的隐藏向量进行知识编辑。具体来说，预计算步骤可以使用原定数量不到0.3%的隐藏向量来完成。这显著节省了预计算时间，并使用户能够在几分钟内开始编辑新的模型。 

---
# Thinking Beyond Visibility: A Near-Optimal Policy Framework for Locally Interdependent Multi-Agent MDPs 

**Title (ZH)**: 超越可见性思考：近最优的局部相互依存多智能体MDP策略框架 

**Authors**: Alex DeWeese, Guannan Qu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04215)  

**Abstract**: Decentralized Partially Observable Markov Decision Processes (Dec-POMDPs) are known to be NEXP-Complete and intractable to solve. However, for problems such as cooperative navigation, obstacle avoidance, and formation control, basic assumptions can be made about local visibility and local dependencies. The work DeWeese and Qu 2024 formalized these assumptions in the construction of the Locally Interdependent Multi-Agent MDP. In this setting, it establishes three closed-form policies that are tractable to compute in various situations and are exponentially close to optimal with respect to visibility. However, it is also shown that these solutions can have poor performance when the visibility is small and fixed, often getting stuck during simulations due to the so called "Penalty Jittering" phenomenon. In this work, we establish the Extended Cutoff Policy Class which is, to the best of our knowledge, the first non-trivial class of near optimal closed-form partially observable policies that are exponentially close to optimal with respect to the visibility for any Locally Interdependent Multi-Agent MDP. These policies are able to remember agents beyond their visibilities which allows them to perform significantly better in many small and fixed visibility settings, resolve Penalty Jittering occurrences, and under certain circumstances guarantee fully observable joint optimal behavior despite the partial observability. We also propose a generalized form of the Locally Interdependent Multi-Agent MDP that allows for transition dependence and extended reward dependence, then replicate our theoretical results in this setting. 

**Abstract (ZH)**: 分布式部分可观测马尔可夫决策过程（Dec-POMDPs）被称为NEXP-完全问题，并且难以求解。然而，对于诸如合作导航、避障和编队控制等问题，可以对局部可见性和局部依赖性做出基本假设。DeWeese和Qu 2024的工作在构建局部依存多智能体MDP的过程中，形式化了这些假设，并建立了三个可计算的闭式策略，在多种情况下这些策略是适用的，并且在可见性方面指数接近最优。然而，当可见性小时，这些解决方案也可能表现不佳，在模拟过程中常常因为所谓的“罚分抖动”现象而陷入困境。在本文中，我们建立了扩展的截止策略类，这是已知的第一个针对任何局部依存多智能体MDP在可见性方面指数接近最优的非平凡的近最优闭式部分可观测策略类。这些策略能够记住超出了其可见范围的代理，使其在许多小且固定的可见性设置中表现出色，可以解决罚分抖动问题，并在某些情况下即使在部分可观测的情况下也能保证完全可观测的联合最优行为。我们还提出了一种局部依存多智能体MDP的广义形式，允许转移依赖和扩展的奖励依赖性，并在该设置中复制了我们的理论结果。 

---
# MACS: Multi-Agent Reinforcement Learning for Optimization of Crystal Structures 

**Title (ZH)**: MACS：多智能体强化学习在晶体结构优化中的应用 

**Authors**: Elena Zamaraeva, Christopher M. Collins, George R. Darling, Matthew S. Dyer, Bei Peng, Rahul Savani, Dmytro Antypov, Vladimir V. Gusev, Judith Clymo, Paul G. Spirakis, Matthew J. Rosseinsky  

**Link**: [PDF](https://arxiv.org/pdf/2506.04195)  

**Abstract**: Geometry optimization of atomic structures is a common and crucial task in computational chemistry and materials design. Following the learning to optimize paradigm, we propose a new multi-agent reinforcement learning method called Multi-Agent Crystal Structure optimization (MACS) to address periodic crystal structure optimization. MACS treats geometry optimization as a partially observable Markov game in which atoms are agents that adjust their positions to collectively discover a stable configuration. We train MACS across various compositions of reported crystalline materials to obtain a policy that successfully optimizes structures from the training compositions as well as structures of larger sizes and unseen compositions, confirming its excellent scalability and zero-shot transferability. We benchmark our approach against a broad range of state-of-the-art optimization methods and demonstrate that MACS optimizes periodic crystal structures significantly faster, with fewer energy calculations, and the lowest failure rate. 

**Abstract (ZH)**: 多代理晶体结构优化（MACS）在周期性晶体结构优化中的应用 

---
# Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints 

**Title (ZH)**: 物理学约束流匹配：具有刚性约束的生成模型采样 

**Authors**: Utkarsh Utkarsh, Pengfei Cai, Alan Edelman, Rafael Gomez-Bombarelli, Christopher Vincent Rackauckas  

**Link**: [PDF](https://arxiv.org/pdf/2506.04171)  

**Abstract**: Deep generative models have recently been applied to physical systems governed by partial differential equations (PDEs), offering scalable simulation and uncertainty-aware inference. However, enforcing physical constraints, such as conservation laws (linear and nonlinear) and physical consistencies, remains challenging. Existing methods often rely on soft penalties or architectural biases that fail to guarantee hard constraints. In this work, we propose Physics-Constrained Flow Matching (PCFM), a zero-shot inference framework that enforces arbitrary nonlinear constraints in pretrained flow-based generative models. PCFM continuously guides the sampling process through physics-based corrections applied to intermediate solution states, while remaining aligned with the learned flow and satisfying physical constraints. Empirically, PCFM outperforms both unconstrained and constrained baselines on a range of PDEs, including those with shocks, discontinuities, and sharp features, while ensuring exact constraint satisfaction at the final solution. Our method provides a general framework for enforcing hard constraints in both scientific and general-purpose generative models, especially in applications where constraint satisfaction is essential. 

**Abstract (ZH)**: 基于物理约束的流匹配（PCFM）：预训练生成模型中任意非线性约束的零样本推理框架 

---
# Horizon Reduction Makes RL Scalable 

**Title (ZH)**: 水平缩减使强化学习更具可扩展性 

**Authors**: Seohong Park, Kevin Frans, Deepinder Mann, Benjamin Eysenbach, Aviral Kumar, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2506.04168)  

**Abstract**: In this work, we study the scalability of offline reinforcement learning (RL) algorithms. In principle, a truly scalable offline RL algorithm should be able to solve any given problem, regardless of its complexity, given sufficient data, compute, and model capacity. We investigate if and how current offline RL algorithms match up to this promise on diverse, challenging, previously unsolved tasks, using datasets up to 1000x larger than typical offline RL datasets. We observe that despite scaling up data, many existing offline RL algorithms exhibit poor scaling behavior, saturating well below the maximum performance. We hypothesize that the horizon is the main cause behind the poor scaling of offline RL. We empirically verify this hypothesis through several analysis experiments, showing that long horizons indeed present a fundamental barrier to scaling up offline RL. We then show that various horizon reduction techniques substantially enhance scalability on challenging tasks. Based on our insights, we also introduce a minimal yet scalable method named SHARSA that effectively reduces the horizon. SHARSA achieves the best asymptotic performance and scaling behavior among our evaluation methods, showing that explicitly reducing the horizon unlocks the scalability of offline RL. Code: this https URL 

**Abstract (ZH)**: 在这项工作中，我们研究了离线强化学习（RL）算法的扩展性。原则上，一个真正意义上的可扩展的离线RL算法应该能够在给定足够数据、计算能力和模型容量的情况下，解决任何复杂的问题。我们调查当前的离线RL算法在使用比标准离线RL数据集大1000倍的数据集时，是否以及如何达到这一承诺，特别是在多样的、具有挑战性的、之前未解决的任务上。我们发现，尽管增加了数据量，许多现有的离线RL算法依然表现出不良的扩展性，性能在远低于最大性能时就达到饱和。我们假设时间 horizon 是导致离线RL不良扩展性的主要原因。通过一系列分析实验，我们实证验证了这一假说，显示较长的时间 horizon 确实是扩展离线RL的一个根本障碍。随后，我们展示了各种时间 horizon 减少技术在具有挑战性的任务上显著提高了可扩展性。基于我们的洞察，我们还引入了一种简单而有效的可扩展方法SHARSA，该方法有效减少了时间 horizon。SHARSA 在我们的评估方法中实现了最好的渐近性能和扩展性表现，表明明确减少时间 horizon 突破了离线RL的可扩展性。代码：这个 https://这个链接Url。 

---
# Plant Bioelectric Early Warning Systems: A Five-Year Investigation into Human-Plant Electromagnetic Communication 

**Title (ZH)**: 植物生物电预警系统：五年来的人-植物电磁通信研究 

**Authors**: Peter A. Gloor  

**Link**: [PDF](https://arxiv.org/pdf/2506.04132)  

**Abstract**: We present a comprehensive investigation into plant bioelectric responses to human presence and emotional states, building on five years of systematic research. Using custom-built plant sensors and machine learning classification, we demonstrate that plants generate distinct bioelectric signals correlating with human proximity, emotional states, and physiological conditions. A deep learning model based on ResNet50 architecture achieved 97% accuracy in classifying human emotional states through plant voltage spectrograms, while control models with shuffled labels achieved only 30% accuracy. This study synthesizes findings from multiple experiments spanning 2020-2025, including individual recognition (66% accuracy), eurythmic gesture detection, stress prediction, and responses to human voice and movement. We propose that these phenomena represent evolved anti-herbivory early warning systems, where plants detect approaching animals through bioelectric field changes before physical contact. Our results challenge conventional understanding of plant sensory capabilities and suggest practical applications in agriculture, healthcare, and human-plant interaction research. 

**Abstract (ZH)**: 我们对植物在人类存在和情绪状态下的生物电响应进行了一项全面调查，基于2020年至2025年间五年系统研究。利用自定义植物传感器和机器学习分类，我们展示了植物生成与人类接近程度、情绪状态和生理条件相关的独特生物电信号。基于ResNet50架构的深度学习模型在通过植物电压光谱分类人类情绪状态方面达到了97%的准确率，而标签打乱的控制模型仅达到了30%的准确率。本研究综合了跨越2020年至2025年期间多项实验的发现，包括个体识别（准确率为66%）、韵律手势检测、压力预测以及对人类声音和运动的响应。我们提出这些现象代表了植物进化出的抗食草防御预警系统，在生物电场变化之前检测即将接近的动物。我们的结果挑战了对植物感觉能力的传统理解，并建议在农业、医疗保健以及人-植物交互研究中具有实际应用价值。 

---
# CLAIM: An Intent-Driven Multi-Agent Framework for Analyzing Manipulation in Courtroom Dialogues 

**Title (ZH)**: CLAIM：一种基于意图的多_agent框架用于法庭对话中的操纵分析 

**Authors**: Disha Sheshanarayana, Tanishka Magar, Ayushi Mittal, Neelam Chaplot  

**Link**: [PDF](https://arxiv.org/pdf/2506.04131)  

**Abstract**: Courtrooms are places where lives are determined and fates are sealed, yet they are not impervious to manipulation. Strategic use of manipulation in legal jargon can sway the opinions of judges and affect the decisions. Despite the growing advancements in NLP, its application in detecting and analyzing manipulation within the legal domain remains largely unexplored. Our work addresses this gap by introducing LegalCon, a dataset of 1,063 annotated courtroom conversations labeled for manipulation detection, identification of primary manipulators, and classification of manipulative techniques, with a focus on long conversations. Furthermore, we propose CLAIM, a two-stage, Intent-driven Multi-agent framework designed to enhance manipulation analysis by enabling context-aware and informed decision-making. Our results highlight the potential of incorporating agentic frameworks to improve fairness and transparency in judicial processes. We hope that this contributes to the broader application of NLP in legal discourse analysis and the development of robust tools to support fairness in legal decision-making. Our code and data are available at this https URL. 

**Abstract (ZH)**: 法庭是决定人们命运和封定前途的地方，但它们并非不受操纵的影响。在法律用语中战略性地使用操纵手段可以影响法官的观点并影响判决。尽管自然语言处理（NLP）取得了越来越大的进展，但在法律领域内检测和分析操纵的应用仍然鲜有研究。我们的工作通过引入包含1063个标注对话的LegalCon数据集来填补这一空白，该数据集涵盖了操纵检测、 primary操纵者识别以及操纵技术分类，并重点关注长对话。此外，我们提出了CLAIM框架，这是一个基于意图的两阶段多智能体系统，旨在通过促进情境感知和知情决策来增强操纵分析。我们的结果强调了整合行动者框架在提高司法过程公平性和透明度方面的潜力。我们希望这能够促进NLP在法律话语分析中的更广泛应用，并推动开发支持法律决策公平性的稳健工具。我们的代码和数据可在以下链接获取。 

---
# A Diffusion-Driven Temporal Super-Resolution and Spatial Consistency Enhancement Framework for 4D MRI imaging 

**Title (ZH)**: 基于扩散驱动的时空超分辨率和空间一致性增强框架用于4D MRI成像 

**Authors**: Xuanru Zhou, Jiarun Liu, Shoujun Yu, Hao Yang, Cheng Li, Tao Tan, Shanshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04116)  

**Abstract**: In medical imaging, 4D MRI enables dynamic 3D visualization, yet the trade-off between spatial and temporal resolution requires prolonged scan time that can compromise temporal fidelity--especially during rapid, large-amplitude motion. Traditional approaches typically rely on registration-based interpolation to generate intermediate frames. However, these methods struggle with large deformations, resulting in misregistration, artifacts, and diminished spatial consistency. To address these challenges, we propose TSSC-Net, a novel framework that generates intermediate frames while preserving spatial consistency. To improve temporal fidelity under fast motion, our diffusion-based temporal super-resolution network generates intermediate frames using the start and end frames as key references, achieving 6x temporal super-resolution in a single inference step. Additionally, we introduce a novel tri-directional Mamba-based module that leverages long-range contextual information to effectively resolve spatial inconsistencies arising from cross-slice misalignment, thereby enhancing volumetric coherence and correcting cross-slice errors. Extensive experiments were performed on the public ACDC cardiac MRI dataset and a real-world dynamic 4D knee joint dataset. The results demonstrate that TSSC-Net can generate high-resolution dynamic MRI from fast-motion data while preserving structural fidelity and spatial consistency. 

**Abstract (ZH)**: 医疗影像中的4D MRI 使动态3D可视化成为可能，但空间分辨率和时间分辨率之间的权衡需要延长扫描时间，尤其是在快速大振幅运动期间会损害时间保真度。传统方法通常依赖于基于注册的插值来生成中间帧，但这些方法在处理大变形时会遇到困难，导致错位、伪影和空间一致性减弱。为了解决这些问题，我们提出了TSSC-Net，这是一种新型框架，可以在保持空间一致性的前提下生成中间帧。为了在快速运动下提高时间保真度，我们的基于扩散的时间超级分辨率网络使用起始帧和结束帧作为关键参考来生成中间帧，实现单一推断步中的6倍时间超级分辨率。此外，我们引入了一种基于Mamba的新三向模块，利用长程上下文信息有效解决切片错位引起的空间不一致问题，从而增强体素相干性并纠正切片误差。在公开的ACDC心脏MRI数据集和实际快速运动的4D膝关节数据集上进行了广泛的实验。结果表明，TSSC-Net可以从快速运动数据中生成高分辨率动态MRI，同时保持结构保真度和空间一致性。 

---
# TextAtari: 100K Frames Game Playing with Language Agents 

**Title (ZH)**: TextAtari: 语言代理下的100K帧游戏玩法 

**Authors**: Wenhao Li, Wenwu Li, Chuyun Shen, Junjie Sheng, Zixiao Huang, Di Wu, Yun Hua, Wei Yin, Xiangfeng Wang, Hongyuan Zha, Bo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.04098)  

**Abstract**: We present TextAtari, a benchmark for evaluating language agents on very long-horizon decision-making tasks spanning up to 100,000 steps. By translating the visual state representations of classic Atari games into rich textual descriptions, TextAtari creates a challenging test bed that bridges sequential decision-making with natural language processing. The benchmark includes nearly 100 distinct tasks with varying complexity, action spaces, and planning horizons, all rendered as text through an unsupervised representation learning framework (AtariARI). We evaluate three open-source large language models (Qwen2.5-7B, Gemma-7B, and Llama3.1-8B) across three agent frameworks (zero-shot, few-shot chain-of-thought, and reflection reasoning) to assess how different forms of prior knowledge affect performance on these long-horizon challenges. Four scenarios-Basic, Obscured, Manual Augmentation, and Reference-based-investigate the impact of semantic understanding, instruction comprehension, and expert demonstrations on agent decision-making. Our results reveal significant performance gaps between language agents and human players in extensive planning tasks, highlighting challenges in sequential reasoning, state tracking, and strategic planning across tens of thousands of steps. TextAtari provides standardized evaluation protocols, baseline implementations, and a framework for advancing research at the intersection of language models and planning. 

**Abstract (ZH)**: TextAtari：一种用于评估语言代理在长达10万步的长期决策任务上的基准测试 

---
# Multimodal Tabular Reasoning with Privileged Structured Information 

**Title (ZH)**: 带特权结构信息的多模态表格推理 

**Authors**: Jun-Peng Jiang, Yu Xia, Hai-Long Sun, Shiyin Lu, Qing-Guo Chen, Weihua Luo, Kaifu Zhang, De-Chuan Zhan, Han-Jia Ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.04088)  

**Abstract**: Tabular reasoning involves multi-step information extraction and logical inference over tabular data. While recent advances have leveraged large language models (LLMs) for reasoning over structured tables, such high-quality textual representations are often unavailable in real-world settings, where tables typically appear as images. In this paper, we tackle the task of tabular reasoning from table images, leveraging privileged structured information available during training to enhance multimodal large language models (MLLMs). The key challenges lie in the complexity of accurately aligning structured information with visual representations, and in effectively transferring structured reasoning skills to MLLMs despite the input modality gap. To address these, we introduce TabUlar Reasoning with Bridged infOrmation ({\sc Turbo}), a new framework for multimodal tabular reasoning with privileged structured tables. {\sc Turbo} benefits from a structure-aware reasoning trace generator based on DeepSeek-R1, contributing to high-quality modality-bridged data. On this basis, {\sc Turbo} repeatedly generates and selects the advantageous reasoning paths, further enhancing the model's tabular reasoning ability. Experimental results demonstrate that, with limited ($9$k) data, {\sc Turbo} achieves state-of-the-art performance ($+7.2\%$ vs. previous SOTA) across multiple datasets. 

**Abstract (ZH)**: 表格式推理涉及多步信息提取和对表格数据进行逻辑推理。虽然后来的工作利用大规模语言模型（LLMs）在结构化表格上进行推理，但在实际场景中，表格通常以图像形式出现，缺乏高质量的文本表示。本文致力于从表格图像进行表格式推理的任务，利用训练过程中可获得的特权结构信息来增强多模态大规模语言模型（MLLMs）。关键挑战在于精确对齐结构化信息与视觉表示的复杂性，以及在输入模态差距下有效转移结构化推理技能。为了解决这些问题，我们提出了一个名为Turbo的新框架，该框架基于DeepSeek-R1实现结构感知的推理轨迹生成器，从而生成高质量的模态桥梁数据。在此基础上，Turbo反复生成和选择有利的推理路径，进一步增强模型的表格式推理能力。实验结果显示，在有限（9k）数据下，Turbo在多个数据集上实现了最先进的性能（对比 previous SOTA 提高了 7.2%）。 

---
# Towards generating more interpretable counterfactuals via concept vectors: a preliminary study on chest X-rays 

**Title (ZH)**: 基于概念向量生成更具解释性的反事实例子：胸部X光片的初步研究 

**Authors**: Bulat Maksudov, Kathleen Curran, Alessandra Mileo  

**Link**: [PDF](https://arxiv.org/pdf/2506.04058)  

**Abstract**: An essential step in deploying medical imaging models is ensuring alignment with clinical knowledge and interpretability. We focus on mapping clinical concepts into the latent space of generative models to identify Concept Activation Vectors (CAVs). Using a simple reconstruction autoencoder, we link user-defined concepts to image-level features without explicit label training. The extracted concepts are stable across datasets, enabling visual explanations that highlight clinically relevant features. By traversing latent space along concept directions, we produce counterfactuals that exaggerate or reduce specific clinical features. Preliminary results on chest X-rays show promise for large pathologies like cardiomegaly, while smaller pathologies remain challenging due to reconstruction limits. Although not outperforming baselines, this approach offers a path toward interpretable, concept-based explanations aligned with clinical knowledge. 

**Abstract (ZH)**: 部署医学成像模型的一个关键步骤是确保与临床知识的对齐和可解释性。我们专注于将临床概念映射到生成模型的潜在空间中以识别概念激活向量（CAVs）。通过一个简单的重建自编码器，我们将用户定义的概念与图像级特征关联起来，而无需显式的标签训练。提取的概念在不同数据集中具有稳定性，从而能够提供视觉解释，突出显示与临床相关的特征。通过沿着概念方向穿越潜在空间，我们生成了反事实样本，以夸大或减少特定的临床特征。初步结果表明，这种方法在胸片上对于大型病理学如心肌肥大具有潜力，而对于小型病理学则因重建限制而面临挑战。尽管不如基准模型优秀，但该方法为基于可解释性的、与临床知识一致的概念驱动解释提供了一条路径。 

---
# Towards Better Disentanglement in Non-Autoregressive Zero-Shot Expressive Voice Conversion 

**Title (ZH)**: 面向更好的非自回归零样本表达语音转换中的独立成分分离 

**Authors**: Seymanur Akti, Tuan Nam Nguyen, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2506.04013)  

**Abstract**: Expressive voice conversion aims to transfer both speaker identity and expressive attributes from a target speech to a given source speech. In this work, we improve over a self-supervised, non-autoregressive framework with a conditional variational autoencoder, focusing on reducing source timbre leakage and improving linguistic-acoustic disentanglement for better style transfer. To minimize style leakage, we use multilingual discrete speech units for content representation and reinforce embeddings with augmentation-based similarity loss and mix-style layer normalization. To enhance expressivity transfer, we incorporate local F0 information via cross-attention and extract style embeddings enriched with global pitch and energy features. Experiments show our model outperforms baselines in emotion and speaker similarity, demonstrating superior style adaptation and reduced source style leakage. 

**Abstract (ZH)**: 表达性语音转换旨在将目标语音的说话人身份和表达属性转移到给定的源语音中。在本文中，我们改进了基于自监督的非自回归框架，使用条件变分自编码器，专注于减少源音色泄漏并提高语言-声学解耦，以实现更好的风格转换。为了最小化风格泄漏，我们使用多语言离散语音单元进行内容表示，并通过基于增强的相似损失和混合风格层归一化增强嵌入。为了增强表达性转移，我们通过交叉注意机制引入局部F0信息，并提取富含全局音高和能量特征的风格嵌入。实验结果显示，我们的模型在情绪和说话人相似度上优于基线模型，证明其具有更好的风格适应性和减少源风格泄漏的能力。 

---
# TransClean: Finding False Positives in Multi-Source Entity Matching under Real-World Conditions via Transitive Consistency 

**Title (ZH)**: TransClean: 在实际条件下通过传递一致性查找多源实体匹配中的假阳性结果 

**Authors**: Fernando de Meer Pardo, Branka Hadji Misheva, Martin Braschler, Kurt Stockinger  

**Link**: [PDF](https://arxiv.org/pdf/2506.04006)  

**Abstract**: We present TransClean, a method for detecting false positive predictions of entity matching algorithms under real-world conditions characterized by large-scale, noisy, and unlabeled multi-source datasets that undergo distributional shifts. TransClean is explicitly designed to operate with multiple data sources in an efficient, robust and fast manner while accounting for edge cases and requiring limited manual labeling. TransClean leverages the Transitive Consistency of a matching, a measure of the consistency of a pairwise matching model f_theta on the matching it produces G_f_theta, based both on its predictions on directly evaluated record pairs and its predictions on implied record pairs. TransClean iteratively modifies a matching through gradually removing false positive matches while removing as few true positive matches as possible. In each of these steps, the estimation of the Transitive Consistency is exclusively done through model evaluations and produces quantities that can be used as proxies of the amounts of true and false positives in the matching while not requiring any manual labeling, producing an estimate of the quality of the matching and indicating which record groups are likely to contain false positives. In our experiments, we compare combining TransClean with a naively trained pairwise matching model (DistilBERT) and with a state-of-the-art end-to-end matching method (CLER) and illustrate the flexibility of TransClean in being able to detect most of the false positives of either setup across a variety of datasets. Our experiments show that TransClean induces an average +24.42 F1 score improvement for entity matching in a multi-source setting when compared to traditional pair-wise matching algorithms. 

**Abstract (ZH)**: TransClean: 一种检测实体匹配算法在大规模、 noisy、未标注多源数据集下的错误正预测的方法 

---
# CARL: Causality-guided Architecture Representation Learning for an Interpretable Performance Predictor 

**Title (ZH)**: CARL: 基于因果性的架构表示学习以构建可解释的性能预测器 

**Authors**: Han Ji, Yuqi Feng, Jiahao Fan, Yanan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.04001)  

**Abstract**: Performance predictors have emerged as a promising method to accelerate the evaluation stage of neural architecture search (NAS). These predictors estimate the performance of unseen architectures by learning from the correlation between a small set of trained architectures and their performance. However, most existing predictors ignore the inherent distribution shift between limited training samples and diverse test samples. Hence, they tend to learn spurious correlations as shortcuts to predictions, leading to poor generalization. To address this, we propose a Causality-guided Architecture Representation Learning (CARL) method aiming to separate critical (causal) and redundant (non-causal) features of architectures for generalizable architecture performance prediction. Specifically, we employ a substructure extractor to split the input architecture into critical and redundant substructures in the latent space. Then, we generate multiple interventional samples by pairing critical representations with diverse redundant representations to prioritize critical features. Extensive experiments on five NAS search spaces demonstrate the state-of-the-art accuracy and superior interpretability of CARL. For instance, CARL achieves 97.67% top-1 accuracy on CIFAR-10 using DARTS. 

**Abstract (ZH)**: 因果引导的架构表示学习（CARL）方法：面向可泛化的架构性能预测 

---
# Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection 

**Title (ZH)**: 因果关系意识对比学习在鲁棒多变量时间序列异常检测中的应用 

**Authors**: HyunGi Kim, Jisoo Mok, Dongjun Lee, Jaihyun Lew, Sungjae Kim, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2506.03964)  

**Abstract**: Utilizing the complex inter-variable causal relationships within multivariate time-series provides a promising avenue toward more robust and reliable multivariate time-series anomaly detection (MTSAD) but remains an underexplored area of research. This paper proposes Causality-Aware contrastive learning for RObust multivariate Time-Series (CAROTS), a novel MTSAD pipeline that incorporates the notion of causality into contrastive learning. CAROTS employs two data augmentors to obtain causality-preserving and -disturbing samples that serve as a wide range of normal variations and synthetic anomalies, respectively. With causality-preserving and -disturbing samples as positives and negatives, CAROTS performs contrastive learning to train an encoder whose latent space separates normal and abnormal samples based on causality. Moreover, CAROTS introduces a similarity-filtered one-class contrastive loss that encourages the contrastive learning process to gradually incorporate more semantically diverse samples with common causal relationships. Extensive experiments on five real-world and two synthetic datasets validate that the integration of causal relationships endows CAROTS with improved MTSAD capabilities. The code is available at this https URL. 

**Abstract (ZH)**: 利用多变量时间序列内的复杂变量因果关系进行鲁棒多变量时间序列异常检测提供了极具前景的研究方向，但这一领域尚未得到充分探索。本文提出了一种因果关系感知对比学习方法，以实现鲁棒多变量时间序列异常检测（CAROTS），该方法将因果关系概念融入对比学习中。CAROTS采用两种数据增强器获取保留因果关系和破坏因果关系的样本，分别作为正常变异和合成异常的广泛范围。通过保留因果关系和破坏因果关系的样本作为正样本和负样本，CAROTS执行对比学习，以训练一个在潜在空间中基于因果关系将正常样本和异常样本分离的编码器。此外，CAROTS引入了一种基于相似性过滤的一类对比损失，鼓励对比学习过程逐渐纳入更多具有共同因果关系的语义多样样本。在五个真实世界和两个合成数据集上的广泛实验验证了因果关系集成赋予CAROTS改进的多变量时间序列异常检测能力。代码可在此链接访问。 

---
# HtFLlib: A Comprehensive Heterogeneous Federated Learning Library and Benchmark 

**Title (ZH)**: HtFLlib: 综合异构联邦学习库及基准 

**Authors**: Jianqing Zhang, Xinghao Wu, Yanbing Zhou, Xiaoting Sun, Qiqi Cai, Yang Liu, Yang Hua, Zhenzhe Zheng, Jian Cao, Qiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03954)  

**Abstract**: As AI evolves, collaboration among heterogeneous models helps overcome data scarcity by enabling knowledge transfer across institutions and devices. Traditional Federated Learning (FL) only supports homogeneous models, limiting collaboration among clients with heterogeneous model architectures. To address this, Heterogeneous Federated Learning (HtFL) methods are developed to enable collaboration across diverse heterogeneous models while tackling the data heterogeneity issue at the same time. However, a comprehensive benchmark for standardized evaluation and analysis of the rapidly growing HtFL methods is lacking. Firstly, the highly varied datasets, model heterogeneity scenarios, and different method implementations become hurdles to making easy and fair comparisons among HtFL methods. Secondly, the effectiveness and robustness of HtFL methods are under-explored in various scenarios, such as the medical domain and sensor signal modality. To fill this gap, we introduce the first Heterogeneous Federated Learning Library (HtFLlib), an easy-to-use and extensible framework that integrates multiple datasets and model heterogeneity scenarios, offering a robust benchmark for research and practical applications. Specifically, HtFLlib integrates (1) 12 datasets spanning various domains, modalities, and data heterogeneity scenarios; (2) 40 model architectures, ranging from small to large, across three modalities; (3) a modularized and easy-to-extend HtFL codebase with implementations of 10 representative HtFL methods; and (4) systematic evaluations in terms of accuracy, convergence, computation costs, and communication costs. We emphasize the advantages and potential of state-of-the-art HtFL methods and hope that HtFLlib will catalyze advancing HtFL research and enable its broader applications. The code is released at this https URL. 

**Abstract (ZH)**: 异构联邦学习库：HtFLlib 

---
# Hanging in the Balance: Pivotal Moments in Crisis Counseling Conversations 

**Title (ZH)**: 悬而未决：危机咨询对话中的关键时刻 

**Authors**: Vivian Nguyen, Lillian Lee, Cristian Danescu-Niculescu-Mizil  

**Link**: [PDF](https://arxiv.org/pdf/2506.03941)  

**Abstract**: During a conversation, there can come certain moments where its outcome hangs in the balance. In these pivotal moments, how one responds can put the conversation on substantially different trajectories leading to significantly different outcomes. Systems that can detect when such moments arise could assist conversationalists in domains with highly consequential outcomes, such as mental health crisis counseling.
In this work, we introduce an unsupervised computational method for detecting such pivotal moments as they happen, in an online fashion. Our approach relies on the intuition that a moment is pivotal if our expectation of the outcome varies widely depending on what might be said next. By applying our method to crisis counseling conversations, we first validate it by showing that it aligns with human perception -- counselors take significantly longer to respond during moments detected by our method -- and with the eventual conversational trajectory -- which is more likely to change course at these times. We then use our framework to explore the relation of the counselor's response during pivotal moments with the eventual outcome of the session. 

**Abstract (ZH)**: 在对话中检测关键时刻的无监督计算方法：以危机咨询为例 

---
# HTSC-2025: A Benchmark Dataset of Ambient-Pressure High-Temperature Superconductors for AI-Driven Critical Temperature Prediction 

**Title (ZH)**: HTSC-2025：高压高温超导体基准数据集，用于AI驱动的临界温度预测 

**Authors**: Xiao-Qi Han, Ze-Feng Gao, Xin-De Wang, Zhenfeng Ouyang, Peng-Jie Guo, Zhong-Yi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03837)  

**Abstract**: The discovery of high-temperature superconducting materials holds great significance for human industry and daily life. In recent years, research on predicting superconducting transition temperatures using artificial intelligence~(AI) has gained popularity, with most of these tools claiming to achieve remarkable accuracy. However, the lack of widely accepted benchmark datasets in this field has severely hindered fair comparisons between different AI algorithms and impeded further advancement of these methods. In this work, we present the HTSC-2025, an ambient-pressure high-temperature superconducting benchmark dataset. This comprehensive compilation encompasses theoretically predicted superconducting materials discovered by theoretical physicists from 2023 to 2025 based on BCS superconductivity theory, including the renowned X$_2$YH$_6$ system, perovskite MXH$_3$ system, M$_3$XH$_8$ system, cage-like BCN-doped metal atomic systems derived from LaH$_{10}$ structural evolution, and two-dimensional honeycomb-structured systems evolving from MgB$_2$. The HTSC-2025 benchmark has been open-sourced at this https URL and will be continuously updated. This benchmark holds significant importance for accelerating the discovery of superconducting materials using AI-based methods. 

**Abstract (ZH)**: 高温超导材料的发现对人类工业和日常生活具有重要意义。近年来，使用人工智能（AI）预测超导转变温度的研究吸引了广泛的关注，大多数工具声称能够达到显著的准确性。然而，该领域缺乏广泛接受的标准数据集严重阻碍了不同AI算法之间的公平比较，阻碍了这些方法的进一步发展。在此工作中，我们提出HTSC-2025，一个常压高温超导标准数据集。该综合编纂包括2023年至2025年由理论物理学家根据BCS超导理论预测的超导材料，包括著名的X₂YH₆系统、钙钛矿MXH₃系统、M₃XH₈系统、从LaH₁₀结构演变来的笼状BCN掺金属原子系统以及从MgB₂演化而来的二维蜂窝结构系统。HTSC-2025基准数据集已在该网址开放源代码，并将持续更新。该基准对于加速使用基于AI的方法发现超导材料具有重要意义。 

---
# Multi-objective Aligned Bidword Generation Model for E-commerce Search Advertising 

**Title (ZH)**: 多目标对齐出价单词生成模型在电子商务搜索广告中的应用 

**Authors**: Zhenhui Liu, Chunyuan Yuan, Ming Pang, Zheng Fang, Li Yuan, Xue Jiang, Changping Peng, Zhangang Lin, Zheng Luo, Jingping Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.03827)  

**Abstract**: Retrieval systems primarily address the challenge of matching user queries with the most relevant advertisements, playing a crucial role in e-commerce search advertising. The diversity of user needs and expressions often produces massive long-tail queries that cannot be matched with merchant bidwords or product titles, which results in some advertisements not being recalled, ultimately harming user experience and search efficiency. Existing query rewriting research focuses on various methods such as query log mining, query-bidword vector matching, or generation-based rewriting. However, these methods often fail to simultaneously optimize the relevance and authenticity of the user's original query and rewrite and maximize the revenue potential of recalled ads.
In this paper, we propose a Multi-objective aligned Bidword Generation Model (MoBGM), which is composed of a discriminator, generator, and preference alignment module, to address these challenges. To simultaneously improve the relevance and authenticity of the query and rewrite and maximize the platform revenue, we design a discriminator to optimize these key objectives. Using the feedback signal of the discriminator, we train a multi-objective aligned bidword generator that aims to maximize the combined effect of the three objectives. Extensive offline and online experiments show that our proposed algorithm significantly outperforms the state of the art. After deployment, the algorithm has created huge commercial value for the platform, further verifying its feasibility and robustness. 

**Abstract (ZH)**: 检索系统主要应对用户查询与最相关广告匹配的挑战，在电子商务搜索广告中发挥关键作用。用户的多样需求和表达方式常常会产生大量的长尾查询，这些查询无法与商家出价词或产品标题匹配，导致部分广告无法召回，最终损害用户体验和搜索效率。现有的查询重写研究侧重于查询日志挖掘、查询-出价词向量匹配或基于生成的重写等多种方法。然而，这些方法往往难以同时优化用户的原始查询和重写的相关性和真实性，并最大化召回广告的潜在收入。

本文提出了一种多目标对齐出价词生成模型（MoBGM），该模型由判别器、生成器和偏好对齐模块组成，以应对这些挑战。为了同时提高查询和重写的相关性和真实性，并最大化平台收入，我们设计了一个判别器来优化这些关键目标。利用判别器的反馈信号，我们训练了一个多目标对齐出价词生成器，旨在最大化这三个目标的复合效果。广泛的离线和在线实验表明，我们提出的算法显著优于现有技术。部署后，该算法为平台创造了巨大的商业价值，进一步验证了其可行性和稳健性。 

---
# When Does Closeness in Distribution Imply Representational Similarity? An Identifiability Perspective 

**Title (ZH)**: 分布接近性是否意味着表征相似性？从可识别性角度探讨 

**Authors**: Beatrix M. G. Nielsen, Emanuele Marconato, Andrea Dittadi, Luigi Gresele  

**Link**: [PDF](https://arxiv.org/pdf/2506.03784)  

**Abstract**: When and why representations learned by different deep neural networks are similar is an active research topic. We choose to address these questions from the perspective of identifiability theory, which suggests that a measure of representational similarity should be invariant to transformations that leave the model distribution unchanged. Focusing on a model family which includes several popular pre-training approaches, e.g., autoregressive language models, we explore when models which generate distributions that are close have similar representations. We prove that a small Kullback-Leibler divergence between the model distributions does not guarantee that the corresponding representations are similar. This has the important corollary that models arbitrarily close to maximizing the likelihood can still learn dissimilar representations, a phenomenon mirrored in our empirical observations on models trained on CIFAR-10. We then define a distributional distance for which closeness implies representational similarity, and in synthetic experiments, we find that wider networks learn distributions which are closer with respect to our distance and have more similar representations. Our results establish a link between closeness in distribution and representational similarity. 

**Abstract (ZH)**: 不同深度神经网络learned表示在何时以及为何相似：从可识别性理论视角探究 

---
# Misalignment or misuse? The AGI alignment tradeoff 

**Title (ZH)**: AGI对齐失调或误用？ 

**Authors**: Max Hellrigel-Holderbaum, Leonard Dung  

**Link**: [PDF](https://arxiv.org/pdf/2506.03755)  

**Abstract**: Creating systems that are aligned with our goals is seen as a leading approach to create safe and beneficial AI in both leading AI companies and the academic field of AI safety. We defend the view that misaligned AGI - future, generally intelligent (robotic) AI agents - poses catastrophic risks. At the same time, we support the view that aligned AGI creates a substantial risk of catastrophic misuse by humans. While both risks are severe and stand in tension with one another, we show that - in principle - there is room for alignment approaches which do not increase misuse risk. We then investigate how the tradeoff between misalignment and misuse looks empirically for different technical approaches to AI alignment. Here, we argue that many current alignment techniques and foreseeable improvements thereof plausibly increase risks of catastrophic misuse. Since the impacts of AI depend on the social context, we close by discussing important social factors and suggest that to reduce the risk of a misuse catastrophe due to aligned AGI, techniques such as robustness, AI control methods and especially good governance seem essential. 

**Abstract (ZH)**: 创建与我们目标一致的系统被视为在领先的人工智能公司和人工智能安全的学术领域中创建安全且有益的人工智能的主要途径。我们认为，未来普遍智能的人工智能代理的失控行为将带来灾难性风险。同时，我们支持观点认为，可控的人工智能代理同样存在因人类滥用而导致灾难性风险的可能性。尽管这两种风险都很严重且相互矛盾，但我们证明，在原则上存在不增加滥用风险的对齐方法。随后，我们探讨了不同类型的人工智能对齐技术在实际应用中失控行为与滥用风险之间的权衡。我们指出，目前许多对齐技术及其可预见的改进很可能是增加了灾难性滥用风险。由于人工智能的影响依赖于社会环境，我们最后讨论了重要社会因素，并建议为了降低因可控人工智能代理的滥用而导致的灾难性风险，必须具备鲁棒性、人工智能控制方法和特别有效治理措施。 

---
# ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices 

**Title (ZH)**: ComRoPE: 可训练共轭角矩阵参数化的可扩展且鲁棒的旋转位置嵌入 

**Authors**: Hao Yu, Tangyu Jiang, Shuning Jia, Shannan Yan, Shunning Liu, Haolong Qian, Guanghao Li, Shuting Dong, Huaisong Zhang, Chun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2506.03737)  

**Abstract**: The Transformer architecture has revolutionized various regions since it was proposed, and its effectiveness largely depends on the ability to encode positional information. Traditional position encoding methods exhibit significant limitations due to lack of robustness and flexibility of position. Therefore, Rotary Positional Encoding (RoPE) was proposed to alleviate these issues, which integrates positional information by rotating the embeddings in the attention mechanism. However, RoPE requires manually defined rotation matrices with limited transformation space, constraining the model's capacity. In this work, we propose ComRoPE, which generalizes RoPE by defining it in terms of trainable commuting angle matrices. Specifically, we demonstrate that pairwise commutativity of these matrices is essential for RoPE to achieve scalability and positional robustness. We formally define the RoPE Equation, which is an essential condition that ensures consistent performance with position offsets. Based on the theoretical analysis, we present two types of trainable commuting angle matrices as sufficient solutions to the RoPE equation, which significantly improve performance, surpassing the current state-of-the-art method by 1.6% at training resolution and 2.9% at higher resolution on the ImageNet-1K dataset. Furthermore, our framework shows versatility in generalizing to existing RoPE formulations and offering new insights for future positional encoding research. To ensure reproducibility, the source code and instructions are available at this https URL 

**Abstract (ZH)**: Transformer架构自提出以来已经革命性地改变了多个领域，其有效性很大程度上取决于编码位置信息的能力。传统的位置编码方法因位置的鲁棒性和灵活性不足而表现出明显的局限性，因此提出了旋转位置编码（RoPE）以缓解这些问题，通过在注意力机制中旋转嵌入来集成位置信息。然而，RoPE需要手动定义的旋转矩阵，其变换空间有限，限制了模型的能力。在此工作中，我们提出ComRoPE，通过将RoPE定义为可训练的可交换角矩阵来推广RoPE。具体而言，我们证明了这些矩阵的成对可交换性是RoPE实现可扩展性和位置鲁棒性的必要条件。基于理论分析，我们提出了两种类型的可训练可交换角矩阵作为RoPE方程的充分解，显著提高了性能，在ImageNet-1K数据集上，与当前最先进的方法相比，在训练分辨率上提高了1.6%，在较高分辨率上提高了2.9%。此外，我们的框架展示了在现有RoPE形式化方法上的一般性和对未来位置编码研究的新见解，以确保可重复性，源代码和说明可在以下网址获取。 

---
# Accelerating SfM-based Pose Estimation with Dominating Set 

**Title (ZH)**: 基于支配集加速结构从运动姿态估计 

**Authors**: Joji Joseph, Bharadwaj Amrutur, Shalabh Bhatnagar  

**Link**: [PDF](https://arxiv.org/pdf/2506.03667)  

**Abstract**: This paper introduces a preprocessing technique to speed up Structure-from-Motion (SfM) based pose estimation, which is critical for real-time applications like augmented reality (AR), virtual reality (VR), and robotics. Our method leverages the concept of a dominating set from graph theory to preprocess SfM models, significantly enhancing the speed of the pose estimation process without losing significant accuracy. Using the OnePose dataset, we evaluated our method across various SfM-based pose estimation techniques. The results demonstrate substantial improvements in processing speed, ranging from 1.5 to 14.48 times, and a reduction in reference images and point cloud size by factors of 17-23 and 2.27-4, respectively. This work offers a promising solution for efficient and accurate 3D pose estimation, balancing speed and accuracy in real-time applications. 

**Abstract (ZH)**: 本文介绍了一种预处理技术，用于加速基于结构从运动（SfM）的姿态估计，这对于增强现实（AR）、虚拟现实（VR）和机器人领域的实时应用至关重要。我们的方法利用图论中的支配集概念对SfM模型进行预处理，显著提高了姿态估计的速度，同时没有丢失显著的准确性。利用OnePose数据集，我们评估了该方法在各种基于SfM的姿态估计技术中的表现。结果表明，在处理速度上取得了显著改进，范围从1.5到14.48倍，并且参考图像和点云大小分别减少了17-23倍和2.27-4倍。本工作提供了一种高效准确的3D姿态估计的 promising 解决方案，在实时应用中平衡了速度和准确性。 

---
# Negative-Guided Subject Fidelity Optimization for Zero-Shot Subject-Driven Generation 

**Title (ZH)**: 负向引导主题保真度优化以实现零样本主题驱动生成 

**Authors**: Chaehun Shin, Jooyoung Choi, Johan Barthelemy, Jungbeom Lee, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2506.03621)  

**Abstract**: We present Subject Fidelity Optimization (SFO), a novel comparative learning framework for zero-shot subject-driven generation that enhances subject fidelity. Beyond supervised fine-tuning methods that rely only on positive targets and use the diffusion loss as in the pre-training stage, SFO introduces synthetic negative targets and explicitly guides the model to favor positives over negatives through pairwise comparison. For negative targets, we propose Condition-Degradation Negative Sampling (CDNS), which automatically generates distinctive and informative negatives by intentionally degrading visual and textual cues without expensive human annotations. Moreover, we reweight the diffusion timesteps to focus finetuning on intermediate steps where subject details emerge. Extensive experiments demonstrate that SFO with CDNS significantly outperforms baselines in terms of both subject fidelity and text alignment on a subject-driven generation benchmark. Project page: this https URL 

**Abstract (ZH)**: 面向主题保真度优化的新型零-shot 主题驱动生成对比学习框架（SFO） 

---
# GCFL: A Gradient Correction-based Federated Learning Framework for Privacy-preserving CPSS 

**Title (ZH)**: GCFL：一种基于梯度校正的隐私保护联邦学习框架 

**Authors**: Jiayi Wan, Xiang Zhu, Fanzhen Liu, Wei Fan, Xiaolong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03618)  

**Abstract**: Federated learning, as a distributed architecture, shows great promise for applications in Cyber-Physical-Social Systems (CPSS). In order to mitigate the privacy risks inherent in CPSS, the integration of differential privacy with federated learning has attracted considerable attention. Existing research mainly focuses on dynamically adjusting the noise added or discarding certain gradients to mitigate the noise introduced by differential privacy. However, these approaches fail to remove the noise that hinders convergence and correct the gradients affected by the noise, which significantly reduces the accuracy of model classification. To overcome these challenges, this paper proposes a novel framework for differentially private federated learning that balances rigorous privacy guarantees with accuracy by introducing a server-side gradient correction mechanism. Specifically, after clients perform gradient clipping and noise perturbation, our framework detects deviations in the noisy local gradients and employs a projection mechanism to correct them, mitigating the negative impact of noise. Simultaneously, gradient projection promotes the alignment of gradients from different clients and guides the model towards convergence to a global optimum. We evaluate our framework on several benchmark datasets, and the experimental results demonstrate that it achieves state-of-the-art performance under the same privacy budget. 

**Abstract (ZH)**: 联邦学习作为分布式架构，展现出在物理-社会系统（CPSS）中的广泛应用前景。为了缓解CPSS中固有的隐私风险，将差分隐私与联邦学习相结合引起了广泛的关注。现有研究主要集中在动态调整添加的噪音或丢弃某些梯度以缓解差分隐私引入的噪音。然而，这些方法无法消除妨碍收敛的噪音并且未能修正受噪音影响的梯度，这显著降低了模型分类的准确性。为克服这些挑战，本文提出了一种新颖的差分隐私联邦学习框架，通过引入服务器端梯度修正机制，平衡严格的隐私保证与准确性。具体而言，客户端执行梯度裁剪和噪音扰动后，该框架检测嘈杂的本地梯度中的偏差，并采用投影机制进行修正，缓解噪音的负面影响。同时，梯度投影促进了不同客户端梯度的对齐，并引导模型向全局最优解收敛。我们在多个基准数据集上评估了该框架，实验结果表明，在相同的隐私预算下，该框架实现了最先进的性能。 

---
# Tone recognition in low-resource languages of North-East India: peeling the layers of SSL-based speech models 

**Title (ZH)**: 东北印度低资源语言的语调识别：基于SSL的语音模型探析 

**Authors**: Parismita Gogoi, Sishir Kalita, Wendy Lalhminghlui, Viyazonuo Terhiija, Moakala Tzudir, Priyankoo Sarmah, S. R. M. Prasanna  

**Link**: [PDF](https://arxiv.org/pdf/2506.03606)  

**Abstract**: This study explores the use of self-supervised learning (SSL) models for tone recognition in three low-resource languages from North Eastern India: Angami, Ao, and Mizo. We evaluate four Wav2vec2.0 base models that were pre-trained on both tonal and non-tonal languages. We analyze tone-wise performance across the layers for all three languages and compare the different models. Our results show that tone recognition works best for Mizo and worst for Angami. The middle layers of the SSL models are the most important for tone recognition, regardless of the pre-training language, i.e. tonal or non-tonal. We have also found that the tone inventory, tone types, and dialectal variations affect tone recognition. These findings provide useful insights into the strengths and weaknesses of SSL-based embeddings for tonal languages and highlight the potential for improving tone recognition in low-resource settings. The source code is available at GitHub 1 . 

**Abstract (ZH)**: 本研究探索了自我监督学习（SSL）模型在印度东北部三种低资源语言（Angami、Ao和Mizo）音调识别中的应用。我们评估了四种在音调和非音调语言上预训练的Wav2vec2.0基础模型。我们分析了所有三种语言各层的音调性能，并比较了不同的模型。研究结果表明，音调识别在Mizo语言中效果最佳，在Angami语言中效果最差。无论预训练语言是音调语言还是非音调语言，SSL模型的中间层对于音调识别都是最重要的。我们还发现，音调Inventory、音调类型和方言变体影响音调识别。这些发现为基于SSL的嵌入在音调语言中的优缺点提供了有价值的见解，并突显了在低资源环境中提高音调识别的潜力。源代码可在GitHub 1获取。 

---
# Adapting Rule Representation With Four-Parameter Beta Distribution for Learning Classifier Systems 

**Title (ZH)**: 适应四参数Beta分布的规则表示学习分类器系统 

**Authors**: Hiroki Shiraishi, Yohei Hayamizu, Tomonori Hashiyama, Keiki Takadama, Hisao Ishibuchi, Masaya Nakata  

**Link**: [PDF](https://arxiv.org/pdf/2506.03602)  

**Abstract**: Rule representations significantly influence the search capabilities and decision boundaries within the search space of Learning Classifier Systems (LCSs), a family of rule-based machine learning systems that evolve interpretable models through evolutionary processes. However, it is very difficult to choose an appropriate rule representation for each problem. Additionally, some problems benefit from using different representations for different subspaces within the input space. Thus, an adaptive mechanism is needed to choose an appropriate rule representation for each rule in LCSs. This article introduces a flexible rule representation using a four-parameter beta distribution and integrates it into a fuzzy-style LCS. The four-parameter beta distribution can form various function shapes, and this flexibility enables our LCS to automatically select appropriate representations for different subspaces. Our rule representation can represent crisp/fuzzy decision boundaries in various boundary shapes, such as rectangles and bells, by controlling four parameters, compared to the standard representations such as trapezoidal ones. Leveraging this flexibility, our LCS is designed to adapt the appropriate rule representation for each subspace. Moreover, our LCS incorporates a generalization bias favoring crisp rules where feasible, enhancing model interpretability without compromising accuracy. Experimental results on real-world classification tasks show that our LCS achieves significantly superior test accuracy and produces more compact rule sets. Our implementation is available at this https URL. An extended abstract related to this work is available at this https URL. 

**Abstract (ZH)**: 基于Beta分布的灵活规则表示及其在模糊风格学习分类系统中的应用 

---
# Auto prompt sql: a resource-efficient architecture for text-to-sql translation in constrained environments 

**Title (ZH)**: 自动提示SQL：受限环境中文本到SQL转换的资源高效架构 

**Authors**: Zetong Tang, Qian Ma, Di Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03598)  

**Abstract**: Using the best Text-to-SQL methods in resource-constrained environments is challenging due to their reliance on resource-intensive open-source models. This paper introduces Auto Prompt SQL(AP-SQL), a novel architecture designed to bridge the gap between resource-efficient small open-source models and the powerful capabilities of large closed-source models for Text-to-SQL translation. Our method decomposes the task into schema filtering, retrieval-augmented text-to-SQL generation based on in-context examples, and prompt-driven schema linking and SQL generation. To improve schema selection accuracy, we fine-tune large language models. Crucially, we also explore the impact of prompt engineering throughout the process, leveraging Chain-of-Thought(CoT) and Graph-of-Thought(GoT) templates to significantly enhance the model's reasoning for accurate SQL generation. Comprehensive evaluations on the Spider benchmarks demonstrate the effectiveness of AP-SQL. 

**Abstract (ZH)**: 在资源受限环境中使用最佳Text-to-SQL方法具有挑战性，因为这些方法依赖于资源密集型开源模型。本文介绍了Auto Prompt SQL (AP-SQL)，这是一种新型架构，旨在弥合资源高效的小开源模型与强大功能的大封闭源模型之间的差距，以实现Text-to-SQL翻译。我们的方法将任务分解为模式过滤、基于上下文示例的检索增强文本到SQL生成以及提示驱动的模式链接和SQL生成。为了提高模式选择准确性，我们对大型语言模型进行了微调。 crucially，我们还在整个过程中探讨了提示工程的影响，利用Chain-of-Thought (CoT) 和 Graph-of-Thought (GoT) 模板显著增强了模型的推理能力以实现准确的SQL生成。在Spider基准上的全面评估证明了AP-SQL的有效性。 

---
# Purifying Shampoo: Investigating Shampoo's Heuristics by Decomposing its Preconditioner 

**Title (ZH)**: 净化洗发水：通过分解其预处理器探讨洗发水的启发式方法 

**Authors**: Runa Eschenhagen, Aaron Defazio, Tsung-Hsien Lee, Richard E. Turner, Hao-Jun Michael Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.03595)  

**Abstract**: The recent success of Shampoo in the AlgoPerf contest has sparked renewed interest in Kronecker-factorization-based optimization algorithms for training neural networks. Despite its success, Shampoo relies heavily on several heuristics such as learning rate grafting and stale preconditioning to achieve performance at-scale. These heuristics increase algorithmic complexity, necessitate further hyperparameter tuning, and lack theoretical justification. This paper investigates these heuristics from the angle of Frobenius norm approximation to full-matrix Adam and decouples the preconditioner's eigenvalues and eigenbasis updates. We show that grafting from Adam mitigates the staleness and mis-scaling of the preconditioner's eigenvalues and how correcting the eigenvalues directly can eliminate the need for learning rate grafting. To manage the error induced by infrequent eigenbasis computations, we propose an adaptive criterion for determining the eigenbasis computation frequency motivated by terminating a warm-started QR algorithm. This criterion decouples the update frequency of different preconditioner matrices and enables us to investigate the impact of approximation error on convergence. These practical techniques offer a principled angle towards removing Shampoo's heuristics and developing improved Kronecker-factorization-based training algorithms. 

**Abstract (ZH)**: Shampoo算法在AlgoPerf竞赛中的最近成功引发了对基于Kronecker因式分解的优化算法在神经网络训练中兴趣的重燃。尽管取得了成功，但Shampoo算法强烈依赖于诸如学习率嫁接和过时预条件处理等启发式方法来实现大规模性能。这些启发式方法增加了算法复杂性，需要进一步调整超参数，并缺乏理论依据。本文从Frobenius范数逼近全矩阵Adam的角度出发，将预条件矩阵的特征值和特征向量分解解耦。我们展示了来自Adam的嫁接如何减轻预条件矩阵特征值的过时和失真，并如何直接矫正特征值可以消除学习率嫁接的需要。为了管理由不频繁计算特征向量引起的误差，我们提出了一种基于终止预热QR算法的自适应准则来确定特征向量计算频率。该准则使不同预条件矩阵的更新频率解耦，并使我们能够研究逼近误差对收敛的影响。这些实用的技术为去除Shampoo的启发式方法并开发改进的基于Kronecker因式分解的训练算法提供了理论视角。 

---
# BiMa: Towards Biases Mitigation for Text-Video Retrieval via Scene Element Guidance 

**Title (ZH)**: BiMa: 基于场景元素指导的文本-视频检索偏差缓解方法 

**Authors**: Huy Le, Nhat Chung, Tung Kieu, Anh Nguyen, Ngan Le  

**Link**: [PDF](https://arxiv.org/pdf/2506.03589)  

**Abstract**: Text-video retrieval (TVR) systems often suffer from visual-linguistic biases present in datasets, which cause pre-trained vision-language models to overlook key details. To address this, we propose BiMa, a novel framework designed to mitigate biases in both visual and textual representations. Our approach begins by generating scene elements that characterize each video by identifying relevant entities/objects and activities. For visual debiasing, we integrate these scene elements into the video embeddings, enhancing them to emphasize fine-grained and salient details. For textual debiasing, we introduce a mechanism to disentangle text features into content and bias components, enabling the model to focus on meaningful content while separately handling biased information. Extensive experiments and ablation studies across five major TVR benchmarks (i.e., MSR-VTT, MSVD, LSMDC, ActivityNet, and DiDeMo) demonstrate the competitive performance of BiMa. Additionally, the model's bias mitigation capability is consistently validated by its strong results on out-of-distribution retrieval tasks. 

**Abstract (ZH)**: 基于视觉-语言偏见的文本-视频检索系统改进：BiMa框架的研究 

---
# A Class Inference Scheme With Dempster-Shafer Theory for Learning Fuzzy-Classifier Systems 

**Title (ZH)**: 基于Dempster-Shafer理论的类别推理方案及其在模糊分类系统中的学习应用 

**Authors**: Hiroki Shiraishi, Hisao Ishibuchi, Masaya Nakata  

**Link**: [PDF](https://arxiv.org/pdf/2506.03588)  

**Abstract**: The decision-making process significantly influences the predictions of machine learning models. This is especially important in rule-based systems such as Learning Fuzzy-Classifier Systems (LFCSs) where the selection and application of rules directly determine prediction accuracy and reliability. LFCSs combine evolutionary algorithms with supervised learning to optimize fuzzy classification rules, offering enhanced interpretability and robustness. Despite these advantages, research on improving decision-making mechanisms (i.e., class inference schemes) in LFCSs remains limited. Most LFCSs use voting-based or single-winner-based inference schemes. These schemes rely on classification performance on training data and may not perform well on unseen data, risking overfitting. To address these limitations, this article introduces a novel class inference scheme for LFCSs based on the Dempster-Shafer Theory of Evidence (DS theory). The proposed scheme handles uncertainty well. By using the DS theory, the scheme calculates belief masses (i.e., measures of belief) for each specific class and the ``I don't know'' state from each fuzzy rule and infers a class from these belief masses. Unlike the conventional schemes, the proposed scheme also considers the ``I don't know'' state that reflects uncertainty, thereby improving the transparency and reliability of LFCSs. Applied to a variant of LFCS (i.e., Fuzzy-UCS), the proposed scheme demonstrates statistically significant improvements in terms of test macro F1 scores across 30 real-world datasets compared to conventional voting-based and single-winner-based fuzzy inference schemes. It forms smoother decision boundaries, provides reliable confidence measures, and enhances the robustness and generalizability of LFCSs in real-world applications. Our implementation is available at this https URL. 

**Abstract (ZH)**: 机器学习模型的决策过程显著影响预测结果。特别是在基于规则的系统如学习模糊分类系统（LFCSs）中，规则的选择和应用直接决定了预测的准确性和可靠性。尽管LFCSs结合了进化算法和监督学习以优化模糊分类规则，提供增强的可解释性和鲁棒性，但关于改进LFCSs决策机制（即类推理方案）的研究仍然有限。大多数LFCSs使用基于投票或单赢者的推理方案。这些方案依赖于在训练数据上的分类性能，可能在未见过的数据上表现不佳，存在过拟合的风险。为应对这些局限，本文提出了一种基于迪斯克耳-肖弗证据理论（DS理论）的新型类推理方案。该方案能良好处理不确定性。通过使用DS理论，该方案计算每个模糊规则及其“不知道”状态下的每种具体类别的信心量（即信念度量），并据此进行类推理。与传统方案不同，该方案还考虑了“不知道”的状态，反映不确定性，从而提高LFCSs的透明度和可靠性。将该方案应用于LFCS的一种变体（即模糊-UCS），在30个真实世界数据集上，相较于传统的基于投票和单赢者的模糊推理方案，测试宏F1分数显示出统计意义上的显著改进。该方案形成更平滑的决策边界，提供可靠的信心度量，并增强LFCSs在实际应用中的鲁棒性和泛化能力。我们的实现可供于此网址。 

---
# ViTSGMM: A Robust Semi-Supervised Image Recognition Network Using Sparse Labels 

**Title (ZH)**: ViTSGMM：一种使用稀疏标签的鲁棒半监督图像识别网络 

**Authors**: Rui Yann, Xianglei Xing  

**Link**: [PDF](https://arxiv.org/pdf/2506.03582)  

**Abstract**: We present ViTSGMM, an image recognition network that leverages semi-supervised learning in a highly efficient manner. Existing works often rely on complex training techniques and architectures, while their generalization ability when dealing with extremely limited labeled data remains to be improved. To address these limitations, we construct a hierarchical mixture density classification decision mechanism by optimizing mutual information between feature representations and target classes, compressing redundant information while retaining crucial discriminative components. Experimental results demonstrate that our method achieves state-of-the-art performance on STL-10 and CIFAR-10/100 datasets when using negligible labeled samples. Notably, this paper also reveals a long-overlooked data leakage issue in the STL-10 dataset for semi-supervised learning tasks and removes duplicates to ensure the reliability of experimental results. Code available at this https URL. 

**Abstract (ZH)**: ViTSGMM：一种高效利用半监督学习的图像识别网络 

---
# KG-BiLM: Knowledge Graph Embedding via Bidirectional Language Models 

**Title (ZH)**: KG-BiLM：基于双向语言模型的知识图谱嵌入 

**Authors**: Zirui Chen, Xin Wang, Zhao Li, Wenbin Guo, Dongxiao He  

**Link**: [PDF](https://arxiv.org/pdf/2506.03576)  

**Abstract**: Recent advances in knowledge representation learning (KRL) highlight the urgent necessity to unify symbolic knowledge graphs (KGs) with language models (LMs) for richer semantic understanding. However, existing approaches typically prioritize either graph structure or textual semantics, leaving a gap: a unified framework that simultaneously captures global KG connectivity, nuanced linguistic context, and discriminative reasoning semantics. To bridge this gap, we introduce KG-BiLM, a bidirectional LM framework that fuses structural cues from KGs with the semantic expressiveness of generative transformers. KG-BiLM incorporates three key components: (i) Bidirectional Knowledge Attention, which removes the causal mask to enable full interaction among all tokens and entities; (ii) Knowledge-Masked Prediction, which encourages the model to leverage both local semantic contexts and global graph connectivity; and (iii) Contrastive Graph Semantic Aggregation, which preserves KG structure via contrastive alignment of sampled sub-graph representations. Extensive experiments on standard benchmarks demonstrate that KG-BiLM outperforms strong baselines in link prediction, especially on large-scale graphs with complex multi-hop relations - validating its effectiveness in unifying structural information and textual semantics. 

**Abstract (ZH)**: 最近关于知识表示学习的进步强调了统一符号知识图谱与语言模型的迫切必要性，以实现更丰富的语义理解。然而，现有方法通常要么侧重于图结构要么侧重于文本语义，留下了统一框架的缺口：一个可以同时捕捉全局知识图谱连接性、细腻的语义上下文和辨别性推理语义的框架。为填补这一缺口，我们提出了KG-BiLM，这是一种融合知识图谱结构线索与生成式变换器语义表现力的双向语言模型框架。KG-BiLM 包含三个关键组件：(i) 双向知识注意，移除因果掩码以实现所有令牌和实体之间的全面交互；(ii) 知识掩蔽预测，促使模型利用局部语义上下文和全局图连接性；(iii) 对比图语义聚合，通过对比子图表示的对齐保留知识图谱结构。在标准基准上的广泛实验表明，KG-BiLM 在链接预测任务中优于强有力的基线模型，尤其是在大型复杂多跳关系图上，验证了其在统一结构信息和文本语义方面的有效性。 

---
# Explainable AI: XAI-Guided Context-Aware Data Augmentation 

**Title (ZH)**: 可解释AI：基于XAI的上下文感知数据增广 

**Authors**: Melkamu Abay Mersha, Mesay Gemeda Yigezu, Atnafu Lambebo Tonja, Hassan Shakil, Samer Iskander, Olga Kolesnikova, Jugal Kalita  

**Link**: [PDF](https://arxiv.org/pdf/2506.03484)  

**Abstract**: Explainable AI (XAI) has emerged as a powerful tool for improving the performance of AI models, going beyond providing model transparency and interpretability. The scarcity of labeled data remains a fundamental challenge in developing robust and generalizable AI models, particularly for low-resource languages. Conventional data augmentation techniques introduce noise, cause semantic drift, disrupt contextual coherence, lack control, and lead to overfitting. To address these challenges, we propose XAI-Guided Context-Aware Data Augmentation. This novel framework leverages XAI techniques to modify less critical features while selectively preserving most task-relevant features. Our approach integrates an iterative feedback loop, which refines augmented data over multiple augmentation cycles based on explainability-driven insights and the model performance gain. Our experimental results demonstrate that XAI-SR-BT and XAI-PR-BT improve the accuracy of models on hate speech and sentiment analysis tasks by 6.6% and 8.1%, respectively, compared to the baseline, using the Amharic dataset with the XLM-R model. XAI-SR-BT and XAI-PR-BT outperform existing augmentation techniques by 4.8% and 5%, respectively, on the same dataset and model. Overall, XAI-SR-BT and XAI-PR-BT consistently outperform both baseline and conventional augmentation techniques across all tasks and models. This study provides a more controlled, interpretable, and context-aware solution to data augmentation, addressing critical limitations of existing augmentation techniques and offering a new paradigm shift for leveraging XAI techniques to enhance AI model training. 

**Abstract (ZH)**: 可解释的人工智能（XAI）已 emerges as a powerful tool for improving the performance of AI models, going beyond providing model transparency and interpretability. The scarcity of labeled data remains a fundamental challenge in developing robust and generalizable AI models, particularly for low-resource languages. Conventional data augmentation techniques introduce noise, cause semantic drift, disrupt contextual coherence, lack control, and lead to overfitting. To address these challenges, we propose XAI-Guided Context-Aware Data Augmentation. This novel framework leverages XAI techniques to modify less critical features while selectively preserving most task-relevant features. Our approach integrates an iterative feedback loop, which refines augmented data over multiple augmentation cycles based on explainability-driven insights and the model performance gain.

可解释的人工智能（XAI）已 Emerged as a Powerful Tool for Improving AI Model Performance: Addressing the Challenges of Labeled Data Scarcity and Conventional Data Augmentation Techniques Through XAI-Guided Context-Aware Data Augmentation 

---
# A Data-Driven Diffusion-based Approach for Audio Deepfake Explanations 

**Title (ZH)**: 基于数据驱动扩散的方法对音频深度假音的解释 

**Authors**: Petr Grinberg, Ankur Kumar, Surya Koppisetti, Gaurav Bharaj  

**Link**: [PDF](https://arxiv.org/pdf/2506.03425)  

**Abstract**: Evaluating explainability techniques, such as SHAP and LRP, in the context of audio deepfake detection is challenging due to lack of clear ground truth annotations. In the cases when we are able to obtain the ground truth, we find that these methods struggle to provide accurate explanations. In this work, we propose a novel data-driven approach to identify artifact regions in deepfake audio. We consider paired real and vocoded audio, and use the difference in time-frequency representation as the ground-truth explanation. The difference signal then serves as a supervision to train a diffusion model to expose the deepfake artifacts in a given vocoded audio. Experimental results on the VocV4 and LibriSeVoc datasets demonstrate that our method outperforms traditional explainability techniques, both qualitatively and quantitatively. 

**Abstract (ZH)**: 在音频深度假音检测中评估SHAP和LRP等可解释性技术具有挑战性，因为缺乏明确的 ground truth 注解。在能够获取 ground truth 的情况下，这些方法也难以提供准确的解释。在这项工作中，我们提出了一种数据驱动的方法来识别深度假音音频中的伪像区域。我们考虑了真实的和声编码音频配对，并使用时频表示的差异作为 ground truth 解释。随后，该差异信号作为监督信息来训练一个扩散模型，以在给定的声编码音频中揭示深度假音伪像。在 VocV4 和 LibriSeVoc 数据集上的实验结果表明，我们的方法在定性和定量上均优于传统可解释性技术。 

---
# Universal Reusability in Recommender Systems: The Case for Dataset- and Task-Independent Frameworks 

**Title (ZH)**: 通用可重用性在推荐系统中的实现：基于数据集和任务独立的框架研究 

**Authors**: Tri Kurniawan Wijaya, Xinyang Shao, Gonzalo Fiz Pontiveros, Edoardo D'Amico  

**Link**: [PDF](https://arxiv.org/pdf/2506.03391)  

**Abstract**: Recommender systems are pivotal in delivering personalized experiences across industries, yet their adoption and scalability remain hindered by the need for extensive dataset- and task-specific configurations. Existing systems often require significant manual intervention, domain expertise, and engineering effort to adapt to new datasets or tasks, creating barriers to entry and limiting reusability. In contrast, recent advancements in large language models (LLMs) have demonstrated the transformative potential of reusable systems, where a single model can handle diverse tasks without significant reconfiguration. Inspired by this paradigm, we propose the Dataset- and Task-Independent Recommender System (DTIRS), a framework aimed at maximizing the reusability of recommender systems while minimizing barriers to entry. Unlike LLMs, which achieve task generalization directly, DTIRS focuses on eliminating the need to rebuild or reconfigure recommendation pipelines for every new dataset or task, even though models may still need retraining on new data. By leveraging the novel Dataset Description Language (DsDL), DTIRS enables standardized dataset descriptions and explicit task definitions, allowing autonomous feature engineering, model selection, and optimization. This paper introduces the concept of DTIRS and establishes a roadmap for transitioning from Level-1 automation (dataset-agnostic but task-specific systems) to Level-2 automation (fully dataset- and task-independent systems). Achieving this paradigm would maximize code reusability and lower barriers to adoption. We discuss key challenges, including the trade-offs between generalization and specialization, computational overhead, and scalability, while presenting DsDL as a foundational tool for this vision. 

**Abstract (ZH)**: 面向数据和任务独立的推荐系统（DTIRS） 

---
# Automated Traffic Incident Response Plans using Generative Artificial Intelligence: Part 1 -- Building the Incident Response Benchmark 

**Title (ZH)**: 使用生成式人工智能自动化的交通事件响应计划：第1部分——建立事件响应基准 

**Authors**: Artur Grigorev, Khaled Saleh, Jiwon Kim, Adriana-Simona Mihaita  

**Link**: [PDF](https://arxiv.org/pdf/2506.03381)  

**Abstract**: Traffic incidents remain a critical public safety concern worldwide, with Australia recording 1,300 road fatalities in 2024, which is the highest toll in 12 years. Similarly, the United States reports approximately 6 million crashes annually, raising significant challenges in terms of a fast reponse time and operational management. Traditional response protocols rely on human decision-making, which introduces potential inconsistencies and delays during critical moments when every minute impacts both safety outcomes and network performance. To address this issue, we propose a novel Incident Response Benchmark that uses generative artificial intelligence to automatically generate response plans for incoming traffic incidents. Our approach aims to significantly reduce incident resolution times by suggesting context-appropriate actions such as variable message sign deployment, lane closures, and emergency resource allocation adapted to specific incident characteristics. First, the proposed methodology uses real-world incident reports from the Performance Measurement System (PeMS) as training and evaluation data. We extract historically implemented actions from these reports and compare them against AI-generated response plans that suggest specific actions, such as lane closures, variable message sign announcements, and/or dispatching appropriate emergency resources. Second, model evaluations reveal that advanced generative AI models like GPT-4o and Grok 2 achieve superior alignment with expert solutions, demonstrated by minimized Hamming distances (averaging 2.96-2.98) and low weighted differences (approximately 0.27-0.28). Conversely, while Gemini 1.5 Pro records the lowest count of missed actions, its extremely high number of unnecessary actions (1547 compared to 225 for GPT-4o) indicates an over-triggering strategy that reduces the overall plan efficiency. 

**Abstract (ZH)**: 基于生成式人工智能的交通事件响应基准研究 

---
# A Foundation Model for Spatial Proteomics 

**Title (ZH)**: 空间蛋白质组学的基石模型 

**Authors**: Muhammad Shaban, Yuzhou Chang, Huaying Qiu, Yao Yu Yeo, Andrew H. Song, Guillaume Jaume, Yuchen Wang, Luca L. Weishaupt, Tong Ding, Anurag Vaidya, Abdallah Lamane, Daniel Shao, Mohammed Zidane, Yunhao Bai, Paige McCallum, Shuli Luo, Wenrui Wu, Yang Wang, Precious Cramer, Chi Ngai Chan, Pierre Stephan, Johanna Schaffenrath, Jia Le Lee, Hendrik A. Michel, Caiwei Tian, Cristina Almagro-Perez, Sophia J. Wagner, Sharifa Sahai, Ming Y. Lu, Richard J. Chen, Andrew Zhang, Mark Edward M. Gonzales, Ahmad Makky, Jia-Ying Joey Lee, Hao Cheng, Nourhan El Ahmar, Sayed Matar, Maximilian Haist, Darci Phillips, Yuqi Tan, Garry P. Nolan, W. Richard Burack, Jacob D. Estes, Jonathan T.C. Liu, Toni K Choueiri, Neeraj Agarwal, Marc Barry, Scott J. Rodig, Long Phi Le, Georg Gerber, Christian M. Schürch, Fabian J. Theis, Youn H Kim, Joe Yeong, Sabina Signoretti, Brooke E. Howitt, Lit-Hsin Loo, Qin Ma, Sizun Jiang, Faisal Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2506.03373)  

**Abstract**: Foundation models have begun to transform image analysis by acting as pretrained generalist backbones that can be adapted to many tasks even when post-training data are limited, yet their impact on spatial proteomics, imaging that maps proteins at single-cell resolution, remains limited. Here, we introduce KRONOS, a foundation model built for spatial proteomics. KRONOS was trained in a self-supervised manner on over 47 million image patches covering 175 protein markers, 16 tissue types, and 8 fluorescence-based imaging platforms. We introduce key architectural adaptations to address the high-dimensional, multi-channel, and heterogeneous nature of multiplex imaging. We demonstrate that KRONOS learns biologically meaningful representations across multiple scales, ranging from cellular and microenvironment to tissue levels, enabling it to address diverse downstream tasks, including cell phenotyping, region classification, and patient stratification. Evaluated across 11 independent cohorts, KRONOS achieves state-of-the-art performance across cell phenotyping, treatment response prediction, and retrieval tasks, and is highly data-efficient. KRONOS also introduces the paradigm of segmentation-free patch-level processing for efficient and scalable spatial proteomics analysis, allowing cross-institutional comparisons, and as an image reverse search engine for spatial patterns. Together, these results position KRONOS as a flexible and scalable tool for spatial proteomics. The model is publicly accessible at this https URL. 

**Abstract (ZH)**: 基础模型已经开始通过充当可以适应多种任务的预训练通才骨干来转变图像分析，即使在后训练数据有限的情况下也如此，但它们对空间蛋白质组学的影响仍未得到充分利用，空间蛋白质组学是通过单细胞分辨率映射蛋白质的成像技术。在这里，我们介绍了KRONOS，一种为空间蛋白质组学构建的基础模型。KRONOS以自我监督的方式在涵盖175种蛋白质标记、16种组织类型和8种基于荧光的成像平台的超过4700万图像片段上进行训练。我们引入了关键的架构调整，以解决多重成像的高维、多通道和异质性问题。我们证明KRONOS能够在从细胞和微环境到组织的不同尺度上学习生物学上意义重大的表示，使其能够解决多种下游任务，包括细胞表型分类、区域分类和患者分层。在11个独立队列中评估，KRONOS在细胞表型分类、治疗反应预测和检索任务上均达到最佳性能，且非常数据高效。KRONOS还引入了无分割的片段级处理范式，实现了高效且可扩展的空间蛋白质组学分析，允许跨机构比较，并作为空间模式的图像逆向搜索引擎。这些结果将KRONOS定位为一个灵活且可扩展的空间蛋白质组学工具。该模型已在此网址公开访问：https://。 

---
# Chipmunk: Training-Free Acceleration of Diffusion Transformers with Dynamic Column-Sparse Deltas 

**Title (ZH)**: Chipmunk: 无需训练的基于动态列稀疏增量的扩散变换器加速方法 

**Authors**: Austin Silveria, Soham V. Govande, Daniel Y. Fu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03275)  

**Abstract**: Diffusion Transformers (DiTs) have achieved state-of-the-art performance in high-quality image and video generation but incur substantial compute cost at inference. A common observation is that DiT latent noise vectors change slowly across inference steps, which suggests that the DiT compute may be redundant across steps. In this paper, we aim to speed up inference by reducing this redundancy, without additional training. We first study how activations change between steps in two state-of-the-art open-source DiTs. We find that just 5-25% of the values in attention and MLP explain 70-90% of the change in activations across steps. This finding motivates our approach, Chipmunk, which uses dynamic sparsity at inference time to recompute only the fastest-changing intermediate activations, while caching the rest. Dynamic sparsity introduces two systems challenges: (1) sparse attention and MLP operations tend to underutilize GPU tensor cores; and (2) computing dynamic sparsity patterns at runtime and caching activations both introduce overhead. To address these challenges, Chipmunk first uses a voxel-based reordering of input tokens to introduce column-wise sparsity. We implement column-sparse kernels utilizing efficient sparse gathers from global to shared GPU memory, achieving a 9.3x speedup at 93% sparsity compared to highly-optimized dense baselines. Second, Chipmunk overlaps the computation of sparsity patterns and cache updates with other parts of the computation (e.g., second layer of the MLP) to hide the extra latency. Chipmunk achieves up to 2.16x speedup on HunyuanVideo and 1.41x on FLUX.1-dev without compromising generation quality. Furthermore, we show that Chipmunk can be stacked on top of full step caching, achieving a 3.72x speedup on HunyuanVideo, a 2.67x speedup on WAN2.1, and a 2.25x speedup on FLUX.1-dev with minimal quality impact. 

**Abstract (ZH)**: Chipmunk: Reducing Redundancy for Efficient Inference of Diffusion Transformers 

---
# UniSite: The First Cross-Structure Dataset and Learning Framework for End-to-End Ligand Binding Site Detection 

**Title (ZH)**: UniSite: 首个多结构域数据集及端到端配体结合位点检测学习框架 

**Authors**: Jigang Fan, Quanlin Wu, Shengjie Luo, Liwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03237)  

**Abstract**: The detection of ligand binding sites for proteins is a fundamental step in Structure-Based Drug Design. Despite notable advances in recent years, existing methods, datasets, and evaluation metrics are confronted with several key challenges: (1) current datasets and methods are centered on individual protein-ligand complexes and neglect that diverse binding sites may exist across multiple complexes of the same protein, introducing significant statistical bias; (2) ligand binding site detection is typically modeled as a discontinuous workflow, employing binary segmentation and subsequent clustering algorithms; (3) traditional evaluation metrics do not adequately reflect the actual performance of different binding site prediction methods. To address these issues, we first introduce UniSite-DS, the first UniProt (Unique Protein)-centric ligand binding site dataset, which contains 4.81 times more multi-site data and 2.08 times more overall data compared to the previously most widely used datasets. We then propose UniSite, the first end-to-end ligand binding site detection framework supervised by set prediction loss with bijective matching. In addition, we introduce Average Precision based on Intersection over Union (IoU) as a more accurate evaluation metric for ligand binding site prediction. Extensive experiments on UniSite-DS and several representative benchmark datasets demonstrate that IoU-based Average Precision provides a more accurate reflection of prediction quality, and that UniSite outperforms current state-of-the-art methods in ligand binding site detection. The dataset and codes will be made publicly available at this https URL. 

**Abstract (ZH)**: 基于蛋白质的配体结合位点检测是结构基于药物设计中的一个基本步骤。尽管近年来取得了显著进展，现有的方法、数据集和评估指标仍面临几个关键挑战：（1）当前的数据集和方法主要集中在单一蛋白质-配体复合物上，忽视了相同蛋白质的不同复合物中可能存在多样化的结合位点，引入了显著的统计偏差；（2）配体结合位点检测通常被认为是断续的工作流程，使用二元分割和后续聚类算法；（3）传统的评估指标未能充分反映不同结合位点预测方法的实际性能。为了应对这些问题，我们首先引入了UniSite-DS，这是第一个以UniProt为中心的配体结合位点数据集，包含比之前最广泛使用的数据集多4.81倍的多位点数据和2.08倍的整体数据。然后，我们提出了UniSite，这是第一个基于集合预测损失且使用双射匹配监督的端到端配体结合位点检测框架。此外，我们引入了基于交并比（IoU）的平均精度作为更准确的配体结合位点预测评估指标。在UniSite-DS和几个代表性基准数据集上的广泛实验表明，基于IoU的平均精度提供了更准确的预测质量反映，且UniSite在配体结合位点检测中超越了当前最先进的方法。数据集和代码将在此处公开。 

---
# Pre-trained Vision-Language Models Assisted Noisy Partial Label Learning 

**Title (ZH)**: 预训练多模态模型辅助噪声_partial_标签学习 

**Authors**: Qian-Wei Wang, Yuqiu Xie, Letian Zhang, Zimo Liu, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2506.03229)  

**Abstract**: In the context of noisy partial label learning (NPLL), each training sample is associated with a set of candidate labels annotated by multiple noisy annotators. With the emergence of high-performance pre-trained vision-language models (VLMs) such as CLIP, LLaVa and GPT-4V, the direction of using these models to replace time-consuming manual annotation workflows and achieve "manual-annotation-free" training for downstream tasks has become a highly promising research avenue. This paper focuses on learning from noisy partial labels annotated by pre-trained VLMs and proposes an innovative collaborative consistency regularization (Co-Reg) method. Unlike the symmetric noise primarily addressed in traditional noisy label learning, the noise generated by pre-trained models is instance-dependent, embodying the underlying patterns of the pre-trained models themselves, which significantly increases the learning difficulty for the model. To address this, we simultaneously train two neural networks that implement collaborative purification of training labels through a "Co-Pseudo-Labeling" mechanism, while enforcing consistency regularization constraints in both the label space and feature representation space. Our method can also leverage few-shot manually annotated valid labels to further enhance its performances. Comparative experiments with different denoising and disambiguation algorithms, annotation manners, and pre-trained model application schemes fully validate the effectiveness of the proposed method, while revealing the broad prospects of integrating weakly-supervised learning techniques into the knowledge distillation process of pre-trained models. 

**Abstract (ZH)**: 基于预训练视觉-语言模型的噪声部分标签学习中的协作一致性正则化方法 

---
# Bridging Neural ODE and ResNet: A Formal Error Bound for Safety Verification 

**Title (ZH)**: 连接神经ODE和ResNet：安全验证的正式误差界 

**Authors**: Abdelrahman Sayed Sayed, Pierre-Jean Meyer, Mohamed Ghazel  

**Link**: [PDF](https://arxiv.org/pdf/2506.03227)  

**Abstract**: A neural ordinary differential equation (neural ODE) is a machine learning model that is commonly described as a continuous depth generalization of a residual network (ResNet) with a single residual block, or conversely, the ResNet can be seen as the Euler discretization of the neural ODE. These two models are therefore strongly related in a way that the behaviors of either model are considered to be an approximation of the behaviors of the other. In this work, we establish a more formal relationship between these two models by bounding the approximation error between two such related models. The obtained error bound then allows us to use one of the models as a verification proxy for the other, without running the verification tools twice: if the reachable output set expanded by the error bound satisfies a safety property on one of the models, this safety property is then guaranteed to be also satisfied on the other model. This feature is fully reversible, and the initial safety verification can be run indifferently on either of the two models. This novel approach is illustrated on a numerical example of a fixed-point attractor system modeled as a neural ODE. 

**Abstract (ZH)**: 一种神经常微分方程（神经ODE）是一种常见的连续深度残差网络（ResNet）单个残差块的连续深度泛化，或者反过来，ResNet 可以被视为神经ODE 的欧拉格式化。这两种模型因此有着密切的关系，即每种模型的行为都被认为是另一种模型行为的近似。在本文中，我们通过界定向相关两种模型之间的近似误差来建立它们之间更正式的关系。得到的误差界允许我们不必运行验证工具两次，即可使用其中一种模型作为另一种模型的验证代理：如果误差界的可达输出集在其中一种模型上满足安全性属性，则这种安全性属性也保证在另一种模型上被满足。这一特性是完全可逆的，初始的安全性验证可以在这两种模型中任意一个上运行。本文以一个固定点吸引子系统作为神经ODE 的数值示例来说明这一新颖的方法。 

---
# Multiple-Frequencies Population-Based Training 

**Title (ZH)**: 多频率基于群体的训练 

**Authors**: Waël Doulazmi, Auguste Lehuger, Marin Toromanoff, Valentin Charraut, Thibault Buhet, Fabien Moutarde  

**Link**: [PDF](https://arxiv.org/pdf/2506.03225)  

**Abstract**: Reinforcement Learning's high sensitivity to hyperparameters is a source of instability and inefficiency, creating significant challenges for practitioners. Hyperparameter Optimization (HPO) algorithms have been developed to address this issue, among them Population-Based Training (PBT) stands out for its ability to generate hyperparameters schedules instead of fixed configurations. PBT trains a population of agents, each with its own hyperparameters, frequently ranking them and replacing the worst performers with mutations of the best agents. These intermediate selection steps can cause PBT to focus on short-term improvements, leading it to get stuck in local optima and eventually fall behind vanilla Random Search over longer timescales. This paper studies how this greediness issue is connected to the choice of evolution frequency, the rate at which the selection is done. We propose Multiple-Frequencies Population-Based Training (MF-PBT), a novel HPO algorithm that addresses greediness by employing sub-populations, each evolving at distinct frequencies. MF-PBT introduces a migration process to transfer information between sub-populations, with an asymmetric design to balance short and long-term optimization. Extensive experiments on the Brax suite demonstrate that MF-PBT improves sample efficiency and long-term performance, even without actually tuning hyperparameters. 

**Abstract (ZH)**: 基于多个频率的 Population-Based Training 算法：一种缓解贪婪问题的新型超参数优化方法 

---
# OpenCarbon: A Contrastive Learning-based Cross-Modality Neural Approach for High-Resolution Carbon Emission Prediction Using Open Data 

**Title (ZH)**: OpenCarbon：基于对比学习的多模态神经网络方法用于开放数据驱动的高分辨率碳排放预测 

**Authors**: Jinwei Zeng, Yu Liu, Guozhen Zhang, Jingtao Ding, Yuming Lin, Jian Yuan, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.03224)  

**Abstract**: Accurately estimating high-resolution carbon emissions is crucial for effective emission governance and mitigation planning. While conventional methods for precise carbon accounting are hindered by substantial data collection efforts, the rise of open data and advanced learning techniques offers a promising solution. Once an open data-based prediction model is developed and trained, it can easily infer emissions for new areas based on available open data. To address this, we incorporate two modalities of open data, satellite images and point-of-interest (POI) data, to predict high-resolution urban carbon emissions, with satellite images providing macroscopic and static and POI data offering fine-grained and relatively dynamic functionality information. However, estimating high-resolution carbon emissions presents two significant challenges: the intertwined and implicit effects of various functionalities on carbon emissions, and the complex spatial contiguity correlations that give rise to the agglomeration effect. Our model, OpenCarbon, features two major designs that target the challenges: a cross-modality information extraction and fusion module to extract complementary functionality information from two modules and model their interactions, and a neighborhood-informed aggregation module to capture the spatial contiguity correlations. Extensive experiments demonstrate our model's superiority, with a significant performance gain of 26.6\% on R2. Further generalizability tests and case studies also show OpenCarbon's capacity to capture the intrinsic relation between urban functionalities and carbon emissions, validating its potential to empower efficient carbon governance and targeted carbon mitigation planning. Codes and data are available: this https URL. 

**Abstract (ZH)**: 基于开放数据的高分辨率城市碳排放准确估计对于有效的排放治理和减缓规划至关重要。尽管传统精细碳核算方法受限于大量数据收集工作，但开放数据和先进学习技术的兴起提供了前景广阔的解决方案。一旦基于开放数据的预测模型得到开发和训练，就可以根据现有开放数据轻松推断新地区的排放情况。为应对这一挑战，我们结合了卫星图像和兴趣点（POI）数据两种开放数据模态，以预测高分辨率城市碳排放，其中卫星图像提供宏观和静态信息，POI数据提供细粒度的相对动态功能信息。然而，高分辨率碳排放估计面临两个重要挑战：多种功能对碳排放的交织和隐含影响，以及复杂的空间连续性关联导致的集聚效应。我们的模型OpenCarbon针对这些挑战进行了两项主要设计：一种跨模态信息提取融合模块，用于从两个模态中提取互补的功能性信息并建模它们的相互作用；一种基于邻域的信息聚合模块，用于捕获空间连续性关联。广泛的实验证明了该模型的优越性，在R2上的性能提升达到26.6%。进一步的泛化测试和案例研究也表明，OpenCarbon能够捕捉城市功能与碳排放之间的内在关系，验证了其在提供有效碳治理和针对性碳减缓规划方面的能力。代码和数据可在以下链接获取：this https URL。 

---
# Beware! The AI Act Can Also Apply to Your AI Research Practices 

**Title (ZH)**: 小心！AI 法案也可能适用于你的 AI 研究实践。 

**Authors**: Alina Wernick, Kristof Meding  

**Link**: [PDF](https://arxiv.org/pdf/2506.03218)  

**Abstract**: The EU has become one of the vanguards in regulating the digital age. A particularly important regulation in the Artificial Intelligence (AI) domain is the EU AI Act, which entered into force in 2024. The AI Act specifies -- due to a risk-based approach -- various obligations for providers of AI systems. These obligations, for example, include a cascade of documentation and compliance measures, which represent a potential obstacle to science. But do these obligations also apply to AI researchers? This position paper argues that, indeed, the AI Act's obligations could apply in many more cases than the AI community is aware of. In our analysis of the AI Act and its applicability, we contribute the following: 1.) We give a high-level introduction to the AI Act aimed at non-legal AI research scientists. 2.) We explain with everyday research examples why the AI Act applies to research. 3.) We analyse the exceptions of the AI Act's applicability and state that especially scientific research exceptions fail to account for current AI research practices. 4.) We propose changes to the AI Act to provide more legal certainty for AI researchers and give two recommendations for AI researchers to reduce the risk of not complying with the AI Act. We see our paper as a starting point for a discussion between policymakers, legal scholars, and AI researchers to avoid unintended side effects of the AI Act on research. 

**Abstract (ZH)**: 欧盟已成为数字时代监管的先锋。特别是在人工智能（AI）领域，欧盟AI法案于2024年生效。该法案由于采取风险为基础的方法，对AI系统的提供者规定了多种义务。这些义务，例如，包括一系列的文件和合规措施，可能成为科学研究的障碍。但这些义务是否也适用于AI研究人员？本文认为，实际上，欧盟AI法案的义务可能比AI社区意识到的更为广泛。在我们对欧盟AI法案及其适用性的分析中，我们做出以下贡献：1）我们为非法律专业背景的AI研究科学家提供AI法案的高层次介绍。2）我们通过日常研究案例解释为什么欧盟AI法案适用于研究。3）我们分析欧盟AI法案的适用例外情况，并指出科学研究的例外情况未能充分考虑到当前的AI研究实践。4）我们提出对欧盟AI法案的修改建议，为AI研究人员提供更多法律确定性，并给出两条建议以降低不符合欧盟AI法案的风险。我们认为，我们的论文是政策制定者、法律学者和AI研究人员之间讨论的起点，以避免欧盟AI法案对研究的意外负面影响。 

---
# A Pre-trained Framework for Multilingual Brain Decoding Using Non-invasive Recordings 

**Title (ZH)**: 使用非侵入性记录数据的多语言脑解码预训练框架 

**Authors**: Yi Guo, Yihang Dong, Michael Kwok-Po Ng, Shuqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03214)  

**Abstract**: Brain-computer interfaces (BCIs) with speech decoding from brain recordings have broad application potential in fields such as clinical rehabilitation and cognitive neuroscience. However, current decoding methods remain limited to single-language, single-subject, and single neuroimaging modality settings, restricting their clinical applicability and generalizability. Here we propose a joint multilingual, multi-subject and multimodal decoding framework. It maps diverse brain recordings into a unified semantic space defined by a pre-trained multilingual model (PMM), enabling decoding across multiple languages, multiple subjects and multiple neuroimaging modalities. The proposed framework is validated using non-invasive brain recordings from 159 participants across four languages. Experimental results show that it exhibits strong generalization across multilingual, multi-subject, and multimodal settings. More importantly, the proposed framework can promote linguistic fairness, which is vital for underrepresented languages in BCI applications. The unified semantic space enables cross-lingual mapping enhancement, allowing the framework to boost the decoding performance of underrepresented languages, thereby promoting linguistic fairness. Overall, the proposed framework establishes a new potential paradigm for brain decoding, opening new paths for broader applications of BCI. 

**Abstract (ZH)**: 基于多语言、多被试和多模态解码的脑机接口框架 

---
# FuXi-Ocean: A Global Ocean Forecasting System with Sub-Daily Resolution 

**Title (ZH)**: FuXi-Ocean: 一个具有亚日分辨率的全球海洋预报系统 

**Authors**: Qiusheng Huang, Yuan Niu, Xiaohui Zhong, Anboyu Guo, Lei Chen, Dianjun Zhang, Xuefeng Zhang, Hao Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.03210)  

**Abstract**: Accurate, high-resolution ocean forecasting is crucial for maritime operations and environmental monitoring. While traditional numerical models are capable of producing sub-daily, eddy-resolving forecasts, they are computationally intensive and face challenges in maintaining accuracy at fine spatial and temporal scales. In contrast, recent data-driven approaches offer improved computational efficiency and emerging potential, yet typically operate at daily resolution and struggle with sub-daily predictions due to error accumulation over time. We introduce FuXi-Ocean, the first data-driven global ocean forecasting model achieving six-hourly predictions at eddy-resolving 1/12° spatial resolution, reaching depths of up to 1500 meters. The model architecture integrates a context-aware feature extraction module with a predictive network employing stacked attention blocks. The core innovation is the Mixture-of-Time (MoT) module, which adaptively integrates predictions from multiple temporal contexts by learning variable-specific reliability , mitigating cumulative errors in sequential forecasting. Through comprehensive experimental evaluation, FuXi-Ocean demonstrates superior skill in predicting key variables, including temperature, salinity, and currents, across multiple depths. 

**Abstract (ZH)**: 准确的高分辨率海洋预报对于海上操作和环境监测至关重要。传统数值模型能够生成亚日尺度、涡动分辨率的预报，但计算密集且在保持细尺度空间和时间分辨率的准确性方面面临挑战。相比之下，近期的数据驱动方法提高了计算效率并展现出了新兴潜力，但通常仅限于日尺度预报，并且在亚日尺度预测中由于时间累积误差而受到限制。我们引入了FuXi-Ocean，这是首个实现每六小时预报、涡动分辨率1/12°空间分辨率的全球海洋预报模型，可达到1500米深海。该模型架构集成了上下文感知特征提取模块和采用堆叠注意块的预测网络。核心创新是Mixture-of-Time (MoT) 模块，该模块通过学习变量特定的可靠性自适应地整合多种时间上下文的预测，从而减轻序列预报中的累积误差。通过全面的实验评估，FuXi-Ocean 在深度多个层次上展示了在预测温度、盐度和流速等关键变量方面的卓越能力。 

---
# Predicting Postoperative Stroke in Elderly SICU Patients: An Interpretable Machine Learning Model Using MIMIC Data 

**Title (ZH)**: 基于MIMIC数据的可解释机器学习模型：用于预测老年SICU术后卒中发生 

**Authors**: Tinghuan Li, Shuheng Chen, Junyi Fan, Elham Pishgar, Kamiar Alaei, Greg Placencia, Maryam Pishgar  

**Link**: [PDF](https://arxiv.org/pdf/2506.03209)  

**Abstract**: Postoperative stroke remains a critical complication in elderly surgical intensive care unit (SICU) patients, contributing to prolonged hospitalization, elevated healthcare costs, and increased mortality. Accurate early risk stratification is essential to enable timely intervention and improve clinical outcomes. We constructed a combined cohort of 19,085 elderly SICU admissions from the MIMIC-III and MIMIC-IV databases and developed an interpretable machine learning (ML) framework to predict in-hospital stroke using clinical data from the first 24 hours of Intensive Care Unit (ICU) stay. The preprocessing pipeline included removal of high-missingness features, iterative Singular Value Decomposition (SVD) imputation, z-score normalization, one-hot encoding, and class imbalance correction via the Adaptive Synthetic Sampling (ADASYN) algorithm. A two-stage feature selection process-combining Recursive Feature Elimination with Cross-Validation (RFECV) and SHapley Additive exPlanations (SHAP)-reduced the initial 80 variables to 20 clinically informative predictors. Among eight ML models evaluated, CatBoost achieved the best performance with an AUROC of 0.8868 (95% CI: 0.8802--0.8937). SHAP analysis and ablation studies identified prior cerebrovascular disease, serum creatinine, and systolic blood pressure as the most influential risk factors. Our results highlight the potential of interpretable ML approaches to support early detection of postoperative stroke and inform decision-making in perioperative critical care. 

**Abstract (ZH)**: 老年手术重症监护病房(SICU)患者术后中风仍然是一个关键并发症，会导致住院时间延长、医疗成本增加以及死亡率提高。准确的早期风险分层是必要的，以便能够及时干预并改善临床结局。我们从MIMIC-III和MIMIC-IV数据库中构建了一个包含19,085例老年SICU入院的联合队列，并开发了一个可解释的机器学习(ML)框架，利用重症监护病房(ICU)住院前24小时的临床数据预测院内中风。预处理管道包括缺失值特征的移除、迭代奇异值分解(SVD)插补、z-score归一化、独热编码以及通过自适应合成采样(ADASYN)算法进行的类别不平衡校正。通过结合递归特征消除与交叉验证(RFECV)和SHapley添加解释(SHAP)的两阶段特征选择过程，从初始的80个变量中选择了20个临床相关的预测变量。在评估的八个ML模型中，CatBoost取得了最佳性能，AUC-ROC为0.8868（95%CI：0.8802-0.8937）。SHAP分析和消融研究确定了既往脑血管疾病、血清肌酐和收缩压是最具影响力的危险因素。我们的研究结果突显了可解释的ML方法在支持术后中风的早期检测和 perioperative 手术期重症监护中的决策制定方面的潜在价值。 

---
# Fingerprinting Deep Learning Models via Network Traffic Patterns in Federated Learning 

**Title (ZH)**: 基于联邦学习中网络流量模式的深度学习模型指纹识别 

**Authors**: Md Nahid Hasan Shuvo, Moinul Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2506.03207)  

**Abstract**: Federated Learning (FL) is increasingly adopted as a decentralized machine learning paradigm due to its capability to preserve data privacy by training models without centralizing user data. However, FL is susceptible to indirect privacy breaches via network traffic analysis-an area not explored in existing research. The primary objective of this research is to study the feasibility of fingerprinting deep learning models deployed within FL environments by analyzing their network-layer traffic information. In this paper, we conduct an experimental evaluation using various deep learning architectures (i.e., CNN, RNN) within a federated learning testbed. We utilize machine learning algorithms, including Support Vector Machines (SVM), Random Forest, and Gradient-Boosting, to fingerprint unique patterns within the traffic data. Our experiments show high fingerprinting accuracy, achieving 100% accuracy using Random Forest and around 95.7% accuracy using SVM and Gradient Boosting classifiers. This analysis suggests that we can identify specific architectures running within the subsection of the network traffic. Hence, if an adversary knows about the underlying DL architecture, they can exploit that information and conduct targeted attacks. These findings suggest a notable security vulnerability in FL systems and the necessity of strengthening it at the network level. 

**Abstract (ZH)**: 联邦学习中的深层学习模型指纹识别研究：基于网络层流量信息的可行性分析 

---
# Infinity Parser: Layout Aware Reinforcement Learning for Scanned Document Parsing 

**Title (ZH)**: 无限解析器：面向布局的强化学习扫描文档解析 

**Authors**: Baode Wang, Biao Wu, Weizhen Li, Meng Fang, Yanjie Liang, Zuming Huang, Haozhe Wang, Jun Huang, Ling Chen, Wei Chu, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.03197)  

**Abstract**: Automated parsing of scanned documents into richly structured, machine-readable formats remains a critical bottleneck in Document AI, as traditional multi-stage pipelines suffer from error propagation and limited adaptability to diverse layouts. We introduce layoutRL, an end-to-end reinforcement learning framework that trains models to be explicitly layout-aware by optimizing a composite reward of normalized edit distance, paragraph count accuracy, and reading order preservation. Leveraging our newly released dataset, Infinity-Doc-55K, which combines 55K high-fidelity synthetic scanned document parsing data with expert-filtered real-world documents, we instantiate layoutRL in a vision-language-model-based parser called Infinity-Parser. Evaluated on English and Chinese benchmarks for OCR, table and formula extraction, and reading order detection, Infinity-Parser achieves new state-of-the-art performance in both accuracy and structural fidelity, outpacing specialist pipelines and general-purpose vision-language models. We will publicly release our code and dataset to accelerate progress in robust document understanding. 

**Abstract (ZH)**: 基于强化学习的自动扫描文档解析方法及其在文档AI中的应用：从传统多阶段管道到布局感知端到端框架 

---
# Encoding of Demographic and Anatomical Information in Chest X-Ray-based Severe Left Ventricular Hypertrophy Classifiers 

**Title (ZH)**: 基于胸部X光的严重左室肥大分类器中的人口统计和解剖信息编码 

**Authors**: Basudha Pal, Rama Chellappa, Muhammad Umair  

**Link**: [PDF](https://arxiv.org/pdf/2506.03192)  

**Abstract**: While echocardiography and MRI are clinical standards for evaluating cardiac structure, their use is limited by cost and this http URL introduce a direct classification framework that predicts severe left ventricular hypertrophy from chest X-rays, without relying on anatomical measurements or demographic inputs. Our approach achieves high AUROC and AUPRC, and employs Mutual Information Neural Estimation to quantify feature expressivity. This reveals clinically meaningful attribute encoding and supports transparent model interpretation. 

**Abstract (ZH)**: 无需左心室解剖测量的胸部X光图像直接分类框架预测严重左心室肥厚 

---
# MINT: Memory-Infused Prompt Tuning at Test-time for CLIP 

**Title (ZH)**: MINT：测试时记忆注入提示调优 for CLIP 

**Authors**: Jiaming Yi, Ruirui Pan, Jishen Yang, Xiulong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03190)  

**Abstract**: Improving the generalization ability of Vision-Language Pre-trained Models (VLMs) under test-time data distribution shifts remains a critical challenge. The existing Test-Time Adaptation (TTA) methods fall short in fully leveraging the model's internal knowledge, particularly in dynamically adapting to complex and hierarchical visual semantic information. In this paper, we propose Memory-Infused Prompt Tuning (MINT), a novel framework to address this issue. Inspired by human associative memory theory, MINT introduces a Memory Prompt Bank (MPB), which stores learnable key-value prompt pairs that work as a memory of previously seen samples. During the test time, relevant prompt pairs in the MPB are retrieved by the hierarchical visual features of test images to dynamically assemble Associative Prompts. The associative prompts are then injected into the image encoder for fine-grained, customized visual contextual guidance. MINT also utilizes learnable text prompts. MINT thus enables rapid, precise VLM adaptation at test time by leveraging this MPB-acquired memory, without source data or retraining. The code is available at this https URL. 

**Abstract (ZH)**: 改进视图-语言预训练模型（VLMs）在测试时数据分布偏移下的泛化能力仍然是一个关键挑战。现有的测试时适调（TTA）方法在充分利用模型内部知识方面存在不足，特别是在动态适应复杂的层次视觉语义信息方面。在本文中，我们提出了一种新的框架——记忆融合提示调优（MINT），以解决这一问题。受人类关联记忆理论的启发，MINT引入了一个记忆提示库（MPB），该库存储可学习的键值提示对，作为之前见过的样本的记忆。在测试时，MPB 中的相关提示对通过测试图像的层次视觉特征进行检索，以动态组装关联提示。这些关联提示随后被注入到图像编码器中，以提供细粒度且定制化的视觉上下文指导。MINT 还利用可学习的文本提示。因此，MINT 通过利用 MPB 获得的记忆，在测试时能够实现快速且精确的 VLM 适调，无需源数据或重新训练。代码可在以下链接获取。 

---
# DLiPath: A Benchmark for the Comprehensive Assessment of Donor Liver Based on Histopathological Image Dataset 

**Title (ZH)**: DLiPath: 基于组织病理图像数据集的供肝综合评估基准 

**Authors**: Liangrui Pan, Xingchen Li, Zhongyi Chen, Ling Chu, Shaoliang Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.03185)  

**Abstract**: Pathologists comprehensive evaluation of donor liver biopsies provides crucial information for accepting or discarding potential grafts. However, rapidly and accurately obtaining these assessments intraoperatively poses a significant challenge for pathologists. Features in donor liver biopsies, such as portal tract fibrosis, total steatosis, macrovesicular steatosis, and hepatocellular ballooning are correlated with transplant outcomes, yet quantifying these indicators suffers from substantial inter- and intra-observer variability. To address this, we introduce DLiPath, the first benchmark for comprehensive donor liver assessment based on a histopathology image dataset. We collected and publicly released 636 whole slide images from 304 donor liver patients at the Department of Pathology, the Third Xiangya Hospital, with expert annotations for key pathological features (including cholestasis, portal tract fibrosis, portal inflammation, total steatosis, macrovesicular steatosis, and hepatocellular ballooning). We selected nine state-of-the-art multiple-instance learning (MIL) models based on the DLiPath dataset as baselines for extensive comparative analysis. The experimental results demonstrate that several MIL models achieve high accuracy across donor liver assessment indicators on DLiPath, charting a clear course for future automated and intelligent donor liver assessment research. Data and code are available at this https URL. 

**Abstract (ZH)**: 基于组织病理学图像数据集的DLiPath：第一个全面供肝评估基准 

---
# Edge Computing for Physics-Driven AI in Computational MRI: A Feasibility Study 

**Title (ZH)**: 基于物理驱动AI的计算MRI中边缘计算可行性研究 

**Authors**: Yaşar Utku Alçalar, Yu Cao, Mehmet Akçakaya  

**Link**: [PDF](https://arxiv.org/pdf/2506.03183)  

**Abstract**: Physics-driven artificial intelligence (PD-AI) reconstruction methods have emerged as the state-of-the-art for accelerating MRI scans, enabling higher spatial and temporal resolutions. However, the high resolution of these scans generates massive data volumes, leading to challenges in transmission, storage, and real-time processing. This is particularly pronounced in functional MRI, where hundreds of volumetric acquisitions further exacerbate these demands. Edge computing with FPGAs presents a promising solution for enabling PD-AI reconstruction near the MRI sensors, reducing data transfer and storage bottlenecks. However, this requires optimization of PD-AI models for hardware efficiency through quantization and bypassing traditional FFT-based approaches, which can be a limitation due to their computational demands. In this work, we propose a novel PD-AI computational MRI approach optimized for FPGA-based edge computing devices, leveraging 8-bit complex data quantization and eliminating redundant FFT/IFFT operations. Our results show that this strategy improves computational efficiency while maintaining reconstruction quality comparable to conventional PD-AI methods, and outperforms standard clinical methods. Our approach presents an opportunity for high-resolution MRI reconstruction on resource-constrained devices, highlighting its potential for real-world deployment. 

**Abstract (ZH)**: 基于物理驱动的人工智能的医用磁共振成像重建方法：面向FPGA边缘计算的优化 

---
# EdgeVidSum: Real-Time Personalized Video Summarization at the Edge 

**Title (ZH)**: EdgeVidSum: 边缘端个性化视频摘要生成 

**Authors**: Ghulam Mujtaba, Eun-Seok Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03171)  

**Abstract**: EdgeVidSum is a lightweight method that generates personalized, fast-forward summaries of long-form videos directly on edge devices. The proposed approach enables real-time video summarization while safeguarding user privacy through local data processing using innovative thumbnail-based techniques and efficient neural architectures. Unlike conventional methods that process entire videos frame by frame, the proposed method uses thumbnail containers to significantly reduce computational complexity without sacrificing semantic relevance. The framework employs a hierarchical analysis approach, where a lightweight 2D CNN model identifies user-preferred content from thumbnails and generates timestamps to create fast-forward summaries. Our interactive demo highlights the system's ability to create tailored video summaries for long-form videos, such as movies, sports events, and TV shows, based on individual user preferences. The entire computation occurs seamlessly on resource-constrained devices like Jetson Nano, demonstrating how EdgeVidSum addresses the critical challenges of computational efficiency, personalization, and privacy in modern video consumption environments. 

**Abstract (ZH)**: EdgeVidSum是一种轻量级方法，可直接在边缘设备上生成长格式视频的个性化快进摘要。该提出的方案通过使用创新性的缩略图为基础的技术和高效神经架构，在本地进行数据处理，从而实现实时视频摘要生成，并保护用户隐私。与传统的逐帧处理视频的方法不同，提出的方案使用缩略图容器显著减少了计算复杂性，同时保持语义相关性。该框架采用分层分析方法，使用轻量级的2D CNN模型从缩略图中识别用户偏好内容，并生成时间戳以创建快进摘要。我们的互动演示展示了系统能够根据个人用户偏好为长格式视频（如电影、体育赛事和电视节目）创建定制视频摘要的能力。整个计算过程无缝地在资源受限的设备（如Jetson Nano）上完成，展示了EdgeVidSum如何在现代视频消费环境中解决计算效率、个性化和隐私的关键挑战。 

---
# Hierarchical Relational Learning for Few-Shot Knowledge Graph Completion 

**Title (ZH)**: few-shot 知识图 completion 的分层关系学习 

**Authors**: Han Wu, Jie Yin, Bala Rajaratnam, Jianyuan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2209.01205)  

**Abstract**: Knowledge graphs (KGs) are powerful in terms of their inference abilities, but are also notorious for their incompleteness and long-tail distribution of relations. To address these challenges and expand the coverage of KGs, few-shot KG completion aims to make predictions for triplets involving novel relations when only a few training triplets are provided as reference. Previous methods have focused on designing local neighbor aggregators to learn entity-level information and/or imposing a potentially invalid sequential dependency assumption at the triplet level to learn meta relation information. However, pairwise triplet-level interactions and context-level relational information have been largely overlooked for learning meta representations of few-shot relations. In this paper, we propose a hierarchical relational learning method (HiRe) for few-shot KG completion. By jointly capturing three levels of relational information (entity-level, triplet-level and context-level), HiRe can effectively learn and refine meta representations of few-shot relations, and thus generalize well to new unseen relations. Extensive experiments on benchmark datasets validate the superiority of HiRe over state-of-the-art methods. The code can be found in this https URL. 

**Abstract (ZH)**: Few-shot知识图谱完成中的层级关系学习方法 

---
