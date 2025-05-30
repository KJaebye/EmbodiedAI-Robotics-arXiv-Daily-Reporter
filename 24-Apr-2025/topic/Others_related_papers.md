# Fast Online Adaptive Neural MPC via Meta-Learning 

**Title (ZH)**: 快速在线自适应神经MPC通过元学习 

**Authors**: Yu Mei, Xinyu Zhou, Shuyang Yu, Vaibhav Srivastava, Xiaobo Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.16369)  

**Abstract**: Data-driven model predictive control (MPC) has demonstrated significant potential for improving robot control performance in the presence of model uncertainties. However, existing approaches often require extensive offline data collection and computationally intensive training, limiting their ability to adapt online. To address these challenges, this paper presents a fast online adaptive MPC framework that leverages neural networks integrated with Model-Agnostic Meta-Learning (MAML). Our approach focuses on few-shot adaptation of residual dynamics - capturing the discrepancy between nominal and true system behavior - using minimal online data and gradient steps. By embedding these meta-learned residual models into a computationally efficient L4CasADi-based MPC pipeline, the proposed method enables rapid model correction, enhances predictive accuracy, and improves real-time control performance. We validate the framework through simulation studies on a Van der Pol oscillator, a Cart-Pole system, and a 2D quadrotor. Results show significant gains in adaptation speed and prediction accuracy over both nominal MPC and nominal MPC augmented with a freshly initialized neural network, underscoring the effectiveness of our approach for real-time adaptive robot control. 

**Abstract (ZH)**: 基于数据驱动模型预测控制的快速在线自适应框架：融合模型无感知元学习的残差动力学适应 

---
# Insect-Computer Hybrid Speaker: Speaker using Chirp of the Cicada Controlled by Electrical Muscle Stimulation 

**Title (ZH)**: 昆虫-计算机混合语音装置：通过电肌肉刺激控制的蝉鸣语音发生器 

**Authors**: Yuga Tsukuda, Naoto Nishida, Jun Lu, Yoichi Ochiai  

**Link**: [PDF](https://arxiv.org/pdf/2504.16459)  

**Abstract**: We propose "Insect-Computer Hybrid Speaker", which enables us to make musics made from combinations of computer and insects. Lots of studies have proposed methods and interfaces for controlling insects and obtaining feedback. However, there have been less research on the use of insects for interaction with third parties. In this paper, we propose a method in which cicadas are used as speakers triggered by using Electrical Muscle Stimulation (EMS). We explored and investigated the suitable waveform of chirp to be controlled, the appropriate voltage range, and the maximum pitch at which cicadas can chirp. 

**Abstract (ZH)**: 昆虫-计算机混合音箱：基于电肌肉刺激的蝉鸣音乐生成方法 

---
# Eigendecomposition Parameterization of Penalty Matrices for Enhanced Control Design: Aerospace Applications 

**Title (ZH)**: Penalty矩阵特征分解参数化方法在航空航天控制设计中的增强应用 

**Authors**: Nicholas P. Nurre, Ehsan Taheri  

**Link**: [PDF](https://arxiv.org/pdf/2504.16328)  

**Abstract**: Modern control algorithms require tuning of square weight/penalty matrices appearing in quadratic functions/costs to improve performance and/or stability output. Due to simplicity in gain-tuning and enforcing positive-definiteness, diagonal penalty matrices are used extensively in control methods such as linear quadratic regulator (LQR), model predictive control, and Lyapunov-based control. In this paper, we propose an eigendecomposition approach to parameterize penalty matrices, allowing positive-definiteness with non-zero off-diagonal entries to be implicitly satisfied, which not only offers notable computational and implementation advantages, but broadens the class of achievable controls. We solve three control problems: 1) a variation of Zermelo's navigation problem, 2) minimum-energy spacecraft attitude control using both LQR and Lyapunov-based methods, and 3) minimum-fuel and minimum-time Lyapunov-based low-thrust trajectory design. Particle swarm optimization is used to optimize the decision variables, which will parameterize the penalty matrices. The results demonstrate improvements of up to 65% in the performance objective in the example problems utilizing the proposed method. 

**Abstract (ZH)**: 现代控制算法需要调节出现在二次函数/成本中的方权重/惩罚矩阵以提高性能和/或稳定性输出。由于增益调节简单且能保证正定性，对角惩罚矩阵在线性二次调节（LQR）、模型预测控制和Lyapunov基于的控制方法中广泛使用。本文提出了一种特征值分解方法来参数化惩罚矩阵，允许在非零非对角元素下隐式满足正定性，不仅提供了显著的计算和实现优势，还扩大了可实现控制的范围。我们解决了三个控制问题：1) Zermelo航行问题的变体；2) 使用LQR和Lyapunov基于方法的最小能量航天器姿态控制；3) 最小燃料和最小时间Lyapunov基于的小推力轨道设计。使用粒子群优化来优化决策变量，这些变量将参数化惩罚矩阵。结果显示，在使用所提方法的示例问题中，性能目标提高了最多65%。 

---
# MonoTher-Depth: Enhancing Thermal Depth Estimation via Confidence-Aware Distillation 

**Title (ZH)**: MonoTher-Depth：通过信心 Awareness 蒸馏增强热深度估计 

**Authors**: Xingxing Zuo, Nikhil Ranganathan, Connor Lee, Georgia Gkioxari, Soon-Jo Chung  

**Link**: [PDF](https://arxiv.org/pdf/2504.16127)  

**Abstract**: Monocular depth estimation (MDE) from thermal images is a crucial technology for robotic systems operating in challenging conditions such as fog, smoke, and low light. The limited availability of labeled thermal data constrains the generalization capabilities of thermal MDE models compared to foundational RGB MDE models, which benefit from datasets of millions of images across diverse scenarios. To address this challenge, we introduce a novel pipeline that enhances thermal MDE through knowledge distillation from a versatile RGB MDE model. Our approach features a confidence-aware distillation method that utilizes the predicted confidence of the RGB MDE to selectively strengthen the thermal MDE model, capitalizing on the strengths of the RGB model while mitigating its weaknesses. Our method significantly improves the accuracy of the thermal MDE, independent of the availability of labeled depth supervision, and greatly expands its applicability to new scenarios. In our experiments on new scenarios without labeled depth, the proposed confidence-aware distillation method reduces the absolute relative error of thermal MDE by 22.88\% compared to the baseline without distillation. 

**Abstract (ZH)**: 单目热成像深度估计：通过知识蒸馏提高在复杂环境中的鲁棒性 

---
# AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models with OpenMathReasoning dataset 

**Title (ZH)**: AIMO-2 获胜方案：基于 OpenMathReasoning 数据集构建领先数学推理模型 

**Authors**: Ivan Moshkov, Darragh Hanley, Ivan Sorokin, Shubham Toshniwal, Christof Henkel, Benedikt Schifferer, Wei Du, Igor Gitman  

**Link**: [PDF](https://arxiv.org/pdf/2504.16891)  

**Abstract**: This paper presents our winning submission to the AI Mathematical Olympiad - Progress Prize 2 (AIMO-2) competition. Our recipe for building state-of-the-art mathematical reasoning models relies on three key pillars. First, we create a large-scale dataset comprising 540K unique high-quality math problems, including olympiad-level problems, and their 3.2M long-reasoning solutions. Second, we develop a novel method to integrate code execution with long reasoning models through iterative training, generation, and quality filtering, resulting in 1.7M high-quality Tool-Integrated Reasoning solutions. Third, we create a pipeline to train models to select the most promising solution from many candidates. We show that such generative solution selection (GenSelect) can significantly improve upon majority voting baseline. Combining these ideas, we train a series of models that achieve state-of-the-art results on mathematical reasoning benchmarks. To facilitate further research, we release our code, models, and the complete OpenMathReasoning dataset under a commercially permissive license. 

**Abstract (ZH)**: 本文呈现了我们参加AI数学奥林匹克竞赛-进展奖（AIMO-2）的获奖提交。构建最先进的数学推理模型的关键在于三个方面。首先，我们构建了一个包含54万个高质量数学问题的大规模数据集，其中包括奥林匹克级别的问题及其320万字长推理解决方案。其次，我们开发了一种新颖方法，通过迭代训练、生成和质量过滤，将代码执行与长推理模型集成，生成170万高质量工具集成推理解决方案。第三，我们创建了一个流水线，用于训练模型从众多候选方案中选择最有潜力的方案。我们表明，生成性解决方案选择（GenSelect）可以显著优于多数投票基准。结合这些想法，我们训练了一系列模型，在数学推理基准测试中达到了最先进的成果。为了促进进一步的研究，我们以商业友好的许可证发布了我们的代码、模型和完整的OpenMathReasoning数据集。 

---
# Bridging Econometrics and AI: VaR Estimation via Reinforcement Learning and GARCH Models 

**Title (ZH)**: 经济计量学与人工智能的桥梁：基于强化学习和GARCH模型的VaR估计 

**Authors**: Fredy Pokou, Jules Sadefo Kamdem, François Benhmad  

**Link**: [PDF](https://arxiv.org/pdf/2504.16635)  

**Abstract**: In an environment of increasingly volatile financial markets, the accurate estimation of risk remains a major challenge. Traditional econometric models, such as GARCH and its variants, are based on assumptions that are often too rigid to adapt to the complexity of the current market dynamics. To overcome these limitations, we propose a hybrid framework for Value-at-Risk (VaR) estimation, combining GARCH volatility models with deep reinforcement learning. Our approach incorporates directional market forecasting using the Double Deep Q-Network (DDQN) model, treating the task as an imbalanced classification problem. This architecture enables the dynamic adjustment of risk-level forecasts according to market conditions. Empirical validation on daily Eurostoxx 50 data covering periods of crisis and high volatility shows a significant improvement in the accuracy of VaR estimates, as well as a reduction in the number of breaches and also in capital requirements, while respecting regulatory risk thresholds. The ability of the model to adjust risk levels in real time reinforces its relevance to modern and proactive risk management. 

**Abstract (ZH)**: 在日益波动的金融市场环境中，风险准确估计仍然是一个重大挑战。传统的经济计量模型，如GARCH及其变体，基于的假设往往过于僵化，难以适应当前市场动态的复杂性。为克服这些局限，我们提出了一个VaR估计的混合框架，结合了GARCH波动模型与深度强化学习。该方法使用双深Q网络（DDQN）模型进行市场方向预测，将任务视为不平衡分类问题。该架构能够根据市场条件动态调整风险水平预测。实证结果表明，该方法在危机时期和高波动期的日Eurostoxx 50数据上的VaR估计准确性有显著提高，减少了违约次数和资本要求，同时遵守监管风险阈值。模型能够实时调整风险水平，增强了其对现代和前瞻风险管理的相关性。 

---
# HTN Plan Repair Algorithms Compared: Strengths and Weaknesses of Different Methods 

**Title (ZH)**: HTN规划修复算法比较：不同方法的优势与弱点 

**Authors**: Paul Zaidins, Robert P. Goldman, Ugur Kuter, Dana Nau, Mark Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2504.16209)  

**Abstract**: This paper provides theoretical and empirical comparisons of three recent hierarchical plan repair algorithms: SHOPFixer, IPyHOPPER, and Rewrite. Our theoretical results show that the three algorithms correspond to three different definitions of the plan repair problem, leading to differences in the algorithms' search spaces, the repair problems they can solve, and the kinds of repairs they can make. Understanding these distinctions is important when choosing a repair method for any given application.
Building on the theoretical results, we evaluate the algorithms empirically in a series of benchmark planning problems. Our empirical results provide more detailed insight into the runtime repair performance of these systems and the coverage of the repair problems solved, based on algorithmic properties such as replanning, chronological backtracking, and backjumping over plan trees. 

**Abstract (ZH)**: 本文提供了三种近期层次计划修复算法的理论和实证比较：SHOPFixer、IPyHOPPER和Rewrite。我们的理论结果表明，这三种算法对应于计划修复问题的三种不同定义，导致它们的搜索空间、可解决的修复问题类型以及可施行的修复类型存在差异。在选择适用于特定应用的修复方法时，理解这些区别非常重要。基于算法特性（如重新规划、时间顺序回溯和计划树上的跨越回溯），我们进一步通过一系列基准规划问题对这些算法进行了实证评估，提供了关于这些系统运行时修复性能和解决的修复问题覆盖范围的更详细见解。 

---
# A Framework for Objective-Driven Dynamical Stochastic Fields 

**Title (ZH)**: 基于目标导向的动力学随机场框架 

**Authors**: Yibo Jacky Zhang, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2504.16115)  

**Abstract**: Fields offer a versatile approach for describing complex systems composed of interacting and dynamic components. In particular, some of these dynamical and stochastic systems may exhibit goal-directed behaviors aimed at achieving specific objectives, which we refer to as $\textit{intelligent fields}$. However, due to their inherent complexity, it remains challenging to develop a formal theoretical description of such systems and to effectively translate these descriptions into practical applications. In this paper, we propose three fundamental principles -- complete configuration, locality, and purposefulness -- to establish a theoretical framework for understanding intelligent fields. Moreover, we explore methodologies for designing such fields from the perspective of artificial intelligence applications. This initial investigation aims to lay the groundwork for future theoretical developments and practical advances in understanding and harnessing the potential of such objective-driven dynamical stochastic fields. 

**Abstract (ZH)**: 领域为描述由互动和动态组件组成的复杂系统提供了一种灵活的方法。特别是，这些动力学和随机系统中的一些可能表现出旨在实现特定目标的有目标导向的行为，我们称之为智能领域。然而，由于它们固有的复杂性，开发这些系统的正式理论描述并有效将其翻译为实际应用仍然具有挑战性。在本文中，我们提出三条基本原理——完整配置、局部性、和目的性——来建立理解智能领域的理论框架。此外，我们从人工智能应用的角度探讨了设计此类领域的方法论。这项初步研究旨在为未来理论发展和理解及利用此类目标导向的动力学随机领域的能力奠定基础。 

---
# I-Con: A Unifying Framework for Representation Learning 

**Title (ZH)**: I-Con：一种统一的表示学习框架 

**Authors**: Shaden Alshammari, John Hershey, Axel Feldmann, William T. Freeman, Mark Hamilton  

**Link**: [PDF](https://arxiv.org/pdf/2504.16929)  

**Abstract**: As the field of representation learning grows, there has been a proliferation of different loss functions to solve different classes of problems. We introduce a single information-theoretic equation that generalizes a large collection of modern loss functions in machine learning. In particular, we introduce a framework that shows that several broad classes of machine learning methods are precisely minimizing an integrated KL divergence between two conditional distributions: the supervisory and learned representations. This viewpoint exposes a hidden information geometry underlying clustering, spectral methods, dimensionality reduction, contrastive learning, and supervised learning. This framework enables the development of new loss functions by combining successful techniques from across the literature. We not only present a wide array of proofs, connecting over 23 different approaches, but we also leverage these theoretical results to create state-of-the-art unsupervised image classifiers that achieve a +8% improvement over the prior state-of-the-art on unsupervised classification on ImageNet-1K. We also demonstrate that I-Con can be used to derive principled debiasing methods which improve contrastive representation learners. 

**Abstract (ZH)**: 随着表示学习领域的发展，出现了多种不同的损失函数以解决不同类别的问题。我们引入了一个信息论方程来泛化机器学习中的多种现代损失函数。特别地，我们提出了一种框架，表明多种机器学习方法实际上是精确地最小化两个条件分布之间的集成KL散度：监督表示和学习到的表示。这一观点揭示了聚类、谱方法、降维、对比学习和监督学习背后的隐藏信息几何结构。该框架通过跨文献的成功技术结合，使开发新的损失函数成为可能。我们不仅呈现了将超过23种不同方法联系起来的广泛证明，还利用这些理论结果创建了在ImageNet-1K的无监督分类中比 previous state-of-the-art 提高了8% 的无监督图像分类器。我们还展示了I-Con可以用来推导出有效的去偏差方法，从而改进对比表示学习器。 

---
# Latent Diffusion Planning for Imitation Learning 

**Title (ZH)**: 潜在扩散规划用于模仿学习 

**Authors**: Amber Xie, Oleh Rybkin, Dorsa Sadigh, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2504.16925)  

**Abstract**: Recent progress in imitation learning has been enabled by policy architectures that scale to complex visuomotor tasks, multimodal distributions, and large datasets. However, these methods often rely on learning from large amount of expert demonstrations. To address these shortcomings, we propose Latent Diffusion Planning (LDP), a modular approach consisting of a planner which can leverage action-free demonstrations, and an inverse dynamics model which can leverage suboptimal data, that both operate over a learned latent space. First, we learn a compact latent space through a variational autoencoder, enabling effective forecasting of future states in image-based domains. Then, we train a planner and an inverse dynamics model with diffusion objectives. By separating planning from action prediction, LDP can benefit from the denser supervision signals of suboptimal and action-free data. On simulated visual robotic manipulation tasks, LDP outperforms state-of-the-art imitation learning approaches, as they cannot leverage such additional data. 

**Abstract (ZH)**: 近期模仿学习的进步得益于能够处理复杂视觉运动任务、多模态分布和大数据集的策略架构。然而，这些方法通常依赖于从大量专家演示中学习。为了解决这些不足，我们提出了一种模块化方法——潜在扩散规划（LDP），它由一个可以利用无动作演示的规划器和一个可以利用非最优数据的动力学逆模型组成，两者均在学习得到的潜在空间中运行。首先，通过变分自编码器学习一个紧凑的潜在空间，从而使基于图像的域中未来的状态预测变得有效。然后，我们使用扩散目标训练规划器和动力学逆模型。通过分离规划与动作预测，LDP可以从非最优和无动作数据的更密集监督信号中受益。在模拟的视觉机器人 manipulation 任务中，LDP 比最先进的模仿学习方法表现更优，因为后者无法利用此类额外数据。 

---
# BadVideo: Stealthy Backdoor Attack against Text-to-Video Generation 

**Title (ZH)**: BadVideo：针对文本生成视频的隐蔽后门攻击 

**Authors**: Ruotong Wang, Mingli Zhu, Jiarong Ou, Rui Chen, Xin Tao, Pengfei Wan, Baoyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.16907)  

**Abstract**: Text-to-video (T2V) generative models have rapidly advanced and found widespread applications across fields like entertainment, education, and marketing. However, the adversarial vulnerabilities of these models remain rarely explored. We observe that in T2V generation tasks, the generated videos often contain substantial redundant information not explicitly specified in the text prompts, such as environmental elements, secondary objects, and additional details, providing opportunities for malicious attackers to embed hidden harmful content. Exploiting this inherent redundancy, we introduce BadVideo, the first backdoor attack framework tailored for T2V generation. Our attack focuses on designing target adversarial outputs through two key strategies: (1) Spatio-Temporal Composition, which combines different spatiotemporal features to encode malicious information; (2) Dynamic Element Transformation, which introduces transformations in redundant elements over time to convey malicious information. Based on these strategies, the attacker's malicious target seamlessly integrates with the user's textual instructions, providing high stealthiness. Moreover, by exploiting the temporal dimension of videos, our attack successfully evades traditional content moderation systems that primarily analyze spatial information within individual frames. Extensive experiments demonstrate that BadVideo achieves high attack success rates while preserving original semantics and maintaining excellent performance on clean inputs. Overall, our work reveals the adversarial vulnerability of T2V models, calling attention to potential risks and misuse. Our project page is at this https URL. 

**Abstract (ZH)**: Text-to-video (T2V) 生成模型已迅速发展并在娱乐、教育和营销等领域找到了广泛的应用。然而，这些模型的对抗性漏洞仍然很少被探索。我们观察到，在T2V生成任务中，生成的视频中常常包含大量的冗余信息，这些信息并未明确地出现在文本提示中，如环境元素、次要对象和额外细节，为恶意攻击者提供了嵌入隐藏有害内容的机会。利用这种固有的冗余性，我们引入了BadVideo，这是首个针对T2V生成的后门攻击框架。我们的攻击主要通过两种关键策略设计目标对抗输出：(1) 空间-时间合成，将不同的空间-时间特征结合起来编码恶意信息；(2) 动态元素转变，通过在冗余元素上引入时间上的变换来传达恶意信息。基于这些策略，攻击者的恶意目标能够无缝地与用户的文本指令整合，提供极高的隐蔽性。此外，通过利用视频的时间维度，我们的攻击成功规避了主要分析单帧内空间信息的传统内容审核系统。广泛实验结果表明，BadVideo 在保持原始语义的同时，对抗成功率高且在干净输入上维持了优异的性能。整体而言，我们的工作揭示了T2V模型的对抗性漏洞，引起了潜在风险和滥用的注意。我们的项目页面在此 [链接]。 

---
# Building A Secure Agentic AI Application Leveraging A2A Protocol 

**Title (ZH)**: 基于A2A协议构建一个安全自主的人工智能应用 

**Authors**: Idan Habler, Ken Huang, Vineeth Sai Narajala, Prashant Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2504.16902)  

**Abstract**: As Agentic AI systems evolve from basic workflows to complex multi agent collaboration, robust protocols such as Google's Agent2Agent (A2A) become essential enablers. To foster secure adoption and ensure the reliability of these complex interactions, understanding the secure implementation of A2A is essential. This paper addresses this goal by providing a comprehensive security analysis centered on the A2A protocol. We examine its fundamental elements and operational dynamics, situating it within the framework of agent communication development. Utilizing the MAESTRO framework, specifically designed for AI risks, we apply proactive threat modeling to assess potential security issues in A2A deployments, focusing on aspects such as Agent Card management, task execution integrity, and authentication methodologies.
Based on these insights, we recommend practical secure development methodologies and architectural best practices designed to build resilient and effective A2A systems. Our analysis also explores how the synergy between A2A and the Model Context Protocol (MCP) can further enhance secure interoperability. This paper equips developers and architects with the knowledge and practical guidance needed to confidently leverage the A2A protocol for building robust and secure next generation agentic applications. 

**Abstract (ZH)**: 随着代理型人工智能系统从基本的工作流发展到复杂的多代理协作，像Google的Agent2Agent (A2A)这样的 robust 协议变得至关重要。为了促进安全采用并确保这些复杂交互的可靠性，理解A2A的安全实现是必不可少的。本文通过提供以A2A协议为中心的全面安全分析来实现这一目标，我们考察其基本要素和操作动态，并将其置于代理通信发展的框架中。利用专门设计用于AI风险的MAESTRO框架，我们应用前瞻性的威胁建模来评估A2A部署中的潜在安全问题，重点关注代理卡管理、任务执行完整性和认证方法等方面。

基于这些见解，我们建议实用的安全开发方法和架构最佳实践，以构建坚固且有效的A2A系统。我们的分析还探讨了A2A与模型上下文协议（MCP）之间的协同作用如何进一步增强安全互操作性。本文为开发者和架构师提供了所需的知识和实用指导，使他们能够自信地利用A2A协议构建坚固且安全的下一代代理型应用。 

---
# Approximating Optimal Labelings for Temporal Connectivity 

**Title (ZH)**: 接近最优标记以实现临时连通性 

**Authors**: Daniele Carnevale, Gianlorenzo D'Angelo, Martin Olsen  

**Link**: [PDF](https://arxiv.org/pdf/2504.16837)  

**Abstract**: In a temporal graph the edge set dynamically changes over time according to a set of time-labels associated with each edge that indicates at which time-steps the edge is available. Two vertices are connected if there is a path connecting them in which the edges are traversed in increasing order of their labels. We study the problem of scheduling the availability time of the edges of a temporal graph in such a way that all pairs of vertices are connected within a given maximum allowed time $a$ and the overall number of labels is minimized.
The problem, known as \emph{Minimum Aged Labeling} (MAL), has several applications in logistics, distribution scheduling, and information spreading in social networks, where carefully choosing the time-labels can significantly reduce infrastructure costs, fuel consumption, or greenhouse gases.
The problem MAL has previously been proved to be NP-complete on undirected graphs and \APX-hard on directed graphs. In this paper, we extend our knowledge on the complexity and approximability of MAL in several directions. We first show that the problem cannot be approximated within a factor better than $O(\log n)$ when $a\geq 2$, unless $\text{P} = \text{NP}$, and a factor better than $2^{\log ^{1-\epsilon} n}$ when $a\geq 3$, unless $\text{NP}\subseteq \text{DTIME}(2^{\text{polylog}(n)})$, where $n$ is the number of vertices in the graph. Then we give a set of approximation algorithms that, under some conditions, almost match these lower bounds. In particular, we show that the approximation depends on a relation between $a$ and the diameter of the input graph.
We further establish a connection with a foundational optimization problem on static graphs called \emph{Diameter Constrained Spanning Subgraph} (DCSS) and show that our hardness results also apply to DCSS. 

**Abstract (ZH)**: 最小年龄标签问题的调度 

---
# Radiometer Calibration using Machine Learning 

**Title (ZH)**: 使用机器学习进行辐射计校准 

**Authors**: S. A. K. Leeney, H. T. J. Bevins, E. de Lera Acedo, W. J. Handley, C. Kirkham, R. S. Patel, J. Zhu, D. Molnar, J. Cumner, D. Anstey, K. Artuc, G. Bernardi, M. Bucher, S. Carey, J. Cavillot, R. Chiello, W. Croukamp, D. I. L. de Villiers, J. A. Ely, A. Fialkov, T. Gessey-Jones, G. Kulkarni, A. Magro, P. D. Meerburg, S. Mittal, J. H. N. Pattison, S. Pegwal, C. M. Pieterse, J. R. Pritchard, E. Puchwein, N. Razavi-Ghods, I. L. V. Roque, A. Saxena, K. H. Scheutwinkel, P. Scott, E. Shen, P. H. Sims, M. Spinelli  

**Link**: [PDF](https://arxiv.org/pdf/2504.16791)  

**Abstract**: Radiometers are crucial instruments in radio astronomy, forming the primary component of nearly all radio telescopes. They measure the intensity of electromagnetic radiation, converting this radiation into electrical signals. A radiometer's primary components are an antenna and a Low Noise Amplifier (LNA), which is the core of the ``receiver'' chain. Instrumental effects introduced by the receiver are typically corrected or removed during calibration. However, impedance mismatches between the antenna and receiver can introduce unwanted signal reflections and distortions. Traditional calibration methods, such as Dicke switching, alternate the receiver input between the antenna and a well-characterised reference source to mitigate errors by comparison. Recent advances in Machine Learning (ML) offer promising alternatives. Neural networks, which are trained using known signal sources, provide a powerful means to model and calibrate complex systems where traditional analytical approaches struggle. These methods are especially relevant for detecting the faint sky-averaged 21-cm signal from atomic hydrogen at high redshifts. This is one of the main challenges in observational Cosmology today. Here, for the first time, we introduce and test a machine learning-based calibration framework capable of achieving the precision required for radiometric experiments aiming to detect the 21-cm line. 

**Abstract (ZH)**: 射电望远镜是射电天文学中至关重要的仪器，构成了几乎全部射电望远镜的主要组件。它们测量电磁辐射的强度，将这种辐射转换为电信号。射电望远镜的主要组件是天线和低噪声放大器（LNA），它是“接收器”链的核心。由接收器引入的仪器效应通常在校准过程中被修正或移除。然而，天线和接收器之间的阻抗失配可能会引入不需要的信号反射和失真。传统的校准方法，如狄克开关，通过交替接收入射天线和一个具有良好表征的参考源来减轻比较方法中的错误。近年来，机器学习的进步提出了有希望的替代方案。使用已知信号源训练的神经网络为建模和校准传统分析方法难以处理的复杂系统提供了强大手段。这些方法特别适用于检测高红移时平均化后的稀弱21厘米线信号。这是当今观测 cosmology 中的主要挑战之一。在这里，我们首次引入并测试了一种基于机器学习的校准框架，能够实现检测21厘米线所需的高度精确性。 

---
# Credible plan-driven RAG method for Multi-hop Question Answering 

**Title (ZH)**: 基于计划驱动的可信多跳问答RAG方法 

**Authors**: Ningning Zhang, Chi Zhang, Zhizhong Tan, Xingxing Yang, Weiping Deng, Wenyong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.16787)  

**Abstract**: Multi-hop question answering (QA) presents a considerable challenge for Retrieval-Augmented Generation (RAG), requiring the structured decomposition of complex queries into logical reasoning paths and the generation of dependable intermediate results. However, deviations in reasoning paths or errors in intermediate results, which are common in current RAG methods, may propagate and accumulate throughout the reasoning process, diminishing the accuracy of the answer to complex queries. To address this challenge, we propose the Plan-then-Act-and-Review (PAR RAG) framework, which is organized into three key stages: planning, act, and review, and aims to offer an interpretable and incremental reasoning paradigm for accurate and reliable multi-hop question answering by mitigating error this http URL RAG initially applies a top-down problem decomposition strategy, formulating a comprehensive plan that integrates multiple executable steps from a holistic viewpoint. This approach avoids the pitfalls of local optima common in traditional RAG methods, ensuring the accuracy of the entire reasoning path. Subsequently, PAR RAG incorporates a plan execution mechanism based on multi-granularity verification. By utilizing both coarse-grained similarity information and fine-grained relevant data, the framework thoroughly checks and adjusts intermediate results, ensuring process accuracy while effectively managing error propagation and amplification. Experimental results on multi-hop QA datasets demonstrate that the PAR RAG framework substantially outperforms existing state-of-the-art methods in key metrics, including EM and F1 scores. 

**Abstract (ZH)**: 多跳问答（多跳QA）对检索增强生成（RAG）构成了显著挑战，需要将复杂查询结构化分解为逻辑推理路径，并生成可靠的中间结果。然而，当前RAG方法中推理路径中的偏差或中间结果中的错误可能在整个推理过程中传递和累积，降低复杂查询答案的准确性。为应对这一挑战，我们提出了计划-执行-回顾（PAR RAG）框架，该框架分为计划、执行和回顾三个关键阶段，旨在通过缓解错误提供可解释的逐步推理范式，以实现准确可靠的多跳问答。RAG最初采用自上而下的问题分解策略，从全局视角综合多个可执行步骤，避免传统RAG方法中常见的局部最优解问题，确保整体推理路径的准确性。随后，PAR RAG结合多粒度验证机制，通过利用粗粒度相似性和细粒度相关数据，彻底检查和调整中间结果，确保过程准确同时有效管理错误传播和放大。实验结果表明，PAR RAG框架在关键指标（包括EM和F1分数）上显著优于现有最先进的方法。 

---
# Evaluation Framework for AI Systems in "the Wild" 

**Title (ZH)**: “野生”环境中AI系统的评估框架 

**Authors**: Sarah Jabbour, Trenton Chang, Anindya Das Antar, Joseph Peper, Insu Jang, Jiachen Liu, Jae-Won Chung, Shiqi He, Michael Wellman, Bryan Goodman, Elizabeth Bondi-Kelly, Kevin Samy, Rada Mihalcea, Mosharaf Chowhury, David Jurgens, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.16778)  

**Abstract**: Generative AI (GenAI) models have become vital across industries, yet current evaluation methods have not adapted to their widespread use. Traditional evaluations often rely on benchmarks and fixed datasets, frequently failing to reflect real-world performance, which creates a gap between lab-tested outcomes and practical applications. This white paper proposes a comprehensive framework for how we should evaluate real-world GenAI systems, emphasizing diverse, evolving inputs and holistic, dynamic, and ongoing assessment approaches. The paper offers guidance for practitioners on how to design evaluation methods that accurately reflect real-time capabilities, and provides policymakers with recommendations for crafting GenAI policies focused on societal impacts, rather than fixed performance numbers or parameter sizes. We advocate for holistic frameworks that integrate performance, fairness, and ethics and the use of continuous, outcome-oriented methods that combine human and automated assessments while also being transparent to foster trust among stakeholders. Implementing these strategies ensures GenAI models are not only technically proficient but also ethically responsible and impactful. 

**Abstract (ZH)**: 面向现实应用的生成式AI系统评估框架 

---
# Noise-Tolerant Coreset-Based Class Incremental Continual Learning 

**Title (ZH)**: 噪声鲁棒核子集基于类增量连续学习 

**Authors**: Edison Mucllari, Aswin Raghavan, Zachary Alan Daniels  

**Link**: [PDF](https://arxiv.org/pdf/2504.16763)  

**Abstract**: Many applications of computer vision require the ability to adapt to novel data distributions after deployment. Adaptation requires algorithms capable of continual learning (CL). Continual learners must be plastic to adapt to novel tasks while minimizing forgetting of previous this http URL, CL opens up avenues for noise to enter the training pipeline and disrupt the CL. This work focuses on label noise and instance noise in the context of class-incremental learning (CIL), where new classes are added to a classifier over time, and there is no access to external data from past classes. We aim to understand the sensitivity of CL methods that work by replaying items from a memory constructed using the idea of Coresets. We derive a new bound for the robustness of such a method to uncorrelated instance noise under a general additive noise threat model, revealing several insights. Putting the theory into practice, we create two continual learning algorithms to construct noise-tolerant replay buffers. We empirically compare the effectiveness of prior memory-based continual learners and the proposed algorithms under label and uncorrelated instance noise on five diverse datasets. We show that existing memory-based CL are not robust whereas the proposed methods exhibit significant improvements in maximizing classification accuracy and minimizing forgetting in the noisy CIL setting. 

**Abstract (ZH)**: 许多计算机视觉应用要求在部署后能够适应新的数据分布。适应性需要具备持续学习能力的算法。持续学习者必须具有可塑性，以便适应新任务的同时尽量减少对先前任务的遗忘。持续学习可能为噪声进入训练管道并扰乱持续学习打开途径。本工作专注于类别增量学习（CIL）中的标签噪声和实例噪声，其中随着时间的推移，新的类别被添加到分类器中，且无法获取过去类别的外部数据。我们旨在理解通过使用Coreset思想构建记忆库并重复播放其中项的持续学习方法在无关联实例噪声情况下的鲁棒性。我们推导出该方法在一般加性噪声威胁模型下的新鲁棒性界限，揭示了若干洞察。将理论付诸实践，我们创建了两个持续学习算法以构建噪声容忍的重复播放缓存。我们在标签噪声和无关联实例噪声环境下，通过五个不同的数据集评估了现有记忆型持续学习方法和所提出算法的有效性。我们表明，现有记忆型持续学习方法不具有鲁棒性，而所提出的方法在噪声类别增量学习环境中显著提升了分类准确率并减少了遗忘。 

---
# Representation Learning via Non-Contrastive Mutual Information 

**Title (ZH)**: 非对比 Mutual Information 的表示学习 

**Authors**: Zhaohan Daniel Guo, Bernardo Avila Pires, Khimya Khetarpal, Dale Schuurmans, Bo Dai  

**Link**: [PDF](https://arxiv.org/pdf/2504.16667)  

**Abstract**: Labeling data is often very time consuming and expensive, leaving us with a majority of unlabeled data. Self-supervised representation learning methods such as SimCLR (Chen et al., 2020) or BYOL (Grill et al., 2020) have been very successful at learning meaningful latent representations from unlabeled image data, resulting in much more general and transferable representations for downstream tasks. Broadly, self-supervised methods fall into two types: 1) Contrastive methods, such as SimCLR; and 2) Non-Contrastive methods, such as BYOL. Contrastive methods are generally trying to maximize mutual information between related data points, so they need to compare every data point to every other data point, resulting in high variance, and thus requiring large batch sizes to work well. Non-contrastive methods like BYOL have much lower variance as they do not need to make pairwise comparisons, but are much trickier to implement as they have the possibility of collapsing to a constant vector. In this paper, we aim to develop a self-supervised objective that combines the strength of both types. We start with a particular contrastive method called the Spectral Contrastive Loss (HaoChen et al., 2021; Lu et al., 2024), and we convert it into a more general non-contrastive form; this removes the pairwise comparisons resulting in lower variance, but keeps the mutual information formulation of the contrastive method preventing collapse. We call our new objective the Mutual Information Non-Contrastive (MINC) loss. We test MINC by learning image representations on ImageNet (similar to SimCLR and BYOL) and show that it consistently improves upon the Spectral Contrastive loss baseline. 

**Abstract (ZH)**: 无监督标签学习既耗时又昂贵，导致大量数据未标注。SimCLR（Chen等，2020）或BYOL（Grill等，2020）等自监督表示学习方法能够从未标注图像数据中学习到有意义的潜在表示，从而为下游任务提供了更加通用和可迁移的表示。自监督方法大致可分为两大类：1）对比方法，如SimCLR；2）非对比方法，如BYOL。对比方法通常试图最大化相关数据点之间的互信息，因此需要将每一对数据点与其他所有数据点进行比较，导致高方差，从而需要大批次大小才能有效工作。非对比方法如BYOL的方差更低，因为它们不需要进行成对比较，但实现起来更为棘手，因为存在退化为固定向量的可能性。在本文中，我们旨在开发一种结合了这两种方法优势的自监督目标函数。我们以特定的对比方法谱对比损失（Spectral Contrastive Loss，HaoChen等，2021；Lu等，2024）为起点，将其转换为一种更通用的非对比形式；这消除了成对比较，从而降低了方差，但保留了对比方法的互信息形式，防止退化。我们称这一新目标函数为互信息非对比损失（Mutual Information Non-Contrastive，MINC）损失。我们通过在ImageNet上学习图像表示（类似于SimCLR和BYOL）来测试MINC，并证明它可以持续改进谱对比损失的基线效果。 

---
# MAYA: Addressing Inconsistencies in Generative Password Guessing through a Unified Benchmark 

**Title (ZH)**: MAYA: 通过统一基准解决生成密码猜测中的一致性问题 

**Authors**: William Corrias, Fabio De Gaspari, Dorjan Hitaj, Luigi V. Mancini  

**Link**: [PDF](https://arxiv.org/pdf/2504.16651)  

**Abstract**: The rapid evolution of generative models has led to their integration across various fields, including password guessing, aiming to generate passwords that resemble human-created ones in complexity, structure, and patterns. Despite generative model's promise, inconsistencies in prior research and a lack of rigorous evaluation have hindered a comprehensive understanding of their true potential. In this paper, we introduce MAYA, a unified, customizable, plug-and-play password benchmarking framework. MAYA provides a standardized approach for evaluating generative password-guessing models through a rigorous set of advanced testing scenarios and a collection of eight real-life password datasets. Using MAYA, we comprehensively evaluate six state-of-the-art approaches, which have been re-implemented and adapted to ensure standardization, for a total of over 15,000 hours of computation. Our findings indicate that these models effectively capture different aspects of human password distribution and exhibit strong generalization capabilities. However, their effectiveness varies significantly with long and complex passwords. Through our evaluation, sequential models consistently outperform other generative architectures and traditional password-guessing tools, demonstrating unique capabilities in generating accurate and complex guesses. Moreover, models learn and generate different password distributions, enabling a multi-model attack that outperforms the best individual model. By releasing MAYA, we aim to foster further research, providing the community with a new tool to consistently and reliably benchmark password-generation techniques. Our framework is publicly available at this https URL 

**Abstract (ZH)**: 生成模型的快速演进已将其整合到各种领域中，包括密码猜测，旨在生成与人类创建的密码在复杂性、结构和模式上相似的密码。尽管生成模型具有巨大的潜力，但由于先前研究中的不一致性以及缺乏严格的评估，阻碍了对其真正潜力的全面理解。在本文中，我们介绍了MAYA，一个统一、可定制、即插即用的密码基准测试框架。MAYA提供了一种通过一系列严格的高级测试场景和八个真实密码数据集来评估生成式密码猜测模型的标准方法。使用MAYA，我们全面评估了六种最先进的方法，这些方法已重新实现和调整以确保标准化，总共进行了超过15,000小时的计算。我们的研究发现，这些模型能有效捕捉人类密码分布的不同方面，并展现出强大的泛化能力。然而，它们在长而复杂的密码上的效果差异显著。通过我们的评估，序列模型普遍优于其他生成架构和传统密码猜测工具，展示了在生成准确且复杂的猜测方面的独特能力。此外，模型学习并生成不同的密码分布，使多模型攻击优于单一最佳模型。通过发布MAYA，我们旨在促进进一步的研究，为社区提供一个新的工具来一致且可靠地评估密码生成技术。我们的框架可在以下网址公开获取：这个 https URL。 

---
# SSLR: A Semi-Supervised Learning Method for Isolated Sign Language Recognition 

**Title (ZH)**: SSLR：孤立手语识别的半监督学习方法 

**Authors**: Hasan Algafri, Hamzah Luqman, Sarah Alyami, Issam Laradji  

**Link**: [PDF](https://arxiv.org/pdf/2504.16640)  

**Abstract**: Sign language is the primary communication language for people with disabling hearing loss. Sign language recognition (SLR) systems aim to recognize sign gestures and translate them into spoken language. One of the main challenges in SLR is the scarcity of annotated datasets. To address this issue, we propose a semi-supervised learning (SSL) approach for SLR (SSLR), employing a pseudo-label method to annotate unlabeled samples. The sign gestures are represented using pose information that encodes the signer's skeletal joint points. This information is used as input for the Transformer backbone model utilized in the proposed approach. To demonstrate the learning capabilities of SSL across various labeled data sizes, several experiments were conducted using different percentages of labeled data with varying numbers of classes. The performance of the SSL approach was compared with a fully supervised learning-based model on the WLASL-100 dataset. The obtained results of the SSL model outperformed the supervised learning-based model with less labeled data in many cases. 

**Abstract (ZH)**: 基于半监督学习的 sign 语言识别方法 

---
# MMHCL: Multi-Modal Hypergraph Contrastive Learning for Recommendation 

**Title (ZH)**: MMHCL：多模态超图对比学习推荐 

**Authors**: Xu Guo, Tong Zhang, Fuyun Wang, Xudong Wang, Xiaoya Zhang, Xin Liu, Zhen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2504.16576)  

**Abstract**: The burgeoning presence of multimodal content-sharing platforms propels the development of personalized recommender systems. Previous works usually suffer from data sparsity and cold-start problems, and may fail to adequately explore semantic user-product associations from multimodal data. To address these issues, we propose a novel Multi-Modal Hypergraph Contrastive Learning (MMHCL) framework for user recommendation. For a comprehensive information exploration from user-product relations, we construct two hypergraphs, i.e. a user-to-user (u2u) hypergraph and an item-to-item (i2i) hypergraph, to mine shared preferences among users and intricate multimodal semantic resemblance among items, respectively. This process yields denser second-order semantics that are fused with first-order user-item interaction as complementary to alleviate the data sparsity issue. Then, we design a contrastive feature enhancement paradigm by applying synergistic contrastive learning. By maximizing/minimizing the mutual information between second-order (e.g. shared preference pattern for users) and first-order (information of selected items for users) embeddings of the same/different users and items, the feature distinguishability can be effectively enhanced. Compared with using sparse primary user-item interaction only, our MMHCL obtains denser second-order hypergraphs and excavates more abundant shared attributes to explore the user-product associations, which to a certain extent alleviates the problems of data sparsity and cold-start. Extensive experiments have comprehensively demonstrated the effectiveness of our method. Our code is publicly available at: this https URL. 

**Abstract (ZH)**: 多模态超图对比学习的用户推荐框架 

---
# Transformers for Complex Query Answering over Knowledge Hypergraphs 

**Title (ZH)**: Transformer在知识超图上的复杂查询回答中应用 

**Authors**: Hong Ting Tsang, Zihao Wang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.16537)  

**Abstract**: Complex Query Answering (CQA) has been extensively studied in recent years. In order to model data that is closer to real-world distribution, knowledge graphs with different modalities have been introduced. Triple KGs, as the classic KGs composed of entities and relations of arity 2, have limited representation of real-world facts. Real-world data is more sophisticated. While hyper-relational graphs have been introduced, there are limitations in representing relationships of varying arity that contain entities with equal contributions. To address this gap, we sampled new CQA datasets: JF17k-HCQA and M-FB15k-HCQA. Each dataset contains various query types that include logical operations such as projection, negation, conjunction, and disjunction. In order to answer knowledge hypergraph (KHG) existential first-order queries, we propose a two-stage transformer model, the Logical Knowledge Hypergraph Transformer (LKHGT), which consists of a Projection Encoder for atomic projection and a Logical Encoder for complex logical operations. Both encoders are equipped with Type Aware Bias (TAB) for capturing token interactions. Experimental results on CQA datasets show that LKHGT is a state-of-the-art CQA method over KHG and is able to generalize to out-of-distribution query types. 

**Abstract (ZH)**: 复杂查询回答（CQA）近年来受到了广泛研究。为了更好地模拟现实世界的分布，引入了不同模态的知识图谱。三元组KG作为经典的二目关联实体和关系的知识图谱，对现实世界的事实表示能力有限。现实世界的数据更为复杂。尽管引入了超关系图，但在表示包含等贡献实体的多种 arity 关系时仍有限制。为解决这一问题，我们采样了新的CQA数据集：JF17k-HCQA和M-FB15k-HCQA。每个数据集包含多种查询类型，包括逻辑操作如投影、否定、合取和析取。为了回答知识超图（KHG）的存在性一阶查询，我们提出了一种两阶段的变压器模型，即逻辑知识超图变换器（LKHGT），该模型包括用于原子投影的投影编码器和用于复杂逻辑操作的逻辑编码器。两个编码器均配备了类型感知偏差（TAB）以捕捉 token 交互。在CQA数据集上的实验结果表明，LKHGT 是KHG上最先进的CQA方法，并且能够泛化到未出分布的查询类型。 

---
# Federated Learning of Low-Rank One-Shot Image Detection Models in Edge Devices with Scalable Accuracy and Compute Complexity 

**Title (ZH)**: 边缘设备上具有可扩展准确度和计算复杂度的低秩单次图像检测模型的联邦学习 

**Authors**: Abdul Hannaan, Zubair Shah, Aiman Erbad, Amr Mohamed, Ali Safa  

**Link**: [PDF](https://arxiv.org/pdf/2504.16515)  

**Abstract**: This paper introduces a novel federated learning framework termed LoRa-FL designed for training low-rank one-shot image detection models deployed on edge devices. By incorporating low-rank adaptation techniques into one-shot detection architectures, our method significantly reduces both computational and communication overhead while maintaining scalable accuracy. The proposed framework leverages federated learning to collaboratively train lightweight image recognition models, enabling rapid adaptation and efficient deployment across heterogeneous, resource-constrained devices. Experimental evaluations on the MNIST and CIFAR10 benchmark datasets, both in an independent-and-identically-distributed (IID) and non-IID setting, demonstrate that our approach achieves competitive detection performance while significantly reducing communication bandwidth and compute complexity. This makes it a promising solution for adaptively reducing the communication and compute power overheads, while not sacrificing model accuracy. 

**Abstract (ZH)**: 一种用于边缘设备上训练低秩单-shot图像检测模型的新型联邦学习框架：LoRa-FL 

---
# On Developers' Self-Declaration of AI-Generated Code: An Analysis of Practices 

**Title (ZH)**: 开发者对自己声明的AI生成代码的研究：实践分析 

**Authors**: Syed Mohammad Kashif, Peng Liang, Amjed Tahir  

**Link**: [PDF](https://arxiv.org/pdf/2504.16485)  

**Abstract**: AI code generation tools have gained significant popularity among developers, who use them to assist in software development due to their capability to generate code. Existing studies mainly explored the quality, e.g., correctness and security, of AI-generated code, while in real-world software development, the prerequisite is to distinguish AI-generated code from human-written code, which emphasizes the need to explicitly declare AI-generated code by developers. To this end, this study intends to understand the ways developers use to self-declare AI-generated code and explore the reasons why developers choose to self-declare or not. We conducted a mixed-methods study consisting of two phases. In the first phase, we mined GitHub repositories and collected 613 instances of AI-generated code snippets. In the second phase, we conducted a follow-up industrial survey, which received 111 valid responses. Our research revealed the practices followed by developers to self-declare AI-generated code. Most practitioners (76.6%) always or sometimes self-declare AI-generated code. In contrast, other practitioners (23.4%) noted that they never self-declare AI-generated code. The reasons for self-declaring AI-generated code include the need to track and monitor the code for future review and debugging, and ethical considerations. The reasons for not self-declaring AI-generated code include extensive modifications to AI-generated code and the developers' perception that self-declaration is an unnecessary activity. We finally provided guidelines for practitioners to self-declare AI-generated code, addressing ethical and code quality concerns. 

**Abstract (ZH)**: AI代码生成工具在开发者中的应用及其自声明方式的研究 

---
# The Dance of Atoms-De Novo Protein Design with Diffusion Model 

**Title (ZH)**: 原子之舞-基于扩散模型的从头蛋白质设计 

**Authors**: Yujie Qin, Ming He, Changyong Yu, Ming Ni, Xian Liu, Xiaochen Bo  

**Link**: [PDF](https://arxiv.org/pdf/2504.16479)  

**Abstract**: The de novo design of proteins refers to creating proteins with specific structures and functions that do not naturally exist. In recent years, the accumulation of high-quality protein structure and sequence data and technological advancements have paved the way for the successful application of generative artificial intelligence (AI) models in protein design. These models have surpassed traditional approaches that rely on fragments and bioinformatics. They have significantly enhanced the success rate of de novo protein design, and reduced experimental costs, leading to breakthroughs in the field. Among various generative AI models, diffusion models have yielded the most promising results in protein design. In the past two to three years, more than ten protein design models based on diffusion models have emerged. Among them, the representative model, RFDiffusion, has demonstrated success rates in 25 protein design tasks that far exceed those of traditional methods, and other AI-based approaches like RFjoint and hallucination. This review will systematically examine the application of diffusion models in generating protein backbones and sequences. We will explore the strengths and limitations of different models, summarize successful cases of protein design using diffusion models, and discuss future development directions. 

**Abstract (ZH)**: 从头设计蛋白质是指设计具有特定结构和功能且自然界中不存在的蛋白质。近年来，高质量的蛋白质结构和序列数据的积累及技术的进步为生成性人工智能（AI）模型在蛋白质设计中的成功应用铺平了道路。这些模型超越了依赖片段和生物信息学的传统方法，显著提高了从头设计蛋白质的成功率，降低了实验成本，推动了该领域的突破。在各种生成性AI模型中，扩散模型在蛋白质设计领域取得了最令人鼓舞的结果。在过去两到三年中，基于扩散模型的蛋白质设计模型超过了十个。其中，代表性的模型RFDiffusion在25项蛋白质设计任务中的成功率远超传统方法及其他基于AI的方法（如RFjoint和hallucination）。本文将系统地探讨扩散模型在生成蛋白质主链和序列中的应用。我们将分析不同模型的优势和局限性，总结使用扩散模型进行蛋白质设计的成功案例，并讨论未来的发展方向。 

---
# Private Federated Learning using Preference-Optimized Synthetic Data 

**Title (ZH)**: 基于偏好优化合成数据的隐私联邦学习 

**Authors**: Charlie Hou, Mei-Yu Wang, Yige Zhu, Daniel Lazar, Giulia Fanti  

**Link**: [PDF](https://arxiv.org/pdf/2504.16438)  

**Abstract**: In practical settings, differentially private Federated learning (DP-FL) is the dominant method for training models from private, on-device client data. Recent work has suggested that DP-FL may be enhanced or outperformed by methods that use DP synthetic data (Wu et al., 2024; Hou et al., 2024). The primary algorithms for generating DP synthetic data for FL applications require careful prompt engineering based on public information and/or iterative private client feedback. Our key insight is that the private client feedback collected by prior DP synthetic data methods (Hou et al., 2024; Xie et al., 2024) can be viewed as a preference ranking. Our algorithm, Preference Optimization for Private Client Data (POPri) harnesses client feedback using preference optimization algorithms such as Direct Preference Optimization (DPO) to fine-tune LLMs to generate high-quality DP synthetic data. To evaluate POPri, we release LargeFedBench, a new federated text benchmark for uncontaminated LLM evaluations on federated client data. POPri substantially improves the utility of DP synthetic data relative to prior work on LargeFedBench datasets and an existing benchmark from Xie et al. (2024). POPri closes the gap between next-token prediction accuracy in the fully-private and non-private settings by up to 68%, compared to 52% for prior synthetic data methods, and 10% for state-of-the-art DP federated learning methods. The code and data are available at this https URL. 

**Abstract (ZH)**: 差分隐私联邦学习（DP-FL）在实际应用中是训练设备端私有数据模型的主要方法。近年来的研究表明，使用差分隐私合成数据的方法可能会增强或超越DP-FL（Wu et al., 2024；Hou et al., 2024）。用于生成差分隐私合成数据的主要算法需要基于公共信息和/或迭代的私有客户端反馈进行精心的设计。我们的关键见解是，先前使用差分隐私合成数据方法收集的客户端反馈可以被视为偏好排名。我们的算法，基于偏好优化的私有客户端数据（POPri），利用偏好优化算法（如直接偏好优化DPO）来微调大语言模型（LLMs），以生成高质量的差分隐私合成数据。为了评估POPri，我们发布了LargeFedBench，一个新的联邦文本基准，用于评估联邦客户端数据上的大语言模型。与LargeFedBench数据集及Xie et al.（2024）的现有基准相比，POPri显著提高了差分隐私合成数据的实用性。与先前的合成数据方法相比，POPri在完全私有和非私有设置下的下一个词预测准确性差距缩小了高达68%，先前的方法为52%，最先进的差分隐私联邦学习方法为10%。代码和数据可在以下链接获取。 

---
# iTFKAN: Interpretable Time Series Forecasting with Kolmogorov-Arnold Network 

**Title (ZH)**: iTFKAN：可解释的时间序列预测与柯尔莫戈罗夫-阿诺尔德网络 

**Authors**: Ziran Liang, Rui An, Wenqi Fan, Yanghui Rao, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2504.16432)  

**Abstract**: As time evolves, data within specific domains exhibit predictability that motivates time series forecasting to predict future trends from historical data. However, current deep forecasting methods can achieve promising performance but generally lack interpretability, hindering trustworthiness and practical deployment in safety-critical applications such as auto-driving and healthcare. In this paper, we propose a novel interpretable model, iTFKAN, for credible time series forecasting. iTFKAN enables further exploration of model decision rationales and underlying data patterns due to its interpretability achieved through model symbolization. Besides, iTFKAN develops two strategies, prior knowledge injection, and time-frequency synergy learning, to effectively guide model learning under complex intertwined time series data. Extensive experimental results demonstrated that iTFKAN can achieve promising forecasting performance while simultaneously possessing high interpretive capabilities. 

**Abstract (ZH)**: 随时间演进，特定领域内的数据展现出可预测性，这促使时间序列预测从历史数据中预测未来趋势。然而，当前的深度预测方法虽然能够取得显著性能，但在可解释性方面 generally 缺乏，这妨碍了其在自动驾驶和医疗健康等关键安全应用中的信任度和实际部署。在本文中，我们提出了一种新颖的可解释模型 iTFKAN，以实现可信的时间序列预测。iTFKAN 通过模型符号化实现的可解释性，能够进一步探索模型决策依据和数据的基本模式。此外，iTFKAN 开发了先验知识注入和时频协同学习两种策略，以有效引导在复杂交织的时间序列数据下的模型学习。广泛实验结果表明，iTFKAN 可同时实现优异的预测性能和高可解释能力。 

---
# A Survey of Foundation Model-Powered Recommender Systems: From Feature-Based, Generative to Agentic Paradigms 

**Title (ZH)**: 基于基础模型的推荐系统综述：从特征导向、生成导向到能动导向 paradigm 

**Authors**: Chengkai Huang, Hongtao Huang, Tong Yu, Kaige Xie, Junda Wu, Shuai Zhang, Julian Mcauley, Dietmar Jannach, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.16420)  

**Abstract**: Recommender systems (RS) have become essential in filtering information and personalizing content for users. RS techniques have traditionally relied on modeling interactions between users and items as well as the features of content using models specific to each task. The emergence of foundation models (FMs), large scale models trained on vast amounts of data such as GPT, LLaMA and CLIP, is reshaping the recommendation paradigm. This survey provides a comprehensive overview of the Foundation Models for Recommender Systems (FM4RecSys), covering their integration in three paradigms: (1) Feature-Based augmentation of representations, (2) Generative recommendation approaches, and (3) Agentic interactive systems. We first review the data foundations of RS, from traditional explicit or implicit feedback to multimodal content sources. We then introduce FMs and their capabilities for representation learning, natural language understanding, and multi-modal reasoning in RS contexts. The core of the survey discusses how FMs enhance RS under different paradigms. Afterward, we examine FM applications in various recommendation tasks. Through an analysis of recent research, we highlight key opportunities that have been realized as well as challenges encountered. Finally, we outline open research directions and technical challenges for next-generation FM4RecSys. This survey not only reviews the state-of-the-art methods but also provides a critical analysis of the trade-offs among the feature-based, the generative, and the agentic paradigms, outlining key open issues and future research directions. 

**Abstract (ZH)**: 基础模型在推荐系统中的应用：一种涵盖特征增强表示、生成推荐方法和自主交互系统的全面概述 

---
# PixelWeb: The First Web GUI Dataset with Pixel-Wise Labels 

**Title (ZH)**: PixelWeb: 首个具有像素级标签的Web GUI数据集 

**Authors**: Qi Yang, Weichen Bi, Haiyang Shen, Yaoqi Guo, Yun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.16419)  

**Abstract**: Graphical User Interface (GUI) datasets are crucial for various downstream tasks. However, GUI datasets often generate annotation information through automatic labeling, which commonly results in inaccurate GUI element BBox annotations, including missing, duplicate, or meaningless BBoxes. These issues can degrade the performance of models trained on these datasets, limiting their effectiveness in real-world applications. Additionally, existing GUI datasets only provide BBox annotations visually, which restricts the development of visually related GUI downstream tasks. To address these issues, we introduce PixelWeb, a large-scale GUI dataset containing over 100,000 annotated web pages. PixelWeb is constructed using a novel automatic annotation approach that integrates visual feature extraction and Document Object Model (DOM) structure analysis through two core modules: channel derivation and layer analysis. Channel derivation ensures accurate localization of GUI elements in cases of occlusion and overlapping elements by extracting BGRA four-channel bitmap annotations. Layer analysis uses the DOM to determine the visibility and stacking order of elements, providing precise BBox annotations. Additionally, PixelWeb includes comprehensive metadata such as element images, contours, and mask annotations. Manual verification by three independent annotators confirms the high quality and accuracy of PixelWeb annotations. Experimental results on GUI element detection tasks show that PixelWeb achieves performance on the mAP95 metric that is 3-7 times better than existing datasets. We believe that PixelWeb has great potential for performance improvement in downstream tasks such as GUI generation and automated user interaction. 

**Abstract (ZH)**: 图形用户界面（GUI）数据集对于各种下游任务至关重要。然而，GUI数据集往往通过自动标注生成注释信息，这通常会导致GUI元素边界框（BBox）注释不准确，包括缺失、重复或无意义的边界框。这些问题会降低基于这些数据集训练的模型的性能，限制其在实际应用中的效果。此外，现有的GUI数据集仅提供可视化的边界框注释，这限制了与GUI相关的视觉下游任务的发展。为了解决这些问题，我们引入了PixelWeb，这是一个包含超过100,000个标注网页的大规模GUI数据集。PixelWeb利用一种新颖的自动标注方法构建，该方法结合了视觉特征提取和文档对象模型（DOM）结构分析，通过两个核心模块：通道衍生和层分析来实现。通道衍生通过提取BGRA四通道位图注释确保GUI元素的准确定位，尤其是在遮挡和重叠的情况下。层分析利用DOM来确定元素的可见性和堆叠顺序，提供精确的边界框注释。此外，PixelWeb还包括元素图像、轮廓和掩码注释等全面的元数据。三名独立注释者的手动验证证实了PixelWeb注释的高质量和准确性。在GUI元素检测任务上的实验结果表明，PixelWeb在mAP95指标上的性能比现有数据集高出3到7倍。我们相信，PixelWeb在GUI生成和自动用户交互等下游任务中的性能改进方面具有巨大潜力。 

---
# FeedQUAC: Quick Unobtrusive AI-Generated Commentary 

**Title (ZH)**: FeedQUAC: 快速不显干预的AI生成评论 

**Authors**: Tao Long, Kendra Wannamaker, Jo Vermeulen, George Fitzmaurice, Justin Matejka  

**Link**: [PDF](https://arxiv.org/pdf/2504.16416)  

**Abstract**: Design thrives on feedback. However, gathering constant feedback throughout the design process can be labor-intensive and disruptive. We explore how AI can bridge this gap by providing effortless, ambient feedback. We introduce FeedQUAC, a design companion that delivers real-time AI-generated commentary from a variety of perspectives through different personas. A design probe study with eight participants highlights how designers can leverage quick yet ambient AI feedback to enhance their creative workflows. Participants highlight benefits such as convenience, playfulness, confidence boost, and inspiration from this lightweight feedback agent, while suggesting additional features, like chat interaction and context curation. We discuss the role of AI feedback, its strengths and limitations, and how to integrate it into existing design workflows while balancing user involvement. Our findings also suggest that ambient interaction is a valuable consideration for both the design and evaluation of future creativity support systems. 

**Abstract (ZH)**: 设计依赖于反馈。然而，在设计过程中不断收集反馈可能是劳动密集型且具有干扰性的。我们探索AI如何通过提供轻松且无感知的反馈来弥合这一差距。我们介绍了FeedQUAC，这是一种设计伴侣，能够通过不同的角色提供实时的AI生成的多视角评论。一项涉及八名参与者的探索性设计实验突显了设计者如何利用快速且无感知的AI反馈来增强其创意工作流程。参与者强调了这种轻量级反馈代理的便利性、趣味性、信心提升和灵感，并建议增加诸如聊天交互和内容筛选等功能。我们讨论了AI反馈的作用、优势和局限性，以及如何在平衡用户参与的情况下将其整合到现有的设计工作流程中。研究结果还表明，无感知交互对于未来创意支持系统的构思和评估均是值得考虑的重要因素。 

---
# PINN-MEP: Continuous Neural Representations for Minimum-Energy Path Discovery in Molecular Systems 

**Title (ZH)**: PINN-MEP：分子系统中最小能量路径发现的连续神经表示方法 

**Authors**: Magnus Petersen, Roberto Covino  

**Link**: [PDF](https://arxiv.org/pdf/2504.16381)  

**Abstract**: Characterizing conformational transitions in physical systems remains a fundamental challenge in the computational sciences. Traditional sampling methods like molecular dynamics (MD) or MCMC often struggle with the high-dimensional nature of molecular systems and the high energy barriers of transitions between stable states. While these transitions are rare events in simulation timescales, they often represent the most biologically significant processes - for example, the conformational change of an ion channel protein from its closed to open state, which controls cellular ion flow and is crucial for neural signaling. Such transitions in real systems may take milliseconds to seconds but could require months or years of continuous simulation to observe even once. We present a method that reformulates transition path generation as a continuous optimization problem solved through physics-informed neural networks (PINNs) inspired by string methods for minimum-energy path (MEP) generation. By representing transition paths as implicit neural functions and leveraging automatic differentiation with differentiable molecular dynamics force fields, our method enables the efficient discovery of physically realistic transition pathways without requiring expensive path sampling. We demonstrate our method's effectiveness on two proteins, including an explicitly hydrated bovine pancreatic trypsin inhibitor (BPTI) system with over 8,300 atoms. 

**Abstract (ZH)**: 物理系统构象转变的特性描述仍然是计算科学中的一个基础挑战。传统的采样方法，如分子动力学（MD）或MCMC，在处理分子系统的高维性质以及稳定状态间高能垒的过渡时常常力不从心。虽然这些过渡事件在模拟时间尺度上是罕见的，但它们往往代表了生物上最显著的过程——例如离子通道蛋白从关闭状态到打开状态的构象变化，这控制着细胞离子流，并对于神经信号传导至关重要。这类过程在实际系统中可能需要毫秒到秒的时间，但在长时间连续模拟中观测到这些过程却可能需要数月至数年。我们提出了一种方法，将过渡路径生成重新表述为通过物理启发式神经网络（PINNs）求解的连续优化问题，该方法受最低能量路径（MEP）生成的串行方法启发。通过将过渡路径表示为隐式神经函数，并利用自动微分和可微分子动力学势场，我们的方法能够在无需昂贵路径采样的情况下高效发现物理上可行的过渡路径。我们在两个蛋白质上展示了该方法的有效性，包括含有超过8,300个原子的明确水化的牛胰糜蛋白酶抑制剂（BPTI）系统。 

---
# Cyberoception: Finding a Painlessly-Measurable New Sense in the Cyberworld Towards Emotion-Awareness in Computing 

**Title (ZH)**: 网络感知：在网络世界中寻找一种无痛可测的新感觉，以实现计算中的情感意识 

**Authors**: Tadashi Okoshi, Zexiong Gao, Tan Yi Zhen, Takumi Karasawa, Takeshi Miki, Wataru Sasaki, Rajesh K. Balan  

**Link**: [PDF](https://arxiv.org/pdf/2504.16378)  

**Abstract**: In Affective computing, recognizing users' emotions accurately is the basis of affective human-computer interaction. Understanding users' interoception contributes to a better understanding of individually different emotional abilities, which is essential for achieving inter-individually accurate emotion estimation. However, existing interoception measurement methods, such as the heart rate discrimination task, have several limitations, including their dependence on a well-controlled laboratory environment and precision apparatus, making monitoring users' interoception challenging. This study aims to determine other forms of data that can explain users' interoceptive or similar states in their real-world lives and propose a novel hypothetical concept "cyberoception," a new sense (1) which has properties similar to interoception in terms of the correlation with other emotion-related abilities, and (2) which can be measured only by the sensors embedded inside commodity smartphone devices in users' daily lives. Results from a 10-day-long in-lab/in-the-wild hybrid experiment reveal a specific cyberoception type "Turn On" (users' subjective sensory perception about the frequency of turning-on behavior on their smartphones), significantly related to participants' emotional valence. We anticipate that cyberoception to serve as a fundamental building block for developing more "emotion-aware", user-friendly applications and services. 

**Abstract (ZH)**: 在情感计算中，准确识别用户的情绪是实现情感人机交互的基础。理解用户的内感受有助于更好地理解个体之间不同的情感能力，这对于实现个体之间准确的情绪估计至关重要。然而，现有的内感受测量方法，如心率辨别任务，存在一些局限性，包括依赖于受控实验室环境和精确的测量仪器，这使得监测用户的内感受具有挑战性。本研究旨在确定其他形式的数据，以解释用户在现实生活中的内感受或类似状态，并提出一个全新的假设概念“赛博感受”，这是一种新的感觉（1）其与与情绪相关的其他能力的相关性类似于内感受的属性；（2）仅通过用户日常生活中内置在智能手机设备中的传感器进行测量。一项为期10天的室内/室外混合实验表明，“开启”（用户对其智能手机开机行为频率的主观感知感受）这一赛博感受类型与参与者的情绪效价显著相关。我们期待赛博感受成为开发更多“情绪感知”、用户友好型应用和服务的基础构建块。 

---
# CLPSTNet: A Progressive Multi-Scale Convolutional Steganography Model Integrating Curriculum Learning 

**Title (ZH)**: CLPSTNet：一种融合分级学习的 Progressive 多尺度卷积隐写模型 

**Authors**: Fengchun Liu, Tong Zhang, Chunying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.16364)  

**Abstract**: In recent years, a large number of works have introduced Convolutional Neural Networks (CNNs) into image steganography, which transform traditional steganography methods such as hand-crafted features and prior knowledge design into steganography methods that neural networks autonomically learn information embedding. However, due to the inherent complexity of digital images, issues of invisibility and security persist when using CNN models for information embedding. In this paper, we propose Curriculum Learning Progressive Steganophy Network (CLPSTNet). The network consists of multiple progressive multi-scale convolutional modules that integrate Inception structures and dilated convolutions. The module contains multiple branching pathways, starting from a smaller convolutional kernel and dilatation rate, extracting the basic, local feature information from the feature map, and gradually expanding to the convolution with a larger convolutional kernel and dilatation rate for perceiving the feature information of a larger receptive field, so as to realize the multi-scale feature extraction from shallow to deep, and from fine to coarse, allowing the shallow secret information features to be refined in different fusion stages. The experimental results show that the proposed CLPSTNet not only has high PSNR , SSIM metrics and decoding accuracy on three large public datasets, ALASKA2, VOC2012 and ImageNet, but also the steganographic images generated by CLPSTNet have low steganalysis this http URL can find our code at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于逐层学习的多尺度卷积秘密嵌入网络（CLPSTNet） 

---
# DP2FL: Dual Prompt Personalized Federated Learning in Foundation Models 

**Title (ZH)**: DP2FL: 基础模型中的双重提示个性化联邦学习 

**Authors**: Ying Chang, Xiaohu Shi, Xiaohui Zhao, Zhaohuang Chen, Deyin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.16357)  

**Abstract**: Personalized federated learning (PFL) has garnered significant attention for its ability to address heterogeneous client data distributions while preserving data privacy. However, when local client data is limited, deep learning models often suffer from insufficient training, leading to suboptimal performance. Foundation models, such as CLIP (Contrastive Language-Image Pretraining), exhibit strong feature extraction capabilities and can alleviate this issue by fine-tuning on limited local data. Despite their potential, foundation models are rarely utilized in federated learning scenarios, and challenges related to integrating new clients remain largely unresolved. To address these challenges, we propose the Dual Prompt Personalized Federated Learning (DP2FL) framework, which introduces dual prompts and an adaptive aggregation strategy. DP2FL combines global task awareness with local data-driven insights, enabling local models to achieve effective generalization while remaining adaptable to specific data distributions. Moreover, DP2FL introduces a global model that enables prediction on new data sources and seamlessly integrates newly added clients without requiring retraining. Experimental results in highly heterogeneous environments validate the effectiveness of DP2FL's prompt design and aggregation strategy, underscoring the advantages of prediction on novel data sources and demonstrating the seamless integration of new clients into the federated learning framework. 

**Abstract (ZH)**: 双提示个性化联邦学习（DP2FL）框架 

---
# Transformer-Based Extraction of Statutory Definitions from the U.S. Code 

**Title (ZH)**: 基于变压器的美国法典法定定义提取 

**Authors**: Arpana Hosabettu, Harsh Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.16353)  

**Abstract**: Automatic extraction of definitions from legal texts is critical for enhancing the comprehension and clarity of complex legal corpora such as the United States Code (U.S.C.). We present an advanced NLP system leveraging transformer-based architectures to automatically extract defined terms, their definitions, and their scope from the U.S.C. We address the challenges of automatically identifying legal definitions, extracting defined terms, and determining their scope within this complex corpus of over 200,000 pages of federal statutory law. Building upon previous feature-based machine learning methods, our updated model employs domain-specific transformers (Legal-BERT) fine-tuned specifically for statutory texts, significantly improving extraction accuracy. Our work implements a multi-stage pipeline that combines document structure analysis with state-of-the-art language models to process legal text from the XML version of the U.S. Code. Each paragraph is first classified using a fine-tuned legal domain BERT model to determine if it contains a definition. Our system then aggregates related paragraphs into coherent definitional units and applies a combination of attention mechanisms and rule-based patterns to extract defined terms and their jurisdictional scope. The definition extraction system is evaluated on multiple titles of the U.S. Code containing thousands of definitions, demonstrating significant improvements over previous approaches. Our best model achieves 96.8% precision and 98.9% recall (98.2% F1-score), substantially outperforming traditional machine learning classifiers. This work contributes to improving accessibility and understanding of legal information while establishing a foundation for downstream legal reasoning tasks. 

**Abstract (ZH)**: 自动从法律文本中提取定义对于增强如美国法典（U.S.C.）等复杂法律文集的可理解性和清晰度至关重要。我们提出了一种基于变换器架构的高级自然语言处理系统，用于自动从美国法典中提取定义术语、其定义及其适用范围。我们解决了自动识别法律定义、提取定义术语以及确定其在复杂文集中的适用范围的挑战，该文集包含超过20万页的联邦立法法条。基于以前的基于特征的机器学习方法，我们的更新模型采用特定领域变换器（Legal-BERT）进行微调，专为立法文本设计，显著提高了提取准确性。我们的工作实施了一个多阶段管道，结合文档结构分析和最先进的语言模型来处理来自美国法典XML版本的法律文本。每个段落首先使用微调的法律领域BERT模型进行分类，以确定是否包含定义。系统随后将相关段落聚合为连贯的定义单元，并应用注意力机制和基于规则的模式组合来提取定义术语及其管辖范围。定义提取系统在包含数千个定义的美国法典多个标题上进行了评估，展示了与之前方法相比的重大改进。我们的最佳模型实现96.8%的精确率和98.9%的召回率（F1分数为98.2%），显著优于传统机器学习分类器。本工作有助于提高法律信息的可访问性和理解，同时为后续法律推理任务奠定基础。 

---
# Disentangling and Generating Modalities for Recommendation in Missing Modality Scenarios 

**Title (ZH)**: 在缺失模态场景中解耦和生成模态进行推荐 

**Authors**: Jiwan Kim, Hongseok Kang, Sein Kim, Kibum Kim, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.16352)  

**Abstract**: Multi-modal recommender systems (MRSs) have achieved notable success in improving personalization by leveraging diverse modalities such as images, text, and audio. However, two key challenges remain insufficiently addressed: (1) Insufficient consideration of missing modality scenarios and (2) the overlooking of unique characteristics of modality features. These challenges result in significant performance degradation in realistic situations where modalities are missing. To address these issues, we propose Disentangling and Generating Modality Recommender (DGMRec), a novel framework tailored for missing modality scenarios. DGMRec disentangles modality features into general and specific modality features from an information-based perspective, enabling richer representations for recommendation. Building on this, it generates missing modality features by integrating aligned features from other modalities and leveraging user modality preferences. Extensive experiments show that DGMRec consistently outperforms state-of-the-art MRSs in challenging scenarios, including missing modalities and new item settings as well as diverse missing ratios and varying levels of missing modalities. Moreover, DGMRec's generation-based approach enables cross-modal retrieval, a task inapplicable for existing MRSs, highlighting its adaptability and potential for real-world applications. Our code is available at this https URL. 

**Abstract (ZH)**: 多模态推荐系统（MRSs）通过利用图像、文本和音频等多种模态取得了显著的个性化提升。然而，仍存在两个关键挑战：（1）对缺失模态场景考虑不足；（2）忽视了模态特征的独特性。这些挑战导致在实际情况下模态缺失时性能显著下降。为解决这些问题，我们提出了基于信息视角解耦和生成模态推荐（DGMRec）的新框架，该框架专门针对缺失模态场景。DGMRec从信息视角将模态特征解耦为通用和特定的模态特征，从而为推荐提供更丰富的表示。在此基础上，它通过整合其他模态的对齐特征并利用用户的模态偏好生成缺失模态特征。广泛的实验表明，DGMRec在包含缺失模态和新项目设置等各种具有挑战性的场景中，以及在不同的缺失比例和不同的模态缺失程度下，均优于现有的MRSs。此外，DGMRec的生成方法使其能够进行跨模态检索，这是现有MRSs无法实现的任务，凸显了其适应性和在实际应用中的潜力。我们的代码可在以下链接获取：this https URL。 

---
# QAOA-GPT: Efficient Generation of Adaptive and Regular Quantum Approximate Optimization Algorithm Circuits 

**Title (ZH)**: QAOA-GPT: 高效生成自适应和规则量子近似优化算法电路 

**Authors**: Ilya Tyagin, Marwa H. Farag, Kyle Sherbert, Karunya Shirali, Yuri Alexeev, Ilya Safro  

**Link**: [PDF](https://arxiv.org/pdf/2504.16350)  

**Abstract**: Quantum computing has the potential to improve our ability to solve certain optimization problems that are computationally difficult for classical computers, by offering new algorithmic approaches that may provide speedups under specific conditions. In this work, we introduce QAOA-GPT, a generative framework that leverages Generative Pretrained Transformers (GPT) to directly synthesize quantum circuits for solving quadratic unconstrained binary optimization problems, and demonstrate it on the MaxCut problem on graphs. To diversify the training circuits and ensure their quality, we have generated a synthetic dataset using the adaptive QAOA approach, a method that incrementally builds and optimizes problem-specific circuits. The experiments conducted on a curated set of graph instances demonstrate that QAOA-GPT, generates high quality quantum circuits for new problem instances unseen in the training as well as successfully parametrizes QAOA. Our results show that using QAOA-GPT to generate quantum circuits will significantly decrease both the computational overhead of classical QAOA and adaptive approaches that often use gradient evaluation to generate the circuit and the classical optimization of the circuit parameters. Our work shows that generative AI could be a promising avenue to generate compact quantum circuits in a scalable way. 

**Abstract (ZH)**: 量子计算有望通过提供在特定条件下可能加速的新算法方法，提高我们解决某些计算上具有挑战性的优化问题的能力。在此工作中，我们介绍了QAOA-GPT，这是一种利用生成预训练变换器（GPT）直接合成求解无约束二次二元优化问题的量子电路的生成框架，并在图的MaxCut问题上进行了演示。为确保训练电路的质量并增加多样性，我们使用自适应QAOA方法生成了一个合成数据集，该方法能够逐步构建和优化针对特定问题的电路。针对精心挑选的图实例进行的实验表明，QAOA-GPT能够生成高质量的量子电路，适用于未在训练中出现的新问题实例，并成功参数化了QAOA。我们的结果表明，使用QAOA-GPT生成量子电路将大幅减少经典QAOA及其通常使用梯度评估生成电路和经典优化电路参数的自适应方法的计算开销。我们的工作表明，生成式AI可能是以可扩展的方式生成紧凑量子电路的一个有前途的方向。 

---
# Mining Software Repositories for Expert Recommendation 

**Title (ZH)**: 从软件仓库中挖掘专家推荐 

**Authors**: Chad Marshall, Andrew Barovic, Armin Moin  

**Link**: [PDF](https://arxiv.org/pdf/2504.16343)  

**Abstract**: We propose an automated approach to bug assignment to developers in large open-source software projects. This way, we assist human bug triagers who are in charge of finding the best developer with the right level of expertise in a particular area to be assigned to a newly reported issue. Our approach is based on the history of software development as documented in the issue tracking systems. We deploy BERTopic and techniques from TopicMiner. Our approach works based on the bug reports' features, such as the corresponding products and components, as well as their priority and severity levels. We sort developers based on their experience with specific combinations of new reports. The evaluation is performed using Top-k accuracy, and the results are compared with the reported results in prior work, namely TopicMiner MTM, BUGZIE, Bug triaging via deep Reinforcement Learning BT-RL, and LDA-SVM. The evaluation data come from various Eclipse and Mozilla projects, such as JDT, Firefox, and Thunderbird. 

**Abstract (ZH)**: 我们提出了一种自动化的方法，用于在大型开源软件项目中自动分配错误给开发者。通过这种方法，我们协助管理人员错误的人员找到最适合解决特定问题的具有适当专业知识水平的开发者。我们的方法基于问题跟踪系统中记录的软件开发历史。我们部署了BERTopic和TopicMiner的技术。我们的方法基于错误报告的特征，如相关的产物和组件，以及它们的优先级和严重程度。我们将开发者根据其对新报告的特定组合的经验进行排序。评估使用Top-k准确率进行，结果与先前的工作，即TopicMiner MTM、BUGZIE、基于深度强化学习的错误处理BT-RL以及LDA-SVM的比较。评估数据来自多个Eclipse和Mozilla项目，如JDT、Firefox和Thunderbird。 

---
# On the Consistency of GNN Explanations for Malware Detection 

**Title (ZH)**: 基于图神经网络恶意软件检测解释的一致性研究 

**Authors**: Hossein Shokouhinejad, Griffin Higgins, Roozbeh Razavi-Far, Hesamodin Mohammadian, Ali A. Ghorbani  

**Link**: [PDF](https://arxiv.org/pdf/2504.16316)  

**Abstract**: Control Flow Graphs (CFGs) are critical for analyzing program execution and characterizing malware behavior. With the growing adoption of Graph Neural Networks (GNNs), CFG-based representations have proven highly effective for malware detection. This study proposes a novel framework that dynamically constructs CFGs and embeds node features using a hybrid approach combining rule-based encoding and autoencoder-based embedding. A GNN-based classifier is then constructed to detect malicious behavior from the resulting graph representations. To improve model interpretability, we apply state-of-the-art explainability techniques, including GNNExplainer, PGExplainer, and CaptumExplainer, the latter is utilized three attribution methods: Integrated Gradients, Guided Backpropagation, and Saliency. In addition, we introduce a novel aggregation method, called RankFusion, that integrates the outputs of the top-performing explainers to enhance the explanation quality. We also evaluate explanations using two subgraph extraction strategies, including the proposed Greedy Edge-wise Composition (GEC) method for improved structural coherence. A comprehensive evaluation using accuracy, fidelity, and consistency metrics demonstrates the effectiveness of the proposed framework in terms of accurate identification of malware samples and generating reliable and interpretable explanations. 

**Abstract (ZH)**: 基于控制流图的图神经网络新型框架：结合规则编码和自编码嵌入的恶意软件检测与解释 

---
# DataS^3: Dataset Subset Selection for Specialization 

**Title (ZH)**: DataS³: 数据子集选择以实现专业化 

**Authors**: Neha Hulkund, Alaa Maalouf, Levi Cai, Daniel Yang, Tsun-Hsuan Wang, Abigail O'Neil, Timm Haucke, Sandeep Mukherjee, Vikram Ramaswamy, Judy Hansen Shen, Gabriel Tseng, Mike Walmsley, Daniela Rus, Ken Goldberg, Hannah Kerner, Irene Chen, Yogesh Girdhar, Sara Beery  

**Link**: [PDF](https://arxiv.org/pdf/2504.16277)  

**Abstract**: In many real-world machine learning (ML) applications (e.g. detecting broken bones in x-ray images, detecting species in camera traps), in practice models need to perform well on specific deployments (e.g. a specific hospital, a specific national park) rather than the domain broadly. However, deployments often have imbalanced, unique data distributions. Discrepancy between the training distribution and the deployment distribution can lead to suboptimal performance, highlighting the need to select deployment-specialized subsets from the available training data. We formalize dataset subset selection for specialization (DS3): given a training set drawn from a general distribution and a (potentially unlabeled) query set drawn from the desired deployment-specific distribution, the goal is to select a subset of the training data that optimizes deployment performance.
We introduce DataS^3; the first dataset and benchmark designed specifically for the DS3 problem. DataS^3 encompasses diverse real-world application domains, each with a set of distinct deployments to specialize in. We conduct a comprehensive study evaluating algorithms from various families--including coresets, data filtering, and data curation--on DataS^3, and find that general-distribution methods consistently fail on deployment-specific tasks. Additionally, we demonstrate the existence of manually curated (deployment-specific) expert subsets that outperform training on all available data with accuracy gains up to 51.3 percent. Our benchmark highlights the critical role of tailored dataset curation in enhancing performance and training efficiency on deployment-specific distributions, which we posit will only become more important as global, public datasets become available across domains and ML models are deployed in the real world. 

**Abstract (ZH)**: 在实际应用场景中（例如，X射线图像中的骨折检测，相机陷阱中的物种检测），模型需要在特定部署（如特定医院，特定国家公园）上表现良好，而不是在广泛的领域中。然而，部署往往具有不平衡且独特的数据分布。训练分布与部署分布之间的差异会导致性能不佳，从而强调需要从可用的训练数据中选择部署专门化的子集。我们形式化了数据子集选择以专门化（Dataset Subset Selection for Specialization, DS3）：给定一个来自一般分布的训练集和一个（可能未标注的）来自目标部署特定分布的查询集，目标是选择一个优化部署性能的训练数据子集。

DataS^3：第一个专门为DS3问题设计的数据集和基准。DataS^3涵盖了多种多样的实际应用场景领域，每个领域都有专门化的部署集。我们在DataS^3上对来自不同家族的算法（包括核集、数据过滤和数据整理）进行了全面评估，发现一般分布方法在特定部署任务上表现不佳。此外，我们证明了手动整理的（部署特定的）专家子集优于在所有可用数据上进行训练，准确率可提高多达51.3%。我们的基准突显了针对特定部署分布进行精心数据整理在提高性能和训练效率方面的重要作用，我们认为随着全球公共数据集在各领域中的普及以及ML模型在现实世界的部署，这一作用将变得更加重要。 

---
# An Automated Pipeline for Few-Shot Bird Call Classification: A Case Study with the Tooth-Billed Pigeon 

**Title (ZH)**: 基于Few-Shot鸟类叫声分类的自动化管道：以齿aileronto鸽为例的研究案例 

**Authors**: Abhishek Jana, Moeumu Uili, James Atherton, Mark O'Brien, Joe Wood, Leandra Brickson  

**Link**: [PDF](https://arxiv.org/pdf/2504.16276)  

**Abstract**: This paper presents an automated one-shot bird call classification pipeline designed for rare species absent from large publicly available classifiers like BirdNET and Perch. While these models excel at detecting common birds with abundant training data, they lack options for species with only 1-3 known recordings-a critical limitation for conservationists monitoring the last remaining individuals of endangered birds. To address this, we leverage the embedding space of large bird classification networks and develop a classifier using cosine similarity, combined with filtering and denoising preprocessing techniques, to optimize detection with minimal training data. We evaluate various embedding spaces using clustering metrics and validate our approach in both a simulated scenario with Xeno-Canto recordings and a real-world test on the critically endangered tooth-billed pigeon (Didunculus strigirostris), which has no existing classifiers and only three confirmed recordings. The final model achieved 1.0 recall and 0.95 accuracy in detecting tooth-billed pigeon calls, making it practical for use in the field. This open-source system provides a practical tool for conservationists seeking to detect and monitor rare species on the brink of extinction. 

**Abstract (ZH)**: 一种针对大型公开分类器（如BirdNET和Perch）未包含的稀有鸟类设计的自动化单次鸟类叫声分类管道的研究 

---
# Quantum Doubly Stochastic Transformers 

**Title (ZH)**: 量子双随机变换器 

**Authors**: Jannis Born, Filip Skogh, Kahn Rhrissorrakrai, Filippo Utro, Nico Wagner, Aleksandros Sobczyk  

**Link**: [PDF](https://arxiv.org/pdf/2504.16275)  

**Abstract**: At the core of the Transformer, the Softmax normalizes the attention matrix to be right stochastic. Previous research has shown that this often destabilizes training and that enforcing the attention matrix to be doubly stochastic (through Sinkhorn's algorithm) consistently improves performance across different tasks, domains and Transformer flavors. However, Sinkhorn's algorithm is iterative, approximative, non-parametric and thus inflexible w.r.t. the obtained doubly stochastic matrix (DSM). Recently, it has been proven that DSMs can be obtained with a parametric quantum circuit, yielding a novel quantum inductive bias for DSMs with no known classical analogue. Motivated by this, we demonstrate the feasibility of a hybrid classical-quantum doubly stochastic Transformer (QDSFormer) that replaces the Softmax in the self-attention layer with a variational quantum circuit. We study the expressive power of the circuit and find that it yields more diverse DSMs that better preserve information than classical operators. Across multiple small-scale object recognition tasks, we find that our QDSFormer consistently surpasses both a standard Vision Transformer and other doubly stochastic Transformers. Beyond the established Sinkformer, this comparison includes a novel quantum-inspired doubly stochastic Transformer (based on QR decomposition) that can be of independent interest. The QDSFormer also shows improved training stability and lower performance variation suggesting that it may mitigate the notoriously unstable training of ViTs on small-scale data. 

**Abstract (ZH)**: 基于变换器的软-max归一化可将注意力矩阵标准化为正确的随机矩阵。前期研究显示，这经常导致训练不稳定，而通过Sinkhorn算法强制注意力矩阵成为双随机矩阵（DSM）则在不同任务、领域和变换器版本中均能一致地提升性能。然而，Sinkhorn算法是迭代的、近似的、非参数化的，因此在获得双随机矩阵方面不够灵活。最近的研究证明，可以通过参数化的量子电路获得DSM，这为DSM提供了一种新型的量子归纳偏置，而这种偏置在经典的类比中尚不存在。受此启发，我们证明了一种混合经典-量子双随机变换器（QDSFormer）的可行性，其中自注意力层中的Softmax被变分量子电路替代。我们研究了该电路的表达能力，发现其产生的DSM更加多样，更能保留信息，而不同于经典的算子。在多个小型对象识别任务中，我们发现我们的QDSFormer在性能上始终优于标准视觉变换器和其他双随机变换器。除了现有的Sinkformer之外，此比较还包括一种基于QR分解的新型量子启发双随机变换器，该变换器也具有独立的研究价值。QDSFormer还显示出改进的训练稳定性和较低的性能变异性，表明它可能缓解视觉变换器在小规模数据上的 notoriously 不稳定训练问题。 

---
# Boosting Classifier Performance with Opposition-Based Data Transformation 

**Title (ZH)**: 基于反对面数据转换的分类器性能提升方法 

**Authors**: Abdesslem Layeb  

**Link**: [PDF](https://arxiv.org/pdf/2504.16268)  

**Abstract**: In this paper, we introduce a novel data transformation framework based on Opposition-Based Learning (OBL) to boost the performance of traditional classification algorithms. Originally developed to accelerate convergence in optimization tasks, OBL is leveraged here to generate synthetic opposite samples that replace the acutely training data and improve decision boundary formation. We explore three OBL variants; Global OBL, Class-Wise OBL, and Localized Class-Wise OBL; and integrate them with several widely used classifiers, including K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Logistic Regression (LR), and Decision Tree (DT). Extensive experiments conducted on 26 heterogeneous and high-dimensional datasets demonstrate that OBL-enhanced classifiers consistently outperform their standard counterparts in terms of accuracy and F1-score, frequently achieving near-perfect or perfect classification. Furthermore, OBL contributes to improved computational efficiency, particularly in SVM and LR. These findings underscore the potential of OBL as a lightweight yet powerful data transformation strategy for enhancing classification performance, especially in complex or sparse learning environments. 

**Abstract (ZH)**: 基于反对学习的数据转换框架在提升传统分类算法性能中的应用 

---
# Gradient-Optimized Fuzzy Classifier: A Benchmark Study Against State-of-the-Art Models 

**Title (ZH)**: 基于梯度优化的模糊分类器：与先进模型的基准研究 

**Authors**: Magnus Sieverding, Nathan Steffen, Kelly Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2504.16263)  

**Abstract**: This paper presents a performance benchmarking study of a Gradient-Optimized Fuzzy Inference System (GF) classifier against several state-of-the-art machine learning models, including Random Forest, XGBoost, Logistic Regression, Support Vector Machines, and Neural Networks. The evaluation was conducted across five datasets from the UCI Machine Learning Repository, each chosen for their diversity in input types, class distributions, and classification complexity. Unlike traditional Fuzzy Inference Systems that rely on derivative-free optimization methods, the GF leverages gradient descent to significantly improving training efficiency and predictive performance. Results demonstrate that the GF model achieved competitive, and in several cases superior, classification accuracy while maintaining high precision and exceptionally low training times. In particular, the GF exhibited strong consistency across folds and datasets, underscoring its robustness in handling noisy data and variable feature sets. These findings support the potential of gradient optimized fuzzy systems as interpretable, efficient, and adaptable alternatives to more complex deep learning models in supervised learning tasks. 

**Abstract (ZH)**: 基于梯度优化的模糊推理系统(GF)分类器与多种先进机器学习模型的性能基准研究 

---
# Blockchain Meets Adaptive Honeypots: A Trust-Aware Approach to Next-Gen IoT Security 

**Title (ZH)**: 区块链与自适应蜜罐相结合：一种基于信任的物联网安全下一代方案 

**Authors**: Yazan Otoum, Arghavan Asad, Amiya Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2504.16226)  

**Abstract**: Edge computing-based Next-Generation Wireless Networks (NGWN)-IoT offer enhanced bandwidth capacity for large-scale service provisioning but remain vulnerable to evolving cyber threats. Existing intrusion detection and prevention methods provide limited security as adversaries continually adapt their attack strategies. We propose a dynamic attack detection and prevention approach to address this challenge. First, blockchain-based authentication uses the Deoxys Authentication Algorithm (DAA) to verify IoT device legitimacy before data transmission. Next, a bi-stage intrusion detection system is introduced: the first stage uses signature-based detection via an Improved Random Forest (IRF) algorithm. In contrast, the second stage applies feature-based anomaly detection using a Diffusion Convolution Recurrent Neural Network (DCRNN). To ensure Quality of Service (QoS) and maintain Service Level Agreements (SLA), trust-aware service migration is performed using Heap-Based Optimization (HBO). Additionally, on-demand virtual High-Interaction honeypots deceive attackers and extract attack patterns, which are securely stored using the Bimodal Lattice Signature Scheme (BLISS) to enhance signature-based Intrusion Detection Systems (IDS). The proposed framework is implemented in the NS3 simulation environment and evaluated against existing methods across multiple performance metrics, including accuracy, attack detection rate, false negative rate, precision, recall, ROC curve, memory usage, CPU usage, and execution time. Experimental results demonstrate that the framework significantly outperforms existing approaches, reinforcing the security of NGWN-enabled IoT ecosystems 

**Abstract (ZH)**: 基于边缘计算的下一代无线网络（NGWN）-IoT提供大规模服务供应的增强带宽容量，但仍易受不断演变的网络安全威胁。现有的入侵检测与预防方法提供有限的安全性，因为攻击者不断调整其攻击策略。我们提出了一种动态攻击检测与预防方法以应对这一挑战。首先，基于区块链的认证使用Deoxys认证算法（DAA）在数据传输前验证物联网设备的合法性。其次，引入了一种两阶段入侵检测系统：第一阶段使用改进的随机森林（IRF）算法进行签名检测；第二阶段采用扩散卷积循环神经网络（DCRNN）进行基于特征的异常检测。为确保服务质量（QoS）并维持服务级别协议（SLA），使用堆基优化（HBO）进行信任感知的服务迁移。此外，基于需求的虚拟高互动蜜罐欺骗攻击者并提取攻击模式，这些模式通过二模格签名方案（BLISS）安全存储，以增强基于签名的入侵检测系统。所提出的框架在NS3仿真环境中实现，并通过多个性能指标与现有方法进行评估，包括准确性、攻击检测率、假阴性率、精确率、召回率、ROC曲线、内存使用率、CPU使用率和执行时间。实验结果表明，该框架显著优于现有方法，增强了NGWN驱动的物联网生态系统安全性。 

---
# Hexcute: A Tile-based Programming Language with Automatic Layout and Task-Mapping Synthesis 

**Title (ZH)**: Hexcute: 基于瓷砖的编程语言，自动布局与任务映射合成 

**Authors**: Xiao Zhang, Yaoyao Ding, Yang Hu, Gennady Pekhimenko  

**Link**: [PDF](https://arxiv.org/pdf/2504.16214)  

**Abstract**: Deep learning (DL) workloads mainly run on accelerators like GPUs. Recent DL quantization techniques demand a new matrix multiplication operator with mixed input data types, further complicating GPU optimization. Prior high-level compilers like Triton lack the expressiveness to implement key optimizations like fine-grained data pipelines and hardware-friendly memory layouts for these operators, while low-level programming models, such as Hidet, Graphene, and CUTLASS, require significant programming efforts. To balance expressiveness with engineering effort, we propose Hexcute, a tile-based programming language that exposes shared memory and register abstractions to enable fine-grained optimization for these operators. Additionally, Hexcute leverages task mapping to schedule the GPU program, and to reduce programming efforts, it automates layout and task mapping synthesis with a novel type-inference-based algorithm. Our evaluation shows that Hexcute generalizes to a wide range of DL operators, achieves 1.7-11.28$\times$ speedup over existing DL compilers for mixed-type operators, and brings up to 2.91$\times$ speedup in the end-to-end evaluation. 

**Abstract (ZH)**: 基于 Tiles 的 Hexcute 编程语言：一种细粒度优化的深度学习矩阵乘法操作符编程方法 

---
# TinyML for Speech Recognition 

**Title (ZH)**: TinyML for Speech Recognition 

**Authors**: Andrew Barovic, Armin Moin  

**Link**: [PDF](https://arxiv.org/pdf/2504.16213)  

**Abstract**: We train and deploy a quantized 1D convolutional neural network model to conduct speech recognition on a highly resource-constrained IoT edge device. This can be useful in various Internet of Things (IoT) applications, such as smart homes and ambient assisted living for the elderly and people with disabilities, just to name a few examples. In this paper, we first create a new dataset with over one hour of audio data that enables our research and will be useful to future studies in this field. Second, we utilize the technologies provided by Edge Impulse to enhance our model's performance and achieve a high Accuracy of up to 97% on our dataset. For the validation, we implement our prototype using the Arduino Nano 33 BLE Sense microcontroller board. This microcontroller board is specifically designed for IoT and AI applications, making it an ideal choice for our target use case scenarios. While most existing research focuses on a limited set of keywords, our model can process 23 different keywords, enabling complex commands. 

**Abstract (ZH)**: 我们训练并部署了一个量化的一维卷积神经网络模型，以在资源极度受限的物联网边缘设备上进行语音识别。这在智能家居、辅助生活等物联网应用中具有实用价值。在本文中，我们首先创建了一个包含超过一小时音频数据的新数据集，以支持我们的研究，并对未来的研究具有参考价值。其次，我们利用Edge Impulse提供的技术优化了模型性能，并在数据集上达到了高达97%的准确率。在验证过程中，我们使用Arduino Nano 33 BLE Sense微控制器板实现我们的原型。该微控制器板专门设计用于物联网和人工智能应用，使其成为我们的目标应用场景的理想选择。与现有研究主要关注有限的关键词集不同，我们的模型可以处理23个不同的关键词，从而实现复杂的命令处理。 

---
# Reflexive Prompt Engineering: A Framework for Responsible Prompt Engineering and Interaction Design 

**Title (ZH)**: 反思性提示工程：负责任的提示工程与交互设计框架 

**Authors**: Christian Djeffal  

**Link**: [PDF](https://arxiv.org/pdf/2504.16204)  

**Abstract**: Responsible prompt engineering has emerged as a critical framework for ensuring that generative artificial intelligence (AI) systems serve society's needs while minimizing potential harms. As generative AI applications become increasingly powerful and ubiquitous, the way we instruct and interact with them through prompts has profound implications for fairness, accountability, and transparency. This article examines how strategic prompt engineering can embed ethical and legal considerations and societal values directly into AI interactions, moving beyond mere technical optimization for functionality. This article proposes a comprehensive framework for responsible prompt engineering that encompasses five interconnected components: prompt design, system selection, system configuration, performance evaluation, and prompt management. Drawing from empirical evidence, the paper demonstrates how each component can be leveraged to promote improved societal outcomes while mitigating potential risks. The analysis reveals that effective prompt engineering requires a delicate balance between technical precision and ethical consciousness, combining the systematic rigor and focus on functionality with the nuanced understanding of social impact. Through examination of real-world and emerging practices, the article illustrates how responsible prompt engineering serves as a crucial bridge between AI development and deployment, enabling organizations to fine-tune AI outputs without modifying underlying model architectures. This approach aligns with broader "Responsibility by Design" principles, embedding ethical considerations directly into the implementation process rather than treating them as post-hoc additions. The article concludes by identifying key research directions and practical guidelines for advancing the field of responsible prompt engineering. 

**Abstract (ZH)**: 负责任的提示工程已 emerges as a critical framework for ensuring that generative artificial intelligence (AI) systems serve society's needs while minimizing potential harms. 

---
# Quality of explanation of xAI from the prespective of Italian end-users: Italian version of System Causability Scale (SCS) 

**Title (ZH)**: 从意大利终端用户视角解释xAI的质量：System Causability Scale (SCS)的意大利版本 

**Authors**: Carmine Attanasio, Alireza Mortezapour  

**Link**: [PDF](https://arxiv.org/pdf/2504.16193)  

**Abstract**: Background and aim: Considering the scope of the application of artificial intelligence beyond the field of computer science, one of the concerns of researchers is to provide quality explanations about the functioning of algorithms based on artificial intelligence and the data extracted from it. The purpose of the present study is to validate the Italian version of system causability scale (I-SCS) to measure the quality of explanations provided in a xAI.
Method: For this purpose, the English version, initially provided in 2020 in coordination with the main developer, was utilized. The forward-backward translation method was applied to ensure accuracy. Finally, these nine steps were completed by calculating the content validity index/ratio and conducting cognitive interviews with representative end users.
Results: The original version of the questionnaire consisted of 10 questions. However, based on the obtained indexes (CVR below 0.49), one question (Question 8) was entirely removed. After completing the aforementioned steps, the Italian version contained 9 questions. The representative sample of Italian end users fully comprehended the meaning and content of the questions in the Italian version.
Conclusion: The Italian version obtained in this study can be used in future research studies as well as in the field by xAI developers. This tool can be used to measure the quality of explanations provided for an xAI system in Italian culture. 

**Abstract (ZH)**: 背景与目的：考虑到人工智能在计算机科学领域之外的应用范围，研究人员的一个关切是如何提供高质量的解释，这些解释基于人工智能算法及其提取的数据。本研究的目的在于验证系统因果量表的意大利语版本（I-SCS），以衡量xAI提供的解释质量。方法：为此目的，使用了2020年与主要开发者协调提供的英语版本，并采用了正向与反向翻译的方法以确保准确性。最终通过计算内容效度指数/比率并进行认知访谈，完成了这九个步骤。结果：原问卷包含10个问题。然而，根据获得的指数（CVR低于0.49），完全删除了一个问题（第8题）。完成上述步骤后，意大利语版本包含9个问题。代表性意大利终端用户完全理解了意大利语版本中问题的意义和内容。结论：本研究获得的意大利语版本可以在未来的研究中以及xAI开发者领域使用。此工具可以用于衡量意大利文化背景下xAI系统提供的解释质量。 

---
# FinNLI: Novel Dataset for Multi-Genre Financial Natural Language Inference Benchmarking 

**Title (ZH)**: FinNLI: 新颖的数据集用于多文体金融自然语言推理基准测试 

**Authors**: Jabez Magomere, Elena Kochkina, Samuel Mensah, Simerjot Kaur, Charese H. Smiley  

**Link**: [PDF](https://arxiv.org/pdf/2504.16188)  

**Abstract**: We introduce FinNLI, a benchmark dataset for Financial Natural Language Inference (FinNLI) across diverse financial texts like SEC Filings, Annual Reports, and Earnings Call transcripts. Our dataset framework ensures diverse premise-hypothesis pairs while minimizing spurious correlations. FinNLI comprises 21,304 pairs, including a high-quality test set of 3,304 instances annotated by finance experts. Evaluations show that domain shift significantly degrades general-domain NLI performance. The highest Macro F1 scores for pre-trained (PLMs) and large language models (LLMs) baselines are 74.57% and 78.62%, respectively, highlighting the dataset's difficulty. Surprisingly, instruction-tuned financial LLMs perform poorly, suggesting limited generalizability. FinNLI exposes weaknesses in current LLMs for financial reasoning, indicating room for improvement. 

**Abstract (ZH)**: Financial Natural Language Inference Benchmark Dataset (FinNLI) Across Diverse Financial Texts 

---
# FPGA-Based Neural Network Accelerators for Space Applications: A Survey 

**Title (ZH)**: 基于FPGA的太空应用神经网络加速器：一个综述 

**Authors**: Pedro Antunes, Artur Podobas  

**Link**: [PDF](https://arxiv.org/pdf/2504.16173)  

**Abstract**: Space missions are becoming increasingly ambitious, necessitating high-performance onboard spacecraft computing systems. In response, field-programmable gate arrays (FPGAs) have garnered significant interest due to their flexibility, cost-effectiveness, and radiation tolerance potential. Concurrently, neural networks (NNs) are being recognized for their capability to execute space mission tasks such as autonomous operations, sensor data analysis, and data compression. This survey serves as a valuable resource for researchers aiming to implement FPGA-based NN accelerators in space applications. By analyzing existing literature, identifying trends and gaps, and proposing future research directions, this work highlights the potential of these accelerators to enhance onboard computing systems. 

**Abstract (ZH)**: 空间任务变得越来越具雄心，需要高性能的机载航天器计算系统。为响应这一需求，现场可编程门阵列（FPGAs）因其灵活性、成本效益以及辐射耐受潜力而引起广泛关注。同时，神经网络（NNs）正被认可为执行空间任务的关键技术，例如自主操作、传感器数据处理和数据压缩。本文综述旨在为研究人员实施基于FPGA的NN加速器提供有价值的资源。通过分析现有文献、识别趋势和空白，以及提出未来的研究方向，本文突显了这些加速器增强机载计算系统的潜力。 

---
# Physics-Informed Inference Time Scaling via Simulation-Calibrated Scientific Machine Learning 

**Title (ZH)**: 基于物理信息的推断时间缩放通过仿真校准的科学机器学习 

**Authors**: Zexi Fan, Yan Sun, Shihao Yang, Yiping Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.16172)  

**Abstract**: High-dimensional partial differential equations (PDEs) pose significant computational challenges across fields ranging from quantum chemistry to economics and finance. Although scientific machine learning (SciML) techniques offer approximate solutions, they often suffer from bias and neglect crucial physical insights. Inspired by inference-time scaling strategies in language models, we propose Simulation-Calibrated Scientific Machine Learning (SCaSML), a physics-informed framework that dynamically refines and debiases the SCiML predictions during inference by enforcing the physical laws. SCaSML leverages derived new physical laws that quantifies systematic errors and employs Monte Carlo solvers based on the Feynman-Kac and Elworthy-Bismut-Li formulas to dynamically correct the prediction. Both numerical and theoretical analysis confirms enhanced convergence rates via compute-optimal inference methods. Our numerical experiments demonstrate that SCaSML reduces errors by 20-50% compared to the base surrogate model, establishing it as the first algorithm to refine approximated solutions to high-dimensional PDE during inference. Code of SCaSML is available at this https URL. 

**Abstract (ZH)**: 高维偏微分方程的物理校准科学机器学习（SCaSML） 

---
# Leveraging Social Media Analytics for Sustainability Trend Detection in Saudi Arabias Evolving Market 

**Title (ZH)**: 利用社交媒体分析检测Saudi Arabia evolving市场可持续性趋势 

**Authors**: Kanwal Aalijah  

**Link**: [PDF](https://arxiv.org/pdf/2504.16153)  

**Abstract**: Saudi Arabias rapid economic growth and social evolution under Vision 2030 present a unique opportunity to track emerging trends in real time. Uncovering trends in real time can open up new avenues for business and investment opportunities. This paper explores how AI and social media analytics can uncover and monitor these trends across sectors like sustainability, construction, food beverages industry, tourism, technology, and entertainment. This paper focus on use of AI-driven methodology to identify sustainability trends across Saudi Arabia. We processed millions of social media posts, news, blogs in order to understand sustainability trends in the region. The paper presents an AI approach that can help economists, businesses, government to understand sustainability trends and make better decisions around them. This approach offers both sector-specific and cross-sector insights, giving decision-makers a reliable, up to date snapshot of Saudi Arabias market shifts. Beyond Saudi Arabia, this framework also shows potential for adapting to other regions. Overall, our findings highlight how by using AI-methodologies, give decision makers a reliable method to understand how initiatives are perceived and adopted by the public and understand growth of trends. 

**Abstract (ZH)**: 沙特阿拉伯在Vision 2030愿景下的快速经济成长与社会进化提供了实时追踪新兴趋势的独特机会。实时发现趋势可以开辟新的商业和投资机会。本文探讨了如何通过人工智能和社会媒体分析在可持续性、建筑、食品饮料行业、旅游业、技术和娱乐等领域发现和监控这些趋势。本文重点探讨了利用人工智能驱动的方法在沙特阿拉伯识别可持续性趋势。我们处理了数百万条社交媒体帖子、新闻和博客，以了解该地区的可持续性趋势。本文提出了一种人工智能方法，可以帮助经济学家、企业、政府了解可持续性趋势，并围绕这些趋势做出更好的决策。该方法提供了领域特定和跨领域的洞察，为决策者提供了可靠的、及时的市场转变快照。此外，该框架还展示了适应其他地区的潜力。总体而言，我们的研究结果强调，通过使用人工智能方法，决策者可以获得一种可靠的方法来理解公众对倡议的看法和接受度，以及趋势的增长。 

---
# Heterogeneous networks in drug-target interaction prediction 

**Title (ZH)**: 药物-靶标相互作用预测中的异质网络模型 

**Authors**: Mohammad Molaee, Nasrollah Moghadam Charkari  

**Link**: [PDF](https://arxiv.org/pdf/2504.16152)  

**Abstract**: Drug discovery requires a tremendous amount of time and cost. Computational drug-target interaction prediction, a significant part of this process, can reduce these requirements by narrowing the search space for wet lab experiments. In this survey, we provide comprehensive details of graph machine learning-based methods in predicting drug-target interaction, as they have shown promising results in this field. These details include the overall framework, main contribution, datasets, and their source codes. The selected papers were mainly published from 2020 to 2024. Prior to discussing papers, we briefly introduce the datasets commonly used with these methods and measurements to assess their performance. Finally, future challenges and some crucial areas that need to be explored are discussed. 

**Abstract (ZH)**: 药物发现需要大量的时间和成本。基于图机器学习的药物-靶点相互作用预测，在这一过程中占据显著位置，可以通过缩小湿实验室实验的搜索空间来减少这些需求。在本综述中，我们提供了基于图机器学习方法预测药物-靶点相互作用的全面细节，因为这些方法在该领域展现出了令人鼓舞的结果。这些细节包括整体框架、主要贡献、数据集及其源代码。所选择的论文主要发表于2020年至2024年。在讨论论文之前，我们简要介绍了这些方法常使用的数据集和评估性能的度量标准。最后，讨论了未来的研究挑战和需要探索的关键领域。 

---
# Towards responsible AI for education: Hybrid human-AI to confront the Elephant in the room 

**Title (ZH)**: 面向教育的负责任AI：人机融合共克教育领域的顽疾 

**Authors**: Danial Hooshyar, Gustav Šír, Yeongwook Yang, Eve Kikas, Raija Hämäläinen, Tommi Kärkkäinen, Dragan Gašević, Roger Azevedo  

**Link**: [PDF](https://arxiv.org/pdf/2504.16148)  

**Abstract**: Despite significant advancements in AI-driven educational systems and ongoing calls for responsible AI for education, several critical issues remain unresolved -- acting as the elephant in the room within AI in education, learning analytics, educational data mining, learning sciences, and educational psychology communities. This critical analysis identifies and examines nine persistent challenges that continue to undermine the fairness, transparency, and effectiveness of current AI methods and applications in education. These include: (1) the lack of clarity around what AI for education truly means -- often ignoring the distinct purposes, strengths, and limitations of different AI families -- and the trend of equating it with domain-agnostic, company-driven large language models; (2) the widespread neglect of essential learning processes such as motivation, emotion, and (meta)cognition in AI-driven learner modelling and their contextual nature; (3) limited integration of domain knowledge and lack of stakeholder involvement in AI design and development; (4) continued use of non-sequential machine learning models on temporal educational data; (5) misuse of non-sequential metrics to evaluate sequential models; (6) use of unreliable explainable AI methods to provide explanations for black-box models; (7) ignoring ethical guidelines in addressing data inconsistencies during model training; (8) use of mainstream AI methods for pattern discovery and learning analytics without systematic benchmarking; and (9) overemphasis on global prescriptions while overlooking localised, student-specific recommendations. Supported by theoretical and empirical research, we demonstrate how hybrid AI methods -- specifically neural-symbolic AI -- can address the elephant in the room and serve as the foundation for responsible, trustworthy AI systems in education. 

**Abstract (ZH)**: 尽管在AI驱动教育系统方面取得了显著进展，并不断呼吁负责任的AI应用于教育，但仍存在若干关键问题——这些问题在教育AI、学习分析、教育数据挖掘、学习科学和教育心理学社区中如同未解决的象てしま停滞不前。本文批判性地识别并分析了九个持续存在的挑战，这些挑战继续削弱当前AI方法和在教育中的有效性。这些挑战包括：（1）对教育AI的真正含义缺乏明确性——往往忽视不同AI家庭的独特目的、优势和局限性，并将其与领域无关、企业驱动的大语言模型相提并论；（2）广泛忽视动机、情绪和元认知等基本学习过程在其情境性；（3）领域知识的有限集成和利益相关者在AI设计和开发中的参与不足；（4）继续在时间序列教育数据上使用非顺序机器学习模型；（5）使用非顺序指标评估顺序模型；（6）使用不可靠的解释性AI方法为黑盒模型提供解释；（7）在模型训练过程中忽视伦理准则以解决数据不一致性；（8）未经系统基准测试就使用主流AI方法进行模式发现和学习分析；（9）过分强调全局处方而忽视本地化的、针对学生的具体建议。我们通过理论和实证研究证明，特定类型的混合AI方法——神经符号AI——可以解决上述问题，并为教育中负责任和可信赖的AI系统奠定基础。 

---
# A Non-Invasive Load Monitoring Method for Edge Computing Based on MobileNetV3 and Dynamic Time Regulation 

**Title (ZH)**: 基于MobileNetV3和动态时间规整的边缘计算非侵入式负荷监测方法 

**Authors**: Hangxu Liu, Yaojie Sun, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.16142)  

**Abstract**: In recent years, non-intrusive load monitoring (NILM) technology has attracted much attention in the related research field by virtue of its unique advantage of utilizing single meter data to achieve accurate decomposition of device-level energy consumption. Cutting-edge methods based on machine learning and deep learning have achieved remarkable results in load decomposition accuracy by fusing time-frequency domain features. However, these methods generally suffer from high computational costs and huge memory requirements, which become the main obstacles for their deployment on resource-constrained microcontroller units (MCUs). To address these challenges, this study proposes an innovative Dynamic Time Warping (DTW) algorithm in the time-frequency domain and systematically compares and analyzes the performance of six machine learning techniques in home electricity scenarios. Through complete experimental validation on edge MCUs, this scheme successfully achieves a recognition accuracy of 95%. Meanwhile, this study deeply optimizes the frequency domain feature extraction process, which effectively reduces the running time by 55.55% and the storage overhead by about 34.6%. The algorithm performance will be further optimized in future research work. Considering that the elimination of voltage transformer design can significantly reduce the cost, the subsequent research will focus on this direction, and is committed to providing more cost-effective solutions for the practical application of NILM, and providing a solid theoretical foundation and feasible technical paths for the design of efficient NILM systems in edge computing environments. 

**Abstract (ZH)**: 近年来，非侵入式负荷监测（NILM）技术因其利用单个电表数据实现设备级能源消耗精确分解的独特优势，在相关研究领域引起了广泛关注。基于机器学习和深度学习的先进方法通过时频域特征融合，在负荷分解准确性方面取得了显著成果。然而，这些方法通常面临高计算成本和巨大内存需求的问题，成为其在资源受限微控制器（MCUs）上部署的主要障碍。为应对这些挑战，本研究提出了一种创新的时频域动态时间 warped（DTW）算法，并系统比较和分析了六种机器学习技术在家庭电力场景中的性能。通过在边缘MCUs上进行全面实验验证，该方案成功实现了95%的识别准确率。同时，本研究深入优化了频域特征提取过程，有效减少了55.55%的运行时间并降低了约34.6%的存储开销。未来研究将进一步优化算法性能。鉴于消除电压互感器设计可显著降低成本，后续研究将侧重于此方向，致力于提供更经济有效的非侵入式负荷监测解决方案，并为边缘计算环境中高效非侵入式负荷监测系统的设计提供坚实的理论基础和可行的技术路径。 

---
# SparseJEPA: Sparse Representation Learning of Joint Embedding Predictive Architectures 

**Title (ZH)**: SparseJEPA: 联合嵌入预测架构的稀疏表示学习 

**Authors**: Max Hartman, Lav Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2504.16140)  

**Abstract**: Joint Embedding Predictive Architectures (JEPA) have emerged as a powerful framework for learning general-purpose representations. However, these models often lack interpretability and suffer from inefficiencies due to dense embedding representations. We propose SparseJEPA, an extension that integrates sparse representation learning into the JEPA framework to enhance the quality of learned representations. SparseJEPA employs a penalty method that encourages latent space variables to be shared among data features with strong semantic relationships, while maintaining predictive performance. We demonstrate the effectiveness of SparseJEPA by training on the CIFAR-100 dataset and pre-training a lightweight Vision Transformer. The improved embeddings are utilized in linear-probe transfer learning for both image classification and low-level tasks, showcasing the architecture's versatility across different transfer tasks. Furthermore, we provide a theoretical proof that demonstrates that the grouping mechanism enhances representation quality. This was done by displaying that grouping reduces Multiinformation among latent-variables, including proofing the Data Processing Inequality for Multiinformation. Our results indicate that incorporating sparsity not only refines the latent space but also facilitates the learning of more meaningful and interpretable representations. In further work, hope to further extend this method by finding new ways to leverage the grouping mechanism through object-centric representation learning. 

**Abstract (ZH)**: SparseJEPA：将稀疏表示学习整合到联合嵌入预测架构中以提高表示质量 

---
# Enhancing Trust Through Standards: A Comparative Risk-Impact Framework for Aligning ISO AI Standards with Global Ethical and Regulatory Contexts 

**Title (ZH)**: 通过标准提升信任：一种将ISO AI标准与全球伦理和监管 contexts 对齐的比较风险影响框架 

**Authors**: Sridharan Sankaran  

**Link**: [PDF](https://arxiv.org/pdf/2504.16139)  

**Abstract**: As artificial intelligence (AI) reshapes industries and societies, ensuring its trustworthiness-through mitigating ethical risks like bias, opacity, and accountability deficits-remains a global challenge. International Organization for Standardization (ISO) AI standards, such as ISO/IEC 24027 and 24368, aim to foster responsible development by embedding fairness, transparency, and risk management into AI systems. However, their effectiveness varies across diverse regulatory landscapes, from the EU's risk-based AI Act to China's stability-focused measures and the U.S.'s fragmented state-led initiatives. This paper introduces a novel Comparative Risk-Impact Assessment Framework to evaluate how well ISO standards address ethical risks within these contexts, proposing enhancements to strengthen their global applicability. By mapping ISO standards to the EU AI Act and surveying regulatory frameworks in ten regions-including the UK, Canada, India, Japan, Singapore, South Korea, and Brazil-we establish a baseline for ethical alignment. The framework, applied to case studies in the EU, US-Colorado, and China, reveals gaps: voluntary ISO standards falter in enforcement (e.g., Colorado) and undervalue region-specific risks like privacy (China). We recommend mandatory risk audits, region-specific annexes, and a privacy-focused module to enhance ISO's adaptability. This approach not only synthesizes global trends but also offers a replicable tool for aligning standardization with ethical imperatives, fostering interoperability and trust in AI worldwide. Policymakers and standards bodies can leverage these insights to evolve AI governance, ensuring it meets diverse societal needs as the technology advances. 

**Abstract (ZH)**: 随着人工智能（AI）重塑产业和社會，確保其可信度——通過減輕偏見、不透明和 Accountability 缺陷等道德風險——依然是一個全球性的挑戰。國際标准化組織（ISO）的AI標準，如ISO/IEC 24027和24368，旨在通過將公平、透明和風險管理嵌入AI系統而促進負責開發。然而，這些標準在全球不同的監管環境中的有效性各不相同，從歐盟的風險基輔助AI法到中國的重点穩定措施和美國的碎片化州主导Initiatives。本文引入了一種新的比較風險影響評估框架，以評估ISO標準如何在這些環境中應對道德風險，並提出改進措施以增強其全球適用性。通過將ISO標準映射到歐盟AI法，并對包括英國、加拿大、印度、日本、新加坡、韓國和巴西在內的十個地區的規制框架進行調研，我們建立了符合道德標準的基線。該框架應用於歐盟、美國丹佛和中國的案例研究，揭示了缺口：自愿ISO標準在執行情況下 crumbling（如丹佛）並低估地方特定風險（如中國的隱私）。我們建議實行強制風險審核、地區特定附錄和隱私Focus模塊，以增強ISO的靈活性。該方法不僅綜合了全球趨勢，還提供了一種可複製工具，以使標準化與道德要求對接，促進人工智能的互操作性和可信度。政策制定者和標準機構可以借助這些洞察來進化人工智能治理，確保其滿足技術進步帶來的多樣化社會需求。 

---
# Trends in Frontier AI Model Count: A Forecast to 2028 

**Title (ZH)**: 前沿AI模型数量的发展趋势：至2028年的预测 

**Authors**: Iyngkarran Kumar, Sam Manning  

**Link**: [PDF](https://arxiv.org/pdf/2504.16138)  

**Abstract**: Governments are starting to impose requirements on AI models based on how much compute was used to train them. For example, the EU AI Act imposes requirements on providers of general-purpose AI with systemic risk, which includes systems trained using greater than $10^{25}$ floating point operations (FLOP). In the United States' AI Diffusion Framework, a training compute threshold of $10^{26}$ FLOP is used to identify "controlled models" which face a number of requirements. We explore how many models such training compute thresholds will capture over time. We estimate that by the end of 2028, there will be between 103-306 foundation models exceeding the $10^{25}$ FLOP threshold put forward in the EU AI Act (90% CI), and 45-148 models exceeding the $10^{26}$ FLOP threshold that defines controlled models in the AI Diffusion Framework (90% CI). We also find that the number of models exceeding these absolute compute thresholds each year will increase superlinearly -- that is, each successive year will see more new models captured within the threshold than the year before. Thresholds that are defined with respect to the largest training run to date (for example, such that all models within one order of magnitude of the largest training run to date are captured by the threshold) see a more stable trend, with a median forecast of 14-16 models being captured by this definition annually from 2025-2028. 

**Abstract (ZH)**: 政府开始基于训练时所使用的计算量对AI模型提出要求。例如，欧盟AI法案对具有系统性风险的一般目的AI提出了要求，其中包括使用超过$10^{25}$浮点运算（FLOP）训练的系统。在美国AI扩散框架中，使用$10^{26}$ FLOP的训练计算阈值来识别“受控模型”，这些模型需要满足一系列要求。我们探究了随着时间的推移，将达到这些训练计算阈值的模型数量。我们估计到2028年底，将达到欧盟AI法案提出$10^{25}$ FLOP阈值的103-306个基础模型（90%置信区间），将达到美国AI扩散框架中“受控模型”定义的$10^{26}$ FLOP阈值的45-148个模型（90%置信区间）。我们也发现，每年超过这些绝对计算阈值的模型数量将呈现超线性增长——即每年捕获到阈值内的新模型数量将超过前一年。以迄今为止最大的训练运行为基础定义的阈值（例如，所有在迄今为止最大训练运行数量的一个数量级范围内的模型都被捕获到）呈现出更稳定的趋势，从2025年到2028年，每年平均预测将有14-16个模型被这种定义捕获。 

---
# A Conceptual Framework for AI-based Decision Systems in Critical Infrastructures 

**Title (ZH)**: 基于人工智能的决策系统在关键基础设施中的概念框架 

**Authors**: Milad Leyli-abadi, Ricardo J. Bessa, Jan Viebahn, Daniel Boos, Clark Borst, Alberto Castagna, Ricardo Chavarriaga, Mohamed Hassouna, Bruno Lemetayer, Giulia Leto, Antoine Marot, Maroua Meddeb, Manuel Meyer, Viola Schiaffonati, Manuel Schneider, Toni Waefler  

**Link**: [PDF](https://arxiv.org/pdf/2504.16133)  

**Abstract**: The interaction between humans and AI in safety-critical systems presents a unique set of challenges that remain partially addressed by existing frameworks. These challenges stem from the complex interplay of requirements for transparency, trust, and explainability, coupled with the necessity for robust and safe decision-making. A framework that holistically integrates human and AI capabilities while addressing these concerns is notably required, bridging the critical gaps in designing, deploying, and maintaining safe and effective systems. This paper proposes a holistic conceptual framework for critical infrastructures by adopting an interdisciplinary approach. It integrates traditionally distinct fields such as mathematics, decision theory, computer science, philosophy, psychology, and cognitive engineering and draws on specialized engineering domains, particularly energy, mobility, and aeronautics. The flexibility in its adoption is also demonstrated through its instantiation on an already existing framework. 

**Abstract (ZH)**: 人类与AI在安全关键系统中的交互提出了一个独特的挑战集，现有框架在部分方面仍然未能有效应对。这些挑战源于透明性、信任和可解释性要求与稳健安全决策需求之间的复杂交互。一个能全面整合人类和AI能力并解决这些问题的框架尤为必要，以弥合设计、部署和维护安全有效系统的关键缺口。本文通过多学科方法提出了一种全方位的概念框架，将传统上独立的数学、决策理论、计算机科学、哲学、心理学和认知工程等领域进行综合，并借鉴了能源、移动性、航空等专门工程领域。同时，通过将其应用于一个现有的框架，展示了其采用的灵活性。 

---
# Efficacy of a Computer Tutor that Models Expert Human Tutors 

**Title (ZH)**: 基于专家人类导师模型的计算机导师的效果研究 

**Authors**: Andrew M. Olney, Sidney K. D'Mello, Natalie Person, Whitney Cade, Patrick Hays, Claire W. Dempsey, Blair Lehman, Betsy Williams, Art Graesser  

**Link**: [PDF](https://arxiv.org/pdf/2504.16132)  

**Abstract**: Tutoring is highly effective for promoting learning. However, the contribution of expertise to tutoring effectiveness is unclear and continues to be debated. We conducted a 9-week learning efficacy study of an intelligent tutoring system (ITS) for biology modeled on expert human tutors with two control conditions: human tutors who were experts in the domain but not in tutoring and a no-tutoring condition. All conditions were supplemental to classroom instruction, and students took learning tests immediately before and after tutoring sessions as well as delayed tests 1-2 weeks later. Analysis using logistic mixed-effects modeling indicates significant positive effects on the immediate post-test for the ITS (d =.71) and human tutors (d =.66) which are in the 99th percentile of meta-analytic effects, as well as significant positive effects on the delayed post-test for the ITS (d =.36) and human tutors (d =.39). We discuss implications for the role of expertise in tutoring and the design of future studies. 

**Abstract (ZH)**: 智能辅导系统在促进学习中的有效性研究：基于专家的人工智能辅导和控制条件下的专家人工辅导对比分析及其对未来研究设计的影响 

---
# Introduction to Quantum Machine Learning and Quantum Architecture Search 

**Title (ZH)**: 量子机器学习与量子架构搜索导论 

**Authors**: Samuel Yen-Chi Chen, Zhiding Liang  

**Link**: [PDF](https://arxiv.org/pdf/2504.16131)  

**Abstract**: Recent advancements in quantum computing (QC) and machine learning (ML) have fueled significant research efforts aimed at integrating these two transformative technologies. Quantum machine learning (QML), an emerging interdisciplinary field, leverages quantum principles to enhance the performance of ML algorithms. Concurrently, the exploration of systematic and automated approaches for designing high-performance quantum circuit architectures for QML tasks has gained prominence, as these methods empower researchers outside the quantum computing domain to effectively utilize quantum-enhanced tools. This tutorial will provide an in-depth overview of recent breakthroughs in both areas, highlighting their potential to expand the application landscape of QML across diverse fields. 

**Abstract (ZH)**: 最近在量子计算和机器学习领域的进展推动了将这两种变革性技术整合的研究努力。量子机器学习作为一种新兴的跨学科领域，利用量子原理来提高机器学习算法的性能。同时，系统化和自动化设计高性能量子电路架构以用于量子机器学习任务的方法也逐渐受到关注，这些方法使非量子计算领域的研究者能够有效利用量子增强的工具。本教程将提供这两个领域的最新突破的详细概述，并强调它们在扩展量子机器学习在多个领域的应用前景。 

---
# A Self-supervised Learning Method for Raman Spectroscopy based on Masked Autoencoders 

**Title (ZH)**: 基于遮蔽自动编码器的自监督学习方法用于拉曼光谱分析 

**Authors**: Pengju Ren, Ri-gui Zhou, Yaochong Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.16130)  

**Abstract**: Raman spectroscopy serves as a powerful and reliable tool for analyzing the chemical information of substances. The integration of Raman spectroscopy with deep learning methods enables rapid qualitative and quantitative analysis of materials. Most existing approaches adopt supervised learning methods. Although supervised learning has achieved satisfactory accuracy in spectral analysis, it is still constrained by costly and limited well-annotated spectral datasets for training. When spectral annotation is challenging or the amount of annotated data is insufficient, the performance of supervised learning in spectral material identification declines. In order to address the challenge of feature extraction from unannotated spectra, we propose a self-supervised learning paradigm for Raman Spectroscopy based on a Masked AutoEncoder, termed SMAE. SMAE does not require any spectral annotations during pre-training. By randomly masking and then reconstructing the spectral information, the model learns essential spectral features. The reconstructed spectra exhibit certain denoising properties, improving the signal-to-noise ratio (SNR) by more than twofold. Utilizing the network weights obtained from masked pre-training, SMAE achieves clustering accuracy of over 80% for 30 classes of isolated bacteria in a pathogenic bacterial dataset, demonstrating significant improvements compared to classical unsupervised methods and other state-of-the-art deep clustering methods. After fine-tuning the network with a limited amount of annotated data, SMAE achieves an identification accuracy of 83.90% on the test set, presenting competitive performance against the supervised ResNet (83.40%). 

**Abstract (ZH)**: 基于_MASKED AUTOENCODER_的自我监督学习在拉曼光谱中的应用 

---
# SOTOPIA-S4: a user-friendly system for flexible, customizable, and large-scale social simulation 

**Title (ZH)**: SOTOPIA-S4：一种用户友好的灵活、可定制和大规模社会仿真系统 

**Authors**: Xuhui Zhou, Zhe Su, Sophie Feng, Jiaxu Zhou, Jen-tse Huang, Hsien-Te Kao, Spencer Lynch, Svitlana Volkova, Tongshuang Sherry Wu, Anita Woolley, Hao Zhu, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2504.16122)  

**Abstract**: Social simulation through large language model (LLM) agents is a promising approach to explore and validate hypotheses related to social science questions and LLM agents behavior. We present SOTOPIA-S4, a fast, flexible, and scalable social simulation system that addresses the technical barriers of current frameworks while enabling practitioners to generate multi-turn and multi-party LLM-based interactions with customizable evaluation metrics for hypothesis testing. SOTOPIA-S4 comes as a pip package that contains a simulation engine, an API server with flexible RESTful APIs for simulation management, and a web interface that enables both technical and non-technical users to design, run, and analyze simulations without programming. We demonstrate the usefulness of SOTOPIA-S4 with two use cases involving dyadic hiring negotiation and multi-party planning scenarios. 

**Abstract (ZH)**: 通过大型语言模型（LLM）代理的社会仿真是一种探索和验证与社会科学问题和LLM代理行为相关假设的有前途的方法。我们提出了SOTOPIA-S4，一个快速、灵活、可扩展的社会仿真系统，该系统解决了当前框架的技术障碍，同时使 practitioners 能够生成多轮和多参与者的基于LLM的交互，并使用可定制的评估指标进行假设检验。SOTOPIA-S4 作为一个 pip 包提供，包含仿真引擎、具有灵活 RESTful API 的 API 服务器以及使技术用户和非技术用户无需编程即可设计、运行和分析仿真功能的 Web 界面。我们通过涉及二元招聘谈判和多参与规划场景的两个用例展示了 SOTOPIA-S4 的实用性。 

---
# Towards Explainable and Lightweight AI for Real-Time Cyber Threat Hunting in Edge Networks 

**Title (ZH)**: 面向边缘网络实时网络威胁狩猎的可解释和轻量级AI方法 

**Authors**: Milad Rahmati  

**Link**: [PDF](https://arxiv.org/pdf/2504.16118)  

**Abstract**: As cyber threats continue to evolve, securing edge networks has become increasingly challenging due to their distributed nature and resource limitations. Many AI-driven threat detection systems rely on complex deep learning models, which, despite their high accuracy, suffer from two major drawbacks: lack of interpretability and high computational cost. Black-box AI models make it difficult for security analysts to understand the reasoning behind their predictions, limiting their practical deployment. Moreover, conventional deep learning techniques demand significant computational resources, rendering them unsuitable for edge devices with limited processing power. To address these issues, this study introduces an Explainable and Lightweight AI (ELAI) framework designed for real-time cyber threat detection in edge networks. Our approach integrates interpretable machine learning algorithms with optimized lightweight deep learning techniques, ensuring both transparency and computational efficiency. The proposed system leverages decision trees, attention-based deep learning, and federated learning to enhance detection accuracy while maintaining explainability. We evaluate ELAI using benchmark cybersecurity datasets, such as CICIDS and UNSW-NB15, assessing its performance across diverse cyberattack scenarios. Experimental results demonstrate that the proposed framework achieves high detection rates with minimal false positives, all while significantly reducing computational demands compared to traditional deep learning methods. The key contributions of this work include: (1) a novel interpretable AI-based cybersecurity model tailored for edge computing environments, (2) an optimized lightweight deep learning approach for real-time cyber threat detection, and (3) a comprehensive analysis of explainability techniques in AI-driven cybersecurity applications. 

**Abstract (ZH)**: 随着网络威胁不断演变，由于其分布式特性和资源限制，确保边缘网络的安全变得越来越具挑战性。许多基于AI的威胁检测系统依赖于复杂的深度学习模型，尽管这些模型具有高度的准确性，但也存在两大主要缺点：缺乏可解释性和高计算成本。黑盒AI模型使得安全分析师难以理解其预测背后的推理，从而限制了其实用部署。此外，传统的深度学习技术需要大量的计算资源，这对于处理能力有限的边缘设备来说是不合适的。为了解决这些问题，本研究提出了一种适用于边缘网络实时网络威胁检测的可解释轻量级AI（ELAI）框架。我们的方法结合了可解释的机器学习算法和优化的轻量级深度学习技术，确保了透明性和计算效率。所提出系统利用决策树、基于注意力的深度学习和联邦学习来提高检测准确性的同时保持可解释性。我们使用CICIDS和UNSW-NB15等基准网络安全数据集评估了ELAI，评估了其在不同网络攻击场景下的性能。实验结果表明，与传统的深度学习方法相比，所提出框架能够以较低的计算需求实现高检测率，同时将误报率降至最低。本项研究的主要贡献包括：（1）一种针对边缘计算环境的新型可解释AI网络安全模型；（2）用于实时网络威胁检测的优化轻量级深度学习方法；（3）对AI驱动网络安全应用中可解释性技术的全面分析。 

---
# Context-Awareness and Interpretability of Rare Occurrences for Discovery and Formalization of Critical Failure Modes 

**Title (ZH)**: 基于上下文感知和少见发生事件解释性的关键失败模式发现与形式化 

**Authors**: Sridevi Polavaram, Xin Zhou, Meenu Ravi, Mohammad Zarei, Anmol Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2504.16117)  

**Abstract**: Vision systems are increasingly deployed in critical domains such as surveillance, law enforcement, and transportation. However, their vulnerabilities to rare or unforeseen scenarios pose significant safety risks. To address these challenges, we introduce Context-Awareness and Interpretability of Rare Occurrences (CAIRO), an ontology-based human-assistive discovery framework for failure cases (or CP - Critical Phenomena) detection and formalization. CAIRO by design incentivizes human-in-the-loop for testing and evaluation of criticality that arises from misdetections, adversarial attacks, and hallucinations in AI black-box models. Our robust analysis of object detection model(s) failures in automated driving systems (ADS) showcases scalable and interpretable ways of formalizing the observed gaps between camera perception and real-world contexts, resulting in test cases stored as explicit knowledge graphs (in OWL/XML format) amenable for sharing, downstream analysis, logical reasoning, and accountability. 

**Abstract (ZH)**: 基于上下文意识和稀有事件可解释性的失败案例（或CP-关键现象）检测与 formalization 框架（CAIRO） 

---
# AI-Based Vulnerability Analysis of NFT Smart Contracts 

**Title (ZH)**: 基于AI的NFT智能合约漏洞分析 

**Authors**: Xin Wang, Xiaoqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.16113)  

**Abstract**: In the research experiment of this article, our research work is divided into several stages. Firstly, we collected a large number of smart contract codes and classified them, identifying several common defects, including Risky Mutably Porxy, ERC-721 Recentrancy, Unlimited Mining, Missing Requirements, and Public Burns. Secondly, we used Python to process the smart contracts. On the one hand, we modified the file names, and on the other hand, we batched the process of the content for analysis and application. Next, we built a model of the decision tree. Firstly, we carried out the feature extraction. We selected the algorithm and divided the data. After comparing and processing, we chose the CART classification tree to process. By gene coefficient, we analyzed and sorted the data, and got the initial model of the decision tree. Then, we introduced the random forest model on the basis of the decision tree. From abstracting the same amount of samples to selecting features this http URL adjusting and optimizing parameters to completing the construction of the forest model. Finally, we compared and analyzed the decision tree, random forest, and self-built model in the paper and drew general conclusions. 

**Abstract (ZH)**: 在本文的研究实验中，我们的研究工作分为几个阶段。首先，我们收集了大量的智能合约代码并进行分类，识别出几种常见的缺陷，包括Risky Mutably Proxy、ERC-721 Reentrancy、Unlimited Mining、Missing Requirements和Public Burns。其次，我们使用Python处理智能合约。一方面，我们修改了文件名；另一方面，我们批量处理内容进行分析和应用。接着，我们构建了决策树模型。首先，我们进行了特征提取，选择了算法并划分了数据。在比较和处理后，我们选择了CART分类树进行处理。通过基因系数分析和排序数据，获得了决策树的初步模型。然后，我们在决策树的基础上引入了随机森林模型。从提取相同数量的样本到选择特征，再到调整和优化参数，最终完成森林模型的构建。最后，我们在文章中比较和分析了决策树、随机森林和自建模型，并得出了总体结论。 

---
# Security-First AI: Foundations for Robust and Trustworthy Systems 

**Title (ZH)**: 安全优先的人工智能：坚实可靠和值得信赖系统的基础 

**Authors**: Krti Tallam  

**Link**: [PDF](https://arxiv.org/pdf/2504.16110)  

**Abstract**: The conversation around artificial intelligence (AI) often focuses on safety, transparency, accountability, alignment, and responsibility. However, AI security (i.e., the safeguarding of data, models, and pipelines from adversarial manipulation) underpins all of these efforts. This manuscript posits that AI security must be prioritized as a foundational layer. We present a hierarchical view of AI challenges, distinguishing security from safety, and argue for a security-first approach to enable trustworthy and resilient AI systems. We discuss core threat models, key attack vectors, and emerging defense mechanisms, concluding that a metric-driven approach to AI security is essential for robust AI safety, transparency, and accountability. 

**Abstract (ZH)**: 关于人工智能（AI）的讨论常常聚焦于安全、透明性、问责制、契合度和责任。然而，AI安全（即保护数据、模型和管道免受敌对操纵）是这一切努力的基础。本文认为，AI安全必须作为基础层得到优先考虑。我们提出了一个分层的AI挑战视图，区分安全与安全，主张采取以安全为主的方法以实现可信赖和健壮的AI系统。我们讨论了核心威胁模型、关键攻击向量和新兴防御机制，认为以度量驱动的方法对于实现稳健的AI安全、透明性和问责制是必不可少的。 

---
# xLSTM-ECG: Multi-label ECG Classification via Feature Fusion with xLSTM 

**Title (ZH)**: xLSTM-ECG：基于xLSTM的特征融合多标签心电图分类 

**Authors**: Lei Kang, Xuanshuo Fu, Javier Vazquez-Corral, Ernest Valveny, Dimosthenis Karatzas  

**Link**: [PDF](https://arxiv.org/pdf/2504.16101)  

**Abstract**: Cardiovascular diseases (CVDs) remain the leading cause of mortality worldwide, highlighting the critical need for efficient and accurate diagnostic tools. Electrocardiograms (ECGs) are indispensable in diagnosing various heart conditions; however, their manual interpretation is time-consuming and error-prone. In this paper, we propose xLSTM-ECG, a novel approach that leverages an extended Long Short-Term Memory (xLSTM) network for multi-label classification of ECG signals, using the PTB-XL dataset. To the best of our knowledge, this work represents the first design and application of xLSTM modules specifically adapted for multi-label ECG classification. Our method employs a Short-Time Fourier Transform (STFT) to convert time-series ECG waveforms into the frequency domain, thereby enhancing feature extraction. The xLSTM architecture is specifically tailored to address the complexities of 12-lead ECG recordings by capturing both local and global signal features. Comprehensive experiments on the PTB-XL dataset reveal that our model achieves strong multi-label classification performance, while additional tests on the Georgia 12-Lead dataset underscore its robustness and efficiency. This approach significantly improves ECG classification accuracy, thereby advancing clinical diagnostics and patient care. The code will be publicly available upon acceptance. 

**Abstract (ZH)**: 心血管疾病(CVDs)仍然是全球死亡的主要原因，强调了高效准确诊断工具的迫切需求。心电图(ECGs)在诊断各种心脏状况中不可或缺；然而，其手动解读耗时且易出错。本文提出了一种新的方法xLSTM-ECG，该方法利用扩展长短期记忆(xLSTM)网络进行心电图信号的多标签分类，并使用PTB-XL数据集。据我们所知，这是首次设计和应用专门适应多标签ECG分类的xLSTM模块。该方法采用短时傅里叶变换(STFT)将时间序列心电图波形转换到频域，从而增强特征提取。xLSTM架构特别针对12导联心电图记录的复杂性进行了优化，以捕获局部和全局信号特征。在PTB-XL数据集上的全面实验表明，我们的模型在多标签分类性能上表现出色，而额外的Georgia 12导联数据集测试进一步证明了其稳健性和效率。该方法显著提高了心电图分类准确性，从而推动临床诊断和患者护理的进步。代码将在接受后公开。 

---
# Towards Accurate Forecasting of Renewable Energy : Building Datasets and Benchmarking Machine Learning Models for Solar and Wind Power in France 

**Title (ZH)**: 面向可再生能源准确预测的研究：构建数据集和评估机器学习模型在法国太阳能和风能电力上的标杆研究 

**Authors**: Eloi Lindas, Yannig Goude, Philippe Ciais  

**Link**: [PDF](https://arxiv.org/pdf/2504.16100)  

**Abstract**: Accurate prediction of non-dispatchable renewable energy sources is essential for grid stability and price prediction. Regional power supply forecasts are usually indirect through a bottom-up approach of plant-level forecasts, incorporate lagged power values, and do not use the potential of spatially resolved data. This study presents a comprehensive methodology for predicting solar and wind power production at country scale in France using machine learning models trained with spatially explicit weather data combined with spatial information about production sites capacity. A dataset is built spanning from 2012 to 2023, using daily power production data from RTE (the national grid operator) as the target variable, with daily weather data from ERA5, production sites capacity and location, and electricity prices as input features. Three modeling approaches are explored to handle spatially resolved weather data: spatial averaging over the country, dimension reduction through principal component analysis, and a computer vision architecture to exploit complex spatial relationships. The study benchmarks state-of-the-art machine learning models as well as hyperparameter tuning approaches based on cross-validation methods on daily power production data. Results indicate that cross-validation tailored to time series is best suited to reach low error. We found that neural networks tend to outperform traditional tree-based models, which face challenges in extrapolation due to the increasing renewable capacity over time. Model performance ranges from 4% to 10% in nRMSE for midterm horizon, achieving similar error metrics to local models established at a single-plant level, highlighting the potential of these methods for regional power supply forecasting. 

**Abstract (ZH)**: 准确预测不可调度的可再生能源对于电网稳定性和价格预测至关重要。基于下自上的机组级预测方法，区域电力供应预测通常间接进行，并未充分利用空间解析数据的潜力。本研究提出了一种全面的方法，通过使用结合生产站点空间信息的空间显式气象数据训练的机器学习模型，在法国尺度上预测太阳能和风能生产。研究构建了一个从2012年到2023年的数据集，以法国国家电网运营商RTE的日电力生产数据为目标变量，输入特征包括ERA5的日气象数据、生产站点的产能和位置以及电价。研究探索了三种处理空间解析气象数据的方法：国家尺度的空间平均、主成分分析降维以及利用复杂空间关系的计算机视觉架构。这项研究在每日电力生产数据上基准了先进的机器学习模型及基于交叉验证方法的超参数调优方法。结果表明，针对时间序列进行定制的交叉验证最适合达到较低的误差。研究发现，神经网络倾向于优于传统的树基模型，后者由于可再生能源产能的增加而面临外推难题。在中期展望下，模型性能范围为4%至10%的nRMSE，表明这些方法在区域电力供应预测中的潜力。 

---
# Two-Timescale Joint Transmit and Pinching Beamforming for Pinching-Antenna Systems 

**Title (ZH)**: 两时标联合传输与压缩波束形成算法for压缩天线系统 

**Authors**: Luyuan Zhang, Xidong Mu, An Liu, Yuanwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.16099)  

**Abstract**: Pinching antenna systems (PASS) have been proposed as a revolutionary flexible antenna technology which facilitates line-of-sight links via numerous low-cost pinching antennas with adjustable activation positions over waveguides. This letter proposes a two-timescale joint transmit and pinching beamforming design for the maximization of sum rate of a PASS-based downlink multi-user multiple input single output system. A primal dual decomposition method is developed to decouple the two-timescale problem into two sub-problems: 1) A Karush-Kuhn-Tucker-guided dual learning-based approach is proposed to solve the short-term transmit beamforming design sub-problem; 2) The long-term pinching beamforming design sub-problem is tackled by adopting a stochastic successive convex approximation method. Simulation results demonstrate that the proposed two-timescale algorithm achieves a significant performance gain compared to other baselines. 

**Abstract (ZH)**: 基于PASS的下行多用户单输入多输出系统两时标联合传输与针状波束形成设计 

---
# Efficient Portfolio Selection through Preference Aggregation with Quicksort and the Bradley--Terry Model 

**Title (ZH)**: 基于快速排序和布雷德利-泰尔模型的偏好聚合的有效投资组合选择 

**Authors**: Yurun Ge, Lucas Böttcher, Tom Chou, Maria R. D'Orsogna  

**Link**: [PDF](https://arxiv.org/pdf/2504.16093)  

**Abstract**: How to allocate limited resources to projects that will yield the greatest long-term benefits is a problem that often arises in decision-making under uncertainty. For example, organizations may need to evaluate and select innovation projects with risky returns. Similarly, when allocating resources to research projects, funding agencies are tasked with identifying the most promising proposals based on idiosyncratic criteria. Finally, in participatory budgeting, a local community may need to select a subset of public projects to fund. Regardless of context, agents must estimate the uncertain values of a potentially large number of projects. Developing parsimonious methods to compare these projects, and aggregating agent evaluations so that the overall benefit is maximized, are critical in assembling the best project portfolio. Unlike in standard sorting algorithms, evaluating projects on the basis of uncertain long-term benefits introduces additional complexities. We propose comparison rules based on Quicksort and the Bradley--Terry model, which connects rankings to pairwise "win" probabilities. In our model, each agent determines win probabilities of a pair of projects based on his or her specific evaluation of the projects' long-term benefit. The win probabilities are then appropriately aggregated and used to rank projects. Several of the methods we propose perform better than the two most effective aggregation methods currently available. Additionally, our methods can be combined with sampling techniques to significantly reduce the number of pairwise comparisons. We also discuss how the Bradley--Terry portfolio selection approach can be implemented in practice. 

**Abstract (ZH)**: 如何将有限资源分配给预期能带来最大长期效益的项目是一个在不确定性条件下做决策时常遇到的问题。例如，组织可能需要评估和选择具有风险回报的研发项目。同样，在为研究项目分配资源时，资助机构需根据特定标准识别最有前景的提案。最后，在参与预算制中，当地社区可能需要选择一部分公共项目进行资助。无论在何种情境下，决策者都必须估算大量项目的不确定性价值。开发简洁的方法来比较这些项目，并综合决策者评估以最大化总体效益，对于构建最佳项目组合至关重要。与标准排序算法不同，基于不确定长期效益评估项目引入了额外的复杂性。我们提出了基于Quicksort和Bradley-Terry模型的比较规则，将排名与“胜负”概率联系起来。在我们的模型中，每个决策者根据对项目长期效益的特定评估来确定项目对之间的胜率。然后对这些胜率进行适当综合，并用于项目排序。我们提出的一些方法在综合效果方面优于目前最有效的两种方法。此外，我们的方法可以与采样技术结合使用，显著减少项目对的比较次数。我们还讨论了如何在实践中实现Bradley-Terry项目选择方法。 

---
