# Learning to Optimize Package Picking for Large-Scale, Real-World Robot Induction 

**Title (ZH)**: 学习优化大规模现实世界机器人装箱捡取 

**Authors**: Shuai Li, Azarakhsh Keipour, Sicong Zhao, Srinath Rajagopalan, Charles Swan, Kostas E. Bekris  

**Link**: [PDF](https://arxiv.org/pdf/2506.09765)  

**Abstract**: Warehouse automation plays a pivotal role in enhancing operational efficiency, minimizing costs, and improving resilience to workforce variability. While prior research has demonstrated the potential of machine learning (ML) models to increase picking success rates in large-scale robotic fleets by prioritizing high-probability picks and packages, these efforts primarily focused on predicting success probabilities for picks sampled using heuristic methods. Limited attention has been given, however, to leveraging data-driven approaches to directly optimize sampled picks for better performance at scale. In this study, we propose an ML-based framework that predicts transform adjustments as well as improving the selection of suction cups for multi-suction end effectors for sampled picks to enhance their success probabilities. The framework was integrated and evaluated in test workcells that resemble the operations of Amazon Robotics' Robot Induction (Robin) fleet, which is used for package manipulation. Evaluated on over 2 million picks, the proposed method achieves a 20\% reduction in pick failure rates compared to a heuristic-based pick sampling baseline, demonstrating its effectiveness in large-scale warehouse automation scenarios. 

**Abstract (ZH)**: 仓库自动化在提高运营效率、降低成本和增强对劳动力变动的抵御能力中发挥着关键作用。尽管先前研究已经表明，机器学习模型可以通过优先处理高概率 pick 和包裹来提高大型机器人队列的拣选成功率，但这些努力主要集中在使用启发式方法采样的拣选的成功概率预测上。然而，利用数据驱动的方法直接优化采样的拣选以在大规模场景中获得更好表现的关注较少。在本研究中，我们提出了一种基于机器学习的框架，该框架预测变换调整并改进多吸盘末端执行器的吸盘选择，以提高采样拣选的成功概率。该框架在类似亚马逊机器人Robot Induction（Robin）队列操作的测试工位中进行集成和评估，用于包裹操作。在超过200万次拣选的评估中，所提出的方法将拣选失败率降低了20%，证明了其在大规模仓库自动化场景中的有效性。 

---
# Adv-BMT: Bidirectional Motion Transformer for Safety-Critical Traffic Scenario Generation 

**Title (ZH)**: Adv-BMT: 双向运动变换器用于安全关键交通场景生成 

**Authors**: Yuxin Liu, Zhenghao Peng, Xuanhao Cui, Bolei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.09485)  

**Abstract**: Scenario-based testing is essential for validating the performance of autonomous driving (AD) systems. However, such testing is limited by the scarcity of long-tailed, safety-critical scenarios in existing datasets collected in the real world. To tackle the data issue, we propose the Adv-BMT framework, which augments real-world scenarios with diverse and realistic adversarial interactions. The core component of Adv-BMT is a bidirectional motion transformer (BMT) model to perform inverse traffic motion predictions, which takes agent information in the last time step of the scenario as input, and reconstruct the traffic in the inverse of chronological order until the initial time step. The Adv-BMT framework is a two-staged pipeline: it first conducts adversarial initializations and then inverse motion predictions. Different from previous work, we do not need any collision data for pretraining, and are able to generate realistic and diverse collision interactions. Our experimental results validate the quality of generated collision scenarios by Adv-BMT: training in our augmented dataset would reduce episode collision rates by 20\% compared to previous work. 

**Abstract (ZH)**: 基于场景的测试对于验证自动驾驶（AD）系统的性能至关重要。然而，现有的现实世界收集的数据集中缺乏长尾的安全关键场景，限制了此类测试。为解决数据问题，我们提出了Adv-BMT框架，该框架通过引入多样且现实的对抗性相互作用来扩展现实世界的场景。Adv-BMT的核心组件是一个双向运动变换器（BMT）模型，该模型用于执行逆向交通运动预测，它以场景中最后一个时间步的代理信息为输入，并以逆时间顺序重建交通，直到初始时间步。Adv-BMT框架是一个两阶段的管道：首先进行对抗性初始化，然后进行逆运动预测。与以往工作不同，我们不需要任何碰撞数据进行预训练，并且能够生成真实的多样化的碰撞交互。实验结果验证了Adv-BMT生成的碰撞场景的质量：在我们扩充的数据集中进行训练与以往工作相比，能使场景中的碰撞率降低20%。 

---
# Towards Full-Scenario Safety Evaluation of Automated Vehicles: A Volume-Based Method 

**Title (ZH)**: 面向自动化车辆全场景安全评估的体积基方法 

**Authors**: Hang Zhou, Chengyuan Ma, Shiyu Shen, Xiaopeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09182)  

**Abstract**: With the rapid development of automated vehicles (AVs) in recent years, commercially available AVs are increasingly demonstrating high-level automation capabilities. However, most existing AV safety evaluation methods are primarily designed for simple maneuvers such as car-following and lane-changing. While suitable for basic tests, these methods are insufficient for assessing high-level automation functions deployed in more complex environments. First, these methods typically use crash rate as the evaluation metric, whose accuracy heavily depends on the quality and completeness of naturalistic driving environment data used to estimate scenario probabilities. Such data is often difficult and expensive to collect. Second, when applied to diverse scenarios, these methods suffer from the curse of dimensionality, making large-scale evaluation computationally intractable. To address these challenges, this paper proposes a novel framework for full-scenario AV safety evaluation. A unified model is first introduced to standardize the representation of diverse driving scenarios. This modeling approach constrains the dimension of most scenarios to a regular highway setting with three lanes and six surrounding background vehicles, significantly reducing dimensionality. To further avoid the limitations of probability-based method, we propose a volume-based evaluation method that quantifies the proportion of risky scenarios within the entire scenario space. For car-following scenarios, we prove that the set of safe scenarios is convex under specific settings, enabling exact volume computation. Experimental results validate the effectiveness of the proposed volume-based method using both AV behavior models from existing literature and six production AV models calibrated from field-test trajectory data in the Ultra-AV dataset. Code and data will be made publicly available upon acceptance of this paper. 

**Abstract (ZH)**: 近年来，随着自动驾驶车辆（AVs）的快速发展，商用AVs日益展现出高层次的自动化能力。然而，目前大多数现有的AV安全评估方法主要针对如跟车和变道等简单的操作。虽然适用于基本测试，但这些方法对部署在更复杂环境中的高层次自动化功能评估不足。首先，这些方法通常使用碰撞率作为评估指标，其准确性高度依赖于用于估计场景概率的自然驾驶环境数据的质量和完整性，而这样的数据收集起来往往既困难又昂贵。其次，当应用于多种场景时，这些方法会遭受维度灾难的问题，使得大规模评估在计算上变得不可行。为了解决这些挑战，本文提出了一种新的全场景AV安全评估框架。首先引入了一个统一模型来标准化各类驾驶场景的表示。通过这种方法，可以将大多数场景的维度限制在一个标准的三车道高速公路设置下，显著降低了维度。为了进一步避免基于概率方法的局限性，本文提出了基于体素的评估方法，该方法量化整个场景空间中具有风险的场景的比例。对于跟车场景，我们证明在特定设置下安全场景集合是凸的，能够实现精确的体积计算。实验结果使用文献中现有的自动驾驶车辆行为模型和 Ultra-AV 数据集中现场测试轨迹数据校准的六款生产中自动驾驶车辆模型验证了所提出的方法的有效性。代码和数据将在本文被接受后公开发布。 

---
# How attention simplifies mental representations for planning 

**Title (ZH)**: 注意力如何简化 planning 中的心理表征 

**Authors**: Jason da Silva Castanheira, Nicholas Shea, Stephen M. Fleming  

**Link**: [PDF](https://arxiv.org/pdf/2506.09520)  

**Abstract**: Human planning is efficient -- it frugally deploys limited cognitive resources to accomplish difficult tasks -- and flexible -- adapting to novel problems and environments. Computational approaches suggest that people construct simplified mental representations of their environment, balancing the complexity of a task representation with its utility. These models imply a nested optimisation in which planning shapes perception, and perception shapes planning -- but the perceptual and attentional mechanisms governing how this interaction unfolds remain unknown. Here, we harness virtual maze navigation to characterise how spatial attention controls which aspects of a task representation enter subjective awareness and are available for planning. We find that spatial proximity governs which aspects of a maze are available for planning, and that when task-relevant information follows natural (lateralised) contours of attention, people can more easily construct simplified and useful maze representations. This influence of attention varies considerably across individuals, explaining differences in people's task representations and behaviour. Inspired by the 'spotlight of attention' analogy, we incorporate the effects of visuospatial attention into existing computational accounts of value-guided construal. Together, our work bridges computational perspectives on perception and decision-making to better understand how individuals represent their environments in aid of planning. 

**Abstract (ZH)**: 人类规划既高效又灵活——它精打细算地使用有限的认知资源来完成复杂的任务，并且能够适应新的问题和环境。计算方法表明人们构建了环境的简化心理表征，平衡任务表征的复杂性和实用性。这些模型暗示了一种嵌套的优化过程，在此过程中规划影响感知，而感知又影响规划——但调控这一交互过程的感知和注意力机制仍然未知。在这里，我们利用虚拟迷宫导航来研究空间注意力如何控制哪些任务表征进入主观意识并可供规划使用。我们发现，空间接近度决定了哪些迷宫方面可供规划使用，当与任务相关信息遵循自然（侧向化）的注意力轮廓时，人们更容易构建简化且有用的迷宫表征。这种注意力的影响在个体间差异很大，解释了人们之间任务表征和行为的差异。借鉴“注意力的探照灯”类比，我们将注意力的空间视觉效应纳入现有价值导向构念的计算解释中。我们的研究将关于感知和决策的计算视角结合起来，以更好地理解个体如何代表其环境以助于规划。 

---
# How Do People Revise Inconsistent Beliefs? Examining Belief Revision in Humans with User Studies 

**Title (ZH)**: 人们如何修订不一致的信念？通过用户研究考察人类信念修订過程 

**Authors**: Stylianos Loukas Vasileiou, Antonio Rago, Maria Vanina Martinez, William Yeoh  

**Link**: [PDF](https://arxiv.org/pdf/2506.09977)  

**Abstract**: Understanding how humans revise their beliefs in light of new information is crucial for developing AI systems which can effectively model, and thus align with, human reasoning. While theoretical belief revision frameworks rely on a set of principles that establish how these operations are performed, empirical evidence from cognitive psychology suggests that people may follow different patterns when presented with conflicting information. In this paper, we present three comprehensive user studies showing that people consistently prefer explanation-based revisions, i.e., those which are guided by explanations, that result in changes to their belief systems that are not necessarily captured by classical belief change theory. Our experiments systematically investigate how people revise their beliefs with explanations for inconsistencies, whether they are provided with them or left to formulate them themselves, demonstrating a robust preference for what may seem non-minimal revisions across different types of scenarios. These findings have implications for AI systems designed to model human reasoning or interact with humans, suggesting that such systems should accommodate explanation-based, potentially non-minimal belief revision operators to better align with human cognitive processes. 

**Abstract (ZH)**: 理解人类在新信息面前如何修订信念对于开发能够有效模拟和对齐人类推理的AI系统至关重要。虽然理论性的信念修订框架基于一套原则来确定这些操作的执行方式，但认知心理学的实证证据表明，当人们面临冲突信息时，可能会遵循不同的模式。在本文中，我们展示了三项全面的用户研究，表明人们一致偏好基于解释的修订，即那些受到解释指引、可能导致信念系统发生变化但不一定能被经典信念变更理论捕获的修订。我们的实验系统地探究了人们在不一致情况下如何根据解释修订信念，无论是由提供还是自行构想这些解释，展现了在不同场景类型中对似乎非最小化修订的稳健偏好。这些发现对旨在模拟人类推理或与人类交互的AI系统具有启示意义，建议此类系统应容纳基于解释的、可能非最小化的信念修订操作，以更好地与人类认知过程对齐。 

---
# Fast Monte Carlo Tree Diffusion: 100x Speedup via Parallel Sparse Planning 

**Title (ZH)**: 快速蒙特卡洛树扩散：通过并行稀疏规划实现100倍加速 

**Authors**: Jaesik Yoon, Hyeonseo Cho, Yoshua Bengio, Sungjin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2506.09498)  

**Abstract**: Diffusion models have recently emerged as a powerful approach for trajectory planning. However, their inherently non-sequential nature limits their effectiveness in long-horizon reasoning tasks at test time. The recently proposed Monte Carlo Tree Diffusion (MCTD) offers a promising solution by combining diffusion with tree-based search, achieving state-of-the-art performance on complex planning problems. Despite its strengths, our analysis shows that MCTD incurs substantial computational overhead due to the sequential nature of tree search and the cost of iterative denoising. To address this, we propose Fast-MCTD, a more efficient variant that preserves the strengths of MCTD while significantly improving its speed and scalability. Fast-MCTD integrates two techniques: Parallel MCTD, which enables parallel rollouts via delayed tree updates and redundancy-aware selection; and Sparse MCTD, which reduces rollout length through trajectory coarsening. Experiments show that Fast-MCTD achieves up to 100x speedup over standard MCTD while maintaining or improving planning performance. Remarkably, it even outperforms Diffuser in inference speed on some tasks, despite Diffuser requiring no search and yielding weaker solutions. These results position Fast-MCTD as a practical and scalable solution for diffusion-based inference-time reasoning. 

**Abstract (ZH)**: 基于扩散模型的快速蒙特卡洛树搜索（Fast-MCTD）：一种高效的轨迹规划方法 

---
# Comment on The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity 

**Title (ZH)**: 论思考的幻象：通过问题复杂性的视角理解推理模型的强弱之处 

**Authors**: C. Opus, A. Lawsen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09250)  

**Abstract**: Shojaee et al. (2025) report that Large Reasoning Models (LRMs) exhibit "accuracy collapse" on planning puzzles beyond certain complexity thresholds. We demonstrate that their findings primarily reflect experimental design limitations rather than fundamental reasoning failures. Our analysis reveals three critical issues: (1) Tower of Hanoi experiments systematically exceed model output token limits at reported failure points, with models explicitly acknowledging these constraints in their outputs; (2) The authors' automated evaluation framework fails to distinguish between reasoning failures and practical constraints, leading to misclassification of model capabilities; (3) Most concerningly, their River Crossing benchmarks include mathematically impossible instances for N > 5 due to insufficient boat capacity, yet models are scored as failures for not solving these unsolvable problems. When we control for these experimental artifacts, by requesting generating functions instead of exhaustive move lists, preliminary experiments across multiple models indicate high accuracy on Tower of Hanoi instances previously reported as complete failures. These findings highlight the importance of careful experimental design when evaluating AI reasoning capabilities. 

**Abstract (ZH)**: Shojaee等（2025）报告大型推理模型（LRMs）在超出一定复杂度阈值的规划谜题上表现出“准确性崩溃”。我们证明他们的发现主要反映了实验设计的局限性而非根本性的推理失败。我们的分析揭示了三个关键问题：（1）汉诺塔实验系统地在报告的失败点超过了模型输出的令牌限制，模型在输出中明确承认了这些约束；（2）作者的自动化评估框架无法区分推理失败和实际约束，导致对模型能力的误分类；（3）最令人担忧的是，他们的河流过河基准测试包括了对于N > 5来说数学上不可能的情况，由于小船容量不足，而模型因未能解决这些不可解的问题而被评分失败。当我们通过要求生成函数而非生成完整的移动列表来控制这些实验效应时，多个模型在之前报告为完全失败的汉诺塔实例上的初步实验显示出了高准确性。这些发现强调了评估人工智能推理能力时仔细实验设计的重要性。 

---
# EditInspector: A Benchmark for Evaluation of Text-Guided Image Edits 

**Title (ZH)**: 文本引导图像编辑评估基准：EditInspector 

**Authors**: Ron Yosef, Moran Yanuka, Yonatan Bitton, Dani Lischinski  

**Link**: [PDF](https://arxiv.org/pdf/2506.09988)  

**Abstract**: Text-guided image editing, fueled by recent advancements in generative AI, is becoming increasingly widespread. This trend highlights the need for a comprehensive framework to verify text-guided edits and assess their quality. To address this need, we introduce EditInspector, a novel benchmark for evaluation of text-guided image edits, based on human annotations collected using an extensive template for edit verification. We leverage EditInspector to evaluate the performance of state-of-the-art (SoTA) vision and language models in assessing edits across various dimensions, including accuracy, artifact detection, visual quality, seamless integration with the image scene, adherence to common sense, and the ability to describe edit-induced changes. Our findings indicate that current models struggle to evaluate edits comprehensively and frequently hallucinate when describing the changes. To address these challenges, we propose two novel methods that outperform SoTA models in both artifact detection and difference caption generation. 

**Abstract (ZH)**: 基于文本引导的图像编辑：随着生成AI的 recent 进展日益普及，需构建全面框架进行验证与评估 

---
# The Sample Complexity of Online Strategic Decision Making with Information Asymmetry and Knowledge Transportability 

**Title (ZH)**: 具有信息不对称和知识可转移性的在线战略决策的样本复杂性 

**Authors**: Jiachen Hu, Rui Ai, Han Zhong, Xiaoyu Chen, Liwei Wang, Zhaoran Wang, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09940)  

**Abstract**: Information asymmetry is a pervasive feature of multi-agent systems, especially evident in economics and social sciences. In these settings, agents tailor their actions based on private information to maximize their rewards. These strategic behaviors often introduce complexities due to confounding variables. Simultaneously, knowledge transportability poses another significant challenge, arising from the difficulties of conducting experiments in target environments. It requires transferring knowledge from environments where empirical data is more readily available. Against these backdrops, this paper explores a fundamental question in online learning: Can we employ non-i.i.d. actions to learn about confounders even when requiring knowledge transfer? We present a sample-efficient algorithm designed to accurately identify system dynamics under information asymmetry and to navigate the challenges of knowledge transfer effectively in reinforcement learning, framed within an online strategic interaction model. Our method provably achieves learning of an $\epsilon$-optimal policy with a tight sample complexity of $O(1/\epsilon^2)$. 

**Abstract (ZH)**: 信息不对称是多Agent系统中一个普遍特征，尤其是在经济学和社会科学领域中尤为明显。在这种环境中，代理基于私有信息调整其行为以最大化奖励。这些战略性行为常常由于混杂变量的引入而增加复杂性。同时，知识可转移性也提出了另一个重大挑战，源于在目标环境中进行实验的困难。这需要从更容易获得经验数据的环境中转移知识。在此背景下，本文探讨了在线学习中的一个基础问题：我们是否可以使用非独立同分布（non-i.i.d.）的动作来学习混杂变量，即使在这种情况下需要进行知识转移？我们提出了一种样本高效的算法，旨在在信息不对称条件下准确识别系统动力学，并有效地在强化学习框架内的在线战略交互模型中应对知识转移的挑战。我们的方法能够证明以紧致的样本复杂度$O(1/\epsilon^2)$学习$\epsilon$-最优策略。 

---
# HadaNorm: Diffusion Transformer Quantization through Mean-Centered Transformations 

**Title (ZH)**: HadaNorm：基于均值中心化变换的扩散变压器量化 

**Authors**: Marco Federici, Riccardo Del Chiaro, Boris van Breugel, Paul Whatmough, Markus Nagel  

**Link**: [PDF](https://arxiv.org/pdf/2506.09932)  

**Abstract**: Diffusion models represent the cutting edge in image generation, but their high memory and computational demands hinder deployment on resource-constrained devices. Post-Training Quantization (PTQ) offers a promising solution by reducing the bitwidth of matrix operations. However, standard PTQ methods struggle with outliers, and achieving higher compression often requires transforming model weights and activations before quantization. In this work, we propose HadaNorm, a novel linear transformation that extends existing approaches and effectively mitigates outliers by normalizing activations feature channels before applying Hadamard transformations, enabling more aggressive activation quantization. We demonstrate that HadaNorm consistently reduces quantization error across the various components of transformer blocks, achieving superior efficiency-performance trade-offs when compared to state-of-the-art methods. 

**Abstract (ZH)**: 基于HadaNorm的新型线性变换在变压器块中有效缓解异常值，实现更优的效率-性能 trade-offs 

---
# Causal Climate Emulation with Bayesian Filtering 

**Title (ZH)**: 基于贝叶斯滤波的因果气候模拟 

**Authors**: Sebastian Hickman, Ilija Trajkovic, Julia Kaltenborn, Francis Pelletier, Alex Archibald, Yaniv Gurwicz, Peer Nowack, David Rolnick, Julien Boussard  

**Link**: [PDF](https://arxiv.org/pdf/2506.09891)  

**Abstract**: Traditional models of climate change use complex systems of coupled equations to simulate physical processes across the Earth system. These simulations are highly computationally expensive, limiting our predictions of climate change and analyses of its causes and effects. Machine learning has the potential to quickly emulate data from climate models, but current approaches are not able to incorporate physics-informed causal relationships. Here, we develop an interpretable climate model emulator based on causal representation learning. We derive a physics-informed approach including a Bayesian filter for stable long-term autoregressive emulation. We demonstrate that our emulator learns accurate climate dynamics, and we show the importance of each one of its components on a realistic synthetic dataset and data from two widely deployed climate models. 

**Abstract (ZH)**: 基于因果表示学习的可解释气候模型仿真器：包含物理信息的稳定长期自回归仿真 

---
# Stakeholder Participation for Responsible AI Development: Disconnects Between Guidance and Current Practice 

**Title (ZH)**: 负责任人工智能开发中的利益相关者参与：指南与当前实践之间的差距 

**Authors**: Emma Kallina, Thomas Bohné, Jat Singh  

**Link**: [PDF](https://arxiv.org/pdf/2506.09873)  

**Abstract**: Responsible AI (rAI) guidance increasingly promotes stakeholder involvement (SHI) during AI development. At the same time, SHI is already common in commercial software development, but with potentially different foci. This study clarifies the extent to which established SHI practices are able to contribute to rAI efforts as well as potential disconnects -- essential insights to inform and tailor future interventions that further shift industry practice towards rAI efforts. First, we analysed 56 rAI guidance documents to identify why SHI is recommended (i.e. its expected benefits for rAI) and uncovered goals such as redistributing power, improving socio-technical understandings, anticipating risks, and enhancing public oversight. To understand why and how SHI is currently practised in commercial settings, we then conducted an online survey (n=130) and semi-structured interviews (n=10) with AI practitioners. Our findings reveal that SHI in practice is primarily driven by commercial priorities (e.g. customer value, compliance) and several factors currently discourage more rAI-aligned SHI practices. This suggests that established SHI practices are largely not contributing to rAI efforts. To address this disconnect, we propose interventions and research opportunities to advance rAI development in practice. 

**Abstract (ZH)**: 负责任人工智能（rAI）指导越来越强调在人工智能开发过程中增加利益相关者参与（SHI）。与此同时，利益相关者参与已经在商业软件开发中普遍存在，但可能侧重不同。本研究阐明了现有SHI实践在多大程度上能够促进rAI努力，以及潜在的脱节——这些见解对于指导和定制未来进一步推动行业实践向rAI努力的干预措施至关重要。首先，我们分析了56份rAI指导文档，以确定推荐SHI的原因（即其对rAI的预期益处），并发现了一些目标，如重新分配权力、改善社会技术理解、预见风险和增强公众监督。为了了解商业环境中当前的SHI实践为何以及如何进行，我们随后对130名AI从业者进行了在线调查，并对10名从业者进行了半结构化访谈。我们的研究发现，在实践中，SHI主要由商业优先事项（如客户价值、合规性）驱动，当前有多种因素阻碍了更符合rAI的SHI实践。这表明现有SHI实践并未大量贡献于rAI努力。为了解决这一脱节，我们提出了推进rAI开发的干预措施和研究机会。 

---
# Guided Graph Compression for Quantum Graph Neural Networks 

**Title (ZH)**: 引导图压缩用于量子图神经网络 

**Authors**: Mikel Casals, Vasilis Belis, Elias F. Combarro, Eduard Alarcón, Sofia Vallecorsa, Michele Grossi  

**Link**: [PDF](https://arxiv.org/pdf/2506.09862)  

**Abstract**: Graph Neural Networks (GNNs) are effective for processing graph-structured data but face challenges with large graphs due to high memory requirements and inefficient sparse matrix operations on GPUs. Quantum Computing (QC) offers a promising avenue to address these issues and inspires new algorithmic approaches. In particular, Quantum Graph Neural Networks (QGNNs) have been explored in recent literature. However, current quantum hardware limits the dimension of the data that can be effectively encoded. Existing approaches either simplify datasets manually or use artificial graph datasets. This work introduces the Guided Graph Compression (GGC) framework, which uses a graph autoencoder to reduce both the number of nodes and the dimensionality of node features. The compression is guided to enhance the performance of a downstream classification task, which can be applied either with a quantum or a classical classifier. The framework is evaluated on the Jet Tagging task, a classification problem of fundamental importance in high energy physics that involves distinguishing particle jets initiated by quarks from those by gluons. The GGC is compared against using the autoencoder as a standalone preprocessing step and against a baseline classical GNN classifier. Our numerical results demonstrate that GGC outperforms both alternatives, while also facilitating the testing of novel QGNN ansatzes on realistic datasets. 

**Abstract (ZH)**: Graph神经网络（GNNs）在处理图结构数据方面效果显著，但在大规模图上面临高内存需求和不高效的稀疏矩阵操作问题。量子计算（QC）提供了一种有前景的方法来解决这些问题，并启发了新的算法方法。特别是，近期文献中探讨了量子图神经网络（QGNNs）。然而，当前的量子硬件限制了能有效编码的数据维度。现有方法要么手动简化数据集，要么使用人工生成的图数据集。本文引入了Guided图压缩（GGC）框架，该框架使用图自编码器减少节点数和节点特征的维度。压缩过程受到后续分类任务性能的指导，可以结合量子或经典分类器应用。该框架在高能物理中至关重要的喷流标记任务上进行了评估，该任务涉及区分由夸克和由胶子引起的喷流。GGC与仅作为独立预处理步骤的自编码器以及基准经典GNN分类器进行了比较。我们的数值结果表明，GGC在性能上优于两种替代方案，同时也为在真实数据集上测试新的QGNN变体提供了途径。 

---
# Dataset of News Articles with Provenance Metadata for Media Relevance Assessment 

**Title (ZH)**: 包含来源元数据的新闻文章数据集用于媒体相关性评估 

**Authors**: Tomas Peterka, Matyas Bohacek  

**Link**: [PDF](https://arxiv.org/pdf/2506.09847)  

**Abstract**: Out-of-context and misattributed imagery is the leading form of media manipulation in today's misinformation and disinformation landscape. The existing methods attempting to detect this practice often only consider whether the semantics of the imagery corresponds to the text narrative, missing manipulation so long as the depicted objects or scenes somewhat correspond to the narrative at hand. To tackle this, we introduce News Media Provenance Dataset, a dataset of news articles with provenance-tagged images. We formulate two tasks on this dataset, location of origin relevance (LOR) and date and time of origin relevance (DTOR), and present baseline results on six large language models (LLMs). We identify that, while the zero-shot performance on LOR is promising, the performance on DTOR hinders, leaving room for specialized architectures and future work. 

**Abstract (ZH)**: 脱离上下文和误归因的图像已成为当今信息误导和虚假信息landscape中的主要媒体操纵形式。现有的检测方法往往仅考虑图像的语义是否与文本叙述相符，只要图像中描绘的对象或场景大致符合叙述，就会忽略操纵。为应对这一挑战，我们引入了新闻媒体溯源数据集，该数据集包含带有溯源标记的新闻文章。我们在该数据集上制定了两个任务：起始地点相关性（LOR）和起始时间相关性（DTOR），并在这六个大型语言模型（LLMs）上呈现了基线结果。我们发现，虽然LOR的零样本性能令人鼓舞，但DTOR的性能受限，为专门架构和未来工作留下了空间。 

---
# Learning to Align: Addressing Character Frequency Distribution Shifts in Handwritten Text Recognition 

**Title (ZH)**: 学习对齐：解决手写文本识别中的字符频率分布变化问题 

**Authors**: Panagiotis Kaliosis, John Pavlopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.09846)  

**Abstract**: Handwritten text recognition aims to convert visual input into machine-readable text, and it remains challenging due to the evolving and context-dependent nature of handwriting. Character sets change over time, and character frequency distributions shift across historical periods or regions, often causing models trained on broad, heterogeneous corpora to underperform on specific subsets. To tackle this, we propose a novel loss function that incorporates the Wasserstein distance between the character frequency distribution of the predicted text and a target distribution empirically derived from training data. By penalizing divergence from expected distributions, our approach enhances both accuracy and robustness under temporal and contextual intra-dataset shifts. Furthermore, we demonstrate that character distribution alignment can also improve existing models at inference time without requiring retraining by integrating it as a scoring function in a guided decoding scheme. Experimental results across multiple datasets and architectures confirm the effectiveness of our method in boosting generalization and performance. We open source our code at this https URL. 

**Abstract (ZH)**: 基于手写文本识别旨在将视觉输入转换为机器可读文本，但由于手写笔迹随时间演化和依赖具体情境，这一任务仍然具有挑战性。字符集随时间变化，不同时期或地区字符频率分布发生变化，常常导致在特定子集上表现不佳。为此，我们提出了一种新的损失函数，该函数结合了预测文本中的字符频率分布与基于训练数据经验推导出的目标分布之间的 Wasserstein 距离。通过惩罚与期望分布的偏差，我们的方法在时间性和情境性内部数据集变化下提高了准确性和稳健性。此外，我们展示了字符分布对齐也可以在推理时提高现有模型的性能，而无需重新训练，通过将其整合为引导解码方案中的打分函数来实现。跨多个数据集和架构的实验结果证实了我们方法在增强泛化能力和性能方面的有效性。我们在 GitHub（此链接请替换为实际链接地址）开源了代码。 

---
# EmoNet-Voice: A Fine-Grained, Expert-Verified Benchmark for Speech Emotion Detection 

**Title (ZH)**: EmoNet-Voice：一种细粒度且专家验证的语音情感识别基准。 

**Authors**: Christoph Schuhmann, Robert Kaczmarczyk, Gollam Rabby, Felix Friedrich, Maurice Kraus, Kourosh Nadi, Huu Nguyen, Kristian Kersting, Sören Auer  

**Link**: [PDF](https://arxiv.org/pdf/2506.09827)  

**Abstract**: The advancement of text-to-speech and audio generation models necessitates robust benchmarks for evaluating the emotional understanding capabilities of AI systems. Current speech emotion recognition (SER) datasets often exhibit limitations in emotional granularity, privacy concerns, or reliance on acted portrayals. This paper introduces EmoNet-Voice, a new resource for speech emotion detection, which includes EmoNet-Voice Big, a large-scale pre-training dataset (featuring over 4,500 hours of speech across 11 voices, 40 emotions, and 4 languages), and EmoNet-Voice Bench, a novel benchmark dataset with human expert annotations. EmoNet-Voice is designed to evaluate SER models on a fine-grained spectrum of 40 emotion categories with different levels of intensities. Leveraging state-of-the-art voice generation, we curated synthetic audio snippets simulating actors portraying scenes designed to evoke specific emotions. Crucially, we conducted rigorous validation by psychology experts who assigned perceived intensity labels. This synthetic, privacy-preserving approach allows for the inclusion of sensitive emotional states often absent in existing datasets. Lastly, we introduce Empathic Insight Voice models that set a new standard in speech emotion recognition with high agreement with human experts. Our evaluations across the current model landscape exhibit valuable findings, such as high-arousal emotions like anger being much easier to detect than low-arousal states like concentration. 

**Abstract (ZH)**: 文本转语音和音频生成模型的进步需要稳健的基准来评估AI系统的情感理解能力。现有的语音情感识别（SER）数据集往往在情感精细度、隐私问题或依赖于表演呈现方面存在局限。本文介绍了一种新的语音情感检测资源EmoNet-Voice，其中包括EmoNet-Voice Big（一个大规模预训练数据集，包含超过4500小时的语音数据，涵盖11种声音、40种情感和4种语言）和EmoNet-Voice Bench（一个新型基准数据集，包含人类专家注释）。EmoNet-Voice旨在评估SER模型在40种不同情感强度级别的精细情感维度上的表现。借助最新的语音生成技术，我们精心制作了模拟演员表演特定情感场景的合成音频片段。至关重要的是，我们通过心理学专家的严格验证，分配了感知强度标签。这种合成且保护隐私的方法使得敏感的情感状态能够被包含在内，这些状态在现有数据集中往往缺失。最后，我们引入了Empathic Insight Voice模型，这些模型在语音情感识别方面达到了新的标准，与人类专家的判断高度一致。我们对当前模型景观的评估显示出有价值的结果，例如，高度激动的情感如愤怒比低激动状态如专注更容易被检测到。 

---
# CoRT: Code-integrated Reasoning within Thinking 

**Title (ZH)**: CoRT: 代码集成推理 

**Authors**: Chengpeng Li, Zhengyang Tang, Ziniu Li, Mingfeng Xue, Keqin Bao, Tian Ding, Ruoyu Sun, Benyou Wang, Xiang Wang, Junyang Lin, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09820)  

**Abstract**: Large Reasoning Models (LRMs) like o1 and DeepSeek-R1 have shown remarkable progress in natural language reasoning with long chain-of-thought (CoT), yet they remain inefficient or inaccurate when handling complex mathematical operations. Addressing these limitations through computational tools (e.g., computation libraries and symbolic solvers) is promising, but it introduces a technical challenge: Code Interpreter (CI) brings external knowledge beyond the model's internal text representations, thus the direct combination is not efficient. This paper introduces CoRT, a post-training framework for teaching LRMs to leverage CI effectively and efficiently. As a first step, we address the data scarcity issue by synthesizing code-integrated reasoning data through Hint-Engineering, which strategically inserts different hints at appropriate positions to optimize LRM-CI interaction. We manually create 30 high-quality samples, upon which we post-train models ranging from 1.5B to 32B parameters, with supervised fine-tuning, rejection fine-tuning and reinforcement learning. Our experimental results demonstrate that Hint-Engineering models achieve 4\% and 8\% absolute improvements on DeepSeek-R1-Distill-Qwen-32B and DeepSeek-R1-Distill-Qwen-1.5B respectively, across five challenging mathematical reasoning datasets. Furthermore, Hint-Engineering models use about 30\% fewer tokens for the 32B model and 50\% fewer tokens for the 1.5B model compared with the natural language models. The models and code are available at this https URL. 

**Abstract (ZH)**: CoRT：一种用于教学大型推理模型高效利用代码解释器的后训练框架 

---
# Q-SAM2: Accurate Quantization for Segment Anything Model 2 

**Title (ZH)**: Q-SAM2: 准确量化分割万物模型 

**Authors**: Nicola Farronato, Florian Scheidegger, Mattia Rigotti, Cristiano Malossi, Michele Magno, Haotong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.09782)  

**Abstract**: The Segment Anything Model 2 (SAM2) has gained significant attention as a foundational approach for promptable image and video segmentation. However, its expensive computational and memory consumption poses a severe challenge for its application in resource-constrained scenarios. In this paper, we propose an accurate low-bit quantization method for efficient SAM2, termed Q-SAM2. To address the performance degradation caused by the singularities in weight and activation distributions during quantization, Q-SAM2 introduces two novel technical contributions. We first introduce a linear layer calibration method for low-bit initialization of SAM2, which minimizes the Frobenius norm over a small image batch to reposition weight distributions for improved quantization. We then propose a Quantization-Aware Training (QAT) pipeline that applies clipping to suppress outliers and allows the network to adapt to quantization thresholds during training. Our comprehensive experiments demonstrate that Q-SAM2 allows for highly accurate inference while substantially improving efficiency. Both quantitative and visual results show that our Q-SAM2 surpasses existing state-of-the-art general quantization schemes, especially for ultra-low 2-bit quantization. While designed for quantization-aware training, our proposed calibration technique also proves effective in post-training quantization, achieving up to a 66% mIoU accuracy improvement over non-calibrated models. 

**Abstract (ZH)**: Q-SAM2：一种高效的低比特量化方法 

---
# Load-Aware Training Scheduling for Model Circulation-based Decentralized Federated Learning 

**Title (ZH)**: 基于模型流通的去中心化联邦学习的负载感知训练调度 

**Authors**: Haruki Kainuma, Takayuki Nishio  

**Link**: [PDF](https://arxiv.org/pdf/2506.09769)  

**Abstract**: This paper proposes Load-aware Tram-FL, an extension of Tram-FL that introduces a training scheduling mechanism to minimize total training time in decentralized federated learning by accounting for both computational and communication loads. The scheduling problem is formulated as a global optimization task, which-though intractable in its original form-is made solvable by decomposing it into node-wise subproblems. To promote balanced data utilization under non-IID distributions, a variance constraint is introduced, while the overall training latency, including both computation and communication costs, is minimized through the objective function. Simulation results on MNIST and CIFAR-10 demonstrate that Load-aware Tram-FL significantly reduces training time and accelerates convergence compared to baseline methods. 

**Abstract (ZH)**: Load-aware Tram-FL：一种考虑计算和通信负载的卸载调度机制以减少去中心化联邦学习的总训练时间 

---
# AtmosMJ: Revisiting Gating Mechanism for AI Weather Forecasting Beyond the Year Scale 

**Title (ZH)**: AtmosMJ: 重新审视超越年尺度的AI天气预报中的门控机制 

**Authors**: Minjong Cheon  

**Link**: [PDF](https://arxiv.org/pdf/2506.09733)  

**Abstract**: The advent of Large Weather Models (LWMs) has marked a turning point in data-driven forecasting, with many models now outperforming traditional numerical systems in the medium range. However, achieving stable, long-range autoregressive forecasts beyond a few weeks remains a significant challenge. Prevailing state-of-the-art models that achieve year-long stability, such as SFNO and DLWP-HPX, have relied on transforming input data onto non-standard spatial domains like spherical harmonics or HEALPix meshes. This has led to the prevailing assumption that such representations are necessary to enforce physical consistency and long-term stability. This paper challenges that assumption by investigating whether comparable long-range performance can be achieved on the standard latitude-longitude grid. We introduce AtmosMJ, a deep convolutional network that operates directly on ERA5 data without any spherical remapping. The model's stability is enabled by a novel Gated Residual Fusion (GRF) mechanism, which adaptively moderates feature updates to prevent error accumulation over long recursive simulations. Our results demonstrate that AtmosMJ produces stable and physically plausible forecasts for about 500 days. In quantitative evaluations, it achieves competitive 10-day forecast accuracy against models like Pangu-Weather and GraphCast, all while requiring a remarkably low training budget of 5.7 days on a V100 GPU. Our findings suggest that efficient architectural design, rather than non-standard data representation, can be the key to unlocking stable and computationally efficient long-range weather prediction. 

**Abstract (ZH)**: 大型天气模型的兴起标志着数据驱动预报的一个转折点，许多模型现在在中短期已经超越了传统的数值系统。然而，实现稳定且长期的自回归预报超过几周仍是一项重大挑战。现有的如SFNO和DLWP-HPX等领先的模型依靠将输入数据转换到非标准的空间域，如球谐函数或HEALPix网格，以确保物理一致性并实现长期稳定性。本文通过研究在标准经纬度网格上是否可以达到相似的长期性能来挑战这一假设。我们引入了AtmosMJ，这是一种直接在ERA5数据上运行的深度卷积网络，无需任何球面重构。通过一种新颖的门控残差融合（GRF）机制，该模型能够逐步更新特征以防止长时间递归模拟中的误差累积。我们的结果显示，AtmosMJ能够产生稳定且物理合理的预报长达约500天。在定性评估中，AtmosMJ在10天预报的准确性方面与Pangu-Weather和GraphCast等模型竞争，同时仅需一个V100 GPU训练5.7天的极低训练预算。我们的研究结果表明，有效的架构设计而非非标准数据表示可能是实现稳定且计算高效的长期天气预报的关键。 

---
# Non-Contact Health Monitoring During Daily Personal Care Routines 

**Title (ZH)**: 非接触式日常个人护理中的健康监测 

**Authors**: Xulin Ma, Jiankai Tang, Zhang Jiang, Songqin Cheng, Yuanchun Shi, Dong LI, Xin Liu, Daniel McDuff, Xiaojing Liu, Yuntao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09718)  

**Abstract**: Remote photoplethysmography (rPPG) enables non-contact, continuous monitoring of physiological signals and offers a practical alternative to traditional health sensing methods. Although rPPG is promising for daily health monitoring, its application in long-term personal care scenarios, such as mirror-facing routines in high-altitude environments, remains challenging due to ambient lighting variations, frequent occlusions from hand movements, and dynamic facial postures. To address these challenges, we present LADH (Long-term Altitude Daily Health), the first long-term rPPG dataset containing 240 synchronized RGB and infrared (IR) facial videos from 21 participants across five common personal care scenarios, along with ground-truth PPG, respiration, and blood oxygen signals. Our experiments demonstrate that combining RGB and IR video inputs improves the accuracy and robustness of non-contact physiological monitoring, achieving a mean absolute error (MAE) of 4.99 BPM in heart rate estimation. Furthermore, we find that multi-task learning enhances performance across multiple physiological indicators simultaneously. Dataset and code are open at this https URL. 

**Abstract (ZH)**: 长期内置高度日常健康监控的远程光体积描记术数据集（LADH） 

---
# Towards Practical Alzheimer's Disease Diagnosis: A Lightweight and Interpretable Spiking Neural Model 

**Title (ZH)**: 面向实际应用的阿尔茨海默病诊断：一种轻量级可解释的.spiiking神经网络模型 

**Authors**: Changwei Wu, Yifei Chen, Yuxin Du, Jinying Zong, Jie Dong, Mingxuan Liu, Yong Peng, Jin Fan, Feiwei Qin, Changmiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09695)  

**Abstract**: Early diagnosis of Alzheimer's Disease (AD), especially at the mild cognitive impairment (MCI) stage, is vital yet hindered by subjective assessments and the high cost of multimodal imaging modalities. Although deep learning methods offer automated alternatives, their energy inefficiency and computational demands limit real-world deployment, particularly in resource-constrained settings. As a brain-inspired paradigm, spiking neural networks (SNNs) are inherently well-suited for modeling the sparse, event-driven patterns of neural degeneration in AD, offering a promising foundation for interpretable and low-power medical diagnostics. However, existing SNNs often suffer from weak expressiveness and unstable training, which restrict their effectiveness in complex medical tasks. To address these limitations, we propose FasterSNN, a hybrid neural architecture that integrates biologically inspired LIF neurons with region-adaptive convolution and multi-scale spiking attention. This design enables sparse, efficient processing of 3D MRI while preserving diagnostic accuracy. Experiments on benchmark datasets demonstrate that FasterSNN achieves competitive performance with substantially improved efficiency and stability, supporting its potential for practical AD screening. Our source code is available at this https URL. 

**Abstract (ZH)**: 基于尖峰神经网络的阿尔茨海默病早期诊断：FasterSNN架构的研究 

---
# Reasoning Models Are More Easily Gaslighted Than You Think 

**Title (ZH)**: 推理模型比你想象的更容易受到 Gaslighting 

**Authors**: Bin Zhu, Hailong Yin, Jingjing Chen, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09677)  

**Abstract**: Recent advances in reasoning-centric models promise improved robustness through mechanisms such as chain-of-thought prompting and test-time scaling. However, their ability to withstand misleading user input remains underexplored. In this paper, we conduct a systematic evaluation of three state-of-the-art reasoning models, i.e., OpenAI's o4-mini, Claude-3.7-Sonnet and Gemini-2.5-Flash, across three multimodal benchmarks: MMMU, MathVista, and CharXiv. Our evaluation reveals significant accuracy drops (25-29% on average) following gaslighting negation prompts, indicating that even top-tier reasoning models struggle to preserve correct answers under manipulative user feedback. Built upon the insights of the evaluation and to further probe this vulnerability, we introduce GaslightingBench-R, a new diagnostic benchmark specifically designed to evaluate reasoning models' susceptibility to defend their belief under gaslighting negation prompt. Constructed by filtering and curating 1,025 challenging samples from the existing benchmarks, GaslightingBench-R induces even more dramatic failures, with accuracy drops exceeding 53% on average. Our findings reveal fundamental limitations in the robustness of reasoning models, highlighting the gap between step-by-step reasoning and belief persistence. 

**Abstract (ZH)**: Recent Advances in Reasoning-Centric Models: Evaluating their Robustness to Gaslighting Negation Prompts and Introducing GaslightingBench-R 

---
# Empirical Quantification of Spurious Correlations in Malware Detection 

**Title (ZH)**: 恶意软件检测中虚假相关性的经验量化 

**Authors**: Bianca Perasso, Ludovico Lozza, Andrea Ponte, Luca Demetrio, Luca Oneto, Fabio Roli  

**Link**: [PDF](https://arxiv.org/pdf/2506.09662)  

**Abstract**: End-to-end deep learning exhibits unmatched performance for detecting malware, but such an achievement is reached by exploiting spurious correlations -- features with high relevance at inference time, but known to be useless through domain knowledge. While previous work highlighted that deep networks mainly focus on metadata, none investigated the phenomenon further, without quantifying their impact on the decision. In this work, we deepen our understanding of how spurious correlation affects deep learning for malware detection by highlighting how much models rely on empty spaces left by the compiler, which diminishes the relevance of the compiled code. Through our seminal analysis on a small-scale balanced dataset, we introduce a ranking of two end-to-end models to better understand which is more suitable to be put in production. 

**Abstract (ZH)**: 端到端深度学习在检测恶意软件方面表现出色，但这种成就依赖于错误的相关性——在推理时具有高相关性的特征，但通过领域知识已知是无用的。尽管先前的工作指出深度网络主要关注元数据，但没有进一步探讨其对决策的影响。在本研究中，我们通过强调模型如何依赖编译器留下的空白空间，深化了对错误相关性如何影响恶意软件检测深度学习的理解，这些空白空间降低了编译代码的相关性。通过对小型平衡数据集的开创性分析，我们引入了两个端到端模型的排序，以更好地理解哪个模型更适合投入生产。 

---
# Neural Functions for Learning Periodic Signal 

**Title (ZH)**: 神经网络在学习周期信号中的功能 

**Authors**: Woojin Cho, Minju Jo, Kookjin Lee, Noseong Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.09526)  

**Abstract**: As function approximators, deep neural networks have served as an effective tool to represent various signal types. Recent approaches utilize multi-layer perceptrons (MLPs) to learn a nonlinear mapping from a coordinate to its corresponding signal, facilitating the learning of continuous neural representations from discrete data points. Despite notable successes in learning diverse signal types, coordinate-based MLPs often face issues of overfitting and limited generalizability beyond the training region, resulting in subpar extrapolation performance. This study addresses scenarios where the underlying true signals exhibit periodic properties, either spatially or temporally. We propose a novel network architecture, which extracts periodic patterns from measurements and leverages this information to represent the signal, thereby enhancing generalization and improving extrapolation performance. We demonstrate the efficacy of the proposed method through comprehensive experiments, including the learning of the periodic solutions for differential equations, and time series imputation (interpolation) and forecasting (extrapolation) on real-world datasets. 

**Abstract (ZH)**: 基于坐标的多层感知机在表示具有周期性特性的信号中的应用：一种新型网络架构的提出与验证 

---
# TransXSSM: A Hybrid Transformer State Space Model with Unified Rotary Position Embedding 

**Title (ZH)**: 跨XSSM：一种结合统一旋转位置嵌入的混合变压器状态空间模型 

**Authors**: Bingheng Wu, Jingze Shi, Yifan Wu, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09507)  

**Abstract**: Transformers exhibit proficiency in capturing long-range dependencies, whereas State Space Models (SSMs) facilitate linear-time sequence modeling. Notwithstanding their synergistic potential, the integration of these architectures presents a significant challenge, primarily attributable to a fundamental incongruity in their respective positional encoding mechanisms: Transformers rely on explicit Rotary Position Embeddings (RoPE), while SSMs leverage implicit positional representations via convolutions. This divergence often precipitates discontinuities and suboptimal performance. To address this impediment, we propose a unified rotary position embedding (\textbf{\ourRoPE}) methodology, thereby establishing a consistent positional encoding framework for both self-attention and state-space components. Using this \ourRoPE, we introduce \textbf{\model}, a hybrid architecture that coherently integrates the Transformer and SSM layers under this unified positional encoding scheme. At a 4K sequence length, \model exhibits training and inference speeds that are \textbf{42.3\% and 29.5\% faster}, respectively, relative to standard Transformer models. It also delivers higher accuracy: under comparable settings, it surpasses a Transformer baseline by over 4\% on language modeling benchmarks. \model furthermore scales more effectively: \model-1.3B gains \textbf{7.22\%} in average accuracy over its 320M version (versus about 6\% gains for equivalent Transformers or SSMs). Our results show that unified positional encoding resolves positional incompatibility in hybrid models, enabling efficient, high-performance long-context modeling. 

**Abstract (ZH)**: 统一旋转位置嵌入的变压器-状态空间混合模型 

---
# A Unified Theory of Compositionality, Modularity, and Interpretability in Markov Decision Processes 

**Title (ZH)**: 马尔可夫决策过程中的组成性、模块化和可解释性统一理论 

**Authors**: Thomas J. Ringstrom, Paul R. Schrater  

**Link**: [PDF](https://arxiv.org/pdf/2506.09499)  

**Abstract**: We introduce Option Kernel Bellman Equations (OKBEs) for a new reward-free Markov Decision Process. Rather than a value function, OKBEs directly construct and optimize a predictive map called a state-time option kernel (STOK) to maximize the probability of completing a goal while avoiding constraint violations. STOKs are compositional, modular, and interpretable initiation-to-termination transition kernels for policies in the Options Framework of Reinforcement Learning. This means: 1) STOKs can be composed using Chapman-Kolmogorov equations to make spatiotemporal predictions for multiple policies over long horizons, 2) high-dimensional STOKs can be represented and computed efficiently in a factorized and reconfigurable form, and 3) STOKs record the probabilities of semantically interpretable goal-success and constraint-violation events, needed for formal verification. Given a high-dimensional state-transition model for an intractable planning problem, we can decompose it with local STOKs and goal-conditioned policies that are aggregated into a factorized goal kernel, making it possible to forward-plan at the level of goals in high-dimensions to solve the problem. These properties lead to highly flexible agents that can rapidly synthesize meta-policies, reuse planning representations across many tasks, and justify goals using empowerment, an intrinsic motivation function. We argue that reward-maximization is in conflict with the properties of compositionality, modularity, and interpretability. Alternatively, OKBEs facilitate these properties to support verifiable long-horizon planning and intrinsic motivation that scales to dynamic high-dimensional world-models. 

**Abstract (ZH)**: 基于选项内核贝尔曼方程的无奖励马尔可夫决策过程 

---
# EnerBridge-DPO: Energy-Guided Protein Inverse Folding with Markov Bridges and Direct Preference Optimization 

**Title (ZH)**: EnerBridge-DPO: 能量导向的蛋白质逆折叠方法结合马尔可夫桥和直接偏好优化 

**Authors**: Dingyi Rong, Haotian Lu, Wenzhuo Zheng, Fan Zhang, Shuangjia Zheng, Ning Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09496)  

**Abstract**: Designing protein sequences with optimal energetic stability is a key challenge in protein inverse folding, as current deep learning methods are primarily trained by maximizing sequence recovery rates, often neglecting the energy of the generated sequences. This work aims to overcome this limitation by developing a model that directly generates low-energy, stable protein sequences. We propose EnerBridge-DPO, a novel inverse folding framework focused on generating low-energy, high-stability protein sequences. Our core innovation lies in: First, integrating Markov Bridges with Direct Preference Optimization (DPO), where energy-based preferences are used to fine-tune the Markov Bridge model. The Markov Bridge initiates optimization from an information-rich prior sequence, providing DPO with a pool of structurally plausible sequence candidates. Second, an explicit energy constraint loss is introduced, which enhances the energy-driven nature of DPO based on prior sequences, enabling the model to effectively learn energy representations from a wealth of prior knowledge and directly predict sequence energy values, thereby capturing quantitative features of the energy landscape. Our evaluations demonstrate that EnerBridge-DPO can design protein complex sequences with lower energy while maintaining sequence recovery rates comparable to state-of-the-art models, and accurately predicts $\Delta \Delta G$ values between various sequences. 

**Abstract (ZH)**: 设计具有最优能量稳定性的蛋白质序列是蛋白质逆折叠中的一个关键挑战，当前的深度学习方法主要通过最大化序列恢复率来训练，往往忽视了生成序列的能量。本工作旨在通过开发一个能够直接生成低能稳定蛋白质序列的模型来克服这一限制。我们提出了一种名为EnerBridge-DPO的新型逆折叠框架，专注于生成低能高稳定性的蛋白质序列。我们的核心创新在于：首先，将马尔可夫桥与直接偏好优化（DPO）相结合，使用基于能量的偏好来精细调整马尔可夫桥模型。马尔可夫桥从一个信息丰富的先验序列开始优化，为DPO提供一组结构上可行的序列候选。其次，引入了一个明确的能量约束损失，这增强了基于先验序列的DPO的能量驱动性质，使模型能够从丰富的先验知识中有效学习能量表示，并直接预测序列的能量值，从而捕捉能量景观的定量特征。我们的评估表明，EnerBridge-DPO可以在保持与最新模型相当的序列恢复率的同时，设计出能量较低的蛋白质复合序列，并准确预测不同序列之间的$\Delta \Delta G$值。 

---
# BemaGANv2: A Tutorial and Comparative Survey of GAN-based Vocoders for Long-Term Audio Generation 

**Title (ZH)**: BemaGANv2：基于GAN的长时音频生成 vocoder 的教程与比较综述 

**Authors**: Taesoo Park, Mungwi Jeong, Mingyu Park, Narae Kim, Junyoung Kim, Mujung Kim, Jisang Yoo, Hoyun Lee, Sanghoon Kim, Soonchul Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2506.09487)  

**Abstract**: This paper presents a tutorial-style survey and implementation guide of BemaGANv2, an advanced GAN-based vocoder designed for high-fidelity and long-term audio generation. Built upon the original BemaGAN architecture, BemaGANv2 incorporates major architectural innovations by replacing traditional ResBlocks in the generator with the Anti-aliased Multi-Periodicity composition (AMP) module, which internally applies the Snake activation function to better model periodic structures. In the discriminator framework, we integrate the Multi-Envelope Discriminator (MED), a novel architecture we originally proposed, to extract rich temporal envelope features crucial for periodicity detection. Coupled with the Multi-Resolution Discriminator (MRD), this combination enables more accurate modeling of long-range dependencies in audio. We systematically evaluate various discriminator configurations, including MSD + MED, MSD + MRD, and MPD + MED + MRD, using objective metrics (FAD, SSIM, PLCC, MCD) and subjective evaluations (MOS, SMOS). This paper also provides a comprehensive tutorial on the model architecture, training methodology, and implementation to promote reproducibility. The code and pre-trained models are available at: this https URL. 

**Abstract (ZH)**: 本文提供了BemaGANv2的教程式综述与实现指南，BemaGANv2是基于GAN的 vocoder，旨在实现高保真度和长时间音频生成。BemaGANv2在原始BemaGAN架构的基础上引入了重大架构创新，通过使用抗混叠多周期组成（AMP）模块替代生成器中的传统ResBlock，并在内部应用蛇形激活函数以更好地建模周期结构。在判别框架中，我们结合了我们最初提出的一种新颖架构——多包络判别器（MED），以提取对于周期性检测至关重要的丰富时间包络特征。结合多分辨率判别器（MRD），这一组合能够更准确地建模音频中的长期依赖性。我们使用客观指标（FAD、SSIM、PLCC、MCD）和主观评估（MOS、SMOS）系统性地评估了各种判别器配置。本文还提供了模型架构、训练方法和实现的全面教程，以促进可再现性。可从以下链接获取代码和预训练模型：this https URL。 

---
# Abstraction-Based Proof Production in Formal Verification of Neural Networks 

**Title (ZH)**: 基于抽象的证明生成在神经网络的形式验证中 

**Authors**: Yizhak Yisrael Elboher, Omri Isac, Guy Katz, Tobias Ladner, Haoze Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09455)  

**Abstract**: Modern verification tools for deep neural networks (DNNs) increasingly rely on abstraction to scale to realistic architectures. In parallel, proof production is becoming a critical requirement for increasing the reliability of DNN verification results. However, current proofproducing verifiers do not support abstraction-based reasoning, creating a gap between scalability and provable guarantees. We address this gap by introducing a novel framework for proof-producing abstraction-based DNN verification. Our approach modularly separates the verification task into two components: (i) proving the correctness of an abstract network, and (ii) proving the soundness of the abstraction with respect to the original DNN. The former can be handled by existing proof-producing verifiers, whereas we propose the first method for generating formal proofs for the latter. This preliminary work aims to enable scalable and trustworthy verification by supporting common abstraction techniques within a formal proof framework. 

**Abstract (ZH)**: 现代深度神经网络(DNN)验证工具越来越多地依赖抽象来扩展到现实架构。与此同时，证明生产正逐渐成为提高DNN验证结果可靠性的关键要求。然而，当前的证明生产验证器不支持基于抽象的推理，从而在可扩展性和可证明保证之间造成了差距。我们通过引入一种新的基于抽象的DNN验证的证明生产框架来解决这一差距。我们的方法将验证任务模块化地分为两个部分：（i）证明抽象网络的正确性，以及（ii）证明抽象相对于原始DNN的正确性。前者可以由现有的证明生产验证器处理，而我们则提出了生成形式证明以处理后者的第一种方法。本初步工作旨在通过在形式证明框架中支持常见的抽象技术，实现可扩展和可信赖的验证。 

---
# When Is Diversity Rewarded in Cooperative Multi-Agent Learning? 

**Title (ZH)**: 在合作多智能体学习中，何时会奖励多样性？ 

**Authors**: Michael Amir, Matteo Bettini, Amanda Prorok  

**Link**: [PDF](https://arxiv.org/pdf/2506.09434)  

**Abstract**: The success of teams in robotics, nature, and society often depends on the division of labor among diverse specialists; however, a principled explanation for when such diversity surpasses a homogeneous team is still missing. Focusing on multi-agent task allocation problems, our goal is to study this question from the perspective of reward design: what kinds of objectives are best suited for heterogeneous teams? We first consider an instantaneous, non-spatial setting where the global reward is built by two generalized aggregation operators: an inner operator that maps the $N$ agents' effort allocations on individual tasks to a task score, and an outer operator that merges the $M$ task scores into the global team reward. We prove that the curvature of these operators determines whether heterogeneity can increase reward, and that for broad reward families this collapses to a simple convexity test. Next, we ask what incentivizes heterogeneity to emerge when embodied, time-extended agents must learn an effort allocation policy. To study heterogeneity in such settings, we use multi-agent reinforcement learning (MARL) as our computational paradigm, and introduce Heterogeneous Environment Design (HED), a gradient-based algorithm that optimizes the parameter space of underspecified MARL environments to find scenarios where heterogeneity is advantageous. Experiments in matrix games and an embodied Multi-Goal-Capture environment show that, despite the difference in settings, HED rediscovers the reward regimes predicted by our theory to maximize the advantage of heterogeneity, both validating HED and connecting our theoretical insights to reward design in MARL. Together, these results help us understand when behavioral diversity delivers a measurable benefit. 

**Abstract (ZH)**: 机器人学、自然和社会中团队的成功 often取决于多样专家之间的劳动分工；然而，关于这种多样性何时超越了 homogeneous 团队的规范解释仍然缺失。我们关注多智能体任务分配问题，从奖励设计的角度研究这一问题：什么样的目标最适合 heterogeneous 团队？我们首先考虑一个瞬时、非空间性的设置，其中全局奖励由两个广义聚合操作员构建：一个内操作员将 N 个智能体在单个任务上的努力分配映射到任务分数，另一个外操作员将 M 个任务分数合并成全局团队奖励。我们证明这些操作员的曲率决定了多样性是否能增加奖励，并且对于广泛的奖励家庭，这归结为一个简单的凸性测试。接下来，我们探讨当具身化的、时间延伸的智能体必须学习努力分配策略时，是什么激励了多样性的出现。为了研究这种设置下的多样性，我们以多智能体强化学习（MARL）作为计算范式，并引入多样环境设计（HED），这是一种基于梯度的算法，在欠规定的 MARL 环境的参数空间中进行优化，以找到多样化有益的场景。在矩阵游戏和一个具身的多目标捕捉环境中进行的实验表明，尽管设置不同，HED 重新发现了我们的理论预测的最大化多样性的奖励区域，这验证了 HED 并将我们的理论洞察与 MARL 中的奖励设计联系起来。总之，这些结果帮助我们理解当行为多样性带来可度量的好处时。 

---
# Reasoning as a Resource: Optimizing Fast and Slow Thinking in Code Generation Models 

**Title (ZH)**: 将推理作为一种资源：优化代码生成模型中的快速与慢速思考 

**Authors**: Zongjie Li, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09396)  

**Abstract**: This position paper proposes a fundamental shift in designing code generation models: treating reasoning depth as a controllable resource. Rather than being an incidental byproduct of prompting, we argue that the trade-off between rapid, direct answers ("fast thinking") and elaborate, chain-of-thought deliberation ("slow thinking") must be explicitly managed. We contend that optimizing reasoning budgets across the entire model lifecycle - from synthetic data creation and benchmarking to real-world deploymen - can unlock superior trade-offs among accuracy, latency, and cost. This paper outlines how adaptive control over reasoning can enrich supervision signals, motivate new multi-dimensional benchmarks, and inform cost-aware, security-conscious deployment policies. By viewing fast and slow thinking as complementary modes to be scheduled, we envision coding agents that think deep when necessary and act fast when possible. 

**Abstract (ZH)**: 本论点论文提出在设计代码生成模型时进行根本性的转变：将推理深度视为可控资源。我们认为，快速直接的答案（“快速思考”）与详细逐步的权衡决策（“慢速思考”）之间的权衡不应被视为提示的附带产物，而应明确管理。我们主张在整个模型生命周期中优化推理预算——从合成数据创建和基准测试到实际部署——以解锁在准确率、延迟和成本之间的更优权衡。本文介绍了如何通过适应性控制推理来丰富监督信号、激发新的多维基准测试，并指导成本意识与安全意识的部署策略。通过将快速与慢速思考视为可调度的互补方式，我们设想能够根据需要深入思考并在可能的情况下快速行动的编码代理。 

---
# LPO: Towards Accurate GUI Agent Interaction via Location Preference Optimization 

**Title (ZH)**: LPO：通过位置偏好优化实现精确的GUI代理交互 

**Authors**: Jiaqi Tang, Yu Xia, Yi-Feng Wu, Yuwei Hu, Yuhui Chen, Qing-Guo Chen, Xiaogang Xu, Xiangyu Wu, Hao Lu, Yanqing Ma, Shiyin Lu, Qifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09373)  

**Abstract**: The advent of autonomous agents is transforming interactions with Graphical User Interfaces (GUIs) by employing natural language as a powerful intermediary. Despite the predominance of Supervised Fine-Tuning (SFT) methods in current GUI agents for achieving spatial localization, these methods face substantial challenges due to their limited capacity to accurately perceive positional data. Existing strategies, such as reinforcement learning, often fail to assess positional accuracy effectively, thereby restricting their utility. In response, we introduce Location Preference Optimization (LPO), a novel approach that leverages locational data to optimize interaction preferences. LPO uses information entropy to predict interaction positions by focusing on zones rich in information. Besides, it further introduces a dynamic location reward function based on physical distance, reflecting the varying importance of interaction positions. Supported by Group Relative Preference Optimization (GRPO), LPO facilitates an extensive exploration of GUI environments and significantly enhances interaction precision. Comprehensive experiments demonstrate LPO's superior performance, achieving SOTA results across both offline benchmarks and real-world online evaluations. Our code will be made publicly available soon, at this https URL. 

**Abstract (ZH)**: 自主代理的出现正在通过采用自然语言作为强中介来转变与图形用户界面（GUI）的交互。尽管当前的GUI代理中监督微调（SFT）方法在实现空间定位方面占据主导地位，但由于其感知位置数据能力有限，这些方法面临着重大挑战。现有策略，如强化学习，往往难以有效地评估位置准确性，从而限制了其应用。为此，我们提出了位置偏好优化（LPO）这一新颖方法，该方法利用位置数据来优化交互偏好。LPO 使用信息熵来预测交互位置，重点关注信息丰富的区域。此外，LPO 还引入了基于物理距离的动态位置奖励函数，反映了交互位置的不同重要性。借助群组相对偏好优化（GRPO），LPO 促进了GUI环境的广泛探索，并显著提高了交互精度。全面的实验表明，LPO 在离线基准和实际在线评估中均实现了最先进的性能。我们的代码将很快公开，地址见此 https URL。 

---
# Anomaly Detection and Generation with Diffusion Models: A Survey 

**Title (ZH)**: 基于扩散模型的异常检测与生成：一个综述 

**Authors**: Yang Liu, Jing Liu, Chengfang Li, Rui Xi, Wenchao Li, Liang Cao, Jin Wang, Laurence T. Yang, Junsong Yuan, Wei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.09368)  

**Abstract**: Anomaly detection (AD) plays a pivotal role across diverse domains, including cybersecurity, finance, healthcare, and industrial manufacturing, by identifying unexpected patterns that deviate from established norms in real-world data. Recent advancements in deep learning, specifically diffusion models (DMs), have sparked significant interest due to their ability to learn complex data distributions and generate high-fidelity samples, offering a robust framework for unsupervised AD. In this survey, we comprehensively review anomaly detection and generation with diffusion models (ADGDM), presenting a tutorial-style analysis of the theoretical foundations and practical implementations and spanning images, videos, time series, tabular, and multimodal data. Crucially, unlike existing surveys that often treat anomaly detection and generation as separate problems, we highlight their inherent synergistic relationship. We reveal how DMs enable a reinforcing cycle where generation techniques directly address the fundamental challenge of anomaly data scarcity, while detection methods provide critical feedback to improve generation fidelity and relevance, advancing both capabilities beyond their individual potential. A detailed taxonomy categorizes ADGDM methods based on anomaly scoring mechanisms, conditioning strategies, and architectural designs, analyzing their strengths and limitations. We final discuss key challenges including scalability and computational efficiency, and outline promising future directions such as efficient architectures, conditioning strategies, and integration with foundation models (e.g., visual-language models and large language models). By synthesizing recent advances and outlining open research questions, this survey aims to guide researchers and practitioners in leveraging DMs for innovative AD solutions across diverse applications. 

**Abstract (ZH)**: 基于扩散模型的异常检测与生成综述（Anomaly Detection and Generation with Diffusion Models: A Survey） 

---
# COGENT: A Curriculum-oriented Framework for Generating Grade-appropriate Educational Content 

**Title (ZH)**: COGENT: 以课程为导向的生成适龄教育内容框架 

**Authors**: Zhengyuan Liu, Stella Xin Yin, Dion Hoe-Lian Goh, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09367)  

**Abstract**: While Generative AI has demonstrated strong potential and versatility in content generation, its application to educational contexts presents several challenges. Models often fail to align with curriculum standards and maintain grade-appropriate reading levels consistently. Furthermore, STEM education poses additional challenges in balancing scientific explanations with everyday language when introducing complex and abstract ideas and phenomena to younger students. In this work, we propose COGENT, a curriculum-oriented framework for generating grade-appropriate educational content. We incorporate three curriculum components (science concepts, core ideas, and learning objectives), control readability through length, vocabulary, and sentence complexity, and adopt a ``wonder-based'' approach to increase student engagement and interest. We conduct a multi-dimensional evaluation via both LLM-as-a-judge and human expert analysis. Experimental results show that COGENT consistently produces grade-appropriate passages that are comparable or superior to human references. Our work establishes a viable approach for scaling adaptive and high-quality learning resources. 

**Abstract (ZH)**: 基于课程的生成式教育内容框架：COGENT 

---
# "I Said Things I Needed to Hear Myself": Peer Support as an Emotional, Organisational, and Sociotechnical Practice in Singapore 

**Title (ZH)**: “我说了自己需要听到的话”：朋辈支持在新加坡作为情感的、组织的和社会技术的实践 

**Authors**: Kellie Yu Hui Sim, Kenny Tsu Wei Choo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09362)  

**Abstract**: Peer support plays a vital role in expanding access to mental health care by providing empathetic, community-based support outside formal clinical systems. As digital platforms increasingly mediate such support, the design and impact of these technologies remain under-examined, particularly in Asian contexts. This paper presents findings from an interview study with 20 peer supporters in Singapore, who operate across diverse online, offline, and hybrid environments. Through a thematic analysis, we unpack how participants start, conduct, and sustain peer support, highlighting their motivations, emotional labour, and the sociocultural dimensions shaping their practices. Building on this grounded understanding, we surface design directions for culturally responsive digital tools that scaffold rather than supplant relational care. Drawing insights from qualitative accounts, we offer a situated perspective on how AI might responsibly augment peer support. This research contributes to human-centred computing by articulating the lived realities of peer supporters and proposing design implications for trustworthy and context-sensitive AI in mental health. 

**Abstract (ZH)**: 同伴支持在外形式化临床系统之外提供了具有同理心的社区支持，对于扩大心理健康服务的获取发挥着至关重要的作用。随着数字平台在提供此类支持方面发挥越来越大的作用，这些技术的设计及其影响仍需进一步考察，尤其是在亚洲背景下。本文基于对新加坡20名同伴支持者的研究访谈，探讨他们在多种线上、线下及混合环境中的支持方式。通过主题分析，我们揭示了参与者如何开展和维持同伴支持，强调了他们的动机、情感劳动及其社会文化背景对其实践的影响。基于这一扎根理解，我们提出了文化响应性的数字工具设计方向，旨在搭桥而非取代关系性关怀。结合定性叙述，我们提出了一种负责任地增强同伴支持的在地性视角。本研究通过阐述同伴支持者的生活现实并提出可信赖且情境敏感的AI设计建议，贡献于以人为本的计算科学。 

---
# ErrorEraser: Unlearning Data Bias for Improved Continual Learning 

**Title (ZH)**: ErrorEraser: 消除数据偏见以改善连续学习 

**Authors**: Xuemei Cao, Hanlin Gu, Xin Yang, Bingjun Wei, Haoyang Liang, Xiangkun Wang, Tianrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09347)  

**Abstract**: Continual Learning (CL) primarily aims to retain knowledge to prevent catastrophic forgetting and transfer knowledge to facilitate learning new tasks. Unlike traditional methods, we propose a novel perspective: CL not only needs to prevent forgetting, but also requires intentional this http URL arises from existing CL methods ignoring biases in real-world data, leading the model to learn spurious correlations that transfer and amplify across tasks. From feature extraction and prediction results, we find that data biases simultaneously reduce CL's ability to retain and transfer knowledge. To address this, we propose ErrorEraser, a universal plugin that removes erroneous memories caused by biases in CL, enhancing performance in both new and old tasks. ErrorEraser consists of two modules: Error Identification and Error Erasure. The former learns the probability density distribution of task data in the feature space without prior knowledge, enabling accurate identification of potentially biased samples. The latter ensures only erroneous knowledge is erased by shifting the decision space of representative outlier samples. Additionally, an incremental feature distribution learning strategy is designed to reduce the resource overhead during error identification in downstream tasks. Extensive experimental results show that ErrorEraser significantly mitigates the negative impact of data biases, achieving higher accuracy and lower forgetting rates across three types of CL methods. The code is available at this https URL. 

**Abstract (ZH)**: 持续学习（CL）主要旨在保留知识以防止灾难性遗忘，并转移知识以促进新任务的学习。与传统方法不同，我们提出了一种新的视角：CL不仅需要防止遗忘，还需要有意地消除由于偏见引起的错误记忆。这种偏见来自于现有CL方法忽略现实世界数据中的偏差，导致模型学习转移并放大错误的相关性。通过特征提取和预测结果，我们发现数据偏见同时降低了CL保留和转移知识的能力。为此，我们提出ErrorEraser，这是一种通用插件，它可以消除由CL中偏见引起的各种错误记忆，从而在新任务和旧任务中均提高性能。ErrorEraser由两个模块组成：错误识别和错误消除。前者在无需先验知识的情况下学习特征空间中任务数据的概率密度分布，从而能够准确识别潜在的偏倚样本。后者通过调整代表性离群样本的决策空间，确保仅消除错误的知识。此外，我们设计了增量特征分布学习策略，以减少错误识别在下游任务中的资源开销。广泛的实验结果显示，ErrorEraser显著减轻了数据偏见的负面影响，在三种类型CL方法中均实现了更高的准确率和更低的遗忘率。代码可在以下链接获取：this https URL。 

---
# Intelligent System of Emergent Knowledge: A Coordination Fabric for Billions of Minds 

**Title (ZH)**: emergent知识智能系统：数十亿心灵的协调织 fabric 

**Authors**: Moshi Wei, Sparks Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09335)  

**Abstract**: The Intelligent System of Emergent Knowledge (ISEK) establishes a decentralized network where human and artificial intelligence agents collaborate as peers, forming a self-organizing cognitive ecosystem. Built on Web3 infrastructure, ISEK combines three fundamental principles: (1) a decentralized multi-agent architecture resistant to censorship, (2) symbiotic AI-human collaboration with equal participation rights, and (3) resilient self-adaptation through distributed consensus mechanisms.
The system implements an innovative coordination protocol featuring a six-phase workflow (Publish, Discover, Recruit, Execute, Settle, Feedback) for dynamic task allocation, supported by robust fault tolerance and a multidimensional reputation system. Economic incentives are governed by the native $ISEK token, facilitating micropayments, governance participation, and reputation tracking, while agent sovereignty is maintained through NFT-based identity management.
This synthesis of blockchain technology, artificial intelligence, and incentive engineering creates an infrastructure that actively facilitates emergent intelligence. ISEK represents a paradigm shift from conventional platforms, enabling the organic development of large-scale, decentralized cognitive systems where autonomous agents collectively evolve beyond centralized constraints. 

**Abstract (ZH)**: 基于涌现知识的智能系统（ISEK）构建了一个去中心化的网络，其中人类和人工智能代理作为平级合作，形成一个自我组织的认知生态系统。ISEK基于Web3基础设施，融合了三项基本原则：（1）去中心化的多功能代理架构，具有抗审查性，（2）共生的人工智能与人类协作，参与者拥有平等的参与权，（3）通过分布式共识机制实现的韧性自适应。系统实现了一种创新的协调协议，包含六个工作流程（发布、发现、招募、执行、结算、反馈），并具有强大的容错性和多维度声誉系统。经济激励由原生ISEK代币管理，支持微支付、治理参与和声誉追踪，代理主权则通过基于NFT的身份管理得以维护。这一区块链技术、人工智能与激励工程的综合应用，构建了一种促进涌现智能的动力结构。ISEK代表了从传统平台向一种新型范式的转变，使大规模、去中心化的认知系统能够有机发展，从而超越中心化的限制。 

---
# $(RSA)^2$: A Rhetorical-Strategy-Aware Rational Speech Act Framework for Figurative Language Understanding 

**Title (ZH)**: $(RSA)^2$: 一种考虑修辞策略的语用推理框架用于描绘性语言理解 

**Authors**: Cesare Spinoso-Di Piano, David Austin, Pablo Piantanida, Jackie Chi Kit Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2506.09301)  

**Abstract**: Figurative language (e.g., irony, hyperbole, understatement) is ubiquitous in human communication, resulting in utterances where the literal and the intended meanings do not match. The Rational Speech Act (RSA) framework, which explicitly models speaker intentions, is the most widespread theory of probabilistic pragmatics, but existing implementations are either unable to account for figurative expressions or require modeling the implicit motivations for using figurative language (e.g., to express joy or annoyance) in a setting-specific way. In this paper, we introduce the Rhetorical-Strategy-Aware RSA $(RSA)^2$ framework which models figurative language use by considering a speaker's employed rhetorical strategy. We show that $(RSA)^2$ enables human-compatible interpretations of non-literal utterances without modeling a speaker's motivations for being non-literal. Combined with LLMs, it achieves state-of-the-art performance on the ironic split of PragMega+, a new irony interpretation dataset introduced in this study. 

**Abstract (ZH)**: 具修辞策略意识的理性演讲行为框架 $(RSA)^2$：通过考虑说话人采用的修辞策略建模隐喻语言的使用 

---
# Causal Graph Recovery in Neuroimaging through Answer Set Programming 

**Title (ZH)**: 神经影像中基于回答集编程的因果图恢复 

**Authors**: Mohammadsajad Abavisani, Kseniya Solovyeva, David Danks, Vince Calhoun, Sergey Plis  

**Link**: [PDF](https://arxiv.org/pdf/2506.09286)  

**Abstract**: Learning graphical causal structures from time series data presents significant challenges, especially when the measurement frequency does not match the causal timescale of the system. This often leads to a set of equally possible underlying causal graphs due to information loss from sub-sampling (i.e., not observing all possible states of the system throughout time). Our research addresses this challenge by incorporating the effects of sub-sampling in the derivation of causal graphs, resulting in more accurate and intuitive outcomes. We use a constraint optimization approach, specifically answer set programming (ASP), to find the optimal set of answers. ASP not only identifies the most probable underlying graph, but also provides an equivalence class of possible graphs for expert selection. In addition, using ASP allows us to leverage graph theory to further prune the set of possible solutions, yielding a smaller, more accurate answer set significantly faster than traditional approaches. We validate our approach on both simulated data and empirical structural brain connectivity, and demonstrate its superiority over established methods in these experiments. We further show how our method can be used as a meta-approach on top of established methods to obtain, on average, 12% improvement in F1 score. In addition, we achieved state of the art results in terms of precision and recall of reconstructing causal graph from sub-sampled time series data. Finally, our method shows robustness to varying degrees of sub-sampling on realistic simulations, whereas other methods perform worse for higher rates of sub-sampling. 

**Abstract (ZH)**: 从时间序列数据中学习图形因果结构面临显著挑战，尤其是当测量频率不能匹配系统因果时间尺度时。这常常导致由于子采样造成的的信息丢失（即，没有在整个时间范围内观察到系统的所有可能状态），使得潜在因果图有多种等可能性。我们的研究通过在因果图的推导中纳入子采样的影响，从而实现了更准确和直观的结果。我们使用约束优化方法，具体来说是回答集编程（ASP），来找到最优的答案集。ASP不仅识别出最可能的潜在图，还为专家选择提供了可能图的等价类。此外，使用ASP使得我们可以利用图论进一步修剪可能解的集合，从而比传统方法更快地获得更小、更准确的答案集。我们在模拟数据和实证的结构脑连接中验证了这一方法，并在这些实验中证明了其在F1分数上的优越性。此外，我们在从子采样时间序列数据重建因果图方面的精确度和召回率上达到了最先进的结果。最后，我们的方法在现实模拟中对不同程度的子采样表现出稳健性，而其他方法在更高的子采样率下表现较差。 

---
# Learning The Minimum Action Distance 

**Title (ZH)**: 学习最小动作距离 

**Authors**: Lorenzo Steccanella, Joshua B. Evans, Özgür Şimşek, Anders Jonsson  

**Link**: [PDF](https://arxiv.org/pdf/2506.09276)  

**Abstract**: This paper presents a state representation framework for Markov decision processes (MDPs) that can be learned solely from state trajectories, requiring neither reward signals nor the actions executed by the agent. We propose learning the minimum action distance (MAD), defined as the minimum number of actions required to transition between states, as a fundamental metric that captures the underlying structure of an environment. MAD naturally enables critical downstream tasks such as goal-conditioned reinforcement learning and reward shaping by providing a dense, geometrically meaningful measure of progress. Our self-supervised learning approach constructs an embedding space where the distances between embedded state pairs correspond to their MAD, accommodating both symmetric and asymmetric approximations. We evaluate the framework on a comprehensive suite of environments with known MAD values, encompassing both deterministic and stochastic dynamics, as well as discrete and continuous state spaces, and environments with noisy observations. Empirical results demonstrate that the proposed approach not only efficiently learns accurate MAD representations across these diverse settings but also significantly outperforms existing state representation methods in terms of representation quality. 

**Abstract (ZH)**: 本文提出了一种仅从状态轨迹学习马尔可夫决策过程（MDPs）状态表示的框架，无需奖励信号或智能体执行的动作。我们提出学习最小动作距离（MAD），定义为在两个状态之间进行转换所需的最小动作数量，作为捕获环境潜在结构的基本度量。MAD 自然地支持目标条件强化学习和奖励塑形等关键下游任务，提供了一个密集的、几何意义上具有意义的进步度量。我们提出了一种自监督学习方法，构建了嵌入空间，其中嵌入状态对之间的距离对应于它们的MAD，既支持对称近似也支持非对称近似。我们在包含已知MAD值的广泛环境套件上进行了评估，这些环境涵盖了确定性和随机动力学，以及离散和连续状态空间，还包括具有噪声观测值的环境。实验证明，所提出的方法不仅在这些多样化的环境中高效地学习了准确的MAD表示，而且在表示质量上显著优于现有状态表示方法。 

---
# A Multi-Armed Bandit Framework for Online Optimisation in Green Integrated Terrestrial and Non-Terrestrial Networks 

**Title (ZH)**: 基于多臂bandit框架的绿色集成 terrestrial 和非terrestrial 网络的在线优化 

**Authors**: Henri Alam, Antonio de Domenico, Tareq Si Salem, Florian Kaltenberger  

**Link**: [PDF](https://arxiv.org/pdf/2506.09268)  

**Abstract**: Integrated terrestrial and non-terrestrial network (TN-NTN) architectures offer a promising solution for expanding coverage and improving capacity for the network. While non-terrestrial networks (NTNs) are primarily exploited for these specific reasons, their role in alleviating terrestrial network (TN) load and enabling energy-efficient operation has received comparatively less attention. In light of growing concerns associated with the densification of terrestrial deployments, this work aims to explore the potential of NTNs in supporting a more sustainable network. In this paper, we propose a novel online optimisation framework for integrated TN-NTN architectures, built on a multi-armed bandit (MAB) formulation and leveraging the Bandit-feedback Constrained Online Mirror Descent (BCOMD) algorithm. Our approach adaptively optimises key system parameters--including bandwidth allocation, user equipment (UE) association, and macro base station (MBS) shutdown--to balance network capacity and energy efficiency in real time. Extensive system-level simulations over a 24-hour period show that our framework significantly reduces the proportion of unsatisfied UEs during peak hours and achieves up to 19% throughput gains and 5% energy savings in low-traffic periods, outperforming standard network settings following 3GPP recommendations. 

**Abstract (ZH)**: 集成陆地和非陆地网络(TN-NTN)架构提供了扩展覆盖范围和提高网络容量的有希望的解决方案。尽管非陆地网络(NTNs)主要用于这些特定目的，但它们在缓解陆地网络(TNs)负载和实现节能操作方面的角色受到了较少关注。鉴于对陆地部署密集化的担忧不断增加，本工作旨在探讨NTNs在支持可持续网络方面的潜力。在本文中，我们提出了一种基于多臂 bandit (MAB) 表述并利用 Bandit-feedback Constrained Online Mirror Descent (BCOMD) 算法的新型在线优化框架，以实现集成TN-NTN架构，我们的方法实现实时平衡网络容量和节能。在24小时的系统级仿真中，我们的框架在高峰时段显著减少了未满足用户设备的比例，并在低流量时段实现了高达19%的吞吐量增益和5%的节能效果，优于遵循3GPP建议的标准网络设置。 

---
# Self-Anchored Attention Model for Sample-Efficient Classification of Prosocial Text Chat 

**Title (ZH)**: 基于自我锚定注意力模型的高效分类亲社会文本聊天 

**Authors**: Zhuofang Li, Rafal Kocielnik, Fereshteh Soltani, Penphob, Boonyarungsrit, Animashree Anandkumar, R. Michael Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2506.09259)  

**Abstract**: Millions of players engage daily in competitive online games, communicating through in-game chat. Prior research has focused on detecting relatively small volumes of toxic content using various Natural Language Processing (NLP) techniques for the purpose of moderation. However, recent studies emphasize the importance of detecting prosocial communication, which can be as crucial as identifying toxic interactions. Recognizing prosocial behavior allows for its analysis, rewarding, and promotion. Unlike toxicity, there are limited datasets, models, and resources for identifying prosocial behaviors in game-chat text. In this work, we employed unsupervised discovery combined with game domain expert collaboration to identify and categorize prosocial player behaviors from game chat. We further propose a novel Self-Anchored Attention Model (SAAM) which gives 7.9% improvement compared to the best existing technique. The approach utilizes the entire training set as "anchors" to help improve model performance under the scarcity of training data. This approach led to the development of the first automated system for classifying prosocial behaviors in in-game chats, particularly given the low-resource settings where large-scale labeled data is not available. Our methodology was applied to one of the most popular online gaming titles - Call of Duty(R): Modern Warfare(R)II, showcasing its effectiveness. This research is novel in applying NLP techniques to discover and classify prosocial behaviors in player in-game chat communication. It can help shift the focus of moderation from solely penalizing toxicity to actively encouraging positive interactions on online platforms. 

**Abstract (ZH)**: 大规模游戏玩家每日参与竞争性在线游戏，并通过游戏内聊天进行交流。先前的研究专注于使用各种自然语言处理（NLP）技术检测相对少量的有毒内容以进行管理。然而，近期的研究强调了检测亲社会交流的重要性，这种交流与识别有毒互动同样重要。识别亲社会行为使其能够被分析、奖励和推广。与毒性识别相比，游戏聊天文本中识别亲社会行为的数据集、模型和资源有限。在本工作中，我们结合无监督发现与游戏领域专家合作，从游戏聊天中识别和分类玩家的亲社会行为。进一步提出了一种新颖的自我锚定注意力模型（SAAM），相比现有最佳技术提高了7.9%。该方法利用整个训练集作为“锚点”来帮助改善在数据稀缺情况下的模型性能。该方法开发了第一个自动化的系统用于分类游戏内聊天中的亲社会行为，特别是在大规模标注数据不可用的低资源环境下。该方法被应用于最受欢迎的在线游戏之一——使命召唤®：现代战争®II，展示了其效果。这项研究是将NLP技术应用于发现和分类玩家游戏内聊天中的亲社会行为的先驱工作，有助于将管理的重点从仅仅惩罚毒性转向积极鼓励在线平台上的正面互动。 

---
# Extrapolation by Association: Length Generalization Transfer in Transformers 

**Title (ZH)**: 关联外推：Transformer中的长度泛化迁移学习 

**Authors**: Ziyang Cai, Nayoung Lee, Avi Schwarzschild, Samet Oymak, Dimitris Papailiopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.09251)  

**Abstract**: Transformer language models have demonstrated impressive generalization capabilities in natural language domains, yet we lack a fine-grained understanding of how such generalization arises. In this paper, we investigate length generalization--the ability to extrapolate from shorter to longer inputs--through the lens of \textit{task association}. We find that length generalization can be \textit{transferred} across related tasks. That is, training a model with a longer and related auxiliary task can lead it to generalize to unseen and longer inputs from some other target task. We demonstrate this length generalization transfer across diverse algorithmic tasks, including arithmetic operations, string transformations, and maze navigation. Our results show that transformer models can inherit generalization capabilities from similar tasks when trained jointly. Moreover, we observe similar transfer effects in pretrained language models, suggesting that pretraining equips models with reusable computational scaffolding that facilitates extrapolation in downstream settings. Finally, we provide initial mechanistic evidence that length generalization transfer correlates with the re-use of the same attention heads between the tasks. Together, our findings deepen our understanding of how transformers generalize to out-of-distribution inputs and highlight the compositional reuse of inductive structure across tasks. 

**Abstract (ZH)**: 基于任务关联的Transformer语言模型的长度泛化迁移研究 

---
# Robust Noise Attenuation via Adaptive Pooling of Transformer Outputs 

**Title (ZH)**: 自适应排序变换器输出的稳健噪声抑制 

**Authors**: Greyson Brothers  

**Link**: [PDF](https://arxiv.org/pdf/2506.09215)  

**Abstract**: We investigate the design of pooling methods used to summarize the outputs of transformer embedding models, primarily motivated by reinforcement learning and vision applications. This work considers problems where a subset of the input vectors contains requisite information for a downstream task (signal) while the rest are distractors (noise). By framing pooling as vector quantization with the goal of minimizing signal loss, we demonstrate that the standard methods used to aggregate transformer outputs, AvgPool, MaxPool, and ClsToken, are vulnerable to performance collapse as the signal-to-noise ratio (SNR) of inputs fluctuates. We then show that an attention-based adaptive pooling method can approximate the signal-optimal vector quantizer within derived error bounds for any SNR. Our theoretical results are first validated by supervised experiments on a synthetic dataset designed to isolate the SNR problem, then generalized to standard relational reasoning, multi-agent reinforcement learning, and vision benchmarks with noisy observations, where transformers with adaptive pooling display superior robustness across tasks. 

**Abstract (ZH)**: 我们考察了用于总结变压器嵌入模型输出的池化方法的设计，主要受强化学习和视觉应用的启发。本文考虑了一种情况，即输入向量中的子集包含了下游任务所需的信息（信号），而其余的则为干扰信息（噪声）。通过将池化视为向量量化，并以最小化信号丢失为目 标，我们证明了标准的变压器输出聚合方法，如 AvgPool、MaxPool 和 ClsToken，在输入信噪比（SNR）波动时容易导致性能崩溃。然后我们展示了基于注意力的自适应池化方法可以在任何信噪比下近似信号优化的向量量化器。我们的理论结果首先通过在合成数据集上进行监督实验进行验证，该数据集旨在孤立信噪比问题，随后泛化到标准关系推理、多智能体强化学习和具有嘈杂观察的视觉基准任务，其中使用自适应池化的方法在各种任务上表现出更强的鲁棒性。 

---
# SimClass: A Classroom Speech Dataset Generated via Game Engine Simulation For Automatic Speech Recognition Research 

**Title (ZH)**: SimClass: 通过游戏引擎模拟生成的课堂语音数据集及其在自动语音识别研究中的应用 

**Authors**: Ahmed Adel Attia, Jing Liu, Carl Espy-Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2506.09206)  

**Abstract**: The scarcity of large-scale classroom speech data has hindered the development of AI-driven speech models for education. Public classroom datasets remain limited, and the lack of a dedicated classroom noise corpus prevents the use of standard data augmentation techniques.
In this paper, we introduce a scalable methodology for synthesizing classroom noise using game engines, a framework that extends to other domains. Using this methodology, we present SimClass, a dataset that includes both a synthesized classroom noise corpus and a simulated classroom speech dataset. The speech data is generated by pairing a public children's speech corpus with YouTube lecture videos to approximate real classroom interactions in clean conditions. Our experiments on clean and noisy speech demonstrate that SimClass closely approximates real classroom speech, making it a valuable resource for developing robust speech recognition and enhancement models. 

**Abstract (ZH)**: 大规模课堂语音数据的稀缺性阻碍了基于AI的教育语音模型的发展。公开的课堂数据集仍然有限，缺乏专门的课堂噪声语料库使得标准数据增强技术无法使用。
在本文中，我们介绍了一种使用游戏引擎合成课堂噪声的可扩展方法，该框架适用于其他领域。利用这种方法，我们呈现了SimClass数据集，该数据集包括合成课堂噪声语料库和模拟课堂语音数据集。语音数据是通过将一个公开的儿童语音语料库与YouTube讲座视频配对生成的，以在干净条件下逼近真实课堂互动。我们的实验证明，在干净和嘈杂的语音上，SimClass 都能逼近真实课堂语音，使其成为开发稳健的语音识别和增强模型的重要资源。 

---
# A Topological Improvement of the Overall Performance of Sparse Evolutionary Training: Motif-Based Structural Optimization of Sparse MLPs Project 

**Title (ZH)**: 基于动力学优化的稀疏演化训练整体性能拓扑改进：稀疏MLP结构的模式基优化 

**Authors**: Xiaotian Chen, Hongyun Liu, Seyed Sahand Mohammadi Ziabari  

**Link**: [PDF](https://arxiv.org/pdf/2506.09204)  

**Abstract**: Deep Neural Networks (DNNs) have been proven to be exceptionally effective and have been applied across diverse domains within deep learning. However, as DNN models increase in complexity, the demand for reduced computational costs and memory overheads has become increasingly urgent. Sparsity has emerged as a leading approach in this area. The robustness of sparse Multi-layer Perceptrons (MLPs) for supervised feature selection, along with the application of Sparse Evolutionary Training (SET), illustrates the feasibility of reducing computational costs without compromising accuracy. Moreover, it is believed that the SET algorithm can still be improved through a structural optimization method called motif-based optimization, with potential efficiency gains exceeding 40% and a performance decline of under 4%. This research investigates whether the structural optimization of Sparse Evolutionary Training applied to Multi-layer Perceptrons (SET-MLP) can enhance performance and to what extent this improvement can be achieved. 

**Abstract (ZH)**: 深神经网络（DNNs）已被证明在深度学习的各个领域内表现出色且应用广泛。然而，随着DNN模型复杂性的增加，降低计算成本和内存开销的需求愈发迫切。稀疏性已成为这一领域的主导方法之一。基于监督特征选择的稀疏多层感知机（MLPs）的稳健性以及稀疏进化训练（SET）的应用表明，在不牺牲准确性的前提下削减计算成本的可能性。此外，通过一种称为图样基于优化的结构优化方法，SET算法仍有改进空间，预期效率提升可达40%以上，性能下降不超过4%。本研究旨在探讨将结构优化应用于多层感知机的稀疏进化训练（SET-MLP）是否能提高性能，以及这种改进的程度。 

---
# Policy-Based Trajectory Clustering in Offline Reinforcement Learning 

**Title (ZH)**: 基于策略的轨迹聚类在离线强化学习中 

**Authors**: Hao Hu, Xinqi Wang, Simon Shaolei Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.09202)  

**Abstract**: We introduce a novel task of clustering trajectories from offline reinforcement learning (RL) datasets, where each cluster center represents the policy that generated its trajectories. By leveraging the connection between the KL-divergence of offline trajectory distributions and a mixture of policy-induced distributions, we formulate a natural clustering objective. To solve this, we propose Policy-Guided K-means (PG-Kmeans) and Centroid-Attracted Autoencoder (CAAE). PG-Kmeans iteratively trains behavior cloning (BC) policies and assigns trajectories based on policy generation probabilities, while CAAE resembles the VQ-VAE framework by guiding the latent representations of trajectories toward the vicinity of specific codebook entries to achieve clustering. Theoretically, we prove the finite-step convergence of PG-Kmeans and identify a key challenge in offline trajectory clustering: the inherent ambiguity of optimal solutions due to policy-induced conflicts, which can result in multiple equally valid but structurally distinct clusterings. Experimentally, we validate our methods on the widely used D4RL dataset and custom GridWorld environments. Our results show that both PG-Kmeans and CAAE effectively partition trajectories into meaningful clusters. They offer a promising framework for policy-based trajectory clustering, with broad applications in offline RL and beyond. 

**Abstract (ZH)**: 一种新的离线强化学习轨迹聚类任务及其方法：政策引导的K-means（PG-Kmeans）和中心吸引自编码器（CAAE） 

---
# Graph Attention-based Decentralized Actor-Critic for Dual-Objective Control of Multi-UAV Swarms 

**Title (ZH)**: 基于图注意力的去中心化演员-评论家方法及其在多无人机群双目标控制中的应用 

**Authors**: Haoran Peng, Ying-Jun Angela Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09195)  

**Abstract**: This research focuses on optimizing multi-UAV systems with dual objectives: maximizing service coverage as the primary goal while extending battery lifetime as the secondary objective. We propose a Graph Attention-based Decentralized Actor-Critic (GADC) to optimize the dual objectives. The proposed approach leverages a graph attention network to process UAVs' limited local observation and reduce the dimension of the environment states. Subsequently, an actor-double-critic network is developed to manage dual policies for joint objective optimization. The proposed GADC uses a Kullback-Leibler (KL) divergence factor to balance the tradeoff between coverage performance and battery lifetime in the multi-UAV system. We assess the scalability and efficiency of GADC through comprehensive benchmarking against state-of-the-art methods, considering both theory and experimental aspects. Extensive testing in both ideal settings and NVIDIA Sionna's realistic ray tracing environment demonstrates GADC's superior performance. 

**Abstract (ZH)**: 基于图注意力的解耦actor-critic方法优化多无人机系统的双重目标：最大化服务覆盖与延长电池寿命 

---
# Integration of Contrastive Predictive Coding and Spiking Neural Networks 

**Title (ZH)**: 对比预测编码与尖峰神经网络的集成 

**Authors**: Emirhan Bilgiç, Neslihan Serap Şengör, Namık Berk Yalabık, Yavuz Selim İşler, Aykut Görkem Gelen, Rahmi Elibol  

**Link**: [PDF](https://arxiv.org/pdf/2506.09194)  

**Abstract**: This study examines the integration of Contrastive Predictive Coding (CPC) with Spiking Neural Networks (SNN). While CPC learns the predictive structure of data to generate meaningful representations, SNN mimics the computational processes of biological neural systems over time. In this study, the goal is to develop a predictive coding model with greater biological plausibility by processing inputs and outputs in a spike-based system. The proposed model was tested on the MNIST dataset and achieved a high classification rate in distinguishing positive sequential samples from non-sequential negative samples. The study demonstrates that CPC can be effectively combined with SNN, showing that an SNN trained for classification tasks can also function as an encoding mechanism. Project codes and detailed results can be accessed on our GitHub page: this https URL 

**Abstract (ZH)**: 本研究考察了对比预测编码（CPC）与尖峰神经网络（SNN）的结合。在CPC学习数据的预测结构以生成有意义的表示的同时，SNN模拟了生物神经系统的计算过程。本研究的目标是通过基于尖峰的系统处理输入和输出来开发具有更强生物合理性的预测编码模型。所提出的方法在MNIST数据集上进行了测试，并实现了高分类率，能够区分正序列样本和非序列负样本。研究表明，CPC可以有效地结合到SNN中，表明SNN在训练分类任务时也可以作为编码机制。项目代码和详细结果可访问我们的GitHub页面：this https URL 

---
# Multi-Task Reward Learning from Human Ratings 

**Title (ZH)**: 多任务奖励学习从人类评价 

**Authors**: Mingkang Wu, Devin White, Evelyn Rose, Vernon Lawhern, Nicholas R Waytowich, Yongcan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09183)  

**Abstract**: Reinforcement learning from human feeback (RLHF) has become a key factor in aligning model behavior with users' goals. However, while humans integrate multiple strategies when making decisions, current RLHF approaches often simplify this process by modeling human reasoning through isolated tasks such as classification or regression. In this paper, we propose a novel reinforcement learning (RL) method that mimics human decision-making by jointly considering multiple tasks. Specifically, we leverage human ratings in reward-free environments to infer a reward function, introducing learnable weights that balance the contributions of both classification and regression models. This design captures the inherent uncertainty in human decision-making and allows the model to adaptively emphasize different strategies. We conduct several experiments using synthetic human ratings to validate the effectiveness of the proposed approach. Results show that our method consistently outperforms existing rating-based RL methods, and in some cases, even surpasses traditional RL approaches. 

**Abstract (ZH)**: 基于人类反馈的强化学习（RLHF）已成为使模型行为与用户目标相一致的关键因素。然而，尽管人类在做决策时会整合多种策略，当前的RLHF方法往往通过分类或回归等孤立任务简化这一过程来建模人类推理。在本文中，我们提出了一种新的强化学习（RL）方法，通过同时考虑多个任务来模仿人类的决策过程。具体而言，我们利用奖励免费环境中的人类评分来推断奖励函数，并引入可学习的权重来平衡分类和回归模型的贡献。这种设计捕捉了人类决策过程中的固有不确定性，并允许模型适应性地强调不同的策略。我们使用合成的人类评分进行了多项实验，以验证所提出方法的有效性。结果表明，我们的方法在多个方面都优于现有基于评分的RL方法，并在某些情况下甚至超越了传统RL方法。 

---
# Estimating Visceral Adiposity from Wrist-Worn Accelerometry 

**Title (ZH)**: 从腕戴加速度计估计内脏脂肪量 

**Authors**: James R. Williamson, Andrew Alini, Brian A. Telfer, Adam W. Potter, Karl E. Friedl  

**Link**: [PDF](https://arxiv.org/pdf/2506.09167)  

**Abstract**: Visceral adipose tissue (VAT) is a key marker of both metabolic health and habitual physical activity (PA). Excess VAT is highly correlated with type 2 diabetes and insulin resistance. The mechanistic basis for this pathophysiology relates to overloading the liver with fatty acids. VAT is also a highly labile fat depot, with increased turnover stimulated by catecholamines during exercise. VAT can be measured with sophisticated imaging technologies, but can also be inferred directly from PA. We tested this relationship using National Health and Nutrition Examination Survey (NHANES) data from 2011-2014, for individuals aged 20-60 years with 7 days of accelerometry data (n=2,456 men; 2,427 women) [1]. Two approaches were used for estimating VAT from activity. The first used engineered features based on movements during gait and sleep, and then ridge regression to map summary statistics of these features into a VAT estimate. The second approach used deep neural networks trained on 24 hours of continuous accelerometry. A foundation model first mapped each 10s frame into a high-dimensional feature vector. A transformer model then mapped each day's feature vector time series into a VAT estimate, which were averaged over multiple days. For both approaches, the most accurate estimates were obtained with the addition of covariate information about subject demographics and body measurements. The best performance was obtained by combining the two approaches, resulting in VAT estimates with correlations of r=0.86. These findings demonstrate a strong relationship between PA and VAT and, by extension, between PA and metabolic health risks. 

**Abstract (ZH)**: 内脏脂肪组织（VAT）是代谢健康和习惯性体力活动（PA）的关键标志物。过多的VAT与2型糖尿病和胰岛素抵抗高度相关。这种病理生理机制的基础在于向肝脏过量提供脂肪酸。VAT也是一种高度可变的脂肪储存部位，在运动期间由儿茶酚胺刺激其周转率的增加。VAT可以使用复杂的成像技术进行测量，但也可以通过体力活动直接推断。我们使用2011-2014年全国健康和营养 Examination Survey (NHANES) 数据（年龄在20-60岁之间，男性2,456人；女性2,427人，有7天的加速度计数据）测试了这种关系 [1]。估测VAT的两种方法均使用了体力活动。第一种方法基于行走和睡眠期间的运动工程特征，并使用岭回归将这些特征的摘要统计映射到VAT估计值。第二种方法使用了在24小时连续加速度计数据上训练的深度神经网络。基础模型将每个10秒帧映射到高维特征向量。然后，变压器模型将每个特征向量时间序列映射到VAT估计值，并在多天内进行平均。对于这两种方法，最准确的估计值是通过添加关于受试者的人口统计学和身体测量的协变量信息获得的。通过结合两种方法，性能最佳，结果的VAT估计值的相关性为r=0.86。这些发现表明，体力活动与VAT之间存在强烈关系，进而推断体力活动与代谢健康风险之间的关系。 

---
# Understanding Human-AI Trust in Education 

**Title (ZH)**: 理解教育中的人工智能信任问题 

**Authors**: Griffin Pitts, Sanaz Motamedi  

**Link**: [PDF](https://arxiv.org/pdf/2506.09160)  

**Abstract**: As AI chatbots become increasingly integrated in education, students are turning to these systems for guidance, feedback, and information. However, the anthropomorphic characteristics of these chatbots create ambiguity regarding whether students develop trust toward them as they would a human peer or instructor, based in interpersonal trust, or as they would any other piece of technology, based in technology trust. This ambiguity presents theoretical challenges, as interpersonal trust models may inappropriately ascribe human intentionality and morality to AI, while technology trust models were developed for non-social technologies, leaving their applicability to anthropomorphic systems unclear. To address this gap, we investigate how human-like and system-like trusting beliefs comparatively influence students' perceived enjoyment, trusting intention, behavioral intention to use, and perceived usefulness of an AI chatbot - factors associated with students' engagement and learning outcomes. Through partial least squares structural equation modeling, we found that human-like and system-like trust significantly influenced student perceptions, with varied effects. Human-like trust more strongly predicted trusting intention, while system-like trust better predicted behavioral intention and perceived usefulness. Both had similar effects on perceived enjoyment. Given the partial explanatory power of each type of trust, we propose that students develop a distinct form of trust with AI chatbots (human-AI trust) that differs from human-human and human-technology models of trust. Our findings highlight the need for new theoretical frameworks specific to human-AI trust and offer practical insights for fostering appropriately calibrated trust, which is critical for the effective adoption and pedagogical impact of AI in education. 

**Abstract (ZH)**: 随着AI聊天机器人在教育中的日益集成，学生开始依赖这些系统寻求指导、反馈和信息。然而，这些聊天机器人的拟人特性使得学生是更像对其人类同伴或教师产生人际信任，还是基于技术信任对其产生信任变得模糊。这种模糊性提出了理论挑战，因为人际信任模型可能会不合适地赋予AI人类意图和道德，而技术信任模型是为非社交技术开发的，其适用于拟人系统的效果尚不清楚。为了解决这一差距，我们探讨了拟人信任信念和系统信任信念在比较上如何影响学生对AI聊天机器人的感知愉悦度、信任意图、使用意图和感知有用性，这些因素与学生的参与度和学习成果相关。通过部分最小二乘结构方程建模，我们发现拟人信任和系统信任显著影响了学生的态度，但影响效果有所不同。拟人信任更强烈地预测了信任意图，而系统信任更好地预测了使用意图和感知有用性。两种信任在感知愉悦度上的影响相似。鉴于每种信任的解释力有限，我们认为学生形成了与人类-人类和人类-技术信任模型不同的对AI聊天机器人的信任形式（人类-AI信任）。我们的研究凸显了需要针对人类-AI信任的新理论框架的必要性，并提供了培养适当校准信任的实用见解，这对于AI在教育中的有效采用及其教育影响至关重要。 

---
# FAIRTOPIA: Envisioning Multi-Agent Guardianship for Disrupting Unfair AI Pipelines 

**Title (ZH)**: FAIRTOPIA: 构想多agent监护以颠覆不公平AIpipeline 

**Authors**: Athena Vakali, Ilias Dimitriadis  

**Link**: [PDF](https://arxiv.org/pdf/2506.09107)  

**Abstract**: AI models have become active decision makers, often acting without human supervision. The rapid advancement of AI technology has already caused harmful incidents that have hurt individuals and societies and AI unfairness in heavily criticized. It is urgent to disrupt AI pipelines which largely neglect human principles and focus on computational biases exploration at the data (pre), model(in), and deployment (post) processing stages. We claim that by exploiting the advances of agents technology, we will introduce cautious, prompt, and ongoing fairness watch schemes, under realistic, systematic, and human-centric fairness expectations. We envision agents as fairness guardians, since agents learn from their environment, adapt to new information, and solve complex problems by interacting with external tools and other systems. To set the proper fairness guardrails in the overall AI pipeline, we introduce a fairness-by-design approach which embeds multi-role agents in an end-to-end (human to AI) synergetic scheme. Our position is that we may design adaptive and realistic AI fairness frameworks, and we introduce a generalized algorithm which can be customized to the requirements and goals of each AI decision making scenario. Our proposed, so called FAIRTOPIA framework, is structured over a three-layered architecture, which encapsulates the AI pipeline inside an agentic guardian and a knowledge-based, self-refining layered scheme. Based on our proposition, we enact fairness watch in all of the AI pipeline stages, under robust multi-agent workflows, which will inspire new fairness research hypothesis, heuristics, and methods grounded in human-centric, systematic, interdisciplinary, socio-technical principles. 

**Abstract (ZH)**: 基于代理技术的设计导向公平性框架：FAIRTOPIA 

---
# MetaTT: A Global Tensor-Train Adapter for Parameter-Efficient Fine-Tuning 

**Title (ZH)**: MetaTT：一种全局张量-训练适配器，用于参数高效微调 

**Authors**: Javier Lopez-Piqueres, Pranav Deshpande, Archan Ray, Mattia J. Villani, Marco Pistoia, Niraj Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.09105)  

**Abstract**: We present MetaTT, a unified Tensor Train (TT) adapter framework for global low-rank fine-tuning of pre-trained transformers. Unlike LoRA, which fine-tunes each weight matrix independently, MetaTT uses a single shared TT to factorize all transformer sub-modules -- query, key, value, projection, and feed-forward layers -- by indexing the structural axes like layer and matrix type, and optionally heads and tasks. For a given rank, while LoRA adds parameters proportional to the product across modes, MetaTT only adds parameters proportional to the sum across modes leading to a significantly compressed final adapter. Our benchmarks compare MetaTT with LoRA along with recent state-of-the-art matrix and tensor decomposition based fine-tuning schemes. We observe that when tested on standard language modeling benchmarks, MetaTT leads to the most reduction in the parameters while maintaining similar accuracy to LoRA and even outperforming other tensor-based methods. Unlike CP or other rank-factorizations, the TT ansatz benefits from mature optimization routines -- e.g., DMRG-style rank adaptive minimization in addition to Adam, which we find simplifies training. Because new modes can be appended cheaply, MetaTT naturally extends to shared adapters across many tasks without redesigning the core tensor. 

**Abstract (ZH)**: MetaTT：一种统一的张量列车适配器框架，用于预训练变换器的全局低秩微调 

---
# Revolutionizing Clinical Trials: A Manifesto for AI-Driven Transformation 

**Title (ZH)**: 革新临床试验：基于AI的转型宣言 

**Authors**: Mihaela van der Schaar, Richard Peck, Eoin McKinney, Jim Weatherall, Stuart Bailey, Justine Rochon, Chris Anagnostopoulos, Pierre Marquet, Anthony Wood, Nicky Best, Harry Amad, Julianna Piskorz, Krzysztof Kacprzyk, Rafik Salama, Christina Gunther, Francesca Frau, Antoine Pugeat, Ramon Hernandez  

**Link**: [PDF](https://arxiv.org/pdf/2506.09102)  

**Abstract**: This manifesto represents a collaborative vision forged by leaders in pharmaceuticals, consulting firms, clinical research, and AI. It outlines a roadmap for two AI technologies - causal inference and digital twins - to transform clinical trials, delivering faster, safer, and more personalized outcomes for patients. By focusing on actionable integration within existing regulatory frameworks, we propose a way forward to revolutionize clinical research and redefine the gold standard for clinical trials using AI. 

**Abstract (ZH)**: this manifesto代表了制药业、咨询公司、临床研究和AI领域领导者们的合作愿景，概述了因果推断和数字孪生两种AI技术在临床试验中的应用路线图，旨在实现更快速、更安全和更具个性化的患者结果。通过专注于现有监管框架内的可操作性整合，我们提出了一条通往利用AI革命临床研究并重新定义临床试验黄金标准的道路。 

---
# Feature Shift Localization Network 

**Title (ZH)**: 特征转移定位网络 

**Authors**: Míriam Barrabés, Daniel Mas Montserrat, Kapal Dev, Alexander G. Ioannidis  

**Link**: [PDF](https://arxiv.org/pdf/2506.09101)  

**Abstract**: Feature shifts between data sources are present in many applications involving healthcare, biomedical, socioeconomic, financial, survey, and multi-sensor data, among others, where unharmonized heterogeneous data sources, noisy data measurements, or inconsistent processing and standardization pipelines can lead to erroneous features. Localizing shifted features is important to address the underlying cause of the shift and correct or filter the data to avoid degrading downstream analysis. While many techniques can detect distribution shifts, localizing the features originating them is still challenging, with current solutions being either inaccurate or not scalable to large and high-dimensional datasets. In this work, we introduce the Feature Shift Localization Network (FSL-Net), a neural network that can localize feature shifts in large and high-dimensional datasets in a fast and accurate manner. The network, trained with a large number of datasets, learns to extract the statistical properties of the datasets and can localize feature shifts from previously unseen datasets and shifts without the need for re-training. The code and ready-to-use trained model are available at this https URL. 

**Abstract (ZH)**: 特征在大规模高维数据源之间的移位在当地化定位中的网络方法：Feature Shift Localization Network (FSL-Net) 在大规模高维数据中的快速准确实现 

---
# Merging Smarter, Generalizing Better: Enhancing Model Merging on OOD Data 

**Title (ZH)**: 更聪明地融合，更好地泛化：在OOD数据上增强模型融合 

**Authors**: Bingjie Zhang, Hongkang Li, Changlong Shi, Guowei Rong, He Zhao, Dongsheng Wang, Dandan Guo, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09093)  

**Abstract**: Multi-task learning (MTL) concurrently trains a model on diverse task datasets to exploit common features, thereby improving overall performance across the tasks. Recent studies have dedicated efforts to merging multiple independent model parameters into a unified model for MTL, thus circumventing the need for training data and expanding the scope of applicable scenarios of MTL. However, current approaches to model merging predominantly concentrate on enhancing performance within in-domain (ID) datasets, often overlooking their efficacy on out-of-domain (OOD) datasets. In this work, we proposed LwPTV (Layer-wise Pruning Task Vector) by building a saliency score, measuring the redundancy of parameters in task vectors. Designed in this way ours can achieve mask vector for each task and thus perform layer-wise pruning on the task vectors, only keeping the pre-trained model parameters at the corresponding layer in merged model. Owing to its flexibility, our method can be seamlessly integrated with most of existing model merging methods to improve their performance on OOD tasks. Extensive experiments demonstrate that the application of our method results in substantial enhancements in OOD performance while preserving the ability on ID tasks. 

**Abstract (ZH)**: 基于层-wise剪枝任务向量的多任务学习参数合并方法 

---
# Designing conflict-based communicative tasks in Teaching Chinese as a Foreign Language with ChatGPT 

**Title (ZH)**: 基于冲突的设计型交际任务在外语教学中文言文教学中的应用研究——以ChatGPT为例 

**Authors**: Xia Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09089)  

**Abstract**: In developing the teaching program for a course in Oral Expression in Teaching Chinese as a Foreign Language at the university level, the teacher designs communicative tasks based on conflicts to encourage learners to engage in interactive dynamics and develop their oral interaction skills. During the design of these tasks, the teacher uses ChatGPT to assist in finalizing the program. This article aims to present the key characteristics of the interactions between the teacher and ChatGPT during this program development process, as well as to examine the use of ChatGPT and its impacts in this specific context. 

**Abstract (ZH)**: 在开发高校对外汉语教学课程《口语表达》的教学计划时，教师基于冲突设计沟通任务，以鼓励学生参与互动动态，提升口语互动能力。在任务设计过程中，教师使用ChatGPT协助最终确定教学计划。本文旨在呈现教师与ChatGPT在该教学计划开发过程中的关键互动特征，并探讨在这一特定背景下ChatGPT的使用及其影响。 

---
# FinHEAR: Human Expertise and Adaptive Risk-Aware Temporal Reasoning for Financial Decision-Making 

**Title (ZH)**: FinHEAR: 人类专长与自适应风险意识时序推理在金融决策中的应用 

**Authors**: Jiaxiang Chen, Mingxi Zou, Zhuo Wang, Qifan Wang, Dongning Sun, Chi Zhang, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09080)  

**Abstract**: Financial decision-making presents unique challenges for language models, demanding temporal reasoning, adaptive risk assessment, and responsiveness to dynamic events. While large language models (LLMs) show strong general reasoning capabilities, they often fail to capture behavioral patterns central to human financial decisions-such as expert reliance under information asymmetry, loss-averse sensitivity, and feedback-driven temporal adjustment. We propose FinHEAR, a multi-agent framework for Human Expertise and Adaptive Risk-aware reasoning. FinHEAR orchestrates specialized LLM-based agents to analyze historical trends, interpret current events, and retrieve expert-informed precedents within an event-centric pipeline. Grounded in behavioral economics, it incorporates expert-guided retrieval, confidence-adjusted position sizing, and outcome-based refinement to enhance interpretability and robustness. Empirical results on curated financial datasets show that FinHEAR consistently outperforms strong baselines across trend prediction and trading tasks, achieving higher accuracy and better risk-adjusted returns. 

**Abstract (ZH)**: 金融决策为语言模型提出了独特的挑战，要求其进行时间推理、适应性风险评估以及对动态事件的响应。尽管大型语言模型（LLMs）展现了强大的一般推理能力，但它们往往无法捕捉到构成人类金融决策的关键行为模式，例如在信息不对称下的专家依赖、损失规避敏感性以及基于反馈的时间调整。我们提出了一种多代理框架FinHEAR，用于人类专业知识和适应性风险管理推理。FinHEAR协调基于LLM的专业代理，分析历史趋势、解释当前事件，并在以事件为中心的管道中检索专家指导的先例。基于行为经济学，FinHEAR整合了专家指导的检索、信心调整的仓位 sizing 和结果导向的优化，以增强可解释性和稳健性。在精心策划的金融数据集上的实证结果表明，FinHEAR在趋势预测和交易任务中均优于强基线模型，实现了更高的准确性和更好的风险调整收益。 

---
# ReStNet: A Reusable & Stitchable Network for Dynamic Adaptation on IoT Devices 

**Title (ZH)**: 可重用可拼接的动态适应物联网设备网络：ReStNet 

**Authors**: Maoyu Wang, Yao Lu, Jiaqi Nie, Zeyu Wang, Yun Lin, Qi Xuan, Guan Gui  

**Link**: [PDF](https://arxiv.org/pdf/2506.09066)  

**Abstract**: With the rapid development of deep learning, a growing number of pre-trained models have been publicly available. However, deploying these fixed models in real-world IoT applications is challenging because different devices possess heterogeneous computational and memory resources, making it impossible to deploy a single model across all platforms. Although traditional compression methods, such as pruning, quantization, and knowledge distillation, can improve efficiency, they become inflexible once applied and cannot adapt to changing resource constraints. To address these issues, we propose ReStNet, a Reusable and Stitchable Network that dynamically constructs a hybrid network by stitching two pre-trained models together. Implementing ReStNet requires addressing several key challenges, including how to select the optimal stitching points, determine the stitching order of the two pre-trained models, and choose an effective fine-tuning strategy. To systematically address these challenges and adapt to varying resource constraints, ReStNet determines the stitching point by calculating layer-wise similarity via Centered Kernel Alignment (CKA). It then constructs the hybrid model by retaining early layers from a larger-capacity model and appending deeper layers from a smaller one. To facilitate efficient deployment, only the stitching layer is fine-tuned. This design enables rapid adaptation to changing budgets while fully leveraging available resources. Moreover, ReStNet supports both homogeneous (CNN-CNN, Transformer-Transformer) and heterogeneous (CNN-Transformer) stitching, allowing to combine different model families flexibly. Extensive experiments on multiple benchmarks demonstrate that ReStNet achieve flexible accuracy-efficiency trade-offs at runtime while significantly reducing training cost. 

**Abstract (ZH)**: 基于ReStNet的可重用与缝合网络在物联网应用中的动态构建与优化 

---
# Llama-Affinity: A Predictive Antibody Antigen Binding Model Integrating Antibody Sequences with Llama3 Backbone Architecture 

**Title (ZH)**: llama-affinity: 一种结合抗体序列和 llama3 主干架构的预测性抗体抗原结合模型 

**Authors**: Delower Hossain, Ehsan Saghapour, Kevin Song, Jake Y. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09052)  

**Abstract**: Antibody-facilitated immune responses are central to the body's defense against pathogens, viruses, and other foreign invaders. The ability of antibodies to specifically bind and neutralize antigens is vital for maintaining immunity. Over the past few decades, bioengineering advancements have significantly accelerated therapeutic antibody development. These antibody-derived drugs have shown remarkable efficacy, particularly in treating cancer, SARS-CoV-2, autoimmune disorders, and infectious diseases. Traditionally, experimental methods for affinity measurement have been time-consuming and expensive. With the advent of artificial intelligence, in silico medicine has been revolutionized; recent developments in machine learning, particularly the use of large language models (LLMs) for representing antibodies, have opened up new avenues for AI-based design and improved affinity prediction. Herein, we present an advanced antibody-antigen binding affinity prediction model (LlamaAffinity), leveraging an open-source Llama 3 backbone and antibody sequence data sourced from the Observed Antibody Space (OAS) database. The proposed approach shows significant improvement over existing state-of-the-art (SOTA) methods (AntiFormer, AntiBERTa, AntiBERTy) across multiple evaluation metrics. Specifically, the model achieved an accuracy of 0.9640, an F1-score of 0.9643, a precision of 0.9702, a recall of 0.9586, and an AUC-ROC of 0.9936. Moreover, this strategy unveiled higher computational efficiency, with a five-fold average cumulative training time of only 0.46 hours, significantly lower than in previous studies. 

**Abstract (ZH)**: 抗体介导的免疫反应是机体对抗病原体、病毒和其他外来入侵者的中心机制。抗体特异性结合和中和抗原的能力对于维持免疫至关重要。过去几十年中，生物工程进步显著加速了治疗性抗体的研发。这些抗体衍生的药物在治疗癌症、SARS-CoV-2、自身免疫疾病和传染病方面展现了显著疗效。传统上，亲和力测量的实验方法耗时且昂贵。随着人工智能的兴起，计算医学得到了革命性的变革；特别是大规模语言模型（LLMs）在表示抗体方面的应用，为基于AI的设计和亲和力预测开辟了新的途径。本文提出了一种先进的抗体-抗原结合亲和力预测模型（LlamaAffinity），该模型基于开源的Llama 3架构，并利用Observed Antibody Space（OAS）数据库中的抗体序列数据。所提出的方法在多个评估指标上显著优于现有最先进的（SOTA）方法（AntiFormer, AntiBERTa, AntiBERTy）。具体来说，该模型实现了0.9640的准确率、0.9643的F1分数、0.9702的精确率、0.9586的召回率以及0.9936的AUC-ROC。此外，该策略展示了更高的计算效率，平均累积训练时间为0.46小时，远低于之前的研究。 

---
# RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling 

**Title (ZH)**: RuleReasoner: 基于领域意识动态采样的强化规则推理 

**Authors**: Yang Liu, Jiaqi Li, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.08672)  

**Abstract**: Rule-based reasoning has been acknowledged as one of the fundamental problems in reasoning, while deviations in rule formats, types, and complexity in real-world applications pose severe challenges. Recent studies have shown that large reasoning models (LRMs) have remarkable reasoning capabilities, and their performance is substantially enhanced by reinforcement learning (RL). However, it remains an open question whether small reasoning models (SRMs) can learn rule-based reasoning effectively with robust generalization across diverse tasks and domains. To address this, we introduce Reinforced Rule-based Reasoning, a.k.a. RuleReasoner, a simple yet effective method to conduct rule-based reasoning via a wide collection of curated tasks and a novel domain-aware dynamic sampling approach. Specifically, RuleReasoner resamples each training batch by updating the sampling weights of different domains based on historical rewards. This facilitates domain augmentation and flexible online learning schedules for RL, obviating the need for pre-hoc human-engineered mix-training recipes used in existing methods. Empirical evaluations on in-distribution (ID) and out-of-distribution (OOD) benchmarks reveal that RuleReasoner outperforms frontier LRMs by a significant margin ($\Delta$4.1% average points on eight ID tasks and $\Delta$10.4% average points on three OOD tasks over OpenAI-o1). Notably, our approach also exhibits higher computational efficiency compared to prior dynamic sampling methods for RL. 

**Abstract (ZH)**: 基于规则的推理强化学习方法：RuleReasoner 

---
