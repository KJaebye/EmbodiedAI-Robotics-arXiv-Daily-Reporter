# Sound and Solution-Complete CCBS 

**Title (ZH)**: 声学与解完备CCBS 

**Authors**: Alvin Combrink, Sabino Francesco Roselli, Martin Fabian  

**Link**: [PDF](https://arxiv.org/pdf/2508.16410)  

**Abstract**: Continuous-time Conflict Based-Search (CCBS) has long been viewed as the de-facto optimal solver for multi-agent path finding in continuous time (MAPFR). Recent findings, however, show that the original theoretical variant of CCBS can suffer from non-termination, while the widely used implementation can return sub-optimal solutions. We introduce an analytical framework that yields simple and sufficient conditions under which any CCBS-style algorithm is both sound, i.e., returns only optimal solutions, and solution complete, i.e., terminates on every solvable MAPFR instance. Investigating the publicly available implementation of CCBS reveals that it violates these conditions. Though this merely indicates that CCBS might be unsound, this indication is supported by counter-examples.
Leveraging the analytical framework, we propose a novel branching rule and prove that it satisfies the sufficient conditions, thereby restoring soundness and termination guarantees. Consequently, the resulting CCBS variant is both sound and solution complete, matching the guarantees of the discrete-time CBS for the first time in the continuous domain. We experimentally apply standard CCBS and CCBS under our branching rule to an example problem, with our branching rule returning a solution with lower sum-of-costs than standard CCBS. Because the branching rule largely only affects the branching step, it can be adopted as a drop-in replacement in existing code-bases, as we show in our provided implementation. Beyond CCBS, the analytical framework and termination criterion provide a systematic way to evaluate other CCBS-like MAPFR solvers and future extensions. 

**Abstract (ZH)**: 连续时间冲突基于搜索（CCBS）一直在连续时间多Agent路径寻找（MAPFR）问题中被视为实际上的最优求解器。然而，最新研究发现，原始的理论变体CCBS可能无法终止，而广泛使用的实现则可能会返回次优解。我们提出了一种分析框架，从而为任何CCBS风格的算法提供了简单且充分的条件，使其既能保证正确性，即仅返回最优解，又能保证完备性，即在每一个可解的MAPFR实例上都能终止。通过对公开的CCBS实现进行调查，我们发现它违反了这些条件。尽管这仅表明CCBS可能是不正确的，这一结论也由反例支持。借助分析框架，我们提出了一个新的分支规则，并证明该规则满足充分条件，从而恢复了正确性和终止性保证。因此，由此衍生的CCBS变体在连续域中首次与离散时间CBS的保证相匹配，既确保正确性又确保完备性。实验结果显示，在我们的分支规则下应用标准CCBS和CCBS，能够获得较低总成本的解。由于分支规则主要影响分支步骤，因此可以在现有的代码库中直接替换使用，正如我们提供的实现所示。除了CCBS之外，分析框架和终止准则还为评估其他CCBS类MAPFR求解器及其未来扩展提供了一种系统性方法。 

---
# Constraints-Guided Diffusion Reasoner for Neuro-Symbolic Learning 

**Title (ZH)**: 基于约束引导的扩散推理器促进神经符号学习 

**Authors**: Xuan Zhang, Zhijian Zhou, Weidi Xu, Yanting Miao, Chao Qu, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16524)  

**Abstract**: Enabling neural networks to learn complex logical constraints and fulfill symbolic reasoning is a critical challenge. Bridging this gap often requires guiding the neural network's output distribution to move closer to the symbolic constraints. While diffusion models have shown remarkable generative capability across various domains, we employ the powerful architecture to perform neuro-symbolic learning and solve logical puzzles. Our diffusion-based pipeline adopts a two-stage training strategy: the first stage focuses on cultivating basic reasoning abilities, while the second emphasizes systematic learning of logical constraints. To impose hard constraints on neural outputs in the second stage, we formulate the diffusion reasoner as a Markov decision process and innovatively fine-tune it with an improved proximal policy optimization algorithm. We utilize a rule-based reward signal derived from the logical consistency of neural outputs and adopt a flexible strategy to optimize the diffusion reasoner's policy. We evaluate our methodology on some classical symbolic reasoning benchmarks, including Sudoku, Maze, pathfinding and preference learning. Experimental results demonstrate that our approach achieves outstanding accuracy and logical consistency among neural networks. 

**Abstract (ZH)**: 使神经网络学习复杂逻辑约束并实现符号推理是一个关键挑战。通过引导神经网络的输出分布接近符号约束来弥合这一差距往往需要具备较高的指导能力。尽管扩散模型已经在多个领域展示了出色的生成能力，我们仍采用这一强健架构进行神经-符号学习并解决逻辑谜题。我们的基于扩散的管道采用两阶段训练策略：第一阶段专注于培养基本的推理能力，第二阶段则侧重系统学习逻辑约束。为了在第二阶段对神经输出施加硬约束，我们将扩散推理机形式化为马尔可夫决策过程，并创新性地使用改进的近端策略优化算法进行微调。我们利用基于逻辑一致性的规则奖励信号，并采用灵活策略优化扩散推理机的策略。我们在经典的符号推理基准测试上评估了该方法，包括数独、迷宫、路径finding和偏好学习。实验结果表明，我们的方法在神经网络中实现了卓越的准确性和逻辑一致性。 

---
# Modular Embedding Recomposition for Incremental Learning 

**Title (ZH)**: 模块化嵌入重组用于增量学习 

**Authors**: Aniello Panariello, Emanuele Frascaroli, Pietro Buzzega, Lorenzo Bonicelli, Angelo Porrello, Simone Calderara  

**Link**: [PDF](https://arxiv.org/pdf/2508.16463)  

**Abstract**: The advent of pre-trained Vision-Language Models (VLMs) has significantly transformed Continual Learning (CL), mainly due to their zero-shot classification abilities. Such proficiency makes VLMs well-suited for real-world applications, enabling robust performance on novel unseen classes without requiring adaptation. However, fine-tuning remains essential when downstream tasks deviate significantly from the pre-training domain. Prior CL approaches primarily focus on preserving the zero-shot capabilities of VLMs during incremental fine-tuning on a downstream task. We take a step further by devising an approach that transforms preservation into enhancement of the zero-shot capabilities of VLMs. Our approach, named MoDular Embedding Recomposition (MoDER), introduces a modular framework that trains multiple textual experts, each specialized in a single seen class, and stores them in a foundational hub. At inference time, for each unseen class, we query the hub and compose the retrieved experts to synthesize a refined prototype that improves classification. We show the effectiveness of our method across two popular zero-shot incremental protocols, Class-IL and MTIL, comprising a total of 14 datasets. The codebase is available at this https URL. 

**Abstract (ZH)**: 预训练视觉-语言模型的出现显著改变了持续学习，主要是由于它们的零样本分类能力。然而，当下游任务与预训练领域差异较大时，fine-tuning仍然是必不可少的。先前的持续学习方法主要关注在下游任务的增量fine-tuning过程中保留预训练模型的零样本能力。我们进一步提出了一种方法，将保留转化为增强预训练模型的零样本能力。我们提出的方法名为MoDular Embedding Recomposition（MoDER），引入了一种模块化框架，训练多个专门针对单一已见类别的文本专家，并将它们存储在一个基础枢纽中。在推理时，对于每个未见类别，查询枢纽并组合检索到的专家以合成一个改进分类的精炼原型。我们在两个流行的零样本增量协议Class-IL和MTIL上展示了该方法的有效性，共涉及14个数据集。代码库可在以下链接获取。 

---
# Causal Beam Selection for Reliable Initial Access in AI-driven Beam Management 

**Title (ZH)**: 基于因果性波束选择的可靠初始接入在AI驱动的波束管理中 

**Authors**: Nasir Khan, Asmaa Abdallah, Abdulkadir Celik, Ahmed M. Eltawil, Sinem Coleri  

**Link**: [PDF](https://arxiv.org/pdf/2508.16352)  

**Abstract**: Efficient and reliable beam alignment is a critical requirement for mmWave multiple-input multiple-output (MIMO) systems, especially in 6G and beyond, where communication must be fast, adaptive, and resilient to real-world uncertainties. Existing deep learning (DL)-based beam alignment methods often neglect the underlying causal relationships between inputs and outputs, leading to limited interpretability, poor generalization, and unnecessary beam sweeping overhead. In this work, we propose a causally-aware DL framework that integrates causal discovery into beam management pipeline. Particularly, we propose a novel two-stage causal beam selection algorithm to identify a minimal set of relevant inputs for beam prediction. First, causal discovery learns a Bayesian graph capturing dependencies between received power inputs and the optimal beam. Then, this graph guides causal feature selection for the DL-based classifier. Simulation results reveal that the proposed causal beam selection matches the performance of conventional methods while drastically reducing input selection time by 94.4% and beam sweeping overhead by 59.4% by focusing only on causally relevant features. 

**Abstract (ZH)**: 高效且可靠的波束对准对于毫米波多输入多输出(MIMO)系统至关重要，特别是在6G及更先进的通信系统中，通信必须快速、适应性强且能抵御现实世界的不确定性。现有的基于深度学习(DL)的波束对准方法往往忽略了输入与输出之间的潜在因果关系，导致解释性有限、泛化能力差以及不必要的波束扫掠开销。在本文中，我们提出了一种因果意识的DL框架，将因果发现集成到波束管理管道中。特别地，我们提出了一种新颖的两阶段因果波束选择算法，以识别波束预测的相关输入最小集合。首先，因果发现学习一个贝叶斯图，捕捉接收功率输入与最优波束之间的依赖关系。然后，该图指导基于DL的分类器的因果特征选择。仿真结果表明，提出的因果波束选择在仅关注因果相关特征的前提下，将输入选择时间减少了94.4%，波束扫掠开销减少了59.4%，同时匹配传统方法的性能。 

---
# The next question after Turing's question: Introducing the Grow-AI test 

**Title (ZH)**: 图灵问题之后的问题：Introducing the Grow-AI测试 

**Authors**: Alexandru Tugui  

**Link**: [PDF](https://arxiv.org/pdf/2508.16277)  

**Abstract**: This study aims to extend the framework for assessing artificial intelligence, called GROW-AI (Growth and Realization of Autonomous Wisdom), designed to answer the question "Can machines grow up?" -- a natural successor to the Turing Test. The methodology applied is based on a system of six primary criteria (C1-C6), each assessed through a specific "game", divided into four arenas that explore both the human dimension and its transposition into AI. All decisions and actions of the entity are recorded in a standardized AI Journal, the primary source for calculating composite scores. The assessment uses the prior expert method to establish initial weights, and the global score -- Grow Up Index -- is calculated as the arithmetic mean of the six scores, with interpretation on maturity thresholds. The results show that the methodology allows for a coherent and comparable assessment of the level of "growth" of AI entities, regardless of their type (robots, software agents, LLMs). The multi-game structure highlights strengths and vulnerable areas, and the use of a unified journal guarantees traceability and replicability in the evaluation. The originality of the work lies in the conceptual transposition of the process of "growing" from the human world to that of artificial intelligence, in an integrated testing format that combines perspectives from psychology, robotics, computer science, and ethics. Through this approach, GROW-AI not only measures performance but also captures the evolutionary path of an AI entity towards maturity. 

**Abstract (ZH)**: 本研究旨在扩展一种名为GROW-AI（自主智慧的增长与实现）的框架，该框架旨在回答“机器能否成长？”这一问题——这是图灵测试的自然延续。所采用的方法基于六个主要标准（C1-C6）体系，每个标准通过特定的“游戏”进行评估，分为四个竞技场，既探索人类维度又将其转化为AI。实体的所有决策和行为记录在标准化的AI日志中，这是计算综合得分的主要来源。评估方法采用先验专家法确定初始权重，整体得分——成长指数——是六个得分的算术平均值，并通过成熟度阈值进行解释。研究结果表明，该方法允许对不同类型（机器人、软件代理、大语言模型）的AI实体的“成长”水平进行连贯且可比的评估。多游戏结构突显了优势和脆弱领域，统一的日志使用保证了评估的可追溯性和可重复性。该工作的主要创新之处在于将“成长”这一过程从人类世界概念性地移植到人工智能领域，采用将心理学、机器人学、计算机科学和伦理学视角相结合的综合测试格式。通过这种方式，GROW-AI不仅衡量表现，还捕捉了一个AI实体向成熟演变的道路。 

---
# Competition and Attraction Improve Model Fusion 

**Title (ZH)**: 竞争与吸引力提升模型融合性能 

**Authors**: João Abrantes, Robert Tjarko Lange, Yujin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16204)  

**Abstract**: Model merging is a powerful technique for integrating the specialized knowledge of multiple machine learning models into a single model. However, existing methods require manually partitioning model parameters into fixed groups for merging, which restricts the exploration of potential combinations and limits performance. To overcome these limitations, we propose Model Merging of Natural Niches (M2N2), an evolutionary algorithm with three key features: (1) dynamic adjustment of merging boundaries to progressively explore a broader range of parameter combinations; (2) a diversity preservation mechanism inspired by the competition for resources in nature, to maintain a population of diverse, high-performing models that are particularly well-suited for merging; and (3) a heuristicbased attraction metric to identify the most promising pairs of models for fusion. Our experimental results demonstrate, for the first time, that model merging can be used to evolve models entirely from scratch. Specifically, we apply M2N2 to evolve MNIST classifiers from scratch and achieve performance comparable to CMA-ES, while being computationally more efficient. Furthermore, M2N2 scales to merge specialized language and image generation models, achieving state-of-the-art performance. Notably, it preserves crucial model capabilities beyond those explicitly optimized by the fitness function, highlighting its robustness and versatility. Our code is available at this https URL 

**Abstract (ZH)**: 模型归并自然niche模型合并（M2N2）：一种进化算法 

---
# Extending FKG.in: Towards a Food Claim Traceability Network 

**Title (ZH)**: 扩展FKG.in：迈向食物声明可追溯性网络 

**Authors**: Saransh Kumar Gupta, Rizwan Gulzar Mir, Lipika Dey, Partha Pratim Das, Anirban Sen, Ramesh Jain  

**Link**: [PDF](https://arxiv.org/pdf/2508.16117)  

**Abstract**: The global food landscape is rife with scientific, cultural, and commercial claims about what foods are, what they do, what they should not do, or should not do. These range from rigorously studied health benefits (probiotics improve gut health) and misrepresentations (soaked almonds make one smarter) to vague promises (superfoods boost immunity) and culturally rooted beliefs (cold foods cause coughs). Despite their widespread influence, the infrastructure for tracing, verifying, and contextualizing these claims remains fragmented and underdeveloped. In this paper, we propose a Food Claim-Traceability Network (FCN) as an extension of this http URL, a knowledge graph of Indian food that we have been incrementally building. We also present the ontology design and the semi-automated knowledge curation workflow that we used to develop a proof of concept of this http URL-FCN using Reddit data and Large Language Models. FCN integrates curated data inputs, structured schemas, and provenance-aware pipelines for food-related claim extraction and validation. While directly linked to the Indian food knowledge graph as an application, our methodology remains application-agnostic and adaptable to other geographic, culinary, or regulatory settings. By modeling food claims and their traceability in a structured, verifiable, and explainable way, we aim to contribute to more transparent and accountable food knowledge ecosystems, supporting researchers, policymakers, and most importantly, everyday consumers in navigating a world saturated with dietary assertions. 

**Abstract (ZH)**: 全球食物景观中存在关于食物是什么、它们的作用、不应作用或不应做什么的科学、文化及商业宣称。这些宣称从严格研究的健康益处（益生菌改善肠道健康）和误导性的断言（浸泡后的杏仁使人更聪明），到模糊的承诺（超级食物提高免疫力），再到根植于文化的信仰（冷食会导致咳嗽）不等。尽管这些宣称有广泛的影响，但追踪、验证和语境化这些宣称的基础设施仍然碎片化且不完善。本文中，我们提出了一个食物宣称追踪网络（FCN）作为对这一知识图谱的扩展，该知识图谱是我们在印度食物领域的逐步构建。我们还介绍了用于开发FCN概念验证的本体设计和半自动化知识整理工作流程，其中使用了Reddit数据和大型语言模型。FCN整合了经过整理的数据输入、结构化的模式和带有来源意识的工作流程，以进行与食品安全相关的宣称提取和验证。虽然该方法直接应用于印度食物知识图谱作为应用程序，但我们的方法仍然保持应用程序无关性，并适用于其他地理、烹饪或监管环境。通过以结构化、可验证和可解释的方式建模食物宣称及其追踪，我们旨在为更透明和负责任的食物知识生态系统做出贡献，支持研究人员、政策制定者，最重要的是普通消费者在充满饮食断言的世界中导航。 

---
# Urban Comfort Assessment in the Era of Digital Planning: A Multidimensional, Data-driven, and AI-assisted Framework 

**Title (ZH)**: 数字规划时代的城市舒适度评估：一个多维度、数据驱动和AI辅助的框架 

**Authors**: Sijie Yang, Binyu Lei, Filip Biljecki  

**Link**: [PDF](https://arxiv.org/pdf/2508.16057)  

**Abstract**: Ensuring liveability and comfort is one of the fundamental objectives of urban planning. Numerous studies have employed computational methods to assess and quantify factors related to urban comfort such as greenery coverage, thermal comfort, and walkability. However, a clear definition of urban comfort and its comprehensive evaluation framework remain elusive. Our research explores the theoretical interpretations and methodologies for assessing urban comfort within digital planning, emphasising three key dimensions: multidimensional analysis, data support, and AI assistance. 

**Abstract (ZH)**: 确保宜居性和舒适性是城市规划的基本目标之一。许多研究利用计算方法评估和量化与城市舒适性相关的影响因素，如绿化覆盖率、热舒适性和可达性。然而，城市舒适性的清晰定义及其全面评价框架尚不明确。我们的研究探讨了在数字规划中评估城市舒适性的理论解释和方法论，强调了多维度分析、数据支持和AI辅助这三个关键维度。 

---
# CoFE: A Framework Generating Counterfactual ECG for Explainable Cardiac AI-Diagnostics 

**Title (ZH)**: CoFE: 一种用于可解释心脏AI诊断的反事实心电图生成框架 

**Authors**: Jong-Hwan Jang, Junho Song, Yong-Yeon Jo  

**Link**: [PDF](https://arxiv.org/pdf/2508.16033)  

**Abstract**: Recognizing the need for explainable AI (XAI) approaches to enable the successful integration of AI-based ECG prediction models (AI-ECG) into clinical practice, we introduce a framework generating \textbf{Co}unter\textbf{F}actual \textbf{E}CGs (i,e., named CoFE) to illustrate how specific features, such as amplitudes and intervals, influence the model's predictive decisions. To demonstrate the applicability of the CoFE, we present two case studies: atrial fibrillation classification and potassium level regression models. The CoFE reveals feature changes in ECG signals that align with the established clinical knowledge. By clarifying both \textbf{where valid features appear} in the ECG and \textbf{how they influence the model's predictions}, we anticipate that our framework will enhance the interpretability of AI-ECG models and support more effective clinical decision-making. Our demonstration video is available at: this https URL. 

**Abstract (ZH)**: 基于对可解释人工智能（XAI）方法的需求，以促进基于人工智能的心电图预测模型（AI-ECG）的成功临床集成，我们提出了一种生成对抗性心电图（即，命名为CoFE）的框架，以展示特定特征（如振幅和时间间隔）如何影响模型的预测决策。为了展示CoFE的应用性，我们介绍了心脏颤动分类和钾水平回归模型两个案例研究。CoFE揭示了与临床知识相符的心电图信号特征变化。通过明确有效特征在心电图中的出现位置以及它们如何影响模型的预测，我们期望该框架能够增强AI-ECG模型的可解释性，并支持更有效的临床决策。视频演示可在以下链接查看：this https URL。 

---
# T-ILR: a Neurosymbolic Integration for LTLf 

**Title (ZH)**: T-ILR：一种LTLf的神经符号集成 

**Authors**: Riccardo Andreoni, Andrei Buliga, Alessandro Daniele, Chiara Ghidini, Marco Montali, Massimiliano Ronzani  

**Link**: [PDF](https://arxiv.org/pdf/2508.15943)  

**Abstract**: State-of-the-art approaches for integrating symbolic knowledge with deep learning architectures have demonstrated promising results in static domains. However, methods to handle temporal logic specifications remain underexplored. The only existing approach relies on an explicit representation of a finite-state automaton corresponding to the temporal specification. Instead, we aim at proposing a neurosymbolic framework designed to incorporate temporal logic specifications, expressed in Linear Temporal Logic over finite traces (LTLf), directly into deep learning architectures for sequence-based tasks. We extend the Iterative Local Refinement (ILR) neurosymbolic algorithm, leveraging the recent introduction of fuzzy LTLf interpretations. We name this proposed method Temporal Iterative Local Refinement (T-ILR). We assess T-ILR on an existing benchmark for temporal neurosymbolic architectures, consisting of the classification of image sequences in the presence of temporal knowledge. The results demonstrate improved accuracy and computational efficiency compared to the state-of-the-art method. 

**Abstract (ZH)**: 基于线性时序逻辑的神经符号框架在序列任务中的时间迭代局部精炼方法 

---
# A Disease-Centric Vision-Language Foundation Model for Precision Oncology in Kidney Cancer 

**Title (ZH)**: 面向肾癌精准 Oncology 的疾病导向的视觉-语言基础模型 

**Authors**: Yuhui Tao, Zhongwei Zhao, Zilong Wang, Xufang Luo, Feng Chen, Kang Wang, Chuanfu Wu, Xue Zhang, Shaoting Zhang, Jiaxi Yao, Xingwei Jin, Xinyang Jiang, Yifan Yang, Dongsheng Li, Lili Qiu, Zhiqiang Shao, Jianming Guo, Nengwang Yu, Shuo Wang, Ying Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2508.16569)  

**Abstract**: The non-invasive assessment of increasingly incidentally discovered renal masses is a critical challenge in urologic oncology, where diagnostic uncertainty frequently leads to the overtreatment of benign or indolent tumors. In this study, we developed and validated RenalCLIP using a dataset of 27,866 CT scans from 8,809 patients across nine Chinese medical centers and the public TCIA cohort, a visual-language foundation model for characterization, diagnosis and prognosis of renal mass. The model was developed via a two-stage pre-training strategy that first enhances the image and text encoders with domain-specific knowledge before aligning them through a contrastive learning objective, to create robust representations for superior generalization and diagnostic precision. RenalCLIP achieved better performance and superior generalizability across 10 core tasks spanning the full clinical workflow of kidney cancer, including anatomical assessment, diagnostic classification, and survival prediction, compared with other state-of-the-art general-purpose CT foundation models. Especially, for complicated task like recurrence-free survival prediction in the TCIA cohort, RenalCLIP achieved a C-index of 0.726, representing a substantial improvement of approximately 20% over the leading baselines. Furthermore, RenalCLIP's pre-training imparted remarkable data efficiency; in the diagnostic classification task, it only needs 20% training data to achieve the peak performance of all baseline models even after they were fully fine-tuned on 100% of the data. Additionally, it achieved superior performance in report generation, image-text retrieval and zero-shot diagnosis tasks. Our findings establish that RenalCLIP provides a robust tool with the potential to enhance diagnostic accuracy, refine prognostic stratification, and personalize the management of patients with kidney cancer. 

**Abstract (ZH)**: 无创评估日益增多的偶然发现的肾脏肿块是泌尿肿瘤学中的关键挑战，其中诊断不确定性经常导致对良性或惰性肿瘤的过度治疗。在本研究中，我们使用来自九家中国医疗机构和公共TCIA队列的27,866张CT扫描图像（涉及8,809名患者）研发并验证了RenalCLIP，这是一种用于肾脏肿块表征、诊断和预后的视觉-语言基础模型。该模型通过两阶段预训练策略开发，首先增强图像和文本编码器的领域特定知识，然后通过对比学习目标使它们对齐，以创建稳健的表示，从而实现更强的泛化能力和诊断精确度。RenalCLIP在包括解剖评估、诊断分类和生存预测在内的涵盖肾癌全流程临床工作的10个核心任务上实现了比其他先进的通用CT基础模型更好的性能和更强的泛化能力。特别是在TCIA队列的无复发生存预测这一复杂任务上，RenalCLIP达到了0.726的C指数，相对于领先基线提升了约20%。此外，RenalCLIP的预训练赋予了它显著的数据效率；在诊断分类任务中，即使在使用100%数据完全微调后，它仅需20%的训练数据即可达到所有基线模型的峰值性能。此外，RenalCLIP在报告生成、图像-文本检索和零样本诊断任务上也取得了优异的性能。我们的研究结果表明，RenalCLIP提供了一种稳健的工具，有望提高诊断准确性、细化预后分层并个性化肾癌患者的管理。 

---
# Enhanced NIRMAL Optimizer With Damped Nesterov Acceleration: A Comparative Analysis 

**Title (ZH)**: 带阻尼Nesterov加速的增强NIRMAL优化器：一种比较分析 

**Authors**: Nirmal Gaud, Prasad Krishna Murthy, Mostaque Md. Morshedur Hassan, Abhijit Ganguly, Vinay Mali, Ms Lalita Bhagwat Randive, Abhaypratap Singh  

**Link**: [PDF](https://arxiv.org/pdf/2508.16550)  

**Abstract**: This study introduces the Enhanced NIRMAL (Novel Integrated Robust Multi-Adaptation Learning with Damped Nesterov Acceleration) optimizer, an improved version of the original NIRMAL optimizer. By incorporating an $(\alpha, r)$-damped Nesterov acceleration mechanism, Enhanced NIRMAL improves convergence stability while retaining chess-inspired strategies of gradient descent, momentum, stochastic perturbations, adaptive learning rates, and non-linear transformations.
We evaluate Enhanced NIRMAL against Adam, SGD with Momentum, Nesterov, and the original NIRMAL on four benchmark image classification datasets: MNIST, FashionMNIST, CIFAR-10, and CIFAR-100, using tailored convolutional neural network (CNN) architectures.
Enhanced NIRMAL achieves a test accuracy of 46.06\% and the lowest test loss (1.960435) on CIFAR-100, surpassing the original NIRMAL (44.34\% accuracy) and closely rivaling SGD with Momentum (46.43\% accuracy). These results underscore Enhanced NIRMAL's superior generalization and stability, particularly on complex datasets. 

**Abstract (ZH)**: 这项研究介绍了一种改进的增强NIRMAL (增强型新型集成稳健多适应学习与阻尼Nesterov加速) 优化器，这是原版NIRMAL优化器的改进版本。通过结合$(\alpha, r)$-阻尼Nesterov加速机制，增强NIRMAL在保持梯度下降、动量、随机扰动、自适应学习率和非线性变换的国际象棋启发式策略的同时，提高了收敛稳定性。 

---
# Guiding Diffusion Models with Reinforcement Learning for Stable Molecule Generation 

**Title (ZH)**: 用强化学习引导的扩散模型实现稳定的分子生成 

**Authors**: Zhijian Zhou, Junyi An, Zongkai Liu, Yunfei Shi, Xuan Zhang, Fenglei Cao, Chao Qu, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16521)  

**Abstract**: Generating physically realistic 3D molecular structures remains a core challenge in molecular generative modeling. While diffusion models equipped with equivariant neural networks have made progress in capturing molecular geometries, they often struggle to produce equilibrium structures that adhere to physical principles such as force field consistency. To bridge this gap, we propose Reinforcement Learning with Physical Feedback (RLPF), a novel framework that extends Denoising Diffusion Policy Optimization to 3D molecular generation. RLPF formulates the task as a Markov decision process and applies proximal policy optimization to fine-tune equivariant diffusion models. Crucially, RLPF introduces reward functions derived from force-field evaluations, providing direct physical feedback to guide the generation toward energetically stable and physically meaningful structures. Experiments on the QM9 and GEOM-drug datasets demonstrate that RLPF significantly improves molecular stability compared to existing methods. These results highlight the value of incorporating physics-based feedback into generative modeling. The code is available at: this https URL. 

**Abstract (ZH)**: 生成符合物理现实的3D分子结构仍是分子生成建模中的核心挑战。尽管结合了李群神经网络的扩散模型在捕捉分子几何结构方面取得进展，但在产生符合物理原理（如力场一致性）的平衡结构方面仍然面临挑战。为解决这一问题，我们提出了物理反馈强化学习（RLPF）框架，该框架将去噪扩散策略优化扩展至3D分子生成。RLPF将任务建模为马尔科夫决策过程，并采用近端策略优化微调李群扩散模型。关键地，RLPF引入了基于力场评估的奖赏函数，从而直接提供物理反馈以引导生成能量稳定且具有物理意义的结构。实验结果表明，RLPF在QM9和GEOM-drug数据集上显著提高了分子稳定性，这凸显了将基于物理的反馈纳入生成建模的价值。代码可在以下链接获取：this https URL。 

---
# On Zero-Shot Reinforcement Learning 

**Title (ZH)**: 零-shot 强化学习 

**Authors**: Scott Jeen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16496)  

**Abstract**: Modern reinforcement learning (RL) systems capture deep truths about general, human problem-solving. In domains where new data can be simulated cheaply, these systems uncover sequential decision-making policies that far exceed the ability of any human. Society faces many problems whose solutions require this skill, but they are often in domains where new data cannot be cheaply simulated. In such scenarios, we can learn simulators from existing data, but these will only ever be approximately correct, and can be pathologically incorrect when queried outside of their training distribution. As a result, a misalignment between the environments in which we train our agents and the real-world in which we wish to deploy our agents is inevitable. Dealing with this misalignment is the primary concern of zero-shot reinforcement learning, a problem setting where the agent must generalise to a new task or domain with zero practice shots. Whilst impressive progress has been made on methods that perform zero-shot RL in idealised settings, new work is needed if these results are to be replicated in real-world settings. In this thesis, we argue that doing so requires us to navigate (at least) three constraints. First, the data quality constraint: real-world datasets are small and homogeneous. Second, the observability constraint: states, dynamics and rewards in the real-world are often only partially observed. And third, the data availability constraint: a priori access to data cannot always be assumed. This work proposes a suite of methods that perform zero-shot RL subject to these constraints. In a series of empirical studies we expose the failings of existing methods, and justify our techniques for remedying them. We believe these designs take us a step closer to RL methods that can be deployed to solve real-world problems. 

**Abstract (ZH)**: 现代强化学习（RL）系统揭示了一般的人类问题解决的深层真理。在新数据可以廉价模拟的领域，这些系统发现了超越人类能力的顺序决策策略。社会面临的许多问题需要这种能力，但这些问题往往出现在新数据不能廉价模拟的领域。在这种情况下，我们可以从现有数据中学习模拟器，但这些模拟器只能在训练分布外产生近似正确的结果，甚至可能出现病态的错误。因此，在训练我们的代理所处的环境与我们希望部署代理的现实世界之间存在不可避免的不匹配。解决这一不匹配是零样本强化学习的主要关注点，这是一种代理必须在零次实践的情况下泛化到新任务或领域的设置。虽然在理想化设置下的零样本RL方法取得了显著进展，但如果要在真实世界中复制这些结果，仍需新的工作。在本论文中，我们argue提出，要做到这一点，我们必须至少处理三种约束。首先，数据质量约束：现实世界的数据集较小且同质化严重。其次，可观测性约束：现实世界的状态、动力学和奖励通常只能部分观测到。第三，数据可用性约束：事先获取数据无法总是被假设。本文提出了在这些约束下执行零样本RL的一系列方法。通过一系列实证研究，我们揭示了现有方法的不足，并证明了我们修复这些问题技术的有效性。我们相信这些设计使我们更接近可以在真实世界问题中部署的RL方法。 

---
# Post Hoc Regression Refinement via Pairwise Rankings 

**Title (ZH)**: 基于成对排名的事后回归细化 

**Authors**: Kevin Tirta Wijaya, Michael Sun, Minghao Guo, Hans-Peter Seidel, Wojciech Matusik, Vahid Babaei  

**Link**: [PDF](https://arxiv.org/pdf/2508.16495)  

**Abstract**: Accurate prediction of continuous properties is essential to many scientific and engineering tasks. Although deep-learning regressors excel with abundant labels, their accuracy deteriorates in data-scarce regimes. We introduce RankRefine, a model-agnostic, plug-and-play post hoc method that refines regression with expert knowledge coming from pairwise rankings. Given a query item and a small reference set with known properties, RankRefine combines the base regressor's output with a rank-based estimate via inverse variance weighting, requiring no retraining. In molecular property prediction task, RankRefine achieves up to 10% relative reduction in mean absolute error using only 20 pairwise comparisons obtained through a general-purpose large language model (LLM) with no finetuning. As rankings provided by human experts or general-purpose LLMs are sufficient for improving regression across diverse domains, RankRefine offers practicality and broad applicability, especially in low-data settings. 

**Abstract (ZH)**: 准确预测连续性质对于许多科学和工程任务至关重要。尽管深度学习回归模型在丰富标签情况下表现卓越，但在数据稀缺的情况下其准确性会下降。我们引入了RankRefine，这是一种模型无关的、即插即用的后处理方法，通过结合基于专家对成对排名的先验知识来改进回归预测。给定一个查询项和一个包含已知性质的小参考集，RankRefine 通过逆方差加权结合基模型的输出和基于排名的估计，无需重新训练。在分子性质预测任务中，仅通过一个通用大语言模型（LLM）获得的20个成对比较，RankRefine在平均绝对误差上实现了高达10%的相对降低，且无需微调。由于人类专家或通用大语言模型提供的排名足以在多种领域改善回归性能，RankRefine 具有实用性和广泛适用性，特别是在数据稀缺的情况下。 

---
# SafeSpace: An Integrated Web Application for Digital Safety and Emotional Well-being 

**Title (ZH)**: SafeSpace：一个综合网络应用，旨在保障数字安全与情感健康 

**Authors**: Kayenat Fatmi, Mohammad Abbas  

**Link**: [PDF](https://arxiv.org/pdf/2508.16488)  

**Abstract**: In the digital era, individuals are increasingly exposed to online harms such as toxicity, manipulation, and grooming, which often pose emotional and safety risks. Existing systems for detecting abusive content or issuing safety alerts operate in isolation and rarely combine digital safety with emotional well-being. In this paper, we present SafeSpace, a unified web application that integrates three modules: (1) toxicity detection in chats and screenshots using NLP models and Google's Perspective API, (2) a configurable safety ping system that issues emergency alerts with the user's live location (longitude and latitude) via SMTP-based emails when check-ins are missed or SOS alerts are manually triggered, and (3) a reflective questionnaire that evaluates relationship health and emotional resilience. The system employs Firebase for alert management and a modular architecture designed for usability, privacy, and scalability. The experimental evaluation shows 93% precision in toxicity detection, 100% reliability in safety alerts under emulator tests, and 92% alignment between automated and manual questionnaire scoring. SafeSpace, implemented as a web application, demonstrates the feasibility of integrating detection, protection, and reflection within a single platform, with future deployment envisioned as a mobile application for broader accessibility. 

**Abstract (ZH)**: 在数字时代，个人越来越容易受到网络危害的影响，如毒性、操纵和诱骗，这些往往会对情绪和安全构成风险。现有检测不当内容或发布安全警报的系统通常独立运作，并不常将数字安全与情绪健康相结合。本文介绍了SafeSpace，这是一个统一的网络应用，整合了三个模块：(1) 使用NLP模型和Google的Perspective API检测聊天和截图中的毒性；(2) 可配置的安全提醒系统，通过基于SMTP的电子邮件自动发送紧急警报，包括用户的位置信息（经度和纬度），当检入被错过或手动触发SOS警报时；(3) 反省问卷，评估人际关系健康和情绪韧性。该系统使用Firebase进行警报管理，并采用模块化架构以提高可用性、隐私性和可扩展性。实验评估显示，检测毒性的准确率为93%，安全警报在模拟器测试中可靠率达到100%，自动和手动问卷评分之间的对齐率为92%。作为网络应用实现的SafeSpace展示了在单一平台上整合检测、保护和反省的可行性，未来计划将其部署为移动应用以提高更广泛的可访问性。 

---
# FraPPE: Fast and Efficient Preference-based Pure Exploration 

**Title (ZH)**: FraPPE: 快速且高效的基于偏好纯探索 

**Authors**: Udvas Das, Apurv Shukla, Debabrota Basu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16487)  

**Abstract**: Preference-based Pure Exploration (PrePEx) aims to identify with a given confidence level the set of Pareto optimal arms in a vector-valued (aka multi-objective) bandit, where the reward vectors are ordered via a (given) preference cone $\mathcal{C}$. Though PrePEx and its variants are well-studied, there does not exist a computationally efficient algorithm that can optimally track the existing lower bound for arbitrary preference cones. We successfully fill this gap by efficiently solving the minimisation and maximisation problems in the lower bound. First, we derive three structural properties of the lower bound that yield a computationally tractable reduction of the minimisation problem. Then, we deploy a Frank-Wolfe optimiser to accelerate the maximisation problem in the lower bound. Together, these techniques solve the maxmin optimisation problem in $\mathcal{O}(KL^{2})$ time for a bandit instance with $K$ arms and $L$ dimensional reward, which is a significant acceleration over the literature. We further prove that our proposed PrePEx algorithm, FraPPE, asymptotically achieves the optimal sample complexity. Finally, we perform numerical experiments across synthetic and real datasets demonstrating that FraPPE achieves the lowest sample complexities to identify the exact Pareto set among the existing algorithms. 

**Abstract (ZH)**: 基于偏好的纯探索（PrePEx）旨在通过给定的信心水平，在向量值（即多目标）赌博机中识别出最优臂集合，其中奖励向量通过给定的偏好锥 $\mathcal{C}$ 进行排序。尽管已经研究了PrePEx及其变体，但仍不存在适用于任意偏好锥的计算上高效的最优下界追踪算法。我们通过高效解决下界中的最小化和最大化问题成功填补了这一空白。首先，我们推导出下界的三种结构特性，以实现最小化问题的计算可处理的归约。然后，我们使用Frank-Wolfe优化器加速下界中的最大化问题。 вместе，这些技术在 $\mathcal{O}(KL^{2})$ 时间内解决了一个赌博机实例中的最大化最小化优化问题，这是对文献中方法的显著加速。我们进一步证明，我们提出的基于偏好的纯探索算法FraPPE在渐近意义上实现了最优的样本复杂度。最后，我们在合成和现实数据集上进行的数值实验表明，FraPPE在现有算法中实现了识别出精确帕累托集所需的最低样本复杂度。 

---
# Disentangled Multi-modal Learning of Histology and Transcriptomics for Cancer Characterization 

**Title (ZH)**: 离散多模态学习：病理学和转录组学在癌症表征中的应用 

**Authors**: Yupei Zhang, Xiaofei Wang, Anran Liu, Lequan Yu, Chao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.16479)  

**Abstract**: Histopathology remains the gold standard for cancer diagnosis and prognosis. With the advent of transcriptome profiling, multi-modal learning combining transcriptomics with histology offers more comprehensive information. However, existing multi-modal approaches are challenged by intrinsic multi-modal heterogeneity, insufficient multi-scale integration, and reliance on paired data, restricting clinical applicability. To address these challenges, we propose a disentangled multi-modal framework with four contributions: 1) To mitigate multi-modal heterogeneity, we decompose WSIs and transcriptomes into tumor and microenvironment subspaces using a disentangled multi-modal fusion module, and introduce a confidence-guided gradient coordination strategy to balance subspace optimization. 2) To enhance multi-scale integration, we propose an inter-magnification gene-expression consistency strategy that aligns transcriptomic signals across WSI magnifications. 3) To reduce dependency on paired data, we propose a subspace knowledge distillation strategy enabling transcriptome-agnostic inference through a WSI-only student model. 4) To improve inference efficiency, we propose an informative token aggregation module that suppresses WSI redundancy while preserving subspace semantics. Extensive experiments on cancer diagnosis, prognosis, and survival prediction demonstrate our superiority over state-of-the-art methods across multiple settings. Code is available at this https URL. 

**Abstract (ZH)**: 组织病理学仍然是癌症诊断和预后的黄金标准。随着转录组测序的出现，结合转录组学与组织学的多模态学习提供了更全面的信息。然而，现有的多模态方法受到了固有的多模态异质性、多尺度集成不足以及对配对数据依赖性的挑战，限制了临床应用。为了解决这些挑战，我们提出了一种解耦的多模态框架，包含四个贡献：1）为减轻多模态异质性，我们通过解耦的多模态融合模块将WSIs和转录组分解到肿瘤和微环境子空间，并引入一种基于置信引导的梯度协调策略来平衡子空间优化。2）为增强多尺度集成，我们提出一种跨放大倍数的基因表达一致性策略，以在WSI放大倍数间对接转录组信号。3）为减少对配对数据的依赖，我们提出了一种子空间知识蒸馏策略，通过仅使用WSI的学生模型实现转录组无关的推断。4）为提高推理效率，我们提出了一种信息标记聚合模块，抑制WSI冗余同时保留子空间语义。在癌症诊断、预后及生存预测的广泛实验中，我们的方法在多个设置中优于最先进的方法。代码详见this https URL。 

---
# Domain-aligned generative downscaling enhances projections of extreme climate events 

**Title (ZH)**: 领域对齐生成下scaling增强极端气候事件的 projections 

**Authors**: Ruian Tie, Xiaohui Zhong, Zhengyu Shi, Hao Li, Jun Liu, Wu Libo  

**Link**: [PDF](https://arxiv.org/pdf/2508.16396)  

**Abstract**: Climate change is exacerbating extreme weather events globally, including high temperatures, extreme precipitation, strong winds, and tropical cyclones, posing severe threats to human health, infrastructure, food security, and socio-economic systems. Although existing global climate models (GCMs) provide essential tools for climate prediction, they face limitations such as insufficient resolution and high computational costs when simulating extreme events. To address these issues, this study proposes a spatiotemporal downscaling model based on generative machine learning-the Domain Aligned Climate Downscaling model (DACD), designed to enhance the simulation capabilities for extreme weather events. The proposed model employs domain adaptation tricks and a Flow Matching training framework to transform global low-resolution climate data into high-resolution local-scale climate information while achieving precise simulation of multivariable and temporal scales. The results show that during the historical period (2005-2014), our model outperformed existing methods in simulating high temperatures, extreme precipitation, strong wind, and tropical cyclone tracks, significantly reducing errors and improving the ability to capture extreme events. Under different future scenarios (2015-2100), the model reveals a significant increasing trend in the frequency and intensity of extreme events, particularly under the high-emission scenario (SSP585). Compared to traditional methods, our model more accurately simulates the spatial distribution and dynamic changes of extreme events, providing an essential tool for understanding the impacts of climate change. This study offers a new technological pathway for high-resolution climate analysis and extreme event prediction, providing scientific support for addressing future climate change and formulating adaptation strategies. 

**Abstract (ZH)**: 气候变化加剧了全球极端天气事件，包括高温、极端降水量、强风和热带气旋，对人类健康、基础设施、粮食安全和社会经济系统构成了严重威胁。尽管现有的全球气候变化模型（GCMs）为气候预测提供了基本工具，但在模拟极端事件时仍存在分辨率不足和计算成本高的局限性。为解决这些问题，本研究提出了一种基于生成式机器学习的空间时间降尺度模型——域对齐气候降尺度模型（DACD），旨在增强极端天气事件的模拟能力。该模型采用了域适应技巧和流动匹配训练框架，将全球低分辨率气候数据转换为高分辨率局部尺度气候信息，并实现了多变量和时间尺度的精确模拟。结果显示，在历史时期（2005-2014年），我们的模型在模拟高温、极端降水量、强风和热带气旋路径方面优于现有方法，显著降低了误差并提高了捕捉极端事件的能力。在不同未来情景（2015-2100年）下，模型揭示了极端事件频率和强度的显著增加趋势，特别是在高排放情景（SSP585）下。与传统方法相比，我们的模型更准确地模拟了极端事件的空间分布和动态变化，为理解和应对气候变化提供了重要工具。本研究提供了一条高分辨率气候分析和极端事件预测的新技术路径，并为未来气候变化的应对策略提供了科学支持。 

---
# RoMedQA: The First Benchmark for Romanian Medical Question Answering 

**Title (ZH)**: RoMedQA: 首个罗马尼亚医疗问答基准 

**Authors**: Ana-Cristina Rogoz, Radu Tudor Ionescu, Alexandra-Valentina Anghel, Ionut-Lucian Antone-Iordache, Simona Coniac, Andreea Iuliana Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16390)  

**Abstract**: Question answering (QA) is an actively studied topic, being a core natural language processing (NLP) task that needs to be addressed before achieving Artificial General Intelligence (AGI). However, the lack of QA datasets in specific domains and languages hinders the development of robust AI models able to generalize across various domains and languages. To this end, we introduce RoMedQA, the first Romanian QA benchmark for the medical domain, alongside a comprehensive evaluation of state-of-the-art large language models (LLMs). We construct a high-quality and large-scale dataset comprising 102,646 QA pairs related to cancer patients. The questions regard medical case summaries of 1,011 patients, requiring either keyword extraction or reasoning to be answered correctly. RoMedQA is the result of a time-consuming manual annotation process carried out by seven physicians specialized in oncology or radiotherapy, who spent a total of about 2,100 work hours to generate the QA pairs. We experiment with four LLMs from distinct families of models on RoMedQA. Each model is employed in two scenarios, namely one based on zero-shot prompting and one based on supervised fine-tuning. Our results show that fine-tuned models significantly outperform their zero-shot counterparts, clearly indicating that pretrained models fail to generalize on RoMedQA. Our findings demonstrate the importance of both domain-specific and language-specific fine-tuning for reliable clinical QA in Romanian. We publicly release our dataset and code at this https URL. 

**Abstract (ZH)**: RoMedQA：面向医学领域的 Romanian 问答基准及大语言模型全面评估 

---
# Uppaal Coshy: Automatic Synthesis of Compact Shields for Hybrid Systems 

**Title (ZH)**: Uppaal Coshy: 自动合成紧凑型混合系统防护器 

**Authors**: Asger Horn Brorholt, Andreas Holck Høeg-Petersen, Peter Gjøl Jensen, Kim Guldstrand Larsen, Marius Mikučionis, Christian Schilling, Andrzej Wąsowski  

**Link**: [PDF](https://arxiv.org/pdf/2508.16345)  

**Abstract**: We present Uppaal Coshy, a tool for automatic synthesis of a safety strategy -- or shield -- for Markov decision processes over continuous state spaces and complex hybrid dynamics. The general methodology is to partition the state space and then solve a two-player safety game, which entails a number of algorithmically hard problems such as reachability for hybrid systems. The general philosophy of Uppaal Coshy is to approximate hard-to-obtain solutions using simulations. Our implementation is fully automatic and supports the expressive formalism of Uppaal models, which encompass stochastic hybrid automata. The precision of our partition-based approach benefits from using finer grids, which however are not efficient to store. We include an algorithm called Caap to efficiently compute a compact representation of a shield in the form of a decision tree, which yields significant reductions. 

**Abstract (ZH)**: Uppaal Coshy：一种用于连续状态空间和复杂混合动力学的马尔可夫决策过程自动安全策略合成工具 

---
# Unsupervised Online Detection of Pipe Blockages and Leakages in Water Distribution Networks 

**Title (ZH)**: 无监督在线检测水分配网络中的管道堵塞和泄漏 

**Authors**: Jin Li, Kleanthis Malialis, Stelios G. Vrachimis, Marios M. Polycarpou  

**Link**: [PDF](https://arxiv.org/pdf/2508.16336)  

**Abstract**: Water Distribution Networks (WDNs), critical to public well-being and economic stability, face challenges such as pipe blockages and background leakages, exacerbated by operational constraints such as data non-stationarity and limited labeled data. This paper proposes an unsupervised, online learning framework that aims to detect two types of faults in WDNs: pipe blockages, modeled as collective anomalies, and background leakages, modeled as concept drift. Our approach combines a Long Short-Term Memory Variational Autoencoder (LSTM-VAE) with a dual drift detection mechanism, enabling robust detection and adaptation under non-stationary conditions. Its lightweight, memory-efficient design enables real-time, edge-level monitoring. Experiments on two realistic WDNs show that the proposed approach consistently outperforms strong baselines in detecting anomalies and adapting to recurrent drift, demonstrating its effectiveness in unsupervised event detection for dynamic WDN environments. 

**Abstract (ZH)**: 水分布网络中的无监督在线学习框架：检测管阻和背景泄漏中的集体异常及概念漂移 

---
# Vevo2: Bridging Controllable Speech and Singing Voice Generation via Unified Prosody Learning 

**Title (ZH)**: Vevo2: 统一韵律学习在可控语音与歌声生成中的桥梁作用 

**Authors**: Xueyao Zhang, Junan Zhang, Yuancheng Wang, Chaoren Wang, Yuanzhe Chen, Dongya Jia, Zhuo Chen, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16332)  

**Abstract**: Controllable human voice generation, particularly for expressive domains like singing, remains a significant challenge. This paper introduces Vevo2, a unified framework for controllable speech and singing voice generation. To tackle issues like the scarcity of annotated singing data and to enable flexible controllability, Vevo2 introduces two audio tokenizers: (1) a music-notation-free prosody tokenizer that captures prosody and melody from speech, singing, and even instrumental sounds, and (2) a low-frame-rate (12.5 Hz) content-style tokenizer that encodes linguistic content, prosody, and style for both speech and singing, while enabling timbre disentanglement. Vevo2 consists of an auto-regressive (AR) content-style modeling stage, which aims to enable controllability over text, prosody, and style, as well as a flow-matching acoustic modeling stage that allows for timbre control. Particularly, during pre-training of the AR model, we propose both explicit and implicit prosody learning strategies to bridge speech and singing voice. Moreover, to further enhance the AR model's ability to follow text and prosody, we design a multi-objective post-training task that integrates both intelligibility and prosody similarity alignment. Experimental results show that the unified modeling in Vevo2 brings mutual benefits to both speech and singing voice generation. Additionally, Vevo2's effectiveness across a wide range of synthesis, conversion, and editing tasks for both speech and singing further demonstrates its strong generalization ability and versatility. Audio samples are are available at this https URL. 

**Abstract (ZH)**: 可控的人声生成，特别是在表达领域如唱歌中依然是一项重大挑战。本文介绍了一种统一的可控语音和歌声生成框架Vevo2。为了解决标注歌声数据稀缺的问题以及实现灵活的可控性，Vevo2 引入了两种音频分词器：（1）一种无音乐记号的语调分词器，可以从语音、歌声甚至乐器音效中捕捉语调和旋律；（2）一种低帧率（12.5 Hz）的内容样式分词器，可以编码语音和歌声中的语言内容、语调和风格，同时实现音色解纠缠。Vevo2 包含一个自回归（AR）内容样式建模阶段，旨在实现对文本、语调和风格的控制，以及一个流匹配声学建模阶段，允许实现音色控制。特别是在 AR 模型的预训练过程中，我们提出了显性和隐性的语调学习策略以弥合语音和歌声之间的鸿沟。此外，为了进一步增强 AR 模型跟随文本和语调的能力，我们设计了一种多目标后训练任务，将可懂性和语调相似度对齐相结合。实验结果表明，Vevo2 统一建模对语音和歌声生成都有益处。此外，Vevo2 在语音和歌声合成、转换和编辑等广泛任务中的有效性进一步证明了其强健的泛化能力和 versatility。音频样本请参见此链接：这个 https URL。 

---
# Cyber Physical Awareness via Intent-Driven Threat Assessment: Enhanced Space Networks with Intershell Links 

**Title (ZH)**: 基于意图驱动威胁评估的网络物理意识：通过内壳链接增强的空间网络 

**Authors**: Selen Gecgel Cetin, Tolga Ovatman, Gunes Karabulut Kurt  

**Link**: [PDF](https://arxiv.org/pdf/2508.16314)  

**Abstract**: This letter addresses essential aspects of threat assessment by proposing intent-driven threat models that incorporate both capabilities and intents. We propose a holistic framework for cyber physical awareness (CPA) in space networks, pointing out that analyzing reliability and security separately can lead to overfitting on system-specific criteria. We structure our proposed framework in three main steps. First, we suggest an algorithm that extracts characteristic properties of the received signal to facilitate an intuitive understanding of potential threats. Second, we develop a multitask learning architecture where one task evaluates reliability-related capabilities while the other deciphers the underlying intentions of the signal. Finally, we propose an adaptable threat assessment that aligns with varying security and reliability requirements. The proposed framework enhances the robustness of threat detection and assessment, outperforming conventional sequential methods, and enables space networks with emerging intershell links to effectively address complex threat scenarios. 

**Abstract (ZH)**: 本文通过提出兼顾能力和意图的威胁模型， addresses威胁评估的基本方面。我们提出了一个综合的网络物理空间awareness (CPA)框架，指出单独分析可靠性和安全性可能导致特定系统的过度拟合。我们提议的框架分为三个主要步骤。首先，我们建议一种算法提取接收到的信号的特征属性，以便直观地理解潜在威胁。其次，我们开发了一种多任务学习架构，其中一个任务评估与可靠性相关的能力，另一个任务解析信号背后的意图。最后，我们提出了一种可适应的威胁评估方法，以满足不同的安全和可靠性要求。该提议的框架增强了威胁检测和评估的稳健性，优于传统的顺序方法，并能使拥有新兴层次链接的空间网络有效应对复杂的威胁场景。 

---
# Representation Learning of Auxiliary Concepts for Improved Student Modeling and Exercise Recommendation 

**Title (ZH)**: 辅助概念的表示学习以改进学生建模和练习推荐 

**Authors**: Yahya Badran, Christine Preisach  

**Link**: [PDF](https://arxiv.org/pdf/2508.16269)  

**Abstract**: Personalized recommendation is a key feature of intelligent tutoring systems, typically relying on accurate models of student knowledge. Knowledge Tracing (KT) models enable this by estimating a student's mastery based on their historical interactions. Many KT models rely on human-annotated knowledge concepts (KCs), which tag each exercise with one or more skills or concepts believed to be necessary for solving it. However, these KCs can be incomplete, error-prone, or overly general.
In this paper, we propose a deep learning model that learns sparse binary representations of exercises, where each bit indicates the presence or absence of a latent concept. We refer to these representations as auxiliary KCs. These representations capture conceptual structure beyond human-defined annotations and are compatible with both classical models (e.g., BKT) and modern deep learning KT architectures.
We demonstrate that incorporating auxiliary KCs improves both student modeling and adaptive exercise recommendation. For student modeling, we show that augmenting classical models like BKT with auxiliary KCs leads to improved predictive performance. For recommendation, we show that using auxiliary KCs enhances both reinforcement learning-based policies and a simple planning-based method (expectimax), resulting in measurable gains in student learning outcomes within a simulated student environment. 

**Abstract (ZH)**: 个性化推荐是智能辅导系统的关键功能，通常依赖于对学生知识的准确模型。知识追踪（KT）模型通过根据学生的历史交互估计其掌握程度来实现这一点。许多KT模型依赖于人工标注的知识概念（KCs），将每个练习标记为一个或多个认为对于解决该练习必要的技能或概念。然而，这些KCs可能是不完整、易出错或过于概括的。
在本文中，我们提出了一种深度学习模型，该模型学习稀疏的二进制表示方法，每个位表示一个潜在概念的存在或不存在。我们将这些表示称为辅助KCs。这些表示捕捉了超出人工定义注释的概念结构，并与经典模型（例如BKT）和现代深度学习KT架构兼容。
我们证明了整合辅助KCs可以提高学生建模和自适应练习推荐的效果。对于学生建模，我们展示了将古老的BKT等模型与辅助KCs相结合可以提高预测性能。对于推荐，我们展示了使用辅助KCs可以增强基于强化学习的策略和一种简单的基于规划的方法（期望极大值），从而在模拟学生环境中可测量地改善了学生的学习成果。 

---
# A Reduction of Input/Output Logics to SAT 

**Title (ZH)**: 输入/输出逻辑问题归约至SAT 

**Authors**: Alexander Steen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16242)  

**Abstract**: Deontic logics are formalisms for reasoning over norms, obligations, permissions and prohibitions. Input/Output (I/O) Logics are a particular family of so-called norm-based deontic logics that formalize conditional norms outside of the underlying object logic language, where conditional norms do not carry a truth-value themselves. In this paper, an automation approach for I/O logics is presented that makes use of suitable reductions to (sequences of) propositional satisfiability problems. A prototypical implementation, named rio (reasoner for input/output logics), of the proposed procedures is presented and applied to illustrative examples. 

**Abstract (ZH)**: 义理性逻辑是用于推理规范、义务、许可和禁止的形式化工具。输入/输出(I/O)逻辑是所谓的基于规范的义理性逻辑的一种特定家族，它们在基础对象逻辑语言之外形式化条件规范，这些条件规范本身不带有真值。本文提出了一种用于I/O逻辑的自动化方法，该方法利用适合的归约方法到(一系列)命题可满足性问题。提出了所提议过程的典型实现rio（输入/输出逻辑推理器），并应用于示例说明。 

---
# A XAI-based Framework for Frequency Subband Characterization of Cough Spectrograms in Chronic Respiratory Disease 

**Title (ZH)**: 基于XAI的慢性呼吸道疾病咳嗽频谱频带特征表征框架 

**Authors**: Patricia Amado-Caballero, Luis M. San-José-Revuelta, Xinheng Wang, José Ramón Garmendia-Leiza, Carlos Alberola-López, Pablo Casaseca-de-la-Higuera  

**Link**: [PDF](https://arxiv.org/pdf/2508.16237)  

**Abstract**: This paper presents an explainable artificial intelligence (XAI)-based framework for the spectral analysis of cough sounds associated with chronic respiratory diseases, with a particular focus on Chronic Obstructive Pulmonary Disease (COPD). A Convolutional Neural Network (CNN) is trained on time-frequency representations of cough signals, and occlusion maps are used to identify diagnostically relevant regions within the spectrograms. These highlighted areas are subsequently decomposed into five frequency subbands, enabling targeted spectral feature extraction and analysis. The results reveal that spectral patterns differ across subbands and disease groups, uncovering complementary and compensatory trends across the frequency spectrum. Noteworthy, the approach distinguishes COPD from other respiratory conditions, and chronic from non-chronic patient groups, based on interpretable spectral markers. These findings provide insight into the underlying pathophysiological characteristics of cough acoustics and demonstrate the value of frequency-resolved, XAI-enhanced analysis for biomedical signal interpretation and translational respiratory disease diagnostics. 

**Abstract (ZH)**: 基于解释性人工智能（XAI）的慢性呼吸道疾病咳嗽声音谱分析框架：以慢性阻塞性肺病（COPD）为例 

---
# OmniCache: A Trajectory-Oriented Global Perspective on Training-Free Cache Reuse for Diffusion Transformer Models 

**Title (ZH)**: 全知缓存：一种面向轨迹的全局视角下无需训练的缓存重用方法用于传播变换器模型 

**Authors**: Huanpeng Chu, Wei Wu, Guanyu Fen, Yutao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16212)  

**Abstract**: Diffusion models have emerged as a powerful paradigm for generative tasks such as image synthesis and video generation, with Transformer architectures further enhancing performance. However, the high computational cost of diffusion Transformers-stemming from a large number of sampling steps and complex per-step computations-presents significant challenges for real-time deployment. In this paper, we introduce OmniCache, a training-free acceleration method that exploits the global redundancy inherent in the denoising process. Unlike existing methods that determine caching strategies based on inter-step similarities and tend to prioritize reusing later sampling steps, our approach originates from the sampling perspective of DIT models. We systematically analyze the model's sampling trajectories and strategically distribute cache reuse across the entire sampling process. This global perspective enables more effective utilization of cached computations throughout the diffusion trajectory, rather than concentrating reuse within limited segments of the sampling this http URL addition, during cache reuse, we dynamically estimate the corresponding noise and filter it out to reduce its impact on the sampling this http URL experiments demonstrate that our approach accelerates the sampling process while maintaining competitive generative quality, offering a promising and practical solution for efficient deployment of diffusion-based generative models. 

**Abstract (ZH)**: 面向扩散模型的训练-free加速方法：OmniCache 

---
# Set Transformer Architectures and Synthetic Data Generation for Flow-Guided Nanoscale Localization 

**Title (ZH)**: 基于流引导的纳米级定位的Set Transformer架构与合成数据生成 

**Authors**: Mika Leo Hube, Filip Lemic, Ethungshan Shitiri, Gerard Calvo Bartra, Sergi Abadal, Xavier Costa Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2508.16200)  

**Abstract**: Flow-guided Localization (FGL) enables the identification of spatial regions within the human body that contain an event of diagnostic interest. FGL does that by leveraging the passive movement of energy-constrained nanodevices circulating through the bloodstream. Existing FGL solutions rely on graph models with fixed topologies or handcrafted features, which limit their adaptability to anatomical variability and hinder scalability. In this work, we explore the use of Set Transformer architectures to address these limitations. Our formulation treats nanodevices' circulation time reports as unordered sets, enabling permutation-invariant, variable-length input processing without relying on spatial priors. To improve robustness under data scarcity and class imbalance, we integrate synthetic data generation via deep generative models, including CGAN, WGAN, WGAN-GP, and CVAE. These models are trained to replicate realistic circulation time distributions conditioned on vascular region labels, and are used to augment the training data. Our results show that the Set Transformer achieves comparable classification accuracy compared to Graph Neural Networks (GNN) baselines, while simultaneously providing by-design improved generalization to anatomical variability. The findings highlight the potential of permutation-invariant models and synthetic augmentation for robust and scalable nanoscale localization. 

**Abstract (ZH)**: 基于流引导的定位（FGL） enables 人体内包含诊断兴趣事件的空间区域的识别。FGL 通过利用在血液循环中的能量受限纳米设备的被动运动来实现这一目标。现有 FGL 解决方案依赖于固定拓扑或手工crafted 特征的图形模型，这限制了其对解剖变异性的适应性并阻碍了可扩展性。在本工作中，我们探索使用 Set Transformer 架构来解决这些问题。我们的建模方式将纳米设备的循环时间报告视为无序集合，从而在无需依赖空间先验的情况下实现不变置换和可变长度输入的处理。为了在数据稀疏和类别不平衡条件下提高鲁棒性，我们通过深度生成模型（包括 CGAN、WGAN、WGAN-GP 和 CVAE）整合了合成数据生成，这些模型被训练以根据血管区域标签生成现实的循环时间分布，并用于增强训练数据。我们的结果显示，Set Transformer 的分类准确性与图形神经网络（GNN）基线相当，同时通过设计提高了对解剖变异性的泛化能力。这些发现突显了置换不变模型和合成增强在纳米尺度定位中的潜力。 

---
# A Relay-Chain-Powered Ciphertext-Policy Attribute-Based Encryption in Intelligent Transportation Systems 

**Title (ZH)**: 基于智能交通系统的一种relay-chainpowered密文策略属性基加密方法 

**Authors**: Aparna Singh, Geetanjali Rathee, Chaker Abdelaziz Kerrache, Mohamed Chahine Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2508.16189)  

**Abstract**: The very high growth of Intelligent Transportation Systems (ITS) has generated an urgent requirement for secure, effective, and context-aware data sharing mechanisms, especially over heterogeneous and geographically dispersed settings. This work suggests a new architecture that combines a relay chain-driven encryption system with a modified Ciphertext-Policy Attribute-Based Encryption (CP-ABE) scheme to tackle the double impediment of dynamic access and low-latency communication. The model proposes a context-aware smart contract on a worldwide relay chain that checks against data properties, including event type, time, and geographical region, to specify the suitable level of encryption policy. From such relay-directed judgment, On-Board Units (OBUs) encrypt data end-to-end by utilising CP-ABE and store ciphertext inside localised regional blockchains, preventing dependence on symmetric encryption or off-chain storage. High-sensitivity events are secured with firm, multi-attribute access rules, whereas common updates use light policies to help reduce processing burdens. The crypto system also adds traceability and low-latency revocation, with global enforcement managed through the relay chain. This distributed, scalable model provides a proper balance between responsiveness in real time and security and is extremely apt for next-gen vehicular networks that function across multi-jurisdictional domains. 

**Abstract (ZH)**: 智能交通系统（ITS）的极高速增长迫切需要安全、有效且上下文感知的数据共享机制，特别是在异构且地理上分散的环境中。本文提出了一种新的架构，该架构结合了中继链驱动的加密系统和修改后的密文策略属性基加密（CP-ABE）方案，以解决动态访问和低延迟通信的双重障碍。该模型提出了一种全球中继链上的上下文感知智能合约，根据事件类型、时间和地理区域等数据属性来指定合适的加密策略。依靠这样的中继链驱动判断，车载单元（OBUs）利用CP-ABE对数据进行端到端加密，并将密文存储在局部区域的区块链中，避免了对对称加密或链下存储的依赖。高敏感事件使用严格的多属性访问规则进行保护，而常规更新则采用轻量级策略以减轻处理负担。该加密系统还增加了可追溯性与低延迟撤销，并通过中继链进行全球执行。该分布式可扩展模型在实时响应性和安全性之间提供了适当的平衡，非常适合跨多辖区运行的新一代车辆网络。 

---
# Motor Imagery EEG Signal Classification Using Minimally Random Convolutional Kernel Transform and Hybrid Deep Learning 

**Title (ZH)**: 基于极小子采样卷积核变换与混合深度学习的想象运动EEG信号分类 

**Authors**: Jamal Hwaidi, Mohamed Chahine Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2508.16179)  

**Abstract**: The brain-computer interface (BCI) establishes a non-muscle channel that enables direct communication between the human body and an external device. Electroencephalography (EEG) is a popular non-invasive technique for recording brain signals. It is critical to process and comprehend the hidden patterns linked to a specific cognitive or motor task, for instance, measured through the motor imagery brain-computer interface (MI-BCI). A significant challenge is presented by classifying motor imagery-based electroencephalogram (MI-EEG) tasks, given that EEG signals exhibit nonstationarity, time-variance, and individual diversity. Obtaining good classification accuracy is also very difficult due to the growing number of classes and the natural variability among individuals. To overcome these issues, this paper proposes a novel method for classifying EEG motor imagery signals that extracts features efficiently with Minimally Random Convolutional Kernel Transform (MiniRocket), a linear classifier then uses the extracted features for activity recognition. Furthermore, a novel deep learning based on Convolutional Neural Network (CNN) and Long Short Term Memory (LSTM) architecture to serve as a baseline was proposed and demonstrated that classification via MiniRocket's features achieves higher performance than the best deep learning models at lower computational cost. The PhysioNet dataset was used to evaluate the performance of the proposed approaches. The proposed models achieved mean accuracy values of 98.63% and 98.06% for the MiniRocket and CNN-LSTM, respectively. The findings demonstrate that the proposed approach can significantly enhance motor imagery EEG accuracy and provide new insights into the feature extraction and classification of MI-EEG. 

**Abstract (ZH)**: 脑-计算机接口（BCI）建立了一个人体与外部设备之间的一种非肌肉通道，实现直接通信。脑电图（EEG）是一种常用的无创技术，用于记录脑电信号。通过运动想象脑-计算机接口（MI-BCI），处理和理解与特定认知或运动任务相关的隐藏模式至关重要。由于EEG信号表现出非站定性、时间变异性和个体差异，基于运动想象的脑电图（MI-EEG）任务分类面临巨大挑战。受个体自然变异性增加和分类类别数不断增长的影响，获得良好的分类精度也非常困难。为克服这些问题，本文提出了一种新的方法，利用Minimal Random Convolutional Kernel Transform (MiniRocket) 提取特征，并使用线性分类器对提取的特征进行活动识别。此外，本文还提出了一种基于卷积神经网络（CNN）和长短期记忆（LSTM）架构的新型深度学习模型作为基线，并证明了通过MiniRocket特征进行分类在较低计算成本下优于最佳的深度学习模型。使用PhysioNet数据集评估了所提方法的性能。所提模型在MiniRocket和CNN-LSTM上的平均准确率分别为98.63%和98.06%。研究结果表明，所提方法能够显著提高运动想象EEG的准确性，并为MI-EEG的特征提取和分类提供了新的见解。 

---
# STA-GANN: A Valid and Generalizable Spatio-Temporal Kriging Approach 

**Title (ZH)**: STA-GANN：一种有效的时空通用克里金方法 

**Authors**: Yujie Li, Zezhi Shao, Chengqing Yu, Tangwen Qian, Zhao Zhang, Yifan Du, Shaoming He, Fei Wang, Yongjun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16161)  

**Abstract**: Spatio-temporal tasks often encounter incomplete data arising from missing or inaccessible sensors, making spatio-temporal kriging crucial for inferring the completely missing temporal information. However, current models struggle with ensuring the validity and generalizability of inferred spatio-temporal patterns, especially in capturing dynamic spatial dependencies and temporal shifts, and optimizing the generalizability of unknown sensors. To overcome these limitations, we propose Spatio-Temporal Aware Graph Adversarial Neural Network (STA-GANN), a novel GNN-based kriging framework that improves spatio-temporal pattern validity and generalization. STA-GANN integrates (i) Decoupled Phase Module that senses and adjusts for timestamp shifts. (ii) Dynamic Data-Driven Metadata Graph Modeling to update spatial relationships using temporal data and metadata; (iii) An adversarial transfer learning strategy to ensure generalizability. Extensive validation across nine datasets from four fields and theoretical evidence both demonstrate the superior performance of STA-GANN. 

**Abstract (ZH)**: 基于时空感知的图对抗神经网络（STA-GANN）：一种改进时空插值有效性和泛化的新型GNN框架 

---
# Beyond Human-prompting: Adaptive Prompt Tuning with Semantic Alignment for Anomaly Detection 

**Title (ZH)**: 超越人类提示：基于语义对齐的自适应提示调谐在异常检测中的应用 

**Authors**: Pi-Wei Chen, Jerry Chun-Wei Lin, Wei-Han Chen, Jia Ji, Zih-Ching Chen, Feng-Hao Yeh, Chao-Chun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16157)  

**Abstract**: Pre-trained Vision-Language Models (VLMs) have recently shown promise in detecting anomalies. However, previous approaches are fundamentally limited by their reliance on human-designed prompts and the lack of accessible anomaly samples, leading to significant gaps in context-specific anomaly understanding. In this paper, we propose \textbf{A}daptive \textbf{P}rompt \textbf{T}uning with semantic alignment for anomaly detection (APT), a groundbreaking prior knowledge-free, few-shot framework and overcomes the limitations of traditional prompt-based approaches. APT uses self-generated anomaly samples with noise perturbations to train learnable prompts that capture context-dependent anomalies in different scenarios. To prevent overfitting to synthetic noise, we propose a Self-Optimizing Meta-prompt Guiding Scheme (SMGS) that iteratively aligns the prompts with general anomaly semantics while incorporating diverse synthetic anomaly. Our system not only advances pixel-wise anomaly detection, but also achieves state-of-the-art performance on multiple benchmark datasets without requiring prior knowledge for prompt crafting, establishing a robust and versatile solution for real-world anomaly detection. 

**Abstract (ZH)**: 自适应提示调优与语义对齐的异常检测（APT） 

---
# On the Collapse Errors Induced by the Deterministic Sampler for Diffusion Models 

**Title (ZH)**: 由确定性采样器引起的扩散模型中的塌陷错误 

**Authors**: Yi Zhang, Zhenyu Liao, Jingfeng Wu, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2508.16154)  

**Abstract**: Despite the widespread adoption of deterministic samplers in diffusion models (DMs), their potential limitations remain largely unexplored. In this paper, we identify collapse errors, a previously unrecognized phenomenon in ODE-based diffusion sampling, where the sampled data is overly concentrated in local data space. To quantify this effect, we introduce a novel metric and demonstrate that collapse errors occur across a variety of settings. When investigating its underlying causes, we observe a see-saw effect, where score learning in low noise regimes adversely impacts the one in high noise regimes. This misfitting in high noise regimes, coupled with the dynamics of deterministic samplers, ultimately causes collapse errors. Guided by these insights, we apply existing techniques from sampling, training, and architecture to empirically support our explanation of collapse errors. This work provides intensive empirical evidence of collapse errors in ODE-based diffusion sampling, emphasizing the need for further research into the interplay between score learning and deterministic sampling, an overlooked yet fundamental aspect of diffusion models. 

**Abstract (ZH)**: 尽管确定性采样器在扩散模型（DMs）中的广泛应用，其潜在限制仍未得到充分探索。在本文中，我们识别出在基于ODE的扩散采样中的一种先前未被认识到的现象——集中误差，即采样数据过度集中在局部数据空间中。为了量化这一效应，我们引入了一个新的度量标准，并证明了集中误差在多种设置中都会发生。在探究其根本原因时，我们观察到一种跷跷板效应，即在低噪声环境下得分学习对高噪声环境下的得分学习产生不利影响。高噪声环境下得分学习的这种不匹配，结合确定性采样器的动力学，最终导致了集中误差。根据这些见解，我们应用来自采样、训练和架构的现有技术来实证支持我们对集中误差的解释。本文提供了基于ODE的扩散采样中集中误差的密集实验证据，强调了得分学习与确定性采样之间相互作用的进一步研究需求，这是扩散模型中一个被忽视但根本方面。 

---
# Machine Learning in Micromobility: A Systematic Review of Datasets, Techniques, and Applications 

**Title (ZH)**: 微移动性中的机器学习：数据集、技术及应用的系统性综述 

**Authors**: Sen Yan, Chinmaya Kaundanya, Noel E. O'Connor, Suzanne Little, Mingming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16135)  

**Abstract**: Micromobility systems, which include lightweight and low-speed vehicles such as bicycles, e-bikes, and e-scooters, have become an important part of urban transportation and are used to solve problems such as traffic congestion, air pollution, and high transportation costs. Successful utilisation of micromobilities requires optimisation of complex systems for efficiency, environmental impact mitigation, and overcoming technical challenges for user safety. Machine Learning (ML) methods have been crucial to support these advancements and to address their unique challenges. However, there is insufficient literature addressing the specific issues of ML applications in micromobilities. This survey paper addresses this gap by providing a comprehensive review of datasets, ML techniques, and their specific applications in micromobilities. Specifically, we collect and analyse various micromobility-related datasets and discuss them in terms of spatial, temporal, and feature-based characteristics. In addition, we provide a detailed overview of ML models applied in micromobilities, introducing their advantages, challenges, and specific use cases. Furthermore, we explore multiple ML applications, such as demand prediction, energy management, and safety, focusing on improving efficiency, accuracy, and user experience. Finally, we propose future research directions to address these issues, aiming to help future researchers better understand this field. 

**Abstract (ZH)**: 微移动系统，包括轻型低速交通工具如自行车、电动自行车和电动滑板车，已成为城市交通的重要组成部分，用于解决交通拥堵、空气污染和高交通成本等问题。成功利用微移动性需要优化复杂系统以提高效率、缓解环境影响并克服技术挑战以确保用户安全。机器学习（ML）方法在这些进展中发挥了关键作用，并解决了其独特的挑战。然而，关于ML在微移动性中的具体应用的文献尚显不足。本文通过提供关于数据集、ML技术及其在微移动性中的具体应用的综合回顾，来填补这一空白。具体而言，我们收集和分析了各种与微移动性相关的数据集，并从空间、时间和特征角度讨论它们。此外，我们详细介绍了应用于微移动性的ML模型，介绍了它们的优势、挑战和具体应用案例。进一步地，我们探索了多种ML应用，如需求预测、能源管理与安全，重点关注提高效率、准确性和用户体验。最后，我们提出未来研究方向，旨在帮助未来研究人员更好地理解这一领域。 

---
# Spacetime-GR: A Spacetime-Aware Generative Model for Large Scale Online POI Recommendation 

**Title (ZH)**: 时空-GR：一种面向时空的生成模型，用于大规模在线地点推荐 

**Authors**: Haitao Lin, Zhen Yang, Jiawei Xue, Ziji Zhang, Luzhu Wang, Yikun Gu, Yao Xu, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.16126)  

**Abstract**: Building upon the strong sequence modeling capability, Generative Recommendation (GR) has gradually assumed a dominant position in the application of recommendation tasks (e.g., video and product recommendation). However, the application of Generative Recommendation in Point-of-Interest (POI) recommendation, where user preferences are significantly affected by spatiotemporal variations, remains a challenging open problem. In this paper, we propose Spacetime-GR, the first spacetime-aware generative model for large-scale online POI recommendation. It extends the strong sequence modeling ability of generative models by incorporating flexible spatiotemporal information encoding. Specifically, we first introduce a geographic-aware hierarchical POI indexing strategy to address the challenge of large vocabulary modeling. Subsequently, a novel spatiotemporal encoding module is introduced to seamlessly incorporate spatiotemporal context into user action sequences, thereby enhancing the model's sensitivity to spatiotemporal variations. Furthermore, we incorporate multimodal POI embeddings to enrich the semantic understanding of each POI. Finally, to facilitate practical deployment, we develop a set of post-training adaptation strategies after sufficient pre-training on action sequences. These strategies enable Spacetime-GR to generate outputs in multiple formats (i.e., embeddings, ranking scores and POI candidates) and support a wide range of downstream application scenarios (i.e., ranking and end-to-end recommendation). We evaluate the proposed model on both public benchmark datasets and large-scale industrial datasets, demonstrating its superior performance over existing methods in terms of POI recommendation accuracy and ranking quality. Furthermore, the model is the first generative model deployed in online POI recommendation services that scale to hundreds of millions of POIs and users. 

**Abstract (ZH)**: 基于强大的序列建模能力，生成推荐（GR）已经在视频和产品推荐等推荐任务中占据了主导地位。然而，在用户偏好受时空变化显著影响的点_of_兴趣（POI）推荐中应用生成推荐仍是一个具有挑战性的开放问题。在本文中，我们提出了一种时空意识生成模型Spacetime-GR，这是首个针对大规模在线POI推荐的应用。该模型通过整合灵活的时空信息编码，扩展了生成模型的强大序列建模能力。具体而言，我们首先引入了一种地理意识的分层POI索引策略，以应对大规模词汇量建模的挑战。随后，我们提出了一种新型的时空编码模块，无缝地将时空上下文融入用户行为序列中，从而增强模型对时空变化的敏感性。此外，我们整合了多模态POI嵌入，以丰富每个POI的语义理解。最后，为了促进实际部署，我们开发了一系列在充分预训练后对行为序列的适应性策略。这些策略使Spacetime-GR能够生成多种格式的输出（即嵌入、排名分数和POI候选），并支持广泛的下游应用场景（即排名和端到端推荐）。我们在公共基准数据集和大规模工业数据集上评估了该模型，表明在POI推荐准确性和排名质量方面，其性能优于现有方法。此外，该模型是首个部署于在线POI推荐服务中的生成模型，能够处理数亿个POI和用户。 

---
# ANSC: Probabilistic Capacity Health Scoring for Datacenter-Scale Reliability 

**Title (ZH)**: ANSC：数据中继规模可靠性的情感概率容量健康评分 

**Authors**: Madhava Gaikwad, Abhishek Gandhi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16119)  

**Abstract**: We present ANSC, a probabilistic capacity health scoring framework for hyperscale datacenter fabrics. While existing alerting systems detect individual device or link failures, they do not capture the aggregate risk of cascading capacity shortfalls. ANSC provides a color-coded scoring system that indicates the urgency of issues \emph{not solely by current impact, but by the probability of imminent capacity violations}. Our system accounts for both current residual capacity and the probability of additional failures, normalized at datacenter and regional level. We demonstrate that ANSC enables operators to prioritize remediation across more than 400 datacenters and 60 regions, reducing noise and aligning SRE focus on the most critical risks. 

**Abstract (ZH)**: ANSC：一种面向大规模数据中心 fabrics 的概率容量健康评分框架 

---
# GPLight+: A Genetic Programming Method for Learning Symmetric Traffic Signal Control Policy 

**Title (ZH)**: GPLight+: 基于遗传编程的对称交通信号控制策略学习方法 

**Authors**: Xiao-Cheng Liao, Yi Mei, Mengjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16090)  

**Abstract**: Recently, learning-based approaches, have achieved significant success in automatically devising effective traffic signal control strategies. In particular, as a powerful evolutionary machine learning approach, Genetic Programming (GP) is utilized to evolve human-understandable phase urgency functions to measure the urgency of activating a green light for a specific phase. However, current GP-based methods are unable to treat the common traffic features of different traffic signal phases consistently. To address this issue, we propose to use a symmetric phase urgency function to calculate the phase urgency for a specific phase based on the current road conditions. This is represented as an aggregation of two shared subtrees, each representing the urgency of a turn movement in the phase. We then propose a GP method to evolve the symmetric phase urgency function. We evaluate our proposed method on the well-known cityflow traffic simulator, based on multiple public real-world datasets. The experimental results show that the proposed symmetric urgency function representation can significantly improve the performance of the learned traffic signal control policies over the traditional GP representation on a wide range of scenarios. Further analysis shows that the proposed method can evolve effective, human-understandable and easily deployable traffic signal control policies. 

**Abstract (ZH)**: 基于学习的方法在自动生成有效的交通信号控制策略方面取得了显著成功。特别是，作为一种强大的进化机器学习方法，遗传编程（GP）被用于演化出人类可理解的相位紧迫函数，以衡量为特定相位激活绿灯的紧迫性。然而，当前基于GP的方法无法一致地处理不同交通信号相位的常见交通特征。为解决这一问题，我们提出使用对称相位紧迫函数来基于当前道路条件计算特定相位的紧迫性。这表示为两个共享子树的聚合，每个子树代表相位中转弯运动的紧迫性。然后，我们提出了一种GP方法来演化对称相位紧迫函数。我们在著名的CityFlow交通模拟器和多个公开的真实世界数据集上评估了我们提出的方法。实验结果表明，所提出的对称紧迫性函数表示在广泛的场景下可以显著提高传统GP表示学习到的交通信号控制策略的性能。进一步的分析表明，所提出的方法可以演化出有效的、人类可理解的和易于部署的交通信号控制策略。 

---
# On Task Vectors and Gradients 

**Title (ZH)**: 关于任务向量和梯度 

**Authors**: Luca Zhou, Daniele Solombrino, Donato Crisostomi, Maria Sofia Bucarelli, Giuseppe Alessio D'Inverno, Fabrizio Silvestri, Emanuele Rodolà  

**Link**: [PDF](https://arxiv.org/pdf/2508.16082)  

**Abstract**: Task arithmetic has emerged as a simple yet powerful technique for model merging, enabling the combination of multiple finetuned models into one. Despite its empirical success, a clear theoretical explanation of why and when it works is lacking. This paper provides a rigorous theoretical foundation for task arithmetic by establishing a connection between task vectors and gradients of the task losses. We show that under standard gradient descent, a task vector generated from one epoch of finetuning is exactly equivalent to the negative gradient of the loss, scaled by the learning rate. For the practical multi-epoch setting, we prove that this equivalence holds approximately, with a second-order error term that we explicitly bound for feed-forward networks. Our empirical analysis across seven vision benchmarks corroborates our theory, demonstrating that the first-epoch gradient dominates the finetuning trajectory in both norm and direction. A key implication is that merging models finetuned for only a single epoch often yields performance comparable to merging fully converged models. These findings reframe task arithmetic as a form of approximate multitask learning, providing a clear rationale for its effectiveness and highlighting the critical role of early training dynamics in model merging. 

**Abstract (ZH)**: TASK 算术作为一种连接任务向量与任务损失梯度的理论基础：一个严格的理论框架 

---
# Enhanced predictions of the Madden-Julian oscillation using the FuXi-S2S machine learning model: Insights into physical mechanisms 

**Title (ZH)**: 使用FuXi-S2S机器学习模型增强Madden-Julian振荡的预测：物理机制的见解 

**Authors**: Can Cao, Xiaohui Zhong, Lei Chen, Zhiwei Wua, Hao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.16041)  

**Abstract**: The Madden-Julian Oscillation (MJO) is the dominant mode of tropical atmospheric variability on intraseasonal timescales, and reliable MJO predictions are essential for protecting lives and mitigating impacts on societal assets. However, numerical models still fall short of achieving the theoretical predictability limit for the MJO due to inherent constraints. In an effort to extend the skillful prediction window for the MJO, machine learning (ML) techniques have gained increasing attention. This study examines the MJO prediction performance of the FuXi subseasonal-to-seasonal (S2S) ML model during boreal winter, comparing it with the European Centre for Medium- Range Weather Forecasts S2S model. Results indicate that for the initial strong MJO phase 3, the FuXi-S2S model demonstrates reduced biases in intraseasonal outgoing longwave radiation anomalies averaged over the tropical western Pacific (WP) region during days 15-20, with the convective center located over this area. Analysis of multiscale interactions related to moisture transport suggests that improvements could be attributed to the FuXi-S2S model's more accurate prediction of the area-averaged meridional gradient of low-frequency background moisture over the tropical WP. These findings not only explain the enhanced predictive capability of the FuXi-S2S model but also highlight the potential of ML approaches in advancing the MJO forecasting. 

**Abstract (ZH)**: FuXi 子季节至季节 ML 模型在北半球冬季 MJO 预报性能及其机理分析 

---
# Pareto Actor-Critic for Communication and Computation Co-Optimization in Non-Cooperative Federated Learning Services 

**Title (ZH)**: Pareto 原则Actor-critic方法在非合作联邦学习服务中的通信与计算协同优化 

**Authors**: Renxuan Tan, Rongpeng Li, Xiaoxue Yu, Xianfu Chen, Xing Xu, Zhifeng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.16037)  

**Abstract**: Federated learning (FL) in multi-service provider (SP) ecosystems is fundamentally hampered by non-cooperative dynamics, where privacy constraints and competing interests preclude the centralized optimization of multi-SP communication and computation resources. In this paper, we introduce PAC-MCoFL, a game-theoretic multi-agent reinforcement learning (MARL) framework where SPs act as agents to jointly optimize client assignment, adaptive quantization, and resource allocation. Within the framework, we integrate Pareto Actor-Critic (PAC) principles with expectile regression, enabling agents to conjecture optimal joint policies to achieve Pareto-optimal equilibria while modeling heterogeneous risk profiles. To manage the high-dimensional action space, we devise a ternary Cartesian decomposition (TCAD) mechanism that facilitates fine-grained control. Further, we develop PAC-MCoFL-p, a scalable variant featuring a parameterized conjecture generator that substantially reduces computational complexity with a provably bounded error. Alongside theoretical convergence guarantees, our framework's superiority is validated through extensive simulations -- PAC-MCoFL achieves approximately 5.8% and 4.2% improvements in total reward and hypervolume indicator (HVI), respectively, over the latest MARL solutions. The results also demonstrate that our method can more effectively balance individual SP and system performance in scaled deployments and under diverse data heterogeneity. 

**Abstract (ZH)**: 多服务提供商基于多代理强化学习的联邦学习框架（PAC-MCoFL） 

---
# Time Series Based Network Intrusion Detection using MTF-Aided Transformer 

**Title (ZH)**: 基于时间序列的网络入侵检测：MTF辅助的变压器方法 

**Authors**: Poorvi Joshi, Mohan Gurusamy  

**Link**: [PDF](https://arxiv.org/pdf/2508.16035)  

**Abstract**: This paper introduces a novel approach to time series classification using a Markov Transition Field (MTF)-aided Transformer model, specifically designed for Software-Defined Networks (SDNs). The proposed model integrates the temporal dependency modeling strengths of MTFs with the sophisticated pattern recognition capabilities of Transformer architectures. We evaluate the model's performance using the InSDN dataset, demonstrating that our model outperforms baseline classification models, particularly in data-constrained environments commonly encountered in SDN applications. We also highlight the relationship between the MTF and Transformer components, which leads to better performance, even with limited data. Furthermore, our approach achieves competitive training and inference times, making it an efficient solution for real-world SDN applications. These findings establish the potential of MTF-aided Transformers to address the challenges of time series classification in SDNs, offering a promising path for reliable and scalable analysis in scenarios with sparse data. 

**Abstract (ZH)**: 本文介绍了一种使用Markov Transition Field (MTF)-辅助Transformer模型的新颖时间序列分类方法，特别适用于软件定义网络（SDNs）。所提出模型整合了MTF的时间依赖性建模优势与Transformer架构的复杂模式识别能力。我们使用InSDN数据集评估模型性能，结果显示我们的模型在数据受限环境中尤其优于基础分类模型。我们还强调了MTF和Transformer组件之间的关系，这使得即使是有限数据的情况下模型性能也更好。此外，我们的方法实现了竞争性的训练和推理时间，使其成为实际SDN应用的有效解决方案。这些 findings 确立了MTF-辅助Transformer在解决SDNs中时间序列分类挑战方面的潜力，为在稀疏数据场景下提供可靠和可扩展分析提供了有前途的道路。 

---
# Breaking Barriers in Software Testing: The Power of AI-Driven Automation 

**Title (ZH)**: 突破软件测试的障碍：AI驱动自动化的力量 

**Authors**: Saba Naqvi, Mohammad Baqar  

**Link**: [PDF](https://arxiv.org/pdf/2508.16025)  

**Abstract**: Software testing remains critical for ensuring reliability, yet traditional approaches are slow, costly, and prone to gaps in coverage. This paper presents an AI-driven framework that automates test case generation and validation using natural language processing (NLP), reinforcement learning (RL), and predictive models, embedded within a policy-driven trust and fairness model. The approach translates natural language requirements into executable tests, continuously optimizes them through learning, and validates outcomes with real-time analysis while mitigating bias. Case studies demonstrate measurable gains in defect detection, reduced testing effort, and faster release cycles, showing that AI-enhanced testing improves both efficiency and reliability. By addressing integration and scalability challenges, the framework illustrates how AI can shift testing from a reactive, manual process to a proactive, adaptive system that strengthens software quality in increasingly complex environments. 

**Abstract (ZH)**: 基于AI的框架通过自然语言处理、强化学习和预测模型自动化测试用例的生成与验证，同时嵌入政策驱动的信任和公平性模型，以确保软件可靠性并提高测试效率。 

---
# Strategic Sample Selection for Improved Clean-Label Backdoor Attacks in Text Classification 

**Title (ZH)**: 改进文本分类中纯净标签后门攻击的策略性样本选择 

**Authors**: Onur Alp Kirci, M. Emre Gursoy  

**Link**: [PDF](https://arxiv.org/pdf/2508.15934)  

**Abstract**: Backdoor attacks pose a significant threat to the integrity of text classification models used in natural language processing. While several dirty-label attacks that achieve high attack success rates (ASR) have been proposed, clean-label attacks are inherently more difficult. In this paper, we propose three sample selection strategies to improve attack effectiveness in clean-label scenarios: Minimum, Above50, and Below50. Our strategies identify those samples which the model predicts incorrectly or with low confidence, and by injecting backdoor triggers into such samples, we aim to induce a stronger association between the trigger patterns and the attacker-desired target label. We apply our methods to clean-label variants of four canonical backdoor attacks (InsertSent, WordInj, StyleBkd, SynBkd) and evaluate them on three datasets (IMDB, SST2, HateSpeech) and four model types (LSTM, BERT, DistilBERT, RoBERTa). Results show that the proposed strategies, particularly the Minimum strategy, significantly improve the ASR over random sample selection with little or no degradation in the model's clean accuracy. Furthermore, clean-label attacks enhanced by our strategies outperform BITE, a state of the art clean-label attack method, in many configurations. 

**Abstract (ZH)**: 文本分类模型中 Cleaner 标签后门攻击对自然语言处理中的完整性构成重大威胁。虽然已经提出了多种实现高攻击成功率的脏标签攻击方法，但清洁标签攻击本质上更加困难。本文提出三种样本选择策略以提高清洁标签场景下的攻击有效性：Minimum、Above50 和 Below50。我们的策略识别出模型预测错误或置信度低的样本，并通过在这些样本中注入后门触发器，旨在增强触发模式与攻击者期望的目标标签之间的关联。我们将方法应用于四个经典后门攻击（InsertSent、WordInj、StyleBkd、SynBkd）的清洁标签变体，并在三个数据集（IMDB、SST2、HateSpeech）和四种模型类型（LSTM、BERT、DistilBERT、RoBERTa）上进行了评估。结果表明，所提出的方法，尤其是 Minimum 策略，相对于随机样本选择显著提高了攻击成功率，且几乎不对模型的清洁准确性造成降级。此外，由我们策略增强的清洁标签攻击在许多配置中优于当前最先进的 BITE 方法。 

---
# Probabilistic Forecasting Cryptocurrencies Volatility: From Point to Quantile Forecasts 

**Title (ZH)**: 加密货币波动性的概率预测：从点预测到分位数预测 

**Authors**: Grzegorz Dudek, Witold Orzeszko, Piotr Fiszeder  

**Link**: [PDF](https://arxiv.org/pdf/2508.15922)  

**Abstract**: Cryptocurrency markets are characterized by extreme volatility, making accurate forecasts essential for effective risk management and informed trading strategies. Traditional deterministic (point) forecasting methods are inadequate for capturing the full spectrum of potential volatility outcomes, underscoring the importance of probabilistic approaches. To address this limitation, this paper introduces probabilistic forecasting methods that leverage point forecasts from a wide range of base models, including statistical (HAR, GARCH, ARFIMA) and machine learning (e.g. LASSO, SVR, MLP, Random Forest, LSTM) algorithms, to estimate conditional quantiles of cryptocurrency realized variance. To the best of our knowledge, this is the first study in the literature to propose and systematically evaluate probabilistic forecasts of variance in cryptocurrency markets based on predictions derived from multiple base models. Our empirical results for Bitcoin demonstrate that the Quantile Estimation through Residual Simulation (QRS) method, particularly when applied to linear base models operating on log-transformed realized volatility data, consistently outperforms more sophisticated alternatives. Additionally, we highlight the robustness of the probabilistic stacking framework, providing comprehensive insights into uncertainty and risk inherent in cryptocurrency volatility forecasting. This research fills a significant gap in the literature, contributing practical probabilistic forecasting methodologies tailored specifically to cryptocurrency markets. 

**Abstract (ZH)**: 加密货币市场以极端波动性为特征，准确的预测对于有效的风险管理及知情交易策略至关重要。传统的确定性预测方法难以捕捉潜在波动性的完整谱系，突显了概率方法的重要性。为解决这一局限性，本文引入了利用一系列基础模型（包括统计模型HAR、GARCH、ARFIMA和机器学习算法LASSO、SVR、MLP、随机森林、LSTM）的点预测来估计加密货币实现波动率条件分位数的概率预测方法。据我们所知，这是文献中首次提出并系统评估基于多模型预测的加密货币市场波动率概率预测的研究。我们的实证结果表明，残差模拟分位数估计（QRS）方法，特别是在对数变换实现波动率数据上的线性模型中应用时，一致地优于更复杂的替代方案。此外，我们强调了概率叠加框架的稳健性，提供了对加密货币波动率预测中固有的不确定性和风险的全面见解。本研究填补了文献中的一个重要空白，为加密货币市场提供了实用的概率预测方法。 

---
# Information Ecosystem Reengineering via Public Sector Knowledge Representation 

**Title (ZH)**: 公共部门知识表征驱动的信息生态系统重塑 

**Authors**: Mayukh Bagchi  

**Link**: [PDF](https://arxiv.org/pdf/2508.15916)  

**Abstract**: Information Ecosystem Reengineering (IER) -- the technological reconditioning of information sources, services, and systems within a complex information ecosystem -- is a foundational challenge in the digital transformation of public sector services and smart governance platforms. From a semantic knowledge management perspective, IER becomes especially entangled due to the potentially infinite number of possibilities in its conceptualization, namely, as a result of manifoldness in the multi-level mix of perception, language and conceptual interlinkage implicit in all agents involved in such an effort. This paper proposes a novel approach -- Representation Disentanglement -- to disentangle these multiple layers of knowledge representation complexity hindering effective reengineering decision making. The approach is based on the theoretically grounded and implementationally robust ontology-driven conceptual modeling paradigm which has been widely adopted in systems analysis and (re)engineering. We argue that such a framework is essential to achieve explainability, traceability and semantic transparency in public sector knowledge representation and to support auditable decision workflows in governance ecosystems increasingly driven by Artificial Intelligence (AI) and data-centric architectures. 

**Abstract (ZH)**: 信息生态系统重塑（IER）——在复杂信息生态系统中重塑信息源、服务和系统的技术改造——是公共部门服务和智能治理平台数字化转型中的基础性挑战。从语义知识管理的角度来看，由于其概念化过程中潜在的无限可能性，特别是由于涉及此类努力的所有参与者在感知、语言和概念关联方面的多层次多样性，IER变得尤为复杂。本文提出了一种新颖的方法——表示解纠缠（Representation Disentanglement）——以解开妨碍有效重塑决策的多层知识表示复杂性。该方法基于在系统分析与（重）工程中广泛采用的有理论依据且实施稳健的本体驱动概念建模框架。我们认为，这种框架对于在公共部门知识表示中实现可解释性、可追溯性和语义透明性，以及支持由人工智能（AI）和数据为中心的架构驱动的治理生态系统中的可审计决策流程是必不可少的。 

---
# TPLA: Tensor Parallel Latent Attention for Efficient Disaggregated Prefill \& Decode Inference 

**Title (ZH)**: TPLA：张量并行隐式注意力机制以实现高效的解聚合前填充与解码推理 

**Authors**: Xiaojuan Tang, Fanxu Meng, Pingzhi Tang, Yuxuan Wang, Di Yin, Xing Sun, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15881)  

**Abstract**: Multi-Head Latent Attention (MLA), introduced in DeepSeek-V2, compresses key-value states into a low-rank latent vector, caching only this vector to reduce memory. In tensor parallelism (TP), however, attention heads are computed across multiple devices, and each device must load the full cache, eroding the advantage of MLA over Grouped Query Attention (GQA). We propose Tensor-Parallel Latent Attention (TPLA): a scheme that partitions both the latent representation and each head's input dimension across devices, performs attention independently per shard, and then combines results with an all-reduce. TPLA preserves the benefits of a compressed KV cache while unlocking TP efficiency. Unlike Grouped Latent Attention (GLA), every head in TPLA still leverages the full latent representation, maintaining stronger representational capacity. TPLA is drop-in compatible with models pre-trained using MLA: it supports MLA-style prefilling and enables efficient tensor-parallel decoding without retraining. Applying simple orthogonal transforms -- e.g., the Hadamard transform or PCA -- before TP slicing further mitigates cross-shard interference, yielding minimal accuracy degradation. By reducing the per-device KV cache for DeepSeek-V3 and Kimi-K2, we achieve 1.79x and 1.93x speedups, respectively, at a 32K-token context length while maintaining performance on commonsense and LongBench benchmarks. TPLA can be implemented with FlashAttention-3, enabling practical end-to-end acceleration. 

**Abstract (ZH)**: Tensor-Parallel Latent Attention (TPLA): 一种保持压缩键值缓存优势并提升张量并行效率的方法 

---
# Lean Meets Theoretical Computer Science: Scalable Synthesis of Theorem Proving Challenges in Formal-Informal Pairs 

**Title (ZH)**: Lean 与理论计算机科学相结合：形式化-非形式化pair中定理证明挑战的大规模合成 

**Authors**: Terry Jingchen Zhang, Wenyuan Jiang, Rongchuan Liu, Yisong Wang, Junran Yang, Ning Wang, Nicole Ni, Yinya Huang, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2508.15878)  

**Abstract**: Formal theorem proving (FTP) has emerged as a critical foundation for evaluating the reasoning capabilities of large language models, enabling automated verification of mathematical proofs at scale. However, progress has been constrained by limited datasets due to the high cost of manual curation and the scarcity of challenging problems with verified formal-informal correspondences. We propose leveraging theoretical computer science (TCS) as a scalable source of rigorous proof problems, where algorithmic definitions enable automated generation of arbitrarily many challenging theorem-proof pairs. We demonstrate this approach on two TCS domains: Busy Beaver problems, which involve proving bounds on Turing machine halting behavior, and Mixed Boolean Arithmetic problems, which combine logical and arithmetic reasoning. Our framework automatically synthesizes problems with parallel formal (Lean4) and informal (Markdown) specifications, creating a scalable pipeline for generating verified proof challenges. Evaluation on frontier models reveals substantial gaps in automated theorem proving: while DeepSeekProver-V2-671B achieves 57.5\% success on Busy Beaver problems, it manages only 12\% on Mixed Boolean Arithmetic problems. These results highlight the difficulty of long-form proof generation even for problems that are computationally easy to verify, demonstrating the value of TCS domains for advancing automated reasoning research. 

**Abstract (ZH)**: 形式化定理证明（FTP）已成为评估大型语言模型推理能力的关键基础，能够大规模自动化验证数学证明。然而，由于手动整理数据的成本高昂以及缺乏具有验证形式化对应关系的具有挑战性的问题，进展受到了限制。我们提出利用理论计算机科学（TCS）作为生成严格证明问题的可扩展来源，其中算法定义使能够自动化生成任意多个具有挑战性的定理-证明对。我们在此方法上对两个TCS领域进行了演示：Busy Beaver问题，涉及证明图灵机停止行为的边界；以及混合布尔算术问题，结合了逻辑和算术推理。我们的框架自动综合具有并行形式化（Lean4）和非形式化（Markdown）规范的问题，创建了一个可扩展的生成验证证明挑战的流水线。对前沿模型的评估揭示了自动定理证明中的巨大差距：虽然DeepSeekProver-V2-671B在Busy Beaver问题上的成功率为57.5%，在混合布尔算术问题上的成功率仅为12%。这些结果强调了即使对于计算上易于验证的问题，长形式证明生成的难度，表明TCS领域对于推进自动推理研究的价值。 

---
# Securing Swarms: Cross-Domain Adaptation for ROS2-based CPS Anomaly Detection 

**Title (ZH)**: 基于ROS2的 CPS 异常检测的跨域适应安全策略 

**Authors**: Julia Boone, Fatemeh Afghah  

**Link**: [PDF](https://arxiv.org/pdf/2508.15865)  

**Abstract**: Cyber-physical systems (CPS) are being increasingly utilized for critical applications. CPS combines sensing and computing elements, often having multi-layer designs with networking, computational, and physical interfaces, which provide them with enhanced capabilities for a variety of application scenarios. However, the combination of physical and computational elements also makes CPS more vulnerable to attacks compared to network-only systems, and the resulting impacts of CPS attacks can be substantial. Intelligent intrusion detection systems (IDS) are an effective mechanism by which CPS can be secured, but the majority of current solutions often train and validate on network traffic-only datasets, ignoring the distinct attacks that may occur on other system layers. In order to address this, we develop an adaptable CPS anomaly detection model that can detect attacks within CPS without the need for previously labeled data. To achieve this, we utilize domain adaptation techniques that allow us to transfer known attack knowledge from a network traffic-only environment to a CPS environment. We validate our approach using a state-of-the-art CPS intrusion dataset that combines network, operating system (OS), and Robot Operating System (ROS) data. Through this dataset, we are able to demonstrate the effectiveness of our model across network traffic-only and CPS environments with distinct attack types and its ability to outperform other anomaly detection methods. 

**Abstract (ZH)**: 基于物理与计算系统的智能异常检测模型研究 

---
# Beyond Individuals: Collective Predictive Coding for Memory, Attention, and the Emergence of Language 

**Title (ZH)**: 超越个体：集体预测编码与记忆、注意力及语言的 emergence 

**Authors**: Tadahiro Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2508.15859)  

**Abstract**: This commentary extends the discussion by Parr et al. on memory and attention beyond individual cognitive systems. From the perspective of the Collective Predictive Coding (CPC) hypothesis -- a framework for understanding these faculties and the emergence of language at the group level -- we introduce a hypothetical idea: that language, with its embedded distributional semantics, serves as a collectively formed external representation. CPC generalises the concepts of individual memory and attention to the collective level. This offers a new perspective on how shared linguistic structures, which may embrace collective world models learned through next-word prediction, emerge from and shape group-level cognition. 

**Abstract (ZH)**: 这一评述将帕等人的讨论从个体认知系统扩展至记忆和注意力。从集体预测编码（CPC）假说的角度——这一框架用于理解这些能力以及在其群体层面的涌现——我们提出一个假设：语言，带着嵌入的分布式语义，作为一种集体形成的外部表征而存在。CPC将个体记忆和注意力的概念扩展至群体层面。这提供了一种新的视角，用以理解这些共享的语言结构如何从接下来的词预测中涌现并塑造群体层面的认知。 

---
# MGSC: A Multi-granularity Consistency Framework for Robust End-to-end Asr 

**Title (ZH)**: MGSC：一种多粒度一致性框架用于稳健的端到端ASR 

**Authors**: Xuwen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15853)  

**Abstract**: End-to-end ASR models, despite their success on benchmarks, often pro-duce catastrophic semantic errors in noisy environments. We attribute this fragility to the prevailing 'direct mapping' objective, which solely penalizes final output errors while leaving the model's internal computational pro-cess unconstrained. To address this, we introduce the Multi-Granularity Soft Consistency (MGSC) framework, a model-agnostic, plug-and-play module that enforces internal self-consistency by simultaneously regulariz-ing macro-level sentence semantics and micro-level token alignment. Cru-cially, our work is the first to uncover a powerful synergy between these two consistency granularities: their joint optimization yields robustness gains that significantly surpass the sum of their individual contributions. On a public dataset, MGSC reduces the average Character Error Rate by a relative 8.7% across diverse noise conditions, primarily by preventing se-vere meaning-altering mistakes. Our work demonstrates that enforcing in-ternal consistency is a crucial step towards building more robust and trust-worthy AI. 

**Abstract (ZH)**: 端到端ASR模型在噪声环境下尽管在基准测试中取得成功，但往往会产生灾难性的语义错误。我们将其脆弱性归因于当前流行的“直接映射”目标，该目标仅惩罚最终输出错误，而不限制模型的内部计算过程。为此，我们引入了多粒度软一致性（MGSC）框架，这是一个模型无关的即插即用模块，通过同时正则化宏观级别的句子语义和微观级别的令牌对齐来强制内部自我一致性。关键的是，我们的工作首次揭示了这两种一致性粒度之间的强大 synergy：它们的联合优化产生的鲁棒性增益远远超过了它们个别贡献的总和。在公共数据集上，MGSC在多种噪声条件下相对降低了平均字符错误率8.7%，主要通过防止严重的意义改变错误。我们的工作证明了强制内部一致性是构建更鲁棒和可信AI的关键步骤。 

---
# CIA+TA Risk Assessment for AI Reasoning Vulnerabilities 

**Title (ZH)**: AI推理漏洞的CIA+TA风险评估 

**Authors**: Yuksel Aydin  

**Link**: [PDF](https://arxiv.org/pdf/2508.15839)  

**Abstract**: As AI systems increasingly influence critical decisions, they face threats that exploit reasoning mechanisms rather than technical infrastructure. We present a framework for cognitive cybersecurity, a systematic protection of AI reasoning processes from adversarial manipulation. Our contributions are threefold. First, we establish cognitive cybersecurity as a discipline complementing traditional cybersecurity and AI safety, addressing vulnerabilities where legitimate inputs corrupt reasoning while evading conventional controls. Second, we introduce the CIA+TA, extending traditional Confidentiality, Integrity, and Availability triad with Trust (epistemic validation) and Autonomy (human agency preservation), requirements unique to systems generating knowledge claims and mediating decisions. Third, we present a quantitative risk assessment methodology with empirically-derived coefficients, enabling organizations to measure cognitive security risks. We map our framework to OWASP LLM Top 10 and MITRE ATLAS, facilitating operational integration. Validation through previously published studies (151 human participants; 12,180 AI trials) reveals strong architecture dependence: identical defenses produce effects ranging from 96% reduction to 135% amplification of vulnerabilities. This necessitates pre-deployment Cognitive Penetration Testing as a governance requirement for trustworthy AI deployment. 

**Abstract (ZH)**: 随着AI系统越来越多地影响关键决策，它们面临利用推理机制而非技术基础设施的威胁。我们提出了一种认知网络安全框架，系统性地保护AI推理过程免受 adversarial 操纵。我们的贡献主要有三个方面。首先，我们确立认知网络安全作为一门与传统网络安全和AI安全相补充的学科，重点关注合法输入破坏推理的同时规避常规控制的脆弱性。其次，我们引入了CIA+TA（保密性、完整性、可用性及信任与自主），在传统三元组中增加了知识声明生成系统和决策调解所需的信任与人类代理保存要求。第三，我们提出了一种定量风险评估方法，采用经验得出的系数，使组织能够衡量认知安全风险。我们将我们的框架映射到OWASP LLM Top 10和MITRE ATLAS，促进操作集成。通过之前的发表研究（151名人类参与者；12,180次AI试验）验证表明，认知架构依赖性强：相同的防御措施会产生从96%的漏洞减少到135%的漏洞放大效果。这要求在部署可信赖AI之前进行认知渗透测试作为治理要求。 

---
# Statistical Comparative Analysis of Semantic Similarities and Model Transferability Across Datasets for Short Answer Grading 

**Title (ZH)**: 基于数据集间语义相似性和模型可迁移性的统计比较分析用于简答评分 

**Authors**: Sridevi Bonthu, S.Rama Sree, M.H.M. Krishna Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2508.15837)  

**Abstract**: Developing dataset-specific models involves iterative fine-tuning and optimization, incurring significant costs over time. This study investigates the transferability of state-of-the-art (SOTA) models trained on established datasets to an unexplored text dataset. The key question is whether the knowledge embedded within SOTA models from existing datasets can be harnessed to achieve high-performance results on a new domain. In pursuit of this inquiry, two well-established benchmarks, the STSB and Mohler datasets, are selected, while the recently introduced SPRAG dataset serves as the unexplored domain. By employing robust similarity metrics and statistical techniques, a meticulous comparative analysis of these datasets is conducted. The primary goal of this work is to yield comprehensive insights into the potential applicability and adaptability of SOTA models. The outcomes of this research have the potential to reshape the landscape of natural language processing (NLP) by unlocking the ability to leverage existing models for diverse datasets. This may lead to a reduction in the demand for resource-intensive, dataset-specific training, thereby accelerating advancements in NLP and paving the way for more efficient model deployment. 

**Abstract (ZH)**: 基于最新模型在未探索文本数据集上的迁移性研究 

---
# MorphNAS: Differentiable Architecture Search for Morphologically-Aware Multilingual NER 

**Title (ZH)**: MorphNAS：面向形态意识的多语言NER可微架构搜索 

**Authors**: Prathamesh Devadiga, Omkaar Jayadev Shetty, Hiya Nachnani, Prema R  

**Link**: [PDF](https://arxiv.org/pdf/2508.15836)  

**Abstract**: Morphologically complex languages, particularly multiscript Indian languages, present significant challenges for Natural Language Processing (NLP). This work introduces MorphNAS, a novel differentiable neural architecture search framework designed to address these challenges. MorphNAS enhances Differentiable Architecture Search (DARTS) by incorporating linguistic meta-features such as script type and morphological complexity to optimize neural architectures for Named Entity Recognition (NER). It automatically identifies optimal micro-architectural elements tailored to language-specific morphology. By automating this search, MorphNAS aims to maximize the proficiency of multilingual NLP models, leading to improved comprehension and processing of these complex languages. 

**Abstract (ZH)**: 形态学复杂语言，尤其是多字符集印度语言，给自然语言处理（NLP）带来了显著挑战。本文介绍了一种新的可微神经网络架构搜索框架MorphNAS，旨在应对这些挑战。MorphNAS通过整合如字符类型和形态复杂度等语言元特征，优化神经架构以提高命名实体识别（NER）的效果。该框架自动识别适应特定语言形态的最优微架构元素。通过自动化这一搜索过程，MorphNAS旨在最大化多语言NLP模型的能力，从而提高对这些复杂语言的理解和处理。 

---
# Alvorada-Bench: Can Language Models Solve Brazilian University Entrance Exams? 

**Title (ZH)**: Alvorada-Bench: 语言模型能解决巴西大学入学考试吗？ 

**Authors**: Henrique Godoy  

**Link**: [PDF](https://arxiv.org/pdf/2508.15835)  

**Abstract**: Language models are increasingly used in Brazil, but most evaluation remains English-centric. This paper presents Alvorada-Bench, a 4,515-question, text-only benchmark drawn from five Brazilian university entrance examinations. Evaluating twenty models under zero-shot, role-playing, and chain-of-thought prompting, producing 270,900 responses with structured self-reports of confidence, perceived difficulty, and Bloom level. The top models exceed 94% accuracy overall, but accuracy declines on Mathematics and on the engineering oriented IME and ITA exams, indicating persistent weaknesses in multi-step reasoning. Confidence is well calibrated and correlates with perceived difficulty, revealing that models can accurately assess their own certainty capabilities. A cost accuracy analysis shows that high accuracy is achievable at under $2 per 1K tokens. On ENEM 2024 the top model (O3) achieved perfect scores in Languages subject questions while even the weakest system (GPT-4.1 Nano) only underperforms humans in Mathematics. Through exams that distill decades of Brazilian educational priorities and assess millions of students yearly, Alvorada-Bench establishes whether language models can navigate the intersection of language, culture, and reasoning that defines academic readiness in Brazil. 

**Abstract (ZH)**: 语言模型在巴西的应用日益增多，但大多数评估仍以英语为中心。本文介绍了Alvorada-Bench，这是一个由五个巴西大学入学考试中的4,515个文本问题组成的基准测试。在零样本、角色扮演和思维链提示下评估了二十个模型，生成了270,900个响应，并提供了结构化的自信度、感知难度和布卢姆水平的自我报告。总体而言，顶级模型的准确率超过94%，但在数学以及面向工程的IME和ITA考试中准确率下降，表明多步推理方面存在持续性弱点。自信度得到了良好校准并与感知难度呈正相关，揭示了模型能够准确评估自身的确定性能力。成本准确性分析显示，每千个标记成本低于2美元即可实现高准确率。在2024年高考中，顶级模型（O3）在语言类问题上获得了满分，而最弱系统（GPT-4.1 Nano）在数学上仅表现低于人类。通过提取几十年来巴西教育优先事项的精华并对每年数以百万计的学生进行评估，Alvorada-Bench 确立了语言模型是否能够应对定义巴西学术准备的语文、文化与推理的交叉领域。 

---
# A Functionality-Grounded Benchmark for Evaluating Web Agents in E-commerce Domains 

**Title (ZH)**: 基于功能性的基准测试：评估电子商务领域中的Web代理 

**Authors**: Xianren Zhang, Shreyas Prasad, Di Wang, Qiuhai Zeng, Suhang Wang, Wenbo Yan, Mat Hans  

**Link**: [PDF](https://arxiv.org/pdf/2508.15832)  

**Abstract**: Web agents have shown great promise in performing many tasks on ecommerce website. To assess their capabilities, several benchmarks have been introduced. However, current benchmarks in the e-commerce domain face two major problems. First, they primarily focus on product search tasks (e.g., Find an Apple Watch), failing to capture the broader range of functionalities offered by real-world e-commerce platforms such as Amazon, including account management and gift card operations. Second, existing benchmarks typically evaluate whether the agent completes the user query, but ignore the potential risks involved. In practice, web agents can make unintended changes that negatively impact the user account or status. For instance, an agent might purchase the wrong item, delete a saved address, or incorrectly configure an auto-reload setting. To address these gaps, we propose a new benchmark called Amazon-Bench. To generate user queries that cover a broad range of tasks, we propose a data generation pipeline that leverages webpage content and interactive elements (e.g., buttons, check boxes) to create diverse, functionality-grounded user queries covering tasks such as address management, wish list management, and brand store following. To improve the agent evaluation, we propose an automated evaluation framework that assesses both the performance and the safety of web agents. We systematically evaluate different agents, finding that current agents struggle with complex queries and pose safety risks. These results highlight the need for developing more robust and reliable web agents. 

**Abstract (ZH)**: Web代理在电子商务网站上执行多种任务展现了巨大的潜力。为了评估其能力，已经引入了若干基准。然而，当前电子商务领域的基准面临着两个主要问题。首先，它们主要集中在产品搜索任务（例如，查找Apple Watch）上，未能捕捉到诸如亚马逊等的真实世界电子商务平台提供的更广泛功能，包括账户管理与礼品卡操作。其次，现有的基准通常评估代理是否完成了用户查询，但忽视了潜在的风险。实际上，Web代理可能会做出意外更改，导致用户账户或状态受损。例如，代理可能会购买错误的商品、删除已保存的地址，或错误地配置自动重新装载设置。为了弥补这些空白，我们提出了一个新的基准，称为Amazon-Bench。我们提出了一种数据生成管道，利用网页内容和交互元素（例如，按钮、复选框）来生成涵盖广泛任务的多样、功能导向的用户查询，包括地址管理、愿望列表管理以及品牌商店关注。为了改进代理评估，我们提出了一种自动化评估框架，该框架评估Web代理的性能和安全性。我们系统地评估了不同代理，发现当前代理难以处理复杂查询，并存在安全风险。这些结果突显了开发更加健壮和可靠的Web代理的必要性。 

---
# Straggler-Resilient Federated Learning over A Hybrid Conventional and Pinching Antenna Network 

**Title (ZH)**: 基于混合传统与挤压天线网络的抗拖后腿联邦学习 

**Authors**: Bibo Wu, Fang Fang, Ming Zeng, Xianbin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15821)  

**Abstract**: Leveraging pinching antennas in wireless network enabled federated learning (FL) can effectively mitigate the common "straggler" issue in FL by dynamically establishing strong line-of-sight (LoS) links on demand. This letter proposes a hybrid conventional and pinching antenna network (HCPAN) to significantly improve communication efficiency in the non-orthogonal multiple access (NOMA)-enabled FL system. Within this framework, a fuzzy logic-based client classification scheme is first proposed to effectively balance clients' data contributions and communication conditions. Given this classification, we formulate a total time minimization problem to jointly optimize pinching antenna placement and resource allocation. Due to the complexity of variable coupling and non-convexity, a deep reinforcement learning (DRL)-based algorithm is developed to effectively address this problem. Simulation results validate the superiority of the proposed scheme in enhancing FL performance via the optimized deployment of pinching antenna. 

**Abstract (ZH)**: 利用pinching天线在无线网络辅助的联邦学习中的应用可以有效缓解联邦学习中的常见“ straggler ”问题，通过按需动态建立强视线（LoS）链路。本文提出了一种混合传统和pinching天线网络（HCPAN）以显著提高非正交多访问（NOMA）辅助联邦学习系统中的通信效率。在这一框架下，首先提出了基于模糊逻辑的客户端分类方案，以有效平衡客户端的数据贡献和通信条件。基于此分类，我们构建了一个总时间最小化问题，以联合优化pinching天线布局和资源分配。由于变量耦合的复杂性和非凸性，开发了一种基于深度强化学习（DRL）的算法来有效解决这一问题。仿真结果验证了所提出方案在通过优化pinching天线部署来提升联邦学习性能方面的优越性。 

---
# From Clicks to Preference: A Multi-stage Alignment Framework for Generative Query Suggestion in Conversational System 

**Title (ZH)**: 从点击到偏好：生成查询建议的多阶段对齐框架在对话系统中的应用 

**Authors**: Junhao Yin, Haolin Wang, Peng Bao, Ju Xu, Yongliang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15811)  

**Abstract**: Generative query suggestion using large language models offers a powerful way to enhance conversational systems, but aligning outputs with nuanced user preferences remains a critical challenge. To address this, we introduce a multi-stage framework designed for progressive alignment between the generation policy and user intent. Our pipeline begins with prompt engineering as a cold-start strategy, followed by the Supervised Fine-Tuning stage, in which we introduce a distillation method on click logs to create a robust foundational model. To better model user preferences while capturing their inherent uncertainty, we develop a Gaussian Reward Model (GaRM) that represents user preferences as probability distributions rather than point estimates. Finally, we employ reinforcement learning to align the generation policy with these preferences, guided by a composite reward function that integrates GaRM with auxiliary heuristics to mitigate reward hacking. To maintain training stability, this process is enhanced by a novel out-of-distribution regularization method and a two-stage reward fusion technique. Extensive experiments demonstrate that our framework significantly outperforms baselines on both automatic and human evaluations and yields a 34\% relative increase in user engagement as measured by click-through rate in live A/B tests. 

**Abstract (ZH)**: 使用大规模语言模型生成查询建议为增强对话系统提供了强有力的方法，但将输出与用户的细腻偏好对齐仍然是一个关键挑战。为此，我们介绍了一个多阶段框架，旨在逐步对生成策略与用户意图之间的对齐进行优化。我们的管道以提示工程作为冷启动策略开始，随后是监督微调阶段，在该阶段中，我们通过点击日志引入蒸馏方法来构建一个稳健的基础模型。为了更好地建模用户偏好并捕捉其固有的不确定性，我们开发了一个高斯奖励模型（GaRM），将用户偏好表示为概率分布而非点估计。最后，我们采用强化学习来根据结合GaRM和辅助启发式的复合奖励函数对齐生成策略，以解决奖励劫持问题。为了保持训练稳定性，此过程通过一个新型的离分布正则化方法和两阶段奖励融合技术进行增强。大量实验表明，我们的框架在自动和人工评估中显著优于基线模型，并在实时A/B测试中将点击率用户参与度提高了34%。 

---
# Uplifted Attackers, Human Defenders: The Cyber Offense-Defense Balance for Trailing-Edge Organizations 

**Title (ZH)**: 提升攻击者，人力防守者： trailing-edge 组织的网络攻防平衡 

**Authors**: Benjamin Murphy, Twm Stone  

**Link**: [PDF](https://arxiv.org/pdf/2508.15808)  

**Abstract**: Advances in AI are widely understood to have implications for cybersecurity. Articles have emphasized the effect of AI on the cyber offense-defense balance, and commentators can be found arguing either that cyber will privilege attackers or defenders. For defenders, arguments are often made that AI will enable solutions like formal verification of all software--and for some well-equipped companies, this may be true. This conversation, however, does not match the reality for most companies. "Trailing-edge organizations," as we term them, rely heavily on legacy software, poorly staff security roles, and struggle to implement best practices like rapid deployment of security patches. These decisions may be the result of corporate inertia, but may also be the result of a seemingly-rational calculation that attackers may not bother targeting a firm due to lack of economic incentives, and as a result, underinvestment in defense will not be punished.
This approach to security may have been sufficient prior to the development of AI systems, but it is unlikely to remain viable in the near future. We argue that continuing improvements in AI's capabilities poses additional risks on two fronts: First, increased usage of AI will alter the economics of the marginal cyberattack and expose these trailing-edge organizations to more attackers, more frequently. Second, AI's advances will enable attackers to develop exploits and launch attacks earlier than they can today--meaning that it is insufficient for these companies to attain parity with today's leading defenders, but must instead aim for faster remediation timelines and more resilient software. The situation today portends a dramatically increased number of attacks in the near future. Moving forward, we offer a range of solutions for both organizations and governments to improve the defensive posture of firms which lag behind their peers today. 

**Abstract (ZH)**: Advances in AI对网络安全的影响：落后组织面临的额外风险及应对策略 

---
# InteChar: A Unified Oracle Bone Character List for Ancient Chinese Language Modeling 

**Title (ZH)**: InteChar: 古代汉语建模统一甲骨文字符列表 

**Authors**: Xiaolei Diao, Zhihan Zhou, Lida Shi, Ting Wang, Ruihua Qi, Hao Xu, Daqian Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.15791)  

**Abstract**: Constructing historical language models (LMs) plays a crucial role in aiding archaeological provenance studies and understanding ancient cultures. However, existing resources present major challenges for training effective LMs on historical texts. First, the scarcity of historical language samples renders unsupervised learning approaches based on large text corpora highly inefficient, hindering effective pre-training. Moreover, due to the considerable temporal gap and complex evolution of ancient scripts, the absence of comprehensive character encoding schemes limits the digitization and computational processing of ancient texts, particularly in early Chinese writing. To address these challenges, we introduce InteChar, a unified and extensible character list that integrates unencoded oracle bone characters with traditional and modern Chinese. InteChar enables consistent digitization and representation of historical texts, providing a foundation for robust modeling of ancient scripts. To evaluate the effectiveness of InteChar, we construct the Oracle Corpus Set (OracleCS), an ancient Chinese corpus that combines expert-annotated samples with LLM-assisted data augmentation, centered on Chinese oracle bone inscriptions. Extensive experiments show that models trained with InteChar on OracleCS achieve substantial improvements across various historical language understanding tasks, confirming the effectiveness of our approach and establishing a solid foundation for future research in ancient Chinese NLP. 

**Abstract (ZH)**: 构建历史语言模型在辅助考古来源研究和理解古代文化中起着关键作用。然而，现有的资源对训练有效的历史语言模型构成了重大挑战。首先，历史语言样本的稀缺性使得基于大规模文本语料库的无监督学习方法效率低下，阻碍了有效的预训练。此外，由于古代文字的显著时间间隔和复杂演变，缺乏全面的字符编码方案限制了古代文本的数字化和计算处理，特别是早期中文书写。为了解决这些挑战，我们引入了InteChar，这是一个统一且可扩展的字符列表，将未编码的甲骨文字符与传统和现代中文相结合。InteChar使历史文本的一致数字化和表示成为可能，为古代文字的稳健建模提供了基础。为了评估InteChar的有效性，我们构建了甲骨文语料集(OracleCS)，这是一个结合专家标注样本和LLM辅助数据增强的古代中国语料库，中心围绕着中文甲骨文铭文。大量实验表明，使用InteChar在OracleCS上训练的模型在各种历史语言理解任务中取得了显著改善，证实了我们方法的有效性，并为未来古代中文NLP研究奠定了坚实的基础。 

---
