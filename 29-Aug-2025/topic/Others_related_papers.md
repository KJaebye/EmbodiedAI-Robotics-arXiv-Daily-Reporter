# PLUME: Procedural Layer Underground Modeling Engine 

**Title (ZH)**: PLUME: 基于过程的地下分层建模引擎 

**Authors**: Gabriel Manuel Garcia, Antoine Richard, Miguel Olivares-Mendez  

**Link**: [PDF](https://arxiv.org/pdf/2508.20926)  

**Abstract**: As space exploration advances, underground environments are becoming increasingly attractive due to their potential to provide shelter, easier access to resources, and enhanced scientific opportunities. Although such environments exist on Earth, they are often not easily accessible and do not accurately represent the diversity of underground environments found throughout the solar system. This paper presents PLUME, a procedural generation framework aimed at easily creating 3D underground environments. Its flexible structure allows for the continuous enhancement of various underground features, aligning with our expanding understanding of the solar system. The environments generated using PLUME can be used for AI training, evaluating robotics algorithms, 3D rendering, and facilitating rapid iteration on developed exploration algorithms. In this paper, it is demonstrated that PLUME has been used along with a robotic simulator. PLUME is open source and has been released on Github. this https URL 

**Abstract (ZH)**: 随着太空探索的进展，地下环境因可能提供的庇护所、更便捷的资源获取途径以及增强的科研机会而变得越来越有吸引力。尽管在地球上存在这样的环境，但它们往往难以访问且不能准确代表太阳系中发现的地下环境的多样性。本文介绍了一种名为PLUME的程序生成框架，旨在轻松创建3D地下环境。其灵活结构允许不断增强各种地下特征，以适应我们对太阳系理解的扩展。使用PLUME生成的环境可用于AI训练、机器人算法评估、3D渲染以及促进开发中的探索算法的快速迭代。本文展示了PLUME已与机器人模拟器结合使用。PLUME是开源的，并已在Github上发布。更多详情请参见：这个链接。 

---
# Train-Once Plan-Anywhere Kinodynamic Motion Planning via Diffusion Trees 

**Title (ZH)**: 一次训练，处处规划：基于扩散树的运动规划 

**Authors**: Yaniv Hassidof, Tom Jurgenson, Kiril Solovey  

**Link**: [PDF](https://arxiv.org/pdf/2508.21001)  

**Abstract**: Kinodynamic motion planning is concerned with computing collision-free trajectories while abiding by the robot's dynamic constraints. This critical problem is often tackled using sampling-based planners (SBPs) that explore the robot's high-dimensional state space by constructing a search tree via action propagations. Although SBPs can offer global guarantees on completeness and solution quality, their performance is often hindered by slow exploration due to uninformed action sampling. Learning-based approaches can yield significantly faster runtimes, yet they fail to generalize to out-of-distribution (OOD) scenarios and lack critical guarantees, e.g., safety, thus limiting their deployment on physical robots. We present Diffusion Tree (DiTree): a \emph{provably-generalizable} framework leveraging diffusion policies (DPs) as informed samplers to efficiently guide state-space search within SBPs. DiTree combines DP's ability to model complex distributions of expert trajectories, conditioned on local observations, with the completeness of SBPs to yield \emph{provably-safe} solutions within a few action propagation iterations for complex dynamical systems. We demonstrate DiTree's power with an implementation combining the popular RRT planner with a DP action sampler trained on a \emph{single environment}. In comprehensive evaluations on OOD scenarios, % DiTree has comparable runtimes to a standalone DP (3x faster than classical SBPs), while improving the average success rate over DP and SBPs. DiTree is on average 3x faster than classical SBPs, and outperforms all other approaches by achieving roughly 30\% higher success rate. Project webpage: this https URL. 

**Abstract (ZH)**: 基于扩散策略的可证明通用化动规树规划方法：DiTree 

---
# Encoding Tactile Stimuli for Organoid Intelligence in Braille Recognition 

**Title (ZH)**: 用于盲文识别的组织体触觉编码智能 

**Authors**: Tianyi Liu, Hemma Philamore, Benjamin Ward-Cherrier  

**Link**: [PDF](https://arxiv.org/pdf/2508.20850)  

**Abstract**: This study proposes a generalizable encoding strategy that maps tactile sensor data to electrical stimulation patterns, enabling neural organoids to perform an open-loop artificial tactile Braille classification task. Human forebrain organoids cultured on a low-density microelectrode array (MEA) are systematically stimulated to characterize the relationship between electrical stimulation parameters (number of pulse, phase amplitude, phase duration, and trigger delay) and organoid responses, measured as spike activity and spatial displacement of the center of activity. Implemented on event-based tactile inputs recorded from the Evetac sensor, our system achieved an average Braille letter classification accuracy of 61 percent with a single organoid, which increased significantly to 83 percent when responses from a three-organoid ensemble were combined. Additionally, the multi-organoid configuration demonstrated enhanced robustness against various types of artificially introduced noise. This research demonstrates the potential of organoids as low-power, adaptive bio-hybrid computational elements and provides a foundational encoding framework for future scalable bio-hybrid computing architectures. 

**Abstract (ZH)**: 本研究提出了一种可推广的编码策略，将触觉传感器数据映射到电刺激模式，使神经器官球能够执行开放环人工触觉盲文分类任务。在低密度微电极阵列(MEA)上培养的人类前脑器官球系统地受到刺激，以表征电刺激参数（脉冲数、相位幅度、相位持续时间和触发延迟）与器官球反应之间的关系，器官球反应通过尖峰活动和活动中心的空间位移进行测量。在事件驱动的触觉输入从Evetac传感器记录的数据上实现时，该系统在单个器官球上的平均盲文字母分类准确率为61%，当结合三个器官球的反应时，分类准确率显著提高至83%。此外，多器官球配置展示了对各种人工引入噪声的增强鲁棒性。该研究展示了器官球作为低能耗、自适应的生物杂合计算元件的潜力，并为未来的可扩展生物杂合计算架构提供了一种基础编码框架。 

---
# Efficient Neuro-Symbolic Learning of Constraints and Objective 

**Title (ZH)**: 高效神经符号学习约束与目标 

**Authors**: Marianne Defresne, Romain Gambardella, Sophie Barbe, Thomas Schiex  

**Link**: [PDF](https://arxiv.org/pdf/2508.20978)  

**Abstract**: In the ongoing quest for hybridizing discrete reasoning with neural nets, there is an increasing interest in neural architectures that can learn how to solve discrete reasoning or optimization problems from natural inputs, a task that Large Language Models seem to struggle with.
Objectives: We introduce a differentiable neuro-symbolic architecture and a loss function dedicated to learning how to solve NP-hard reasoning problems.
Methods: Our new probabilistic loss allows for learning both the constraints and the objective, thus delivering a complete model that can be scrutinized and completed with side constraints. By pushing the combinatorial solver out of the training loop, our architecture also offers scalable training while exact inference gives access to maximum accuracy.
Results: We empirically show that it can efficiently learn how to solve NP-hard reasoning problems from natural inputs. On three variants of the Sudoku benchmark -- symbolic, visual, and many-solution --, our approach requires a fraction of training time of other hybrid methods. On a visual Min-Cut/Max-cut task, it optimizes the regret better than a Decision-Focused-Learning regret-dedicated loss. Finally, it efficiently learns the energy optimization formulation of the large real-world problem of designing proteins. 

**Abstract (ZH)**: 在离散推理与神经网络结合的不断探索中，人们对能够从自然输入中学习解决离散推理或优化问题的神经架构越来越感兴趣，而大型语言模型似乎在这方面遇到困难。
目标：我们提出了一种可微神经符号架构和一个专门用于学习解决NP难推理问题的损失函数。
方法：我们新的概率损失函数允许同时学习约束条件和目标，从而提供一个可以审查和补充侧约束的完整模型。通过将组合式求解器从训练循环中移除，该架构还提供了可扩展的训练方法，而精确推断则提供了最大限度的准确性。
结果：我们实验证明，该方法能够高效地从自然输入中学习解决NP难推理问题。在三个版本的数独基准测试中——符号版、视觉版和多解版——我们的方法所需训练时间仅为其他混合方法的一小部分。在视觉最小割/最大割任务中，它比一种决策导向学习的后悔专用损失更好地优化了后悔值。最后，它有效地学习了设计蛋白质这一大规模实际问题的能量优化公式。 

---
# A Multi-Objective Genetic Algorithm for Healthcare Workforce Scheduling 

**Title (ZH)**: 多目标遗传算法在医疗卫生人员排班中的应用 

**Authors**: Vipul Patel, Anirudh Deodhar, Dagnachew Birru  

**Link**: [PDF](https://arxiv.org/pdf/2508.20953)  

**Abstract**: Workforce scheduling in the healthcare sector is a significant operational challenge, characterized by fluctuating patient loads, diverse clinical skills, and the critical need to control labor costs while upholding high standards of patient care. This problem is inherently multi-objective, demanding a delicate balance between competing goals: minimizing payroll, ensuring adequate staffing for patient needs, and accommodating staff preferences to mitigate burnout. We propose a Multi-objective Genetic Algorithm (MOO-GA) that models the hospital unit workforce scheduling problem as a multi-objective optimization task. Our model incorporates real-world complexities, including hourly appointment-driven demand and the use of modular shifts for a multi-skilled workforce. By defining objective functions for cost, patient care coverage, and staff satisfaction, the GA navigates the vast search space to identify a set of high-quality, non-dominated solutions. Demonstrated on datasets representing a typical hospital unit, the results show that our MOO-GA generates robust and balanced schedules. On average, the schedules produced by our algorithm showed a 66\% performance improvement over a baseline that simulates a conventional, manual scheduling process. This approach effectively manages trade-offs between critical operational and staff-centric objectives, providing a practical decision support tool for nurse managers and hospital administrators. 

**Abstract (ZH)**: 医疗保健领域的劳动力排班是一个重要的运营挑战，特征为波动的患者负载、多样化的临床技能以及控制劳动力成本与保持高标准患者护理之间的关键需求。这是一个本质上的多目标问题，需要在竞争的目标之间取得微妙的平衡：最小化工资支出、确保满足患者的护理需求以及兼顾工作人员的偏好以减少职业倦怠。我们提出了一种多目标遗传算法（MOO-GA），将医院单位劳动力排班问题建模为一个多目标优化任务。我们的模型包含了实际的复杂性，包括按小时预约驱动的需求和使用模块化班次来适应多技能劳动力。通过定义成本、患者护理覆盖和员工满意度的目标函数，遗传算法在广阔的搜索空间中导航，以识别一组高质量的非支配解。在典型医院单位的数据集上进行的演示结果显示，我们的MOO-GA生成了稳健且平衡的排班表。与模拟传统手工排班过程的基线相比，由我们算法生成的排班表平均性能提高了66%。该方法有效地管理了关键运营目标与以员工为中心的目标之间的权衡，为护士经理和医院管理人员提供了实用的决策支持工具。 

---
# Transparent Semantic Spaces: A Categorical Approach to Explainable Word Embeddings 

**Title (ZH)**: 透明语义空间：一种可解释词嵌入的范畴论方法 

**Authors**: Ares Fabregat-Hernández, Javier Palanca, Vicent Botti  

**Link**: [PDF](https://arxiv.org/pdf/2508.20701)  

**Abstract**: The paper introduces a novel framework based on category theory to enhance the explainability of artificial intelligence systems, particularly focusing on word embeddings. Key topics include the construction of categories $ Ł_{T} $ and $ ¶_{T} $, providing schematic representations of the semantics of a text $ T $, and reframing the selection of the element with maximum probability as a categorical notion. Additionally, the monoidal category $ ¶_{T} $ is constructed to visualize various methods of extracting semantic information from $ T $, offering a dimension-agnostic definition of semantic spaces reliant solely on information within the text.
Furthermore, the paper defines the categories of configurations $ \Conf $ and word embeddings $ \Emb $, accompanied by the concept of divergence as a decoration on $ \Emb $. It establishes a mathematically precise method for comparing word embeddings, demonstrating the equivalence between the GloVe and Word2Vec algorithms and the metric MDS algorithm, transitioning from neural network algorithms (black box) to a transparent framework. Finally, the paper presents a mathematical approach to computing biases before embedding and offers insights on mitigating biases at the semantic space level, advancing the field of explainable artificial intelligence. 

**Abstract (ZH)**: 基于范畴论的新型框架：提高人工智能系统的可解释性，特别是聚焦于词嵌入。主要内容包括构造范畴$Ł_{T}$和$¶_{T}$，提供文本$T$语义的示意图表示，并将概率最大元素的选择重新定义为范畴论的概念。此外，构造单调范畴$¶_{T}$以可视化从$T$中提取语义信息的各种方法，提供基于文本信息的语义空间的维数无关定义。进一步地，论文定义了配置范畴$\Conf$和词嵌入范畴$\Emb$，并引入偏差作为$\Emb$的装饰概念。建立了词嵌入的精确比较方法，证明GloVe和Word2Vec算法与度量MDS算法之间的等价性，从神经网络算法（黑盒）过渡到透明框架。最后，论文提供了在嵌入前计算偏差的数学方法，并在语义空间层面提供减轻偏差的见解，推动可解释的人工智能领域的发展。 

---
# Bridging Minds and Machines: Toward an Integration of AI and Cognitive Science 

**Title (ZH)**: 大脑与机器的连通：人工智能与认知科学的整合研究 

**Authors**: Rui Mao, Qian Liu, Xiao Li, Erik Cambria, Amir Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2508.20674)  

**Abstract**: Cognitive Science has profoundly shaped disciplines such as Artificial Intelligence (AI), Philosophy, Psychology, Neuroscience, Linguistics, and Culture. Many breakthroughs in AI trace their roots to cognitive theories, while AI itself has become an indispensable tool for advancing cognitive research. This reciprocal relationship motivates a comprehensive review of the intersections between AI and Cognitive Science. By synthesizing key contributions from both perspectives, we observe that AI progress has largely emphasized practical task performance, whereas its cognitive foundations remain conceptually fragmented. We argue that the future of AI within Cognitive Science lies not only in improving performance but also in constructing systems that deepen our understanding of the human mind. Promising directions include aligning AI behaviors with cognitive frameworks, situating AI in embodiment and culture, developing personalized cognitive models, and rethinking AI ethics through cognitive co-evaluation. 

**Abstract (ZH)**: 认知科学深刻地塑造了人工智能、哲学、心理学、神经科学、语言学和文化等学科。许多人工智能领域的突破性进展源于认知理论，而人工智能本身也成为推动认知研究进展不可或缺的工具。这种相互关系促使我们对人工智能与认知科学的交集进行全面回顾。通过综合来自两个领域的关键贡献，我们观察到，人工智能的进步主要集中在实际任务性能上，而其认知基础仍处于概念性的碎片化状态。我们认为，人工智能在认知科学中的未来不仅在于提高性能，还在于构建能够加深我们对人类心智理解的系统。有前景的方向包括使人工智能行为与认知框架相一致、将人工智能置于体认和文化之中、开发个性化的认知模型，并通过认知共评重新思考人工智能伦理。 

---
# Governable AI: Provable Safety Under Extreme Threat Models 

**Title (ZH)**: 可治理的人工智能：在极端威胁模型下的可证明安全性 

**Authors**: Donglin Wang, Weiyun Liang, Chunyuan Chen, Jing Xu, Yulong Fu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20411)  

**Abstract**: As AI rapidly advances, the security risks posed by AI are becoming increasingly severe, especially in critical scenarios, including those posing existential risks. If AI becomes uncontrollable, manipulated, or actively evades safety mechanisms, it could trigger systemic disasters. Existing AI safety approaches-such as model enhancement, value alignment, and human intervention-suffer from fundamental, in-principle limitations when facing AI with extreme motivations and unlimited intelligence, and cannot guarantee security. To address this challenge, we propose a Governable AI (GAI) framework that shifts from traditional internal constraints to externally enforced structural compliance based on cryptographic mechanisms that are computationally infeasible to break, even for future AI, under the defined threat model and well-established cryptographic this http URL GAI framework is composed of a simple yet reliable, fully deterministic, powerful, flexible, and general-purpose rule enforcement module (REM); governance rules; and a governable secure super-platform (GSSP) that offers end-to-end protection against compromise or subversion by AI. The decoupling of the governance rules and the technical platform further enables a feasible and generalizable technical pathway for the safety governance of AI. REM enforces the bottom line defined by governance rules, while GSSP ensures non-bypassability, tamper-resistance, and unforgeability to eliminate all identified attack vectors. This paper also presents a rigorous formal proof of the security properties of this mechanism and demonstrates its effectiveness through a prototype implementation evaluated in representative high-stakes scenarios. 

**Abstract (ZH)**: 随着人工智能迅速发展，AI带来的安全风险日益严重，特别是在存在根本性风险的关键场景中。如果AI变得无法控制、被操控或主动规避安全机制，可能会引发系统性灾难。现有AI安全方法，如模型增强、价值对齐和人工干预，在面对具有极端动机和无限智能的AI时，存在根本性的内在限制，无法确保安全性。为应对这一挑战，我们提出了一种可治理人工智能（GAI）框架，该框架从传统的内部约束转向基于计算上不可破解的加密机制的外部强制结构合规，以应对预定义威胁模型下的未来AI。GAI框架由一个简单可靠、完全确定性强、功能强大、通用性强的规则执行模块（REM）、治理规则以及一个可治理的安全超平台（GSSP）组成，该平台提供端到端的保护，防止AI的中断或篡改。治理规则与技术平台的分离进一步为AI的安全治理提供了一种可行且通用的技术途径。REM执行由治理规则定义的底线，而GSSP确保不可绕过、抗篡改和不可伪造，以消除所有已识别的攻击向量。本文还提出了该机制的安全属性的严格形式证明，并通过在代表性高风险场景中进行原型实现评估，展示了其有效性。 

---
# P2C: Path to Counterfactuals 

**Title (ZH)**: P2C: 背投方案路径 

**Authors**: Sopam Dasgupta, Sadaf MD Halim, Joaquín Arias, Elmer Salazar, Gopal Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2508.20371)  

**Abstract**: Machine-learning models are increasingly driving decisions in high-stakes settings, such as finance, law, and hiring, thus, highlighting the need for transparency. However, the key challenge is to balance transparency -- clarifying `why' a decision was made -- with recourse: providing actionable steps on `how' to achieve a favourable outcome from an unfavourable outcome. Counterfactual explanations reveal `why' an undesired outcome occurred and `how' to reverse it through targeted feature changes (interventions).
Current counterfactual approaches have limitations: 1) they often ignore causal dependencies between features, and 2) they typically assume all interventions can happen simultaneously, an unrealistic assumption in practical scenarios where actions are typically taken in a sequence. As a result, these counterfactuals are often not achievable in the real world.
We present P2C (Path-to-Counterfactuals), a model-agnostic framework that produces a plan (ordered sequence of actions) converting an unfavourable outcome to a causally consistent favourable outcome. P2C addresses both limitations by 1) Explicitly modelling causal relationships between features and 2) Ensuring that each intermediate state in the plan is feasible and causally valid. P2C uses the goal-directed Answer Set Programming system s(CASP) to generate the plan accounting for feature changes that happen automatically due to causal dependencies. Furthermore, P2C refines cost (effort) computation by only counting changes actively made by the user, resulting in realistic cost estimates. Finally, P2C highlights how its causal planner outperforms standard planners, which lack causal knowledge and thus can generate illegal actions. 

**Abstract (ZH)**: 基于路径的反事实解释（Path-to-Counterfactuals）：一种模型无关的框架 

---
# AI reasoning effort mirrors human decision time on content moderation tasks 

**Title (ZH)**: AI推理努力与人类在内容审核任务中的决策时间相 mirror 

**Authors**: Thomas Davidson  

**Link**: [PDF](https://arxiv.org/pdf/2508.20262)  

**Abstract**: Large language models can now generate intermediate reasoning steps before producing answers, improving performance on difficult problems. This study uses a paired conjoint experiment on a content moderation task to examine parallels between human decision times and model reasoning effort. Across three frontier models, reasoning effort consistently predicts human decision time. Both humans and models expended greater effort when important variables were held constant, suggesting similar sensitivity to task difficulty and patterns consistent with dual-process theories of cognition. These findings show that AI reasoning effort mirrors human processing time in subjective judgments and underscores the potential of reasoning traces for interpretability and decision-making. 

**Abstract (ZH)**: 大型语言模型现在可以在生成答案之前生成中间推理步骤，从而在解决困难问题上表现出更高的性能。本研究通过一项关于内容 Moderation 任务的配对联合实验，考察了人类决策时间和模型推理努力之间的相似性。在三种前沿模型中，推理努力始终预测人类决策时间。当重要变量保持不变时，人类和模型都投入了更多的努力，这表明对任务难度的相似敏感性，并与认知的双重过程理论一致。这些发现表明，AI 推理努力在主观判断中与人类处理时间相呼应，并强调了推理痕迹在可解释性和决策中的潜在价值。 

---
# Do Students Rely on AI? Analysis of Student-ChatGPT Conversations from a Field Study 

**Title (ZH)**: 学生依赖AI吗？来自实地研究的大学生与ChatGPT对话分析 

**Authors**: Jiayu Zheng, Lingxin Hao, Kelun Lu, Ashi Garg, Mike Reese, Melo-Jean Yap, I-Jeng Wang, Xingyun Wu, Wenrui Huang, Jenna Hoffman, Ariane Kelly, My Le, Ryan Zhang, Yanyu Lin, Muhammad Faayez, Anqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20244)  

**Abstract**: This study explores how college students interact with generative AI (ChatGPT-4) during educational quizzes, focusing on reliance and predictors of AI adoption. Conducted at the early stages of ChatGPT implementation, when students had limited familiarity with the tool, this field study analyzed 315 student-AI conversations during a brief, quiz-based scenario across various STEM courses. A novel four-stage reliance taxonomy was introduced to capture students' reliance patterns, distinguishing AI competence, relevance, adoption, and students' final answer correctness. Three findings emerged. First, students exhibited overall low reliance on AI and many of them could not effectively use AI for learning. Second, negative reliance patterns often persisted across interactions, highlighting students' difficulty in effectively shifting strategies after unsuccessful initial experiences. Third, certain behavioral metrics strongly predicted AI reliance, highlighting potential behavioral mechanisms to explain AI adoption. The study's findings underline critical implications for ethical AI integration in education and the broader field. It emphasizes the need for enhanced onboarding processes to improve student's familiarity and effective use of AI tools. Furthermore, AI interfaces should be designed with reliance-calibration mechanisms to enhance appropriate reliance. Ultimately, this research advances understanding of AI reliance dynamics, providing foundational insights for ethically sound and cognitively enriching AI practices. 

**Abstract (ZH)**: 本研究探究了大学生在教育测验中使用生成式AI（ChatGPT-4）的交互情况，重点关注对学生依赖程度及其影响因素的分析。在ChatGPT工具实施的初期阶段，学生对该工具的熟悉度有限，本研究通过一项简短的基于测验的场景分析了跨各类STEM课程的315名学生与AI的对话。引入了一种新颖的四阶段依赖分类法来捕捉学生们的依赖模式，区分了AI的熟练度、相关性、采用程度以及学生的最终答案准确性。研究结果包括三个方面。首先，学生们整体上对AI的依赖较低，许多学生无法有效利用AI进行学习。其次，消极的依赖模式往往会在多次交互中持续存在，凸显了学生们在经历不成功的初始体验后难以有效调整策略的困境。第三，某些行为指标强烈预测了AI的依赖程度，揭示了可能的行为机制以解释AI的采用情况。研究结果强调了在教育和更广泛领域中伦理地整合AI的关键意义，强调了需要改进的注册流程以提高学生对AI工具的熟悉度和有效使用，并且AI界面应设计有依赖校准机制以促进合适的依赖。最终，这项研究推进了对AI依赖动态的理解，提供了伦理合理且认知上充实的AI实践的基本见解。 

---
# Array-Based Monte Carlo Tree Search 

**Title (ZH)**: 阵列基于蒙特卡洛树搜索 

**Authors**: James Ragan, Fred Y. Hadaegh, Soon-Jo Chung  

**Link**: [PDF](https://arxiv.org/pdf/2508.20140)  

**Abstract**: Monte Carlo Tree Search is a popular method for solving decision making problems. Faster implementations allow for more simulations within the same wall clock time, directly improving search performance. To this end, we present an alternative array-based implementation of the classic Upper Confidence bounds applied to Trees algorithm. Our method preserves the logic of the original algorithm, but eliminates the need for branch prediction, enabling faster performance on pipelined processors, and up to a factor of 2.8 times better scaling with search depth in our numerical simulations. 

**Abstract (ZH)**: 基于数组的 Upper Confidence bounds applied to Trees 算法的 Monte Carlo Tree Search 并行实现 

---
# FakeParts: a New Family of AI-Generated DeepFakes 

**Title (ZH)**: 假部件：一种新的AI生成的DeepFakes家族 

**Authors**: Gaetan Brison, Soobash Daiboo, Samy Aimeur, Awais Hussain Sani, Xi Wang, Gianni Franchi, Vicky Kalogeiton  

**Link**: [PDF](https://arxiv.org/pdf/2508.21052)  

**Abstract**: We introduce FakeParts, a new class of deepfakes characterized by subtle, localized manipulations to specific spatial regions or temporal segments of otherwise authentic videos. Unlike fully synthetic content, these partial manipulations, ranging from altered facial expressions to object substitutions and background modifications, blend seamlessly with real elements, making them particularly deceptive and difficult to detect. To address the critical gap in detection capabilities, we present FakePartsBench, the first large-scale benchmark dataset specifically designed to capture the full spectrum of partial deepfakes. Comprising over 25K videos with pixel-level and frame-level manipulation annotations, our dataset enables comprehensive evaluation of detection methods. Our user studies demonstrate that FakeParts reduces human detection accuracy by over 30% compared to traditional deepfakes, with similar performance degradation observed in state-of-the-art detection models. This work identifies an urgent vulnerability in current deepfake detection approaches and provides the necessary resources to develop more robust methods for partial video manipulations. 

**Abstract (ZH)**: FakeParts: 新颖的局部操纵深伪类及其检测基准 FakePartsBench 

---
# Understanding, Protecting, and Augmenting Human Cognition with Generative AI: A Synthesis of the CHI 2025 Tools for Thought Workshop 

**Title (ZH)**: 利用生成式AI理解、保护和增强人类认知：CHI 2025 工具思维研讨会综述 

**Authors**: Lev Tankelevitch, Elena L. Glassman, Jessica He, Aniket Kittur, Mina Lee, Srishti Palani, Advait Sarkar, Gonzalo Ramos, Yvonne Rogers, Hari Subramonyam  

**Link**: [PDF](https://arxiv.org/pdf/2508.21036)  

**Abstract**: Generative AI (GenAI) radically expands the scope and capability of automation for work, education, and everyday tasks, a transformation posing both risks and opportunities for human cognition. How will human cognition change, and what opportunities are there for GenAI to augment it? Which theories, metrics, and other tools are needed to address these questions? The CHI 2025 workshop on Tools for Thought aimed to bridge an emerging science of how the use of GenAI affects human thought, from metacognition to critical thinking, memory, and creativity, with an emerging design practice for building GenAI tools that both protect and augment human thought. Fifty-six researchers, designers, and thinkers from across disciplines as well as industry and academia, along with 34 papers and portfolios, seeded a day of discussion, ideation, and community-building. We synthesize this material here to begin mapping the space of research and design opportunities and to catalyze a multidisciplinary community around this pressing area of research. 

**Abstract (ZH)**: Generative AI (GenAI) 对工作、教育和日常任务的自动化范围和能力产生了根本性扩展，这一变革为人类认知带来了风险与机遇。人类认知将如何变化，GenAI又有何增强潜力？需要哪些理论、指标和其他工具来应对这些问题？CHI 2025 工作坊“思维工具”旨在连接关于GenAI使用如何影响人类思维（包括元认知、批判性思维、记忆和创造力）的研究科学与新兴的GenAI工具设计实践，该实践既保护也增强人类思维。来自多个学科及产业和学术界的56名研究人员、设计师和思想家，以及34篇论文和作品集，激发了一天的讨论、创意思考和社区构建。我们在此整理这些材料，以开始绘制研究和设计机会的空间图，并催化围绕这一紧迫研究领域的跨学科社区。 

---
# ChainReaction! Structured Approach with Causal Chains as Intermediate Representations for Improved and Explainable Causal Video Question Answering 

**Title (ZH)**: Chain Reaction！基于因果链作为中间表示的结构化方法以提高可解释的因果视频问答 

**Authors**: Paritosh Parmar, Eric Peh, Basura Fernando  

**Link**: [PDF](https://arxiv.org/pdf/2508.21010)  

**Abstract**: Existing Causal-Why Video Question Answering (VideoQA) models often struggle with higher-order reasoning, relying on opaque, monolithic pipelines that entangle video understanding, causal inference, and answer generation. These black-box approaches offer limited interpretability and tend to depend on shallow heuristics. We propose a novel, modular framework that explicitly decouples causal reasoning from answer generation, introducing natural language causal chains as interpretable intermediate representations. Inspired by human cognitive models, these structured cause-effect sequences bridge low-level video content with high-level causal reasoning, enabling transparent and logically coherent inference. Our two-stage architecture comprises a Causal Chain Extractor (CCE) that generates causal chains from video-question pairs, and a Causal Chain-Driven Answerer (CCDA) that produces answers grounded in these chains. To address the lack of annotated reasoning traces, we introduce a scalable method for generating high-quality causal chains from existing datasets using large language models. We also propose CauCo, a new evaluation metric for causality-oriented captioning. Experiments on three large-scale benchmarks demonstrate that our approach not only outperforms state-of-the-art models, but also yields substantial gains in explainability, user trust, and generalization -- positioning the CCE as a reusable causal reasoning engine across diverse domains. Project page: this https URL 

**Abstract (ZH)**: 现有的因果解释视频问答（VideoQA）模型常常难以处理高阶推理，依赖于将视频理解、因果推理和答案生成纠缠在一起的不透明单一管道。这些黑盒方法提供有限的解释性，并倾向于依赖浅层启发式方法。我们提出了一种新的模块化框架，明确地将因果推理与答案生成分离，并引入可解释的因果链作为中间表示。受人类认知模型的启发，这些结构化的因果序列将低层级视频内容与高层级因果推理相连，使得推理具有透明性和逻辑一致性。我们的两阶段架构包括一个因果链提取器（CCE），它从视频-问题对中生成因果链，以及一个因果链驱动的答案生成器（CCDA），它基于这些链生成答案。为了解决缺少标注推理轨迹的问题，我们提出了使用大型语言模型从现有数据集中生成高质量因果链的可扩展方法。我们还提出了CauCo，一种新的因果导向的标题评估指标。在三个大规模基准上的实验表明，我们的方法不仅优于现有最佳模型，还在解释性、用户信任度和泛化能力方面取得了显著提升——这将CCE定位为跨多种领域的可重用因果推理引擎。 

---
# ExpertSim: Fast Particle Detector Simulation Using Mixture-of-Generative-Experts 

**Title (ZH)**: ExpertSim: 快速粒子探测器仿真用混合生成专家模型 

**Authors**: Patryk Będkowski, Jan Dubiński, Filip Szatkowski, Kamil Deja, Przemysław Rokita, Tomasz Trzciński  

**Link**: [PDF](https://arxiv.org/pdf/2508.20991)  

**Abstract**: Simulating detector responses is a crucial part of understanding the inner workings of particle collisions in the Large Hadron Collider at CERN. Such simulations are currently performed with statistical Monte Carlo methods, which are computationally expensive and put a significant strain on CERN's computational grid. Therefore, recent proposals advocate for generative machine learning methods to enable more efficient simulations. However, the distribution of the data varies significantly across the simulations, which is hard to capture with out-of-the-box methods. In this study, we present ExpertSim - a deep learning simulation approach tailored for the Zero Degree Calorimeter in the ALICE experiment. Our method utilizes a Mixture-of-Generative-Experts architecture, where each expert specializes in simulating a different subset of the data. This allows for a more precise and efficient generation process, as each expert focuses on a specific aspect of the calorimeter response. ExpertSim not only improves accuracy, but also provides a significant speedup compared to the traditional Monte-Carlo methods, offering a promising solution for high-efficiency detector simulations in particle physics experiments at CERN. We make the code available at this https URL. 

**Abstract (ZH)**: 在欧洲核子研究组织（CERN）的大型强子对撞机（LHC）的ALICE实验中，零度角 calorimeter 的深度学习仿真方法 

---
# AI Agentic Vulnerability Injection And Transformation with Optimized Reasoning 

**Title (ZH)**: AI自主漏洞注入与优化推理转化 

**Authors**: Amine Lbath, Massih-Reza Amini, Aurelien Delaitre, Vadim Okun  

**Link**: [PDF](https://arxiv.org/pdf/2508.20866)  

**Abstract**: The increasing complexity of software systems and the sophistication of cyber-attacks have underscored the critical need for effective automated vulnerability detection and repair systems. Traditional methods, such as static program analysis, face significant challenges related to scalability, adaptability, and high false-positive and false-negative rates. AI-driven approaches, particularly those using machine learning and deep learning models, show promise but are heavily reliant on the quality and quantity of training data. This paper introduces a novel framework designed to automatically introduce realistic, category-specific vulnerabilities into secure C/C++ codebases to generate datasets. The proposed approach coordinates multiple AI agents that simulate expert reasoning, along with function agents and traditional code analysis tools. It leverages Retrieval-Augmented Generation for contextual grounding and employs Low-Rank approximation of weights for efficient model fine-tuning. Our experimental study on 116 code samples from three different benchmarks suggests that our approach outperforms other techniques with regard to dataset accuracy, achieving between 89\% and 95\% success rates in injecting vulnerabilities at function level. 

**Abstract (ZH)**: 软件系统日益复杂的程度和网络攻击的 sophistication 加强了高效自动化漏洞检测与修复系统的需求。传统方法，如静态程序分析，面临可扩展性、适应性以及高误报和漏报率的显著挑战。基于 AI 的方法，特别是利用机器学习和深度学习模型的方法显示出潜力，但对训练数据的质量和数量高度依赖。本文提出了一种新型框架，旨在自动生成现实的、类别特定的漏洞，注入到安全的 C/C++ 代码库中以生成数据集。所提出的方法协调多个 AI 代理以模拟专家推理，结合函数代理和传统代码分析工具。该方法利用检索增强生成进行上下文定位，并采用低秩权重逼近进行高效模型微调。在对来自三个不同基准的 116 个代码样本进行的实验研究中，我们的方法在数据集准确性方面优于其他技术，在函数级别注入漏洞的成功率在 89% 至 95% 之间。 

---
# JADES: A Universal Framework for Jailbreak Assessment via Decompositional Scoring 

**Title (ZH)**: JADES: 一种基于分解评分的通用 Jailbreak 评估框架 

**Authors**: Junjie Chu, Mingjie Li, Ziqing Yang, Ye Leng, Chenhao Lin, Chao Shen, Michael Backes, Yun Shen, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20848)  

**Abstract**: Accurately determining whether a jailbreak attempt has succeeded is a fundamental yet unresolved challenge. Existing evaluation methods rely on misaligned proxy indicators or naive holistic judgments. They frequently misinterpret model responses, leading to inconsistent and subjective assessments that misalign with human perception. To address this gap, we introduce JADES (Jailbreak Assessment via Decompositional Scoring), a universal jailbreak evaluation framework. Its key mechanism is to automatically decompose an input harmful question into a set of weighted sub-questions, score each sub-answer, and weight-aggregate the sub-scores into a final decision. JADES also incorporates an optional fact-checking module to strengthen the detection of hallucinations in jailbreak responses. We validate JADES on JailbreakQR, a newly introduced benchmark proposed in this work, consisting of 400 pairs of jailbreak prompts and responses, each meticulously annotated by humans. In a binary setting (success/failure), JADES achieves 98.5% agreement with human evaluators, outperforming strong baselines by over 9%. Re-evaluating five popular attacks on four LLMs reveals substantial overestimation (e.g., LAA's attack success rate on GPT-3.5-Turbo drops from 93% to 69%). Our results show that JADES could deliver accurate, consistent, and interpretable evaluations, providing a reliable basis for measuring future jailbreak attacks. 

**Abstract (ZH)**: 准确确定越狱尝试是否成功是一个基本但尚未解决的挑战。现有的评估方法依赖于不对齐的代理指标或直觉的整体判断。它们经常误解释模型的响应，导致不一致且主观的评估，这与人类感知不符。为解决这一差距，我们引入了JADES（越狱评估通过分解评分），这是一种通用的越狱评估框架。其关键机制是自动将输入的有害问题分解为一组加权子问题，对每个子回答进行评分，并将子评分加权聚合为最终决策。JADES还包含一个可选的事实核查模块，以增强对越狱响应中幻觉的检测。我们在本文提出的新基准JailbreakQR上验证了JADES，该基准包括400对越狱提示和响应，每一对都由人类细致标注。在二元设置（成功/失败）下，JADES在人评分者中的一致率为98.5%，优于强 baselines超过9%。重新评估五个流行的攻击在四个LLM上的效果显示了显著的高估（例如，LAA对GPT-3.5-Turbo的攻击成功率从93%下降到69%）。我们的结果表明，JADES能够提供准确、一致和可解释的评估，为衡量未来越狱攻击提供可靠的依据。 

---
# Multi-Agent Penetration Testing AI for the Web 

**Title (ZH)**: 网络空间多代理渗透测试人工智能 

**Authors**: Isaac David, Arthur Gervais  

**Link**: [PDF](https://arxiv.org/pdf/2508.20816)  

**Abstract**: AI-powered development platforms are making software creation accessible to a broader audience, but this democratization has triggered a scalability crisis in security auditing. With studies showing that up to 40% of AI-generated code contains vulnerabilities, the pace of development now vastly outstrips the capacity for thorough security assessment.
We present MAPTA, a multi-agent system for autonomous web application security assessment that combines large language model orchestration with tool-grounded execution and end-to-end exploit validation. On the 104-challenge XBOW benchmark, MAPTA achieves 76.9% overall success with perfect performance on SSRF and misconfiguration vulnerabilities, 83% success on broken authorization, and strong results on injection attacks including server-side template injection (85%) and SQL injection (83%). Cross-site scripting (57%) and blind SQL injection (0%) remain challenging. Our comprehensive cost analysis across all challenges totals $21.38 with a median cost of $0.073 for successful attempts versus $0.357 for failures. Success correlates strongly with resource efficiency, enabling practical early-stopping thresholds at approximately 40 tool calls or $0.30 per challenge.
MAPTA's real-world findings are impactful given both the popularity of the respective scanned GitHub repositories (8K-70K stars) and MAPTA's low average operating cost of $3.67 per open-source assessment: MAPTA discovered critical vulnerabilities including RCEs, command injections, secret exposure, and arbitrary file write vulnerabilities. Findings are responsibly disclosed, 10 findings are under CVE review. 

**Abstract (ZH)**: 基于多代理系统的自主网页应用安全评估平台MAPTA 

---
# Evaluating Compositional Generalisation in VLMs and Diffusion Models 

**Title (ZH)**: 评估VLMs和扩散模型的组合泛化能力 

**Authors**: Beth Pearson, Bilal Boulbarss, Michael Wray, Martha Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2508.20783)  

**Abstract**: A fundamental aspect of the semantics of natural language is that novel meanings can be formed from the composition of previously known parts. Vision-language models (VLMs) have made significant progress in recent years, however, there is evidence that they are unable to perform this kind of composition. For example, given an image of a red cube and a blue cylinder, a VLM such as CLIP is likely to incorrectly label the image as a red cylinder or a blue cube, indicating it represents the image as a `bag-of-words' and fails to capture compositional semantics. Diffusion models have recently gained significant attention for their impressive generative abilities, and zero-shot classifiers based on diffusion models have been shown to perform competitively with CLIP in certain compositional tasks. In this work we explore whether the generative Diffusion Classifier has improved compositional generalisation abilities compared to discriminative models. We assess three models -- Diffusion Classifier, CLIP, and ViLT -- on their ability to bind objects with attributes and relations in both zero-shot learning (ZSL) and generalised zero-shot learning (GZSL) settings. Our results show that the Diffusion Classifier and ViLT perform well at concept binding tasks, but that all models struggle significantly with the relational GZSL task, underscoring the broader challenges VLMs face with relational reasoning. Analysis of CLIP embeddings suggests that the difficulty may stem from overly similar representations of relational concepts such as left and right. Code and dataset are available at: this https URL 

**Abstract (ZH)**: 自然语言语义的一个基本方面是从先前已知部分的组合中形成新的含义。视觉-语言模型（VLMs）在近年来取得了显著进展，然而，证据表明它们无法执行这种组合。扩散模型近年来因其出色的生成能力而受到了广泛关注，基于扩散模型的零样本分类器已被证明在某些组合任务中能与CLIP竞争。在此项工作中，我们探讨了生成扩散分类器是否在组合泛化能力方面优于判别模型。我们评估了三种模型——扩散分类器、CLIP和ViLT——在零样本学习（ZSL）和广义零样本学习（GZSL）设置中将物体与其属性和关系相结合的能力。我们的结果表明，扩散分类器和ViLT在概念绑定任务中表现良好，但所有模型在关系GZSL任务中都面临重大挑战，突显了VLMs在关系推理方面面临的更广泛挑战。CLIP嵌入分析表明，困难可能源于关系概念如左和右的表示过于相似。代码和数据集可在以下链接获取：this https URL。 

---
# Safer Skin Lesion Classification with Global Class Activation Probability Map Evaluation and SafeML 

**Title (ZH)**: 全球类激活概率图评估与SafeML的皮肤病变分类安全性 

**Authors**: Kuniko Paxton, Koorosh Aslansefat, Amila Akagić, Dhavalkumar Thakker, Yiannis Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2508.20776)  

**Abstract**: Recent advancements in skin lesion classification models have significantly improved accuracy, with some models even surpassing dermatologists' diagnostic performance. However, in medical practice, distrust in AI models remains a challenge. Beyond high accuracy, trustworthy, explainable diagnoses are essential. Existing explainability methods have reliability issues, with LIME-based methods suffering from inconsistency, while CAM-based methods failing to consider all classes. To address these limitations, we propose Global Class Activation Probabilistic Map Evaluation, a method that analyses all classes' activation probability maps probabilistically and at a pixel level. By visualizing the diagnostic process in a unified manner, it helps reduce the risk of misdiagnosis. Furthermore, the application of SafeML enhances the detection of false diagnoses and issues warnings to doctors and patients as needed, improving diagnostic reliability and ultimately patient safety. We evaluated our method using the ISIC datasets with MobileNetV2 and Vision Transformers. 

**Abstract (ZH)**: 近期皮肤病变分类模型的进展显著提高了准确性，某些模型甚至超越了皮肤科医生的诊断性能。然而，在医疗实践中，对AI模型的信任依然是一个挑战。除了高准确性，可信赖且可解释的诊断结果至关重要。现有的可解释性方法存在可靠性问题，基于LIME的方法表现出不一致性，而基于CAM的方法未能考虑所有类别。为解决这些限制，我们提出全局类别激活概率图评估方法，该方法在像素级和概率层面分析所有类别的激活概率图。通过统一可视化诊断过程，有助于降低误诊风险。此外，SafeML的应用可以检测假性诊断并根据需要向医生和患者发出警告，从而提高诊断可靠性，最终保障患者安全。我们使用ISIC数据集和MobileNetV2及Vision Transformers对我们的方法进行了评估。 

---
# Unleashing Uncertainty: Efficient Machine Unlearning for Generative AI 

**Title (ZH)**: 解锁不确定性：生成式AI的高效机器遗忘技术 

**Authors**: Christoforos N. Spartalis, Theodoros Semertzidis, Petros Daras, Efstratios Gavves  

**Link**: [PDF](https://arxiv.org/pdf/2508.20773)  

**Abstract**: We introduce SAFEMax, a novel method for Machine Unlearning in diffusion models. Grounded in information-theoretic principles, SAFEMax maximizes the entropy in generated images, causing the model to generate Gaussian noise when conditioned on impermissible classes by ultimately halting its denoising process. Also, our method controls the balance between forgetting and retention by selectively focusing on the early diffusion steps, where class-specific information is prominent. Our results demonstrate the effectiveness of SAFEMax and highlight its substantial efficiency gains over state-of-the-art methods. 

**Abstract (ZH)**: SAFEMax：一种基于信息论原则的扩散模型机器遗忘新方法 

---
# Signs of Struggle: Spotting Cognitive Distortions across Language and Register 

**Title (ZH)**: 挣扎的迹象：跨语言和体裁识别认知扭曲 

**Authors**: Abhishek Kuber, Enrico Liscio, Ruixuan Zhang, Caroline Figueroa, Pradeep K. Murukannaiah  

**Link**: [PDF](https://arxiv.org/pdf/2508.20771)  

**Abstract**: Rising mental health issues among youth have increased interest in automated approaches for detecting early signs of psychological distress in digital text. One key focus is the identification of cognitive distortions, irrational thought patterns that have a role in aggravating mental distress. Early detection of these distortions may enable timely, low-cost interventions. While prior work has focused on English clinical data, we present the first in-depth study of cross-lingual and cross-register generalization of cognitive distortion detection, analyzing forum posts written by Dutch adolescents. Our findings show that while changes in language and writing style can significantly affect model performance, domain adaptation methods show the most promise. 

**Abstract (ZH)**: 青少年心理健康问题上升激发了对自动检测数字文本早期心理压力迹象方法的兴趣。关键焦点在于认知 distortions 的识别，这些不合理的思维模式在加剧心理压力中起作用。早期检测这些 distorting 可能能实现及时、低成本的干预。虽然之前的工作主要集中在英语临床数据上，我们首次进行了跨语言和跨文体一般性的认知 distortions 检测深入研究，分析了荷兰青少年在论坛上撰写的文章。研究发现，尽管语言和写作风格的变化会显著影响模型性能，但领域适应方法显示出最大的潜力。 

---
# EEGDM: Learning EEG Representation with Latent Diffusion Model 

**Title (ZH)**: EEGDM：基于潜在扩散模型的EEG表示学习 

**Authors**: Shaocong Wang, Tong Liu, Ming Li, Minjing Yu, Yong-Jin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20705)  

**Abstract**: While electroencephalography (EEG) signal analysis using deep learning has shown great promise, existing approaches still face significant challenges in learning generalizable representations that perform well across diverse tasks, particularly when training data is limited. Current EEG representation learning methods including EEGPT and LaBraM typically rely on simple masked reconstruction objective, which may not fully capture the rich semantic information and complex patterns inherent in EEG signals. In this paper, we propose EEGDM, a novel self-supervised EEG representation learning method based on the latent diffusion model, which leverages EEG signal generation as a self-supervised objective, turning the diffusion model into a strong representation learner capable of capturing EEG semantics. EEGDM incorporates an EEG encoder that distills EEG signals and their channel augmentations into a compact representation, acting as conditional information to guide the diffusion model for generating EEG signals. This design endows EEGDM with a compact latent space, which not only offers ample control over the generative process but also can be leveraged for downstream tasks. Experimental results show that EEGDM (1) can reconstruct high-quality EEG signals, (2) effectively learns robust representations, and (3) achieves competitive performance with modest pre-training data size across diverse downstream tasks, underscoring its generalizability and practical utility. 

**Abstract (ZH)**: 基于潜扩散模型的自监督EEG表示学习方法EEGDMSelf-supervised EEG Representation Learning Method Based on Latent Diffusion Model: EEGDM 

---
# Generative Annotation for ASR Named Entity Correction 

**Title (ZH)**: ASR命名实体识别的生成性标注方法 

**Authors**: Yuanchang Luo, Daimeng Wei, Shaojun Li, Hengchao Shang, Jiaxin Guo, Zongyao Li, Zhanglin Wu, Xiaoyu Chen, Zhiqiang Rao, Jinlong Yang, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20700)  

**Abstract**: End-to-end automatic speech recognition systems often fail to transcribe domain-specific named entities, causing catastrophic failures in downstream tasks. Numerous fast and lightweight named entity correction (NEC) models have been proposed in recent years. These models, mainly leveraging phonetic-level edit distance algorithms, have shown impressive performances. However, when the forms of the wrongly-transcribed words(s) and the ground-truth entity are significantly different, these methods often fail to locate the wrongly transcribed words in hypothesis, thus limiting their usage. We propose a novel NEC method that utilizes speech sound features to retrieve candidate entities. With speech sound features and candidate entities, we inovatively design a generative method to annotate entity errors in ASR transcripts and replace the text with correct entities. This method is effective in scenarios of word form difference. We test our method using open-source and self-constructed test sets. The results demonstrate that our NEC method can bring significant improvement to entity accuracy. We will open source our self-constructed test set and training data. 

**Abstract (ZH)**: 端到端自动语音识别系统常在转录领域特异性命名实体时失败，导致下游任务出现灾难性失败。近年来，提出了许多快速且轻量级的命名实体修正模型。这些模型主要利用音素级编辑距离算法，表现出色。然而，当错误转录词的形式与真实实体形式差异显著时，这些方法往往无法在假设中定位错误转录的词，从而限制了其应用。我们提出了一种新颖的命名实体修正方法，利用语音音素特征来检索候选实体。借助语音音素特征和候选实体，我们创新设计了一种生成方法，用于标注ASR转录中的实体错误，并用正确实体替换文本。该方法在词形差异场景中有效。我们使用开源和自行构建的测试集对方法进行了测试。结果表明，我们的命名实体修正方法能显著提高实体准确性。我们将开放我们自行构建的测试集和训练数据。 

---
# Amadeus: Autoregressive Model with Bidirectional Attribute Modelling for Symbolic Music 

**Title (ZH)**: Amadeus: 自回归模型结合双向属性建模的符号音乐生成 

**Authors**: Hongju Su, Ke Li, Lan Yang, Honggang Zhang, Yi-Zhe Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.20665)  

**Abstract**: Existing state-of-the-art symbolic music generation models predominantly adopt autoregressive or hierarchical autoregressive architectures, modelling symbolic music as a sequence of attribute tokens with unidirectional temporal dependencies, under the assumption of a fixed, strict dependency structure among these attributes. However, we observe that using different attributes as the initial token in these models leads to comparable performance. This suggests that the attributes of a musical note are, in essence, a concurrent and unordered set, rather than a temporally dependent sequence. Based on this insight, we introduce Amadeus, a novel symbolic music generation framework. Amadeus adopts a two-level architecture: an autoregressive model for note sequences and a bidirectional discrete diffusion model for attributes. To enhance performance, we propose Music Latent Space Discriminability Enhancement Strategy(MLSDES), incorporating contrastive learning constraints that amplify discriminability of intermediate music representations. The Conditional Information Enhancement Module (CIEM) simultaneously strengthens note latent vector representation via attention mechanisms, enabling more precise note decoding. We conduct extensive experiments on unconditional and text-conditioned generation tasks. Amadeus significantly outperforms SOTA models across multiple metrics while achieving at least 4$\times$ speed-up. Furthermore, we demonstrate training-free, fine-grained note attribute control feasibility using our model. To explore the upper performance bound of the Amadeus architecture, we compile the largest open-source symbolic music dataset to date, AMD (Amadeus MIDI Dataset), supporting both pre-training and fine-tuning. 

**Abstract (ZH)**: 现有的最先进符号音乐生成模型大多采用自回归或分层自回归架构，将符号音乐建模为具有单向时间依赖性的属性 token 序列，假设这些属性之间存在固定且严格的时间依赖关系。然而，我们观察到，在这些模型中使用不同的属性作为初始 token 并不会导致性能显著差异，这表明音乐符号的属性实际上是一个并发且无序的集合，而非具有时间依赖性的序列。基于这一洞察，我们引入了Amadeus，一种新颖的符号音乐生成框架。Amadeus采用两层架构：对接奏序列的自回归模型和对属性的双向离散扩散模型。为提升性能，我们提出了Music Latent Space Discriminability Enhancement Strategy(MLSDES)，将对比学习约束纳入模型以增强中间音乐表征的可区分性。条件信息增强模块(CIEM)通过注意机制同时增强了对奏符号 latent 向量的表示，从而能够更精确地解码对奏符号。我们在无条件和文本条件生成任务上进行了大量实验证明，Amadeus在多个指标上大幅优于当前最佳模型，且至少快 4 倍。此外，我们展示了使用该模型实现无训练的细粒度对奏属性控制的可能性。为了探索Amadeus架构的性能上限，我们构建了迄今为止最大的开源符号音乐数据集AMD（Amadeus MIDI Dataset），支持预训练和微调。 

---
# Task-Oriented Edge-Assisted Cross-System Design for Real-Time Human-Robot Interaction in Industrial Metaverse 

**Title (ZH)**: 面向任务的边缘辅助跨系统设计：工业元宇宙中的实时人机器人交互 

**Authors**: Kan Chen, Zhen Meng, Xiangmin Xu, Jiaming Yang, Emma Li, Philip G. Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.20664)  

**Abstract**: Real-time human-device interaction in industrial Metaverse faces challenges such as high computational load, limited bandwidth, and strict latency. This paper proposes a task-oriented edge-assisted cross-system framework using digital twins (DTs) to enable responsive interactions. By predicting operator motions, the system supports: 1) proactive Metaverse rendering for visual feedback, and 2) preemptive control of remote devices. The DTs are decoupled into two virtual functions-visual display and robotic control-optimizing both performance and adaptability. To enhance generalizability, we introduce the Human-In-The-Loop Model-Agnostic Meta-Learning (HITL-MAML) algorithm, which dynamically adjusts prediction horizons. Evaluation on two tasks demonstrates the framework's effectiveness: in a Trajectory-Based Drawing Control task, it reduces weighted RMSE from 0.0712 m to 0.0101 m; in a real-time 3D scene representation task for nuclear decommissioning, it achieves a PSNR of 22.11, SSIM of 0.8729, and LPIPS of 0.1298. These results show the framework's capability to ensure spatial precision and visual fidelity in real-time, high-risk industrial environments. 

**Abstract (ZH)**: 基于数字孪生的任务导向边缘辅助跨系统框架在工业元宇宙中的实时人机交互 

---
# ArtFace: Towards Historical Portrait Face Identification via Model Adaptation 

**Title (ZH)**: ArtFace: 基于模型适应的历史人物肖像面部识别 

**Authors**: Francois Poh, Anjith George, Sébastien Marcel  

**Link**: [PDF](https://arxiv.org/pdf/2508.20626)  

**Abstract**: Identifying sitters in historical paintings is a key task for art historians, offering insight into their lives and how they chose to be seen. However, the process is often subjective and limited by the lack of data and stylistic variations. Automated facial recognition is capable of handling challenging conditions and can assist, but while traditional facial recognition models perform well on photographs, they struggle with paintings due to domain shift and high intra-class variation. Artistic factors such as style, skill, intent, and influence from other works further complicate recognition. In this work, we investigate the potential of foundation models to improve facial recognition in artworks. By fine-tuning foundation models and integrating their embeddings with those from conventional facial recognition networks, we demonstrate notable improvements over current state-of-the-art methods. Our results show that foundation models can bridge the gap where traditional methods are ineffective. Paper page at this https URL 

**Abstract (ZH)**: 历史绘画中人物身份识别是艺术史学家的一项关键任务，有助于了解人物的生活及其选择呈现的方式。然而，这一过程往往是主观的，并受到数据不足和风格差异的限制。自动化面部识别能够处理具有挑战性的条件，并提供辅助，但传统面部识别模型在照片上表现良好，但在绘画上则因领域转移和高内类变异性而挣扎。艺术因素如风格、技巧、意图及受其他作品影响进一步加剧了识别难度。在本文中，我们研究了基础模型在提高艺术品面部识别方面潜力。通过微调基础模型并将它们的嵌入与传统面部识别网络的嵌入整合，我们展示了在现有顶级方法上的显著改进。我们的结果表明，基础模型能够弥补传统方法无效的空白。论文页面详见此链接：https://this.is/url。 

---
# Flowing Straighter with Conditional Flow Matching for Accurate Speech Enhancement 

**Title (ZH)**: 基于条件流匹配的精确语音增强 

**Authors**: Mattias Cross, Anton Ragni  

**Link**: [PDF](https://arxiv.org/pdf/2508.20584)  

**Abstract**: Current flow-based generative speech enhancement methods learn curved probability paths which model a mapping between clean and noisy speech. Despite impressive performance, the implications of curved probability paths are unknown. Methods such as Schrodinger bridges focus on curved paths, where time-dependent gradients and variance do not promote straight paths. Findings in machine learning research suggest that straight paths, such as conditional flow matching, are easier to train and offer better generalisation. In this paper we quantify the effect of path straightness on speech enhancement quality. We report experiments with the Schrodinger bridge, where we show that certain configurations lead to straighter paths. Conversely, we propose independent conditional flow-matching for speech enhancement, which models straight paths between noisy and clean speech. We demonstrate empirically that a time-independent variance has a greater effect on sample quality than the gradient. Although conditional flow matching improves several speech quality metrics, it requires multiple inference steps. We rectify this with a one-step solution by inferring the trained flow-based model as if it was directly predictive. Our work suggests that straighter time-independent probability paths improve generative speech enhancement over curved time-dependent paths. 

**Abstract (ZH)**: 基于流的方法在噪声抑制中的生成性增强学习到曲率概率路径，这些路径建模了干净语音和噪声语音之间的映射。尽管表现出色，但曲率概率路径的影响尚不知晓。Schrödinger桥梁等方法专注于曲率路径，其中时间相关的梯度和方差不促进直线路径。机器学习研究中的发现表明，如条件流匹配这样的直线路径更容易训练并提供更好的泛化能力。在本文中，我们量化了路径直线性对语音增强质量的影响。我们使用Schrödinger桥梁进行实验，表明某些配置导致更直线的路径。相反，我们提出了独立条件流匹配方法用于语音增强，该方法建模了噪声语音和干净语音之间的直线路径。实验结果显示，时间无关的方差对样本质量的影响大于梯度。尽管条件流匹配提高了多种语音质量指标，但仍需要多步推理。我们通过将训练中的流模型直接预测来实现一步解决方案。我们的工作表明，时间无关的更直线的概率路径在生成性语音增强中优于时间相关的曲率路径。 

---
# Towards Mechanistic Defenses Against Typographic Attacks in CLIP 

**Title (ZH)**: 面向CLIPagainst语义攻击的机理防御方法探究 

**Authors**: Lorenz Hufe, Constantin Venhoff, Maximilian Dreyer, Sebastian Lapuschkin, Wojciech Samek  

**Link**: [PDF](https://arxiv.org/pdf/2508.20570)  

**Abstract**: Typographic attacks exploit multi-modal systems by injecting text into images, leading to targeted misclassifications, malicious content generation and even Vision-Language Model jailbreaks. In this work, we analyze how CLIP vision encoders behave under typographic attacks, locating specialized attention heads in the latter half of the model's layers that causally extract and transmit typographic information to the cls token. Building on these insights, we introduce a method to defend CLIP models against typographic attacks by selectively ablating a typographic circuit, consisting of attention heads. Without requiring finetuning, our method improves performance by up to 19.6% on a typographic variant of ImageNet-100, while reducing standard ImageNet-100 accuracy by less than 1%. Notably, our training-free approach remains competitive with current state-of-the-art typographic defenses that rely on finetuning. To this end, we release a family of dyslexic CLIP models which are significantly more robust against typographic attacks. These models serve as suitable drop-in replacements for a broad range of safety-critical applications, where the risks of text-based manipulation outweigh the utility of text recognition. 

**Abstract (ZH)**: typographic 攻击通过向图像注入文本来利用多模态系统，导致目标错误分类、恶意内容生成，甚至视觉语言模型的逃逸。在本文中，我们分析了 CLIP 视觉编码器在 typographic 攻击下的行为，定位到模型后半部分层中的专门注意头，这些头因果性地提取并传递 typographic 信息至 cls 标记。基于这些洞见，我们提出了一种方法，通过选择性地消除 typographic 循环（由注意头组成）来防御 CLIP 模型免受 typographic 攻击，无需微调，该方法在 typographic 变体的 ImageNet-100 数据集上性能提升高达 19.6%，同时将标准 ImageNet-100 的准确性降低不到 1%。值得注意的是，我们的无需训练的方法与目前依赖微调的状态最先进 typographic 防御方法具有竞争力。为此，我们发布了家族系列的 dyslexic CLIP 模型，这些模型在很大程度上抵御 typographic 攻击。这些模型适合作为广泛的安全关键应用的合适即插即用替代品，特别是在文本操纵的风险超过文本识别的实用性时。 

---
# AI and Agile Software Development: A Research Roadmap from the XP2025 Workshop 

**Title (ZH)**: AI和敏捷软件开发：来自XP2025研讨会的研究路线图 

**Authors**: Zheying Zhang, Tomas Herda, Victoria Pichler, Pekka Abrahamsson, Geir K. Hanssen, Joshua Kerievsky, Alex Polyakov, Mohit Chandna, Marius Irgens, Kai-Kristian Kemell, Ayman Asad Khan, Crystal Kwok, Evan Leybourn, Munish Malik, Dorota Mleczko, Morteza Moalagh, Christopher Morales, Yuliia Pieskova, Daniel Planötscher, Mika Saari, Anastasiia Tkalich, Karl Josef Gstettner, Xiaofeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20563)  

**Abstract**: This paper synthesizes the key findings from a full-day XP2025 workshop on "AI and Agile: From Frustration to Success", held in Brugg-Windisch, Switzerland. The workshop brought together over 30 interdisciplinary academic researchers and industry practitioners to tackle the concrete challenges and emerging opportunities at the intersection of Generative Artificial Intelligence (GenAI) and agile software development. Through structured, interactive breakout sessions, participants identified shared pain points like tool fragmentation, governance, data quality, and critical skills gaps in AI literacy and prompt engineering. These issues were further analyzed, revealing underlying causes and cross-cutting concerns. The workshop concluded by collaboratively co-creating a multi-thematic research roadmap, articulating both short-term, implementable actions and visionary, long-term research directions. This cohesive agenda aims to guide future investigation and drive the responsible, human-centered integration of GenAI into agile practices. 

**Abstract (ZH)**: 本文综合了在瑞士布吕格-温迪施举行的一整天XP2025研讨会“AI与敏捷：从挫折到成功”的关键发现。该研讨会汇聚了超过30名跨学科的学术研究人员和行业实践者，共同探讨生成式人工智能（GenAI）与敏捷软件开发交汇处的具体挑战和新兴机遇。通过结构化的互动分组讨论，参与者识别出工具碎片化、治理、数据质量以及关键的人工智能素养和提示工程技能缺口等共同痛点。这些问题进一步被分析，揭示了其背后的成因及跨领域的关注点。研讨会最终通过协作共同创建了一个多主题的研究路线图，既包括短期可实施的行动方案，也包括具有前瞻性的长期研究方向。这一统一的议程旨在引导未来的研究，并推动生成式人工智能负责任地融入敏捷实践之中。 

---
# Adaptive Federated Distillation for Multi-Domain Non-IID Textual Data 

**Title (ZH)**: 自适应联邦蒸馏多域非IID文本数据 

**Authors**: Jiahao Xiao, Jiangming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20557)  

**Abstract**: The widespread success of pre-trained language models has established a new training paradigm, where a global PLM is fine-tuned using task-specific data from local clients. The local data are highly different from each other and can not capture the global distribution of the whole data in real world. To address the challenges of non-IID data in real environments, privacy-preserving federated distillation has been proposed and highly investigated. However, previous experimental non-IID scenarios are primarily identified with the label (output) diversity, without considering the diversity of language domains (input) that is crucial in natural language processing. In this paper, we introduce a comprehensive set of multi-domain non-IID scenarios and propose a unified benchmarking framework that includes diverse data. The benchmark can be used to evaluate the federated learning framework in a real environment. To this end, we propose an Adaptive Federated Distillation (AdaFD) framework designed to address multi-domain non-IID challenges in both homogeneous and heterogeneous settings. Experimental results demonstrate that our models capture the diversity of local clients and achieve better performance compared to the existing works. The code for this paper is available at: this https URL. 

**Abstract (ZH)**: 预训练语言模型的广泛应用确立了一种新的训练范式，其中全局预训练模型使用来自本地客户端的任务特定数据进行 fine-tune。本地数据彼此高度不同，无法捕获整个真实世界数据的全局分布。为了应对真实环境中非-IID数据的挑战，隐私保护联邦蒸馏已被提出并进行了深入研究。然而，之前实验中的非-IID场景主要关注标签（输出）多样性，未考虑对自然语言处理至关重要的语言领域多样性。本文引入了一组全面的多领域非-IID场景，并提出了一体化的基准框架，其中包括多样化的数据。该基准可用于评估真实环境中的联邦学习框架。为此，我们提出了一种针对同质性和异质性环境中多领域非-IID挑战的自适应联邦蒸馏（AdaFD）框架。实验结果表明，我们的模型捕捉了本地客户端的多样性，并且在性能上优于现有工作。本文的相关代码可访问：this https URL。 

---
# Overview of BioASQ 2025: The Thirteenth BioASQ Challenge on Large-Scale Biomedical Semantic Indexing and Question Answering 

**Title (ZH)**: BioASQ 2025：第十三届生物医学语义索引和问答挑战赛概述 

**Authors**: Anastasios Nentidis, Georgios Katsimpras, Anastasia Krithara, Martin Krallinger, Miguel Rodríguez-Ortega, Eduard Rodriguez-López, Natalia Loukachevitch, Andrey Sakhovskiy, Elena Tutubalina, Dimitris Dimitriadis, Grigorios Tsoumakas, George Giannakoulas, Alexandra Bekiaridou, Athanasios Samaras, Giorgio Maria Di Nunzio, Nicola Ferro, Stefano Marchesin, Marco Martinelli, Gianmaria Silvello, Georgios Paliouras  

**Link**: [PDF](https://arxiv.org/pdf/2508.20554)  

**Abstract**: This is an overview of the thirteenth edition of the BioASQ challenge in the context of the Conference and Labs of the Evaluation Forum (CLEF) 2025. BioASQ is a series of international challenges promoting advances in large-scale biomedical semantic indexing and question answering. This year, BioASQ consisted of new editions of the two established tasks, b and Synergy, and four new tasks: a) Task MultiClinSum on multilingual clinical summarization. b) Task BioNNE-L on nested named entity linking in Russian and English. c) Task ELCardioCC on clinical coding in cardiology. d) Task GutBrainIE on gut-brain interplay information extraction. In this edition of BioASQ, 83 competing teams participated with more than 1000 distinct submissions in total for the six different shared tasks of the challenge. Similar to previous editions, several participating systems achieved competitive performance, indicating the continuous advancement of the state-of-the-art in the field. 

**Abstract (ZH)**: 这是CLEF 2025会议和评价论坛实验室第十三届BioASQ挑战的概览。BioASQ是一系列促进大规模生物医学语义索引和问答技术发展的国际挑战。今年，BioASQ包括两个既定任务b和Synergy的新版，以及四个新任务：a) 多语言临床总结任务MultiClinSum。b) 俄语和英语中嵌套命名实体链接任务BioNNE-L。c) 心脏病临床编码任务ELCardioCC。d) 肠-脑交互信息提取任务GutBrainIE。在本次BioASQ挑战中，共有83支参赛队伍参与，提交了总计超过1000份不同的共享任务参赛作品。与往届相同，多个参赛系统表现出了竞争力，表明该领域的一流技术持续进步。 

---
# MedGR$^2$: Breaking the Data Barrier for Medical Reasoning via Generative Reward Learning 

**Title (ZH)**: MedGR\$^2\$: 通过生成奖励学习打破医疗推理的数据壁垒 

**Authors**: Weihai Zhi, Jiayan Guo, Shangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.20549)  

**Abstract**: The application of Vision-Language Models (VLMs) in medicine is critically hampered by the scarcity of high-quality, expert-annotated data. Supervised Fine-Tuning (SFT) on existing datasets often leads to poor generalization on unseen modalities and tasks, while Reinforcement Learning (RL), a promising alternative, is stymied by the lack of reliable reward signals in this data-scarce domain. To break this impasse, we introduce Generative Reward Learning for Medical Reasoning (MedGR$^2$), a novel framework that creates a self-improving virtuous cycle. MedGR$^2$ co-develops a data generator and a reward model, enabling the automated, continuous creation of high-quality, multi-modal medical data that serves as both a superior training source for SFT and RL. Our experiments demonstrate that SFT with MedGR$^2$-produced data already surpasses baselines trained on large-scale, human-curated datasets. Crucially, when leveraging this data for RL via Group Relative Policy Optimization (GRPO), our model achieves state-of-the-art cross-modality and cross-task generalization, significantly outperforming specialized RL-based methods. Furthermore, our compact model, empowered by MedGR$^2$, achieves performance competitive with foundation models possessing over 10 times more parameters. MedGR$^2$ presents a new paradigm for data-efficient learning in high-stakes domains, transforming the problem from data scarcity to data generation and unlocking the full potential of RL for building truly generalizable medical AI. 

**Abstract (ZH)**: 医学领域中视觉-语言模型的应用受到高质量专家标注数据稀缺的严重阻碍。现有的监督微调(SFT)往往在未见过的模态和任务上表现出较差的一般化能力，而强化学习(RL)，作为一种有 promise 的替代方法，由于数据稀缺领域的可靠奖励信号缺乏而受阻。为解决这一困境，我们提出了医学推理中的生成奖励学习框架（MedGR$^2$），这是一种新颖的方法，能够创造一个自我改进的良性循环。MedGR$^2$ 共同开发了一个数据生成器和一个奖励模型，使高质量多模态医学数据的自动化和持续生成成为可能，这些数据既可作为监督微调(SFT)和RL的优质训练源。我们的实验证明，使用MedGR$^2$生成的数据进行的监督微调已经超越了在大规模人工标注数据集上训练的基本模型。更关键的是，当利用这些数据通过组相对策略优化(GRPO)进行RL时，我们的模型在跨模态和跨任务一般化方面达到了最先进的性能，显著优于专门的基于RL的方法。此外，我们的紧凑型模型借助MedGR$^2的赋能，性能可与参数量超过其10倍的基模型相匹敌。MedGR$^2”为高风险领域的数据高效学习提供了新的范式，将问题从数据稀缺转变为数据生成，并解锁了RL在构建真正通用的医疗AI方面的全部潜力。 

---
# Overview of BioASQ 2024: The twelfth BioASQ challenge on Large-Scale Biomedical Semantic Indexing and Question Answering 

**Title (ZH)**: BioASQ 2024 生物医学大型语义索引与问答挑战赛十二届概述 

**Authors**: Anastasios Nentidis, Georgios Katsimpras, Anastasia Krithara, Salvador Lima-López, Eulàlia Farré-Maduell, Martin Krallinger, Natalia Loukachevitch, Vera Davydova, Elena Tutubalina, Georgios Paliouras  

**Link**: [PDF](https://arxiv.org/pdf/2508.20532)  

**Abstract**: This is an overview of the twelfth edition of the BioASQ challenge in the context of the Conference and Labs of the Evaluation Forum (CLEF) 2024. BioASQ is a series of international challenges promoting advances in large-scale biomedical semantic indexing and question answering. This year, BioASQ consisted of new editions of the two established tasks b and Synergy, and two new tasks: a) MultiCardioNER on the adaptation of clinical entity detection to the cardiology domain in a multilingual setting, and b) BIONNE on nested NER in Russian and English. In this edition of BioASQ, 37 competing teams participated with more than 700 distinct submissions in total for the four different shared tasks of the challenge. Similarly to previous editions, most of the participating systems achieved competitive performance, suggesting the continuous advancement of the state-of-the-art in the field. 

**Abstract (ZH)**: BioASQ挑战的第十二届概述：CLEF 2024会议与评估论坛实验室上下文中的进展与新任务 

---
# BridgeShield: Enhancing Security for Cross-chain Bridge Applications via Heterogeneous Graph Mining 

**Title (ZH)**: BridgeShield: 通过异构图挖掘增强跨链桥应用的安全性 

**Authors**: Dan Lin, Shunfeng Lu, Ziyan Liu, Jiajing Wu, Junyuan Fang, Kaixin Lin, Bowen Song, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.20517)  

**Abstract**: Cross-chain bridges play a vital role in enabling blockchain interoperability. However, due to the inherent design flaws and the enormous value they hold, they have become prime targets for hacker attacks. Existing detection methods show progress yet remain limited, as they mainly address single-chain behaviors and fail to capture cross-chain semantics. To address this gap, we leverage heterogeneous graph attention networks, which are well-suited for modeling multi-typed entities and relations, to capture the complex execution semantics of cross-chain behaviors. We propose BridgeShield, a detection framework that jointly models the source chain, off-chain coordination, and destination chain within a unified heterogeneous graph representation. BridgeShield incorporates intra-meta-path attention to learn fine-grained dependencies within cross-chain paths and inter-meta-path attention to highlight discriminative cross-chain patterns, thereby enabling precise identification of attack behaviors. Extensive experiments on 51 real-world cross-chain attack events demonstrate that BridgeShield achieves an average F1-score of 92.58%, representing a 24.39% improvement over state-of-the-art baselines. These results validate the effectiveness of BridgeShield as a practical solution for securing cross-chain bridges and enhancing the resilience of multi-chain ecosystems. 

**Abstract (ZH)**: 跨链桥梁在促进区块链互操作性方面发挥着关键作用。然而，由于其固有的设计缺陷和所持有的巨大价值，它们成为黑客攻击的主要目标。现有检测方法虽有进展但仍有限制，因为它们主要关注单链行为而未能捕捉到跨链语义。为弥补这一差距，我们利用适合 modeling 多类型实体和关系的异构图注意力网络来捕捉跨链行为的复杂执行语义。我们提出 BridgeShield，这是一种联合建模源链、链外协调和目的链的检测框架，采用统一的异构图表示。BridgeShield 结合使用 intra-meta-path 注意力学习跨链路径内的细粒度依赖关系，并使用 inter-meta-path 注意力突出跨链模式，从而实现对攻击行为的精确识别。在 51 个真实世界的跨链攻击事件上的广泛实验表明，BridgeShield 的平均 F1 得分为 92.58%，比最先进的基线方法改进了 24.39%。这些结果验证了 BridgeShield 作为跨链桥梁安全解决方案的有效性以及对多链生态系统的增强韧性。 

---
# Languages Still Left Behind: Toward a Better Multilingual Machine Translation Benchmark 

**Title (ZH)**: 语言仍被遗忘：向更好的多语言机器翻译评估标准迈进 

**Authors**: Chihiro Taguchi, Seng Mai, Keita Kurabe, Yusuke Sakai, Georgina Agyei, Soudabeh Eslami, David Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20511)  

**Abstract**: Multilingual machine translation (MT) benchmarks play a central role in evaluating the capabilities of modern MT systems. Among them, the FLORES+ benchmark is widely used, offering English-to-many translation data for over 200 languages, curated with strict quality control protocols. However, we study data in four languages (Asante Twi, Japanese, Jinghpaw, and South Azerbaijani) and uncover critical shortcomings in the benchmark's suitability for truly multilingual evaluation. Human assessments reveal that many translations fall below the claimed 90% quality standard, and the annotators report that source sentences are often too domain-specific and culturally biased toward the English-speaking world. We further demonstrate that simple heuristics, such as copying named entities, can yield non-trivial BLEU scores, suggesting vulnerabilities in the evaluation protocol. Notably, we show that MT models trained on high-quality, naturalistic data perform poorly on FLORES+ while achieving significant gains on our domain-relevant evaluation set. Based on these findings, we advocate for multilingual MT benchmarks that use domain-general and culturally neutral source texts rely less on named entities, in order to better reflect real-world translation challenges. 

**Abstract (ZH)**: 多语言机器翻译基准在评估现代机器翻译系统的能力中发挥着核心作用。其中，FLORES+基准被广泛使用，提供了面向200多种语言的英译多语言数据，并遵循严格的质量控制协议。然而，我们研究了四种语言（阿桑蒂陶伊、日语、景颇语和西南土耳其语）的数据，并发现基准在真正多语言评估方面的适用性存在关键不足。人类评估显示，许多翻译未能达到声称的90%质量标准，且注释人员表示，源句子往往过于领域特定且文化上偏向英语国家。我们进一步证明，简单的启发式方法，如复制命名实体，可以产生非平凡的BLEU分数，这表明评估协议存在漏洞。值得注意的是，我们表明，训练于高质量自然数据的机器翻译模型在FLORES+上表现不佳，但在我们相关的领域评估集上取得了显著进步。基于这些发现，我们建议使用通用领域和文化中立的源文本的多语言机器翻译基准，并减少对命名实体的依赖，以便更好地反映实际翻译挑战。 

---
# Photonic restricted Boltzmann machine for content generation tasks 

**Title (ZH)**: 基于光子的受限制玻尔兹曼机用于内容生成任务 

**Authors**: Li Luo, Yisheng Fang, Wanyi Zhang, Zhichao Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2508.20472)  

**Abstract**: The restricted Boltzmann machine (RBM) is a neural network based on the Ising model, well known for its ability to learn probability distributions and stochastically generate new content. However, the high computational cost of Gibbs sampling in content generation tasks imposes significant bottlenecks on electronic implementations. Here, we propose a photonic restricted Boltzmann machine (PRBM) that leverages photonic computing to accelerate Gibbs sampling, enabling efficient content generation. By introducing an efficient encoding method, the PRBM eliminates the need for computationally intensive matrix decomposition and reduces the computational complexity of Gibbs sampling from $O(N)$ to $O(1)$. Moreover, its non-Von Neumann photonic computing architecture circumvents the memory storage of interaction matrices, providing substantial advantages for large-scale RBMs. We experimentally validate the photonic-accelerated Gibbs sampling by simulating a two-dimensional Ising model, where the observed phase transition temperature closely matches the theoretical predictions. Beyond physics-inspired tasks, the PRBM demonstrates robust capabilities in generating and restoring diverse content, including images and temporal sequences, even in the presence of noise and aberrations. The scalability and reduced training cost of the PRBM framework underscore its potential as a promising pathway for advancing photonic computing in generative artificial intelligence. 

**Abstract (ZH)**: 基于光子计算的受限制玻尔兹曼机（PRBM）及其加速 Gibbs 抽样方法 

---
# Dual-Model Weight Selection and Self-Knowledge Distillation for Medical Image Classification 

**Title (ZH)**: 双模型权重选择与自我知识蒸馏在医学图像分类中的应用 

**Authors**: Ayaka Tsutsumi, Guang Li, Ren Togo, Takahiro Ogawa, Satoshi Kondo, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2508.20461)  

**Abstract**: We propose a novel medical image classification method that integrates dual-model weight selection with self-knowledge distillation (SKD). In real-world medical settings, deploying large-scale models is often limited by computational resource constraints, which pose significant challenges for their practical implementation. Thus, developing lightweight models that achieve comparable performance to large-scale models while maintaining computational efficiency is crucial. To address this, we employ a dual-model weight selection strategy that initializes two lightweight models with weights derived from a large pretrained model, enabling effective knowledge transfer. Next, SKD is applied to these selected models, allowing the use of a broad range of initial weight configurations without imposing additional excessive computational cost, followed by fine-tuning for the target classification tasks. By combining dual-model weight selection with self-knowledge distillation, our method overcomes the limitations of conventional approaches, which often fail to retain critical information in compact models. Extensive experiments on publicly available datasets-chest X-ray images, lung computed tomography scans, and brain magnetic resonance imaging scans-demonstrate the superior performance and robustness of our approach compared to existing methods. 

**Abstract (ZH)**: 我们提出了一种结合双模型权重选择与自我知识蒸馏（SKD）的新型医疗图像分类方法。 

---
# Evaluating Differentially Private Generation of Domain-Specific Text 

**Title (ZH)**: 评价差异隐私生成领域特定文本 

**Authors**: Yidan Sun, Viktor Schlegel, Srinivasan Nandakumar, Iqra Zahid, Yuping Wu, Warren Del-Pinto, Goran Nenadic, Siew-Kei Lam, Jie Zhang, Anil A Bharath  

**Link**: [PDF](https://arxiv.org/pdf/2508.20452)  

**Abstract**: Generative AI offers transformative potential for high-stakes domains such as healthcare and finance, yet privacy and regulatory barriers hinder the use of real-world data. To address this, differentially private synthetic data generation has emerged as a promising alternative. In this work, we introduce a unified benchmark to systematically evaluate the utility and fidelity of text datasets generated under formal Differential Privacy (DP) guarantees. Our benchmark addresses key challenges in domain-specific benchmarking, including choice of representative data and realistic privacy budgets, accounting for pre-training and a variety of evaluation metrics. We assess state-of-the-art privacy-preserving generation methods across five domain-specific datasets, revealing significant utility and fidelity degradation compared to real data, especially under strict privacy constraints. These findings underscore the limitations of current approaches, outline the need for advanced privacy-preserving data sharing methods and set a precedent regarding their evaluation in realistic scenarios. 

**Abstract (ZH)**: 生成式AI在医疗保健和金融等领域提供变革性潜力，但由于隐私和监管障碍，限制了实际数据的使用。为此，差异化隐私合成数据生成已成为一种有前景的替代方案。本研究引入了一个统一基准，系统评估在正式差异隐私（DP）保证下生成的文本数据集的实用性和保真度。该基准解决了领域特定基准测试中的关键挑战，包括代表数据的选择和现实的隐私预算，考虑了预训练并采用多种评估指标。我们评估了五种领域特定数据集上的最新隐私保护生成方法，结果显示，在严格隐私约束下，与真实数据相比，实用性和保真度显著下降。这些发现突显了当前方法的局限性，指出了急需先进的隐私保护数据共享方法，并为其实用场景下的评估设定了范例。 

---
# Uncovering the Spectral Bias in Diagonal State Space Models 

**Title (ZH)**: 揭示对角状态空间模型的频谱偏见 

**Authors**: Ruben Solozabal, Velibor Bojkovic, Hilal AlQuabeh, Kentaro Inui, Martin Takáč  

**Link**: [PDF](https://arxiv.org/pdf/2508.20441)  

**Abstract**: Current methods for initializing state space models (SSMs) parameters mainly rely on the \textit{HiPPO framework}, which is based on an online approximation of orthogonal polynomials. Recently, diagonal alternatives have shown to reach a similar level of performance while being significantly more efficient due to the simplification in the kernel computation. However, the \textit{HiPPO framework} does not explicitly study the role of its diagonal variants. In this paper, we take a further step to investigate the role of diagonal SSM initialization schemes from the frequency perspective. Our work seeks to systematically understand how to parameterize these models and uncover the learning biases inherent in such diagonal state-space models. Based on our observations, we propose a diagonal initialization on the discrete Fourier domain \textit{S4D-DFouT}. The insights in the role of pole placing in the initialization enable us to further scale them and achieve state-of-the-art results on the Long Range Arena benchmark, allowing us to train from scratch on very large datasets as PathX-256. 

**Abstract (ZH)**: 当前用于初始化状态空间模型参数的方法主要依赖于HiPPO框架，该框架基于在线正交多项式的近似计算。近期，对角线替代方法已显示出相似的性能水平，由于核计算的简化而显著更加高效。然而，HiPPO框架并没有明确研究其对角线变体的作用。在本文中，我们从频率角度进一步探讨对角线状态空间模型初始化方案的作用。我们的工作旨在系统地理解如何参数化这些模型，并揭示此类对角线状态空间模型中的学习偏见。基于我们的观察，我们提出了一种基于离散傅里叶域的对角线初始化方法S4D-DFouT。对极点放置在初始化中的作用的洞察使我们能够进一步扩展该方法，并在Long Range Arena基准上取得最先进的结果，使我们能够从头开始在非常大的数据集上进行训练，如PathX-256。 

---
# On Identifying Why and When Foundation Models Perform Well on Time-Series Forecasting Using Automated Explanations and Rating 

**Title (ZH)**: 基于自动化解释和评价，探究基础模型在时间序列预测中表现优异的原因及时机 

**Authors**: Michael Widener, Kausik Lakkaraju, John Aydin, Biplav Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2508.20437)  

**Abstract**: Time-series forecasting models (TSFM) have evolved from classical statistical methods to sophisticated foundation models, yet understanding why and when these models succeed or fail remains challenging. Despite this known limitation, time series forecasting models are increasingly used to generate information that informs real-world actions with equally real consequences. Understanding the complexity, performance variability, and opaque nature of these models then becomes a valuable endeavor to combat serious concerns about how users should interact with and rely on these models' outputs. This work addresses these concerns by combining traditional explainable AI (XAI) methods with Rating Driven Explanations (RDE) to assess TSFM performance and interpretability across diverse domains and use cases. We evaluate four distinct model architectures: ARIMA, Gradient Boosting, Chronos (time-series specific foundation model), Llama (general-purpose; both fine-tuned and base models) on four heterogeneous datasets spanning finance, energy, transportation, and automotive sales domains. In doing so, we demonstrate that feature-engineered models (e.g., Gradient Boosting) consistently outperform foundation models (e.g., Chronos) in volatile or sparse domains (e.g., power, car parts) while providing more interpretable explanations, whereas foundation models excel only in stable or trend-driven contexts (e.g., finance). 

**Abstract (ZH)**: 时间序列预测模型（TSFM）从经典统计方法演进到了复杂的基础模型，但仍对这些模型为何成功或失败缺乏深刻理解。尽管存在这一已知局限性，时间序列预测模型仍被广泛用于生成影响现实世界行动的重要信息。因此，理解这些模型的复杂性、性能变异性及不透明性质，成为了一个有价值的研究领域，以解决用户如何与这些模型的输出进行交互和依赖的问题。本文通过结合传统的可解释人工智能（XAI）方法和评分驱动解释（RDE），评估了TSFM在不同领域的性能和可解释性。我们在金融、能源、交通和汽车销售等多个领域中，使用了四种不同的模型架构（ARIMA、梯度提升、Chronos、Llama）对四个异构数据集进行了评估。结果显示，在波动或稀疏的数据领域（如电力、汽车零部件），特征工程模型（如梯度提升）始终优于基础模型（如Chronos），并且提供更可解释的解释；而在稳定或趋势驱动的背景下（如金融），基础模型则表现出色。 

---
# Rethinking Purity and Diversity in Multi-Behavior Sequential Recommendation from the Frequency Perspective 

**Title (ZH)**: 从频率视角重新思考多行为序列推荐中的纯度与多样性 

**Authors**: Yongqiang Han, Kai Cheng, Kefan Wang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.20427)  

**Abstract**: In recommendation systems, users often exhibit multiple behaviors, such as browsing, clicking, and purchasing. Multi-behavior sequential recommendation (MBSR) aims to consider these different behaviors in an integrated manner to improve the recommendation performance of the target behavior. However, some behavior data will also bring inevitable noise to the modeling of user interests. Some research efforts focus on data denoising from the frequency domain perspective to improve the accuracy of user preference prediction. These studies indicate that low-frequency information tends to be valuable and reliable, while high-frequency information is often associated with noise. In this paper, we argue that high-frequency information is by no means insignificant. Further experimental results highlight that low frequency corresponds to the purity of user interests, while high frequency corresponds to the diversity of user interests. Building upon this finding, we proposed our model PDB4Rec, which efficiently extracts information across various frequency bands and their relationships, and introduces Boostrapping Balancer mechanism to balance their contributions for improved recommendation performance. Sufficient experiments on real-world datasets demonstrate the effectiveness and efficiency of our model. 

**Abstract (ZH)**: 多行为序列推荐中高频率信息的重要性及PDB4Rec模型 

---
# Assessing local deformation and computing scalar curvature with nonlinear conformal regularization of decoders 

**Title (ZH)**: 评估局部变形并计算标量曲率的非线性共形正则化解码器方法 

**Authors**: Benjamin Couéraud, Vikram Sunkara, Christof Schütte  

**Link**: [PDF](https://arxiv.org/pdf/2508.20413)  

**Abstract**: One aim of dimensionality reduction is to discover the main factors that explain the data, and as such is paramount to many applications. When working with high dimensional data, autoencoders offer a simple yet effective approach to learn low-dimensional representations. The two components of a general autoencoder consist first of an encoder that maps the observed data onto a latent space; and second a decoder that maps the latent space back to the original observation space, which allows to learn a low-dimensional manifold representation of the original data. In this article, we introduce a new type of geometric regularization for decoding maps approximated by deep neural networks, namely nonlinear conformal regularization. This regularization procedure permits local variations of the decoder map and comes with a new scalar field called conformal factor which acts as a quantitative indicator of the amount of local deformation sustained by the latent space when mapped into the original data space. We also show that this regularization technique allows the computation of the scalar curvature of the learned manifold. Implementation and experiments on the Swiss roll and CelebA datasets are performed to illustrate how to obtain these quantities from the architecture. 

**Abstract (ZH)**: 降低数据维度的一目的是发现解释数据的主要因素，这对于许多应用至关重要。在处理高维数据时，自编码器提供了一种简单有效的方法来学习低维表示。一个通用自编码器由两个部分组成：首先是一个编码器，将观测数据映射到潜在空间；其次是一个解码器，将潜在空间映射回原始观测空间，从而学习原始数据的低维流形表示。本文介绍了一种新的几何正则化方法，用于近似由深神经网络实现的解码映射，即非线性共形正则化。该正则化过程允许解码映射的局部变化，并伴随一个新标量场称为共形因子，它作为潜在空间在映射到原始数据空间时所承受的局部变形量的定量指标。我们还展示了这种正则化技术如何计算所学习流形的标量曲率。在瑞士卷和CelebA数据集上的实现和实验说明了如何从架构中获取这些量。 

---
# MPFormer: Adaptive Framework for Industrial Multi-Task Personalized Sequential Retriever 

**Title (ZH)**: MPFormer: 自适应工业多任务个性化序列检索框架 

**Authors**: Yijia Sun, Shanshan Huang, Linxiao Che, Haitao Lu, Qiang Luo, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.20400)  

**Abstract**: Modern industrial recommendation systems encounter a core challenge of multi-stage optimization misalignment: a significant semantic gap exists between the multi-objective optimization paradigm widely used in the ranking phase and the single-objective modeling in the retrieve phase. Although the mainstream industry solution achieves multi-objective coverage through parallel multi-path single-objective retrieval, this approach leads to linear growth of training and serving resources with the number of objectives and has inherent limitations in handling loosely coupled objectives. This paper proposes the MPFormer, a dynamic multi-task Transformer framework, which systematically addresses the aforementioned issues through three innovative mechanisms. First, an objective-conditioned transformer that jointly encodes user behavior sequences and multi-task semantics through learnable attention modulation; second, personalized target weights are introduced to achieve dynamic adjustment of retrieval results; finally, user personalization information is incorporated into token representations and the Transformer structure to further enhance the model's representation ability. This framework has been successfully integrated into Kuaishou short video recommendation system, stably serving over 400 million daily active users. It significantly improves user daily engagement and system operational efficiency. Practical deployment verification shows that, compared with traditional solutions, it effectively optimizes the iterative paradigm of multi-objective retrieval while maintaining service response speed, providing a scalable multi-objective solution for industrial recommendation systems. 

**Abstract (ZH)**: 现代工业推荐系统中的多阶段优化不一致性问题：基于排名阶段广泛使用的多目标优化范式与检索阶段的单目标建模之间的显著语义鸿沟的动态多任务Transformer框架 

---
# TF-TransUNet1D: Time-Frequency Guided Transformer U-Net for Robust ECG Denoising in Digital Twin 

**Title (ZH)**: TF-TransUNet1D：基于时频引导的变压器U-网在数字孪生中的稳健心电图去噪 

**Authors**: Shijie Wang, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.20398)  

**Abstract**: Electrocardiogram (ECG) signals serve as a foundational data source for cardiac digital twins, yet their diagnostic utility is frequently compromised by noise and artifacts. To address this issue, we propose TF-TransUNet1D, a novel one-dimensional deep neural network that integrates a U-Net-based encoder-decoder architecture with a Transformer encoder, guided by a hybrid time-frequency domain loss. The model is designed to simultaneously capture local morphological features and long-range temporal dependencies, which are critical for preserving the diagnostic integrity of ECG signals. To enhance denoising robustness, we introduce a dual-domain loss function that jointly optimizes waveform reconstruction in the time domain and spectral fidelity in the frequency domain. In particular, the frequency-domain component effectively suppresses high-frequency noise while maintaining the spectral structure of the signal, enabling recovery of subtle but clinically significant waveform components. We evaluate TF-TransUNet1D using synthetically corrupted signals from the MIT-BIH Arrhythmia Database and the Noise Stress Test Database (NSTDB). Comparative experiments against state-of-the-art baselines demonstrate consistent superiority of our model in terms of SNR improvement and error metrics, achieving a mean absolute error of 0.1285 and Pearson correlation coefficient of 0.9540. By delivering high-precision denoising, this work bridges a critical gap in pre-processing pipelines for cardiac digital twins, enabling more reliable real-time monitoring and personalized modeling. 

**Abstract (ZH)**: 基于时间频率域损失的TF-TransUNet1D用于ECG信号去噪与心脏数字双生预处理 

---
# Adaptive Root Cause Localization for Microservice Systems with Multi-Agent Recursion-of-Thought 

**Title (ZH)**: 支持多Agent递归思考的微服务系统自适应根因定位 

**Authors**: Lingzhe Zhang, Tong Jia, Kangjin Wang, Weijie Hong, Chiming Duan, Minghua He, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.20370)  

**Abstract**: As contemporary microservice systems become increasingly popular and complex-often comprising hundreds or even thousands of fine-grained, interdependent subsystems-they are facing more frequent failures. Ensuring system reliability thus demands accurate root cause localization. While traces and metrics have proven to be effective data sources for this task, existing methods either heavily rely on pre-defined schemas, which struggle to adapt to evolving operational contexts, or lack interpretability in their reasoning process, thereby leaving Site Reliability Engineers (SREs) confused. In this paper, we conduct a comprehensive study on how SREs localize the root cause of failures, drawing insights from multiple professional SREs across different organizations. Our investigation reveals that human root cause analysis exhibits three key characteristics: recursiveness, multi-dimensional expansion, and cross-modal reasoning. Motivated by these findings, we introduce RCLAgent, an adaptive root cause localization method for microservice systems that leverages a multi-agent recursion-of-thought framework. RCLAgent employs a novel recursion-of-thought strategy to guide the LLM's reasoning process, effectively integrating data from multiple agents and tool-assisted analysis to accurately pinpoint the root cause. Experimental evaluations on various public datasets demonstrate that RCLAgent achieves superior performance by localizing the root cause using only a single request-outperforming state-of-the-art methods that depend on aggregating multiple requests. These results underscore the effectiveness of RCLAgent in enhancing the efficiency and precision of root cause localization in complex microservice environments. 

**Abstract (ZH)**: 当代微服务系统因日益流行和复杂（常包含数百甚至数千个细粒度、相互依赖的子系统），故障频率增加。因此，确保系统可靠性需要精确的根因定位。虽然追踪信息和指标已被证明是有效的数据来源，但现有方法要么严重依赖预定义的模式（难以适应不断变化的操作环境），要么在推理过程中缺乏可解释性，导致运维工程师（SREs）困惑。本文通过对来自不同组织的多名专业SRE的深入研究，探讨了如何定位故障的根因。研究表明，人类的根因分析具有递归性、多维度扩展性和跨模态推理的三大特征。基于这些发现，我们提出了RCLAgent——一种适应性微服务系统根因定位方法，利用多智能体思想递归框架。RCLAgent采用一种新颖的思想递归策略来引导大模型的推理过程，有效整合多智能体的数据和工具辅助分析，准确地定位根因。在多种公开数据集上进行的实验评估表明，RCLAgent仅使用单一请求即可超越依赖聚合多个请求的最先进的方法，展现出优越的性能。这些结果突显了RCLAgent在复杂微服务环境中的根因定位效率和精确性方面的有效性。 

---
# Multi-View Graph Convolution Network for Internal Talent Recommendation Based on Enterprise Emails 

**Title (ZH)**: 基于企业邮件的多视图图卷积网络内部人才推荐 

**Authors**: Soo Hyun Kim, Jang-Hyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.20328)  

**Abstract**: Internal talent recommendation is a critical strategy for organizational continuity, yet conventional approaches suffer from structural limitations, often overlooking qualified candidates by relying on the narrow perspective of a few managers. To address this challenge, we propose a novel framework that models two distinct dimensions of an employee's position fit from email data: WHAT they do (semantic similarity of tasks) and HOW they work (structural characteristics of their interactions and collaborations). These dimensions are represented as independent graphs and adaptively fused using a Dual Graph Convolutional Network (GCN) with a gating mechanism. Experiments show that our proposed gating-based fusion model significantly outperforms other fusion strategies and a heuristic baseline, achieving a top performance of 40.9% on Hit@100. Importantly, it is worth noting that the model demonstrates high interpretability by learning distinct, context-aware fusion strategies for different job families. For example, it learned to prioritize relational (HOW) data for 'sales and marketing' job families while applying a balanced approach for 'research' job families. This research offers a quantitative and comprehensive framework for internal talent discovery, minimizing the risk of candidate omission inherent in traditional methods. Its primary contribution lies in its ability to empirically determine the optimal fusion ratio between task alignment (WHAT) and collaborative patterns (HOW), which is required for employees to succeed in the new positions, thereby offering important practical implications. 

**Abstract (ZH)**: 基于电子邮件数据的员工位置匹配新框架：融合任务内容与协作方式 

---
# Differentially Private Federated Quantum Learning via Quantum Noise 

**Title (ZH)**: 差分隐私 Federated 量子学习 via 量子噪声 

**Authors**: Atit Pokharel, Ratun Rahman, Shaba Shaon, Thomas Morris, Dinh C. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.20310)  

**Abstract**: Quantum federated learning (QFL) enables collaborative training of quantum machine learning (QML) models across distributed quantum devices without raw data exchange. However, QFL remains vulnerable to adversarial attacks, where shared QML model updates can be exploited to undermine information privacy. In the context of noisy intermediate-scale quantum (NISQ) devices, a key question arises: How can inherent quantum noise be leveraged to enforce differential privacy (DP) and protect model information during training and communication? This paper explores a novel DP mechanism that harnesses quantum noise to safeguard quantum models throughout the QFL process. By tuning noise variance through measurement shots and depolarizing channel strength, our approach achieves desired DP levels tailored to NISQ constraints. Simulations demonstrate the framework's effectiveness by examining the relationship between differential privacy budget and noise parameters, as well as the trade-off between security and training accuracy. Additionally, we demonstrate the framework's robustness against an adversarial attack designed to compromise model performance using adversarial examples, with evaluations based on critical metrics such as accuracy on adversarial examples, confidence scores for correct predictions, and attack success rates. The results reveal a tunable trade-off between privacy and robustness, providing an efficient solution for secure QFL on NISQ devices with significant potential for reliable quantum computing applications. 

**Abstract (ZH)**: 量子联邦学习中利用固有量子噪声保障差分隐私的研究 

---
# Surveying the Operational Cybersecurity and Supply Chain Threat Landscape when Developing and Deploying AI Systems 

**Title (ZH)**: 开发和部署AI系统时调研操作网络安全与供应链威胁 landscape 

**Authors**: Michael R Smith, Joe Ingram  

**Link**: [PDF](https://arxiv.org/pdf/2508.20307)  

**Abstract**: The rise of AI has transformed the software and hardware landscape, enabling powerful capabilities through specialized infrastructures, large-scale data storage, and advanced hardware. However, these innovations introduce unique attack surfaces and objectives which traditional cybersecurity assessments often overlook. Cyber attackers are shifting their objectives from conventional goals like privilege escalation and network pivoting to manipulating AI outputs to achieve desired system effects, such as slowing system performance, flooding outputs with false positives, or degrading model accuracy. This paper serves to raise awareness of the novel cyber threats that are introduced when incorporating AI into a software system. We explore the operational cybersecurity and supply chain risks across the AI lifecycle, emphasizing the need for tailored security frameworks to address evolving threats in the AI-driven landscape. We highlight previous exploitations and provide insights from working in this area. By understanding these risks, organizations can better protect AI systems and ensure their reliability and resilience. 

**Abstract (ZH)**: 人工智能的兴起已Transformer了软件和硬件landscape，通过专门的基础设施、大规模数据存储和先进的硬件增强了强大能力。然而，这些创新引入了传统网络安全评估经常忽略的独特攻击面和目标。网络攻击者的攻击目标已从传统的特权提升和网络跃变转变为操纵AI输出以实现期望的系统效果，如降低系统性能、泛滥误报或降低模型准确性。本文旨在提高人们对将AI集成到软件系统中时引入的新型网络威胁的认识。我们探讨了AI生命周期中操作网络安全和供应链风险，强调需要定制的安全框架来应对AI驱动场景中的不断演变的威胁。我们强调了之前的安全漏洞并提供了在这个领域工作的见解。通过了解这些风险，组织可以更好地保护AI系统并确保其可靠性和弹性。 

---
# Beacon: Post-Training Quantization with Integrated Grid Selection 

**Title (ZH)**: Beacon:  integral Grid Selection for Post-Training Quantization 

**Authors**: Shihao Zhang, Rayan Saab  

**Link**: [PDF](https://arxiv.org/pdf/2508.20293)  

**Abstract**: Quantization is a widely used compression technique for reducing the memory and computation costs of large pre-trained models. A key challenge in per-channel post-training quantization (PTQ) is selecting appropriate scaling factors to replace weight values with values from a scaled quantization grid. Existing methods typically fix the scale at the outset via heuristic tuning or grid search. In this note, we propose Beacon, a simple and effective algorithm that eliminates the need for such manual tuning. Beacon performs per-channel PTQ directly using a fixed non-scaled alphabet and automatically determines the optimal scaling factors by exploiting the geometry of symmetric scalar quantization. It supports both symmetric and asymmetric quantization with minimal modifications and does not rely on back-propagation or large calibration sets. Despite its simplicity and tuning-free nature, Beacon achieves competitive performance compared to state-of-the-art methods, making it a practical solution for efficient model deployment. 

**Abstract (ZH)**: 量化是一种广泛用于减少大规模预训练模型内存和计算成本的压缩技术。通道后训练量化(PTQ)中的一个关键挑战是选择合适的缩放因子，以用缩放量化网格中的值替换权重值。现有方法通常通过启发式调整或网格搜索在一开始就固定缩放比例。在本文中，我们提出了Beacon，一种简单而有效的算法，无需进行手动调整即可直接进行通道后训练量化。Beacon 利用对称标量量化几何特性自动确定最优的缩放因子，支持对称和非对称量化，并只需进行少量修改即可实现，不依赖于反向传播或大型校准集。尽管结构简单且无需调优，Beacon 的性能与最先进的方法相当，使其成为高效模型部署的实际解决方案。 

---
# Objective Value Change and Shape-Based Accelerated Optimization for the Neural Network Approximation 

**Title (ZH)**: 基于价值变化和形状加速优化的神经网络近似方法 

**Authors**: Pengcheng Xie, Zihao Zhou, Zijian Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.20290)  

**Abstract**: This paper introduce a novel metric of an objective function f, we say VC (value change) to measure the difficulty and approximation affection when conducting an neural network approximation task, and it numerically supports characterizing the local performance and behavior of neural network approximation. Neural networks often suffer from unpredictable local performance, which can hinder their reliability in critical applications. VC addresses this issue by providing a quantifiable measure of local value changes in network behavior, offering insights into the stability and performance for achieving the neural-network approximation. We investigate some fundamental theoretical properties of VC and identified two intriguing phenomena in neural network approximation: the VC-tendency and the minority-tendency. These trends respectively characterize how pointwise errors evolve in relation to the distribution of VC during the approximation this http URL addition, we propose a novel metric based on VC, which measures the distance between two functions from the perspective of variation. Building upon this metric, we further propose a new preprocessing framework for neural network approximation. Numerical results including the real-world experiment and the PDE-related scientific problem support our discovery and pre-processing acceleration method. 

**Abstract (ZH)**: 本文引入了一种新的目标函数度量标准，称为VC（值变化），用于评估神经网络逼近任务中的难度和逼近影响，并从数值上支持刻画神经网络逼近的局部性能和行为。神经网络在局部性能上往往存在不可预测性，这可能在关键应用中影响其可靠性。VC通过提供网络行为中局部值变化的可量化度量，来洞察神经网络逼近的稳定性和性能。我们探讨了VC的一些基本理论性质，并在神经网络逼近中发现了两种有趣的趋势：VC倾向性和少数派倾向性。这些趋势分别描述了在逼近过程中点态误差如何相对于VC分布演变。此外，我们基于VC提出了一种新的基于变异距离的度量方法，并在此基础上提出了一种新的预处理框架。数值结果，包括真实世界实验和与偏微分方程相关的科学问题，支持了我们的发现和预处理加速方法。 

---
# Network-Level Prompt and Trait Leakage in Local Research Agents 

**Title (ZH)**: 网络级提示和特质泄漏在本地研究代理中 

**Authors**: Hyejun Jeong, Mohammadreze Teymoorianfard, Abhinav Kumar, Amir Houmansadr, Eugene Badasarian  

**Link**: [PDF](https://arxiv.org/pdf/2508.20282)  

**Abstract**: We show that Web and Research Agents (WRAs) -- language model-based systems that investigate complex topics on the Internet -- are vulnerable to inference attacks by passive network adversaries such as ISPs. These agents could be deployed \emph{locally} by organizations and individuals for privacy, legal, or financial purposes. Unlike sporadic web browsing by humans, WRAs visit $70{-}140$ domains with distinguishable timing correlations, enabling unique fingerprinting attacks.
Specifically, we demonstrate a novel prompt and user trait leakage attack against WRAs that only leverages their network-level metadata (i.e., visited IP addresses and their timings). We start by building a new dataset of WRA traces based on user search queries and queries generated by synthetic personas. We define a behavioral metric (called OBELS) to comprehensively assess similarity between original and inferred prompts, showing that our attack recovers over 73\% of the functional and domain knowledge of user prompts. Extending to a multi-session setting, we recover up to 19 of 32 latent traits with high accuracy. Our attack remains effective under partial observability and noisy conditions. Finally, we discuss mitigation strategies that constrain domain diversity or obfuscate traces, showing negligible utility impact while reducing attack effectiveness by an average of 29\%. 

**Abstract (ZH)**: 基于网络的代理（WRAs）在被动网络对手（如ISP）的推理攻击下易受攻击——这些代理可以由组织和个人为了隐私、法律或财务目的在本地部署。我们展示了仅利用WRAs的网络层面元数据（即访问的IP地址及其时间戳）就能发起一种新颖的提示和用户特征泄漏攻击。我们通过构建基于用户搜索查询和合成 persona 生成的查询的新数据集来实现这一攻击。我们定义了一个行为指标（称为OBELS），以全面评估原始提示与推断出的提示之间的相似性，结果显示我们的攻击恢复了超过73%的用户提示的功能性和领域知识。在多会话场景中，我们能够以高精度恢复多达19个潜在的用户特征。我们的攻击在网络部分可观测性和噪声条件下仍然有效。最后，我们讨论了限制领域多样性和模糊跟踪的缓解策略，这些策略在几乎不影响实用性的同时，平均减少了29%的攻击效果。 

---
# The Mathematician's Assistant: Integrating AI into Research Practice 

**Title (ZH)**: 数学家的助手：将AI融入研究实践 

**Authors**: Jonas Henkel  

**Link**: [PDF](https://arxiv.org/pdf/2508.20236)  

**Abstract**: The rapid development of artificial intelligence (AI), marked by breakthroughs like 'AlphaEvolve' and 'Gemini Deep Think', is beginning to offer powerful new tools that have the potential to significantly alter the research practice in many areas of mathematics. This paper explores the current landscape of publicly accessible large language models (LLMs) in a mathematical research context, based on developments up to August 2, 2025. Our analysis of recent benchmarks, such as MathArena and the Open Proof Corpus (Balunović et al., 2025; Dekoninck et al., 2025), reveals a complex duality: while state-of-the-art models demonstrate strong abilities in solving problems and evaluating proofs, they also exhibit systematic flaws, including a lack of self-critique and a model depending discrepancy between final-answer accuracy and full-proof validity.
Based on these findings, we propose a durable framework for integrating AI into the research workflow, centered on the principle of the augmented mathematician. In this model, the AI functions as a copilot under the critical guidance of the human researcher, an approach distilled into five guiding principles for effective and responsible use. We then systematically explore seven fundamental ways AI can be applied across the research lifecycle, from creativity and ideation to the final writing process, demonstrating how these principles translate into concrete practice.
We conclude that the primary role of AI is currently augmentation rather than automation. This requires a new skill set focused on strategic prompting, critical verification, and methodological rigor in order to effectively use these powerful tools. 

**Abstract (ZH)**: 人工智能的快速发展，如“AlphaEvolve”和“Gemini Deep Think”等突破，正开始提供强大的新工具，有望在许多数学领域显著改变研究实践。本文基于截至2025年8月2日的最新发展，探讨了数学研究背景下公开可访问的大语言模型的现状。我们的分析揭示了复杂的双重性：尽管最先进的模型在解决问题和评估证明方面表现出强大的能力，但它们也表现出系统性的缺陷，包括缺乏自我批判和最终答案准确性与完整证明有效性之间的模型依赖差异。基于这些发现，我们提出了一种持久框架，将AI集成到研究工作流程中，以增强数学家为核心原则。在这种模型中，AI在人类研究者的关键指导下作为联合飞行员发挥作用，并提炼出五项基本原则以实现有效和负责任的使用。接着，我们系统地探讨了AI在研究生命周期中的七大基本应用方式，从创造力和创意产生到最终的写作过程，展示了这些原则如何转化为具体实践。我们得出结论，当前AI的主要作用是增强而非自动化。这需要一种新的技能组合，集中在战略性提示、批判性验证和方法论严谨性上，以有效地使用这些强大的工具。 

---
# The Role of Teacher Calibration in Knowledge Distillation 

**Title (ZH)**: 教师校准在知识蒸馏中的作用 

**Authors**: Suyoung Kim, Seonguk Park, Junhoo Lee, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2508.20224)  

**Abstract**: Knowledge Distillation (KD) has emerged as an effective model compression technique in deep learning, enabling the transfer of knowledge from a large teacher model to a compact student model. While KD has demonstrated significant success, it is not yet fully understood which factors contribute to improving the student's performance. In this paper, we reveal a strong correlation between the teacher's calibration error and the student's accuracy. Therefore, we claim that the calibration of the teacher model is an important factor for effective KD. Furthermore, we demonstrate that the performance of KD can be improved by simply employing a calibration method that reduces the teacher's calibration error. Our algorithm is versatile, demonstrating effectiveness across various tasks from classification to detection. Moreover, it can be easily integrated with existing state-of-the-art methods, consistently achieving superior performance. 

**Abstract (ZH)**: 知识蒸馏（KD）作为一种有效的模型压缩技术，在深度学习中已得到广泛应用，能够从大型教师模型转移知识到紧凑的学生模型。尽管KD已经取得了显著的成功，但尚未完全理解哪些因素能够改善学生模型的性能。在本文中，我们揭示了教师模型的校准误差与学生模型的准确率之间存在密切的相关性。因此，我们认为教师模型的校准是有效知识蒸馏的重要因素。此外，我们证明通过简单地采用一种减少教师模型校准误差的校准方法，可以提高知识蒸馏的性能。我们的算法具有通用性，能够在从分类到检测的各种任务中有效地提升性能，并且可以很容易地与现有的最先进技术集成，始终实现优越的表现。 

---
# Collaborating with GenAI: Incentives and Replacements 

**Title (ZH)**: 与生成式人工智能协作：激励与替代 

**Authors**: Boaz Taitler, Omer Ben-Porat  

**Link**: [PDF](https://arxiv.org/pdf/2508.20213)  

**Abstract**: The rise of Generative AI (GenAI) is reshaping how workers contribute to shared projects. While workers can use GenAI to boost productivity or reduce effort, managers may use it to replace some workers entirely. We present a theoretical framework to analyze how GenAI affects collaboration in such settings. In our model, the manager selects a team to work on a shared task, with GenAI substituting for unselected workers. Each worker selects how much effort to exert, and incurs a cost that increases with the level of effort. We show that GenAI can lead workers to exert no effort, even if GenAI is almost ineffective. We further show that the manager's optimization problem is NP-complete, and provide an efficient algorithm for the special class of (almost-) linear instances. Our analysis shows that even workers with low individual value may play a critical role in sustaining overall output, and excluding such workers can trigger a cascade. Finally, we conduct extensive simulations to illustrate our theoretical findings. 

**Abstract (ZH)**: 生成式AI的兴起正在重塑工人在共享项目中的贡献方式。虽然工人可以利用生成式AI提高生产力或减少努力，管理者可能利用它完全替代某些工人。我们提出了一种理论框架来分析生成式AI在这些环境中如何影响协作。在我们的模型中，管理者选择一个团队来处理一个共享任务，生成式AI替代未选中的工人。每个工人选择付出多少努力，并且付出努力的水平越高，成本也越高。我们证明生成式AI可能导致工人付出零努力，即便生成式AI几乎无效。进一步证明管理者的优化问题为NP完全问题，并提供了适用于（几乎）线性实例的高效算法。我们的分析表明，即使个体价值较低的工人也可能在维持总体产出中发挥关键作用，排除这些工人可能会引发连锁反应。最后，我们进行了广泛的模拟以说明我们的理论发现。 

---
# Filter then Attend: Improving attention-based Time Series Forecasting with Spectral Filtering 

**Title (ZH)**: 滤波后再关注：基于谱滤波的注意力时序预测改进 

**Authors**: Elisha Dayag, Nhat Thanh Van Tran, Jack Xin  

**Link**: [PDF](https://arxiv.org/pdf/2508.20206)  

**Abstract**: Transformer-based models are at the forefront in long time-series forecasting (LTSF). While in many cases, these models are able to achieve state of the art results, they suffer from a bias toward low-frequencies in the data and high computational and memory requirements. Recent work has established that learnable frequency filters can be an integral part of a deep forecasting model by enhancing the model's spectral utilization. These works choose to use a multilayer perceptron to process their filtered signals and thus do not solve the issues found with transformer-based models. In this paper, we establish that adding a filter to the beginning of transformer-based models enhances their performance in long time-series forecasting. We add learnable filters, which only add an additional $\approx 1000$ parameters to several transformer-based models and observe in multiple instances 5-10 \% relative improvement in forecasting performance. Additionally, we find that with filters added, we are able to decrease the embedding dimension of our models, resulting in transformer-based architectures that are both smaller and more effective than their non-filtering base models. We also conduct synthetic experiments to analyze how the filters enable Transformer-based models to better utilize the full spectrum for forecasting. 

**Abstract (ZH)**: 基于Transformer的模型在长时间序列预测中的前沿地位：添加滤波器提高性能研究 

---
# RelAItionship Building: Analyzing Recruitment Strategies for Participatory AI 

**Title (ZH)**: 关系构建：分析参与式AI的招聘策略 

**Authors**: Eugene Kim, Vaibhav Balloli, Berelian Karimian, Elizabeth Bondi-Kelly, Benjamin Fish  

**Link**: [PDF](https://arxiv.org/pdf/2508.20176)  

**Abstract**: Participatory AI, in which impacted community members and other stakeholders are involved in the design and development of AI systems, holds promise as a way to ensure AI is developed to meet their needs and reflect their values. However, the process of identifying, reaching out, and engaging with all relevant stakeholder groups, which we refer to as recruitment methodology, is still a practical challenge in AI projects striving to adopt participatory practices. In this paper, we investigate the challenges that researchers face when designing and executing recruitment methodology for Participatory AI projects, and the implications of current recruitment practice for Participatory AI. First, we describe the recruitment methodologies used in AI projects using a corpus of 37 projects to capture the diversity of practices in the field and perform an initial analysis on the documentation of recruitment practices, as well as specific strategies that researchers use to meet goals of equity and empowerment. To complement this analysis, we interview five AI researchers to learn about the outcomes of recruitment methodologies. We find that these outcomes are shaped by structural conditions of their work, researchers' own goals and expectations, and the relationships built from the recruitment methodology and subsequent collaboration. Based on these analyses, we provide recommendations for designing and executing relationship-forward recruitment methods, as well as reflexive recruitment documentation practices for Participatory AI researchers. 

**Abstract (ZH)**: 参与式AI中的招募能力研究：一种确保AI系统符合用户需求并反映其价值观的方法，但如何识别、联系并有效参与所有相关利益相关者群体仍是一个实践挑战。本文探讨了研究人员在设计和执行参与式AI项目招募能力方法时遇到的挑战及其现有的招募能力实践对参与式AI的含义，并提供了招募能力方法和反思性招募能力文档实践的建议。 

---
# Navigating the EU AI Act: Foreseeable Challenges in Qualifying Deep Learning-Based Automated Inspections of Class III Medical Devices 

**Title (ZH)**: 欧盟AI法案中基于深度学习的III类医疗器械自动检查的可预见挑战导航 

**Authors**: Julio Zanon Diaz, Tommy Brennan, Peter Corcoran  

**Link**: [PDF](https://arxiv.org/pdf/2508.20144)  

**Abstract**: As deep learning (DL) technologies advance, their application in automated visual inspection for Class III medical devices offers significant potential to enhance quality assurance and reduce human error. However, the adoption of such AI-based systems introduces new regulatory complexities--particularly under the EU Artificial Intelligence (AI) Act, which imposes high-risk system obligations that differ in scope and depth from established regulatory frameworks such as the Medical Device Regulation (MDR) and the U.S. FDA Quality System Regulation (QSR). This paper presents a high-level technical assessment of the foresee-able challenges that manufacturers are likely to encounter when qualifying DL-based automated inspections within the existing medical device compliance landscape. It examines divergences in risk management principles, dataset governance, model validation, explainability requirements, and post-deployment monitoring obligations. The discussion also explores potential implementation strategies and highlights areas of uncertainty, including data retention burdens, global compliance implications, and the practical difficulties of achieving statistical significance in validation with limited defect data. Disclaimer: This publication is in-tended solely as an academic and technical evaluation. It is not a substitute for le-gal advice or official regulatory interpretation. The information presented here should not be relied upon to demonstrate compliance with the EU AI Act or any other statutory obligation. Manufacturers are encouraged to consult appropriate regulatory authorities and legal experts to determine specific compliance pathways. 

**Abstract (ZH)**: 深学习技术在III类医疗设备自动化视觉检测中的应用：现有合规landscape下的挑战与监管复杂性分析 

---
# UltraEar: a multicentric, large-scale database combining ultra-high-resolution computed tomography and clinical data for ear diseases 

**Title (ZH)**: 耳部超高清计算机断层扫描及临床数据多中心大規模数据库：UltraEar 

**Authors**: Ruowei Tang, Pengfei Zhao, Xiaoguang Li, Ning Xu, Yue Cheng, Mengshi Zhang, Zhixiang Wang, Zhengyu Zhang, Hongxia Yin, Heyu Ding, Shusheng Gong, Yuhe Liu, Zhenchang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20141)  

**Abstract**: Ear diseases affect billions of people worldwide, leading to substantial health and socioeconomic burdens. Computed tomography (CT) plays a pivotal role in accurate diagnosis, treatment planning, and outcome evaluation. The objective of this study is to present the establishment and design of UltraEar Database, a large-scale, multicentric repository of isotropic 0.1 mm ultra-high-resolution CT (U-HRCT) images and associated clinical data dedicated to ear diseases. UltraEar recruits patients from 11 tertiary hospitals between October 2020 and October 2035, integrating U-HRCT images, structured CT reports, and comprehensive clinical information, including demographics, audiometric profiles, surgical records, and pathological findings. A broad spectrum of otologic disorders is covered, such as otitis media, cholesteatoma, ossicular chain malformation, temporal bone fracture, inner ear malformation, cochlear aperture stenosis, enlarged vestibular aqueduct, and sigmoid sinus bony deficiency. Standardized preprocessing pipelines have been developed for geometric calibration, image annotation, and multi-structure segmentation. All personal identifiers in DICOM headers and metadata are removed or anonymized to ensure compliance with data privacy regulation. Data collection and curation are coordinated through monthly expert panel meetings, with secure storage on an offline cloud system. UltraEar provides an unprecedented ultra-high-resolution reference atlas with both technical fidelity and clinical relevance. This resource has significant potential to advance radiological research, enable development and validation of AI algorithms, serve as an educational tool for training in otologic imaging, and support multi-institutional collaborative studies. UltraEar will be continuously updated and expanded, ensuring long-term accessibility and usability for the global otologic research community. 

**Abstract (ZH)**: 耳部疾病影响全球数十亿人，导致严重的健康和社会经济负担。计算机断层扫描（CT）在准确诊断、治疗规划和预后评估中发挥着关键作用。本研究的目的是介绍并设计UltraEar数据库，这是一个大规模、多中心的耳部疾病超高清CT（U-HRCT）图像及其相关临床数据的存储库。UltraEar从2020年10月至2035年10月招募来自11家三级医院的患者，整合超高清CT图像、结构化CT报告以及包括人口统计学、听力图谱、手术记录和病理学发现在内的综合临床信息。涵盖了广泛的耳科疾病，如中耳炎、胆固醇瘤、听小骨链畸形、颞骨骨折、内耳畸形、耳蜗孔狭窄、前庭水管扩张和水平半恒骨头缺乏。已开发了标准化的预处理管道，用于几何校准、图像注释和多结构分割。所有DICOM头和元数据中的个人标识符已被删除或匿名化，以确保符合数据隐私法规。数据收集和管理通过每月专家小组会议协调，并在离线云系统中安全存储。UltraEar提供了一个具有技术和临床相关性的前所未有的超高清参考图谱。该资源具有显著潜力，可促进放射学研究、开发和验证人工智能算法，作为耳科成像培训的教育工具，并支持多机构协作研究。UltraEar将继续更新和扩展，确保全球耳科研究社区的长期可访问性和可用性。 

---
# Artificial Intelligence for CRISPR Guide RNA Design: Explainable Models and Off-Target Safety 

**Title (ZH)**: 人工智能在CRISPR导向RNA设计中的应用：可解释模型与脱靶安全性 

**Authors**: Alireza Abbaszadeh, Armita Shahlai  

**Link**: [PDF](https://arxiv.org/pdf/2508.20130)  

**Abstract**: CRISPR-based genome editing has revolutionized biotechnology, yet optimizing guide RNA (gRNA) design for efficiency and safety remains a critical challenge. Recent advances (2020--2025, updated to reflect current year if needed) demonstrate that artificial intelligence (AI), especially deep learning, can markedly improve the prediction of gRNA on-target activity and identify off-target risks. In parallel, emerging explainable AI (XAI) techniques are beginning to illuminate the black-box nature of these models, offering insights into sequence features and genomic contexts that drive Cas enzyme performance. Here we review how state-of-the-art machine learning models are enhancing gRNA design for CRISPR systems, highlight strategies for interpreting model predictions, and discuss new developments in off-target prediction and safety assessment. We emphasize breakthroughs from top-tier journals that underscore an interdisciplinary convergence of AI and genome editing to enable more efficient, specific, and clinically viable CRISPR applications. 

**Abstract (ZH)**: 基于CRISPR的基因编辑技术 telah革命性地改变了生物技术领域，然而，如何优化导向RNA（gRNA）的设计以提高效率和安全性仍是一项关键挑战。近年来（2020-2025年，根据需要更新至当前年份），研究表明，特别是深度学习等人工智能（AI）技术能够显著改善gRNA靶向活性的预测，并识别潜在的脱靶风险。与此同时，新兴的可解释AI（XAI）技术开始揭示这些模型的黑箱性质，提供关于驱动Cas酶性能的序列特征和基因组背景的见解。本文综述了最先进的机器学习模型如何增强CRISPR系统中的gRNA设计，强调了解析模型预测策略，并讨论了脱靶预测和安全性评估的新进展。我们强调了顶级期刊上的突破，突显了人工智能与基因编辑的跨学科融合，以实现更高效、更特异且临床可行的CRISPR应用。 

---
# Improving Liver Disease Diagnosis with SNNDeep: A Custom Spiking Neural Network Using Diverse Learning Algorithms 

**Title (ZH)**: 基于 diverse 学习算法的定制化脉冲神经网络 SNNDeep 以提高肝脏疾病诊断准确性 

**Authors**: Zofia Rudnicka, Janusz Szczepanski, Agnieszka Pregowska  

**Link**: [PDF](https://arxiv.org/pdf/2508.20125)  

**Abstract**: Purpose: Spiking neural networks (SNNs) have recently gained attention as energy-efficient, biologically plausible alternatives to conventional deep learning models. Their application in high-stakes biomedical imaging remains almost entirely unexplored. Methods: This study introduces SNNDeep, the first tailored SNN specifically optimized for binary classification of liver health status from computed tomography (CT) features. To ensure clinical relevance and broad generalizability, the model was developed and evaluated using the Task03\Liver dataset from the Medical Segmentation Decathlon (MSD), a standardized benchmark widely used for assessing performance across diverse medical imaging tasks. We benchmark three fundamentally different learning algorithms, namely Surrogate Gradient Learning, the Tempotron rule, and Bio-Inspired Active Learning across three architectural variants: a fully customized low-level model built from scratch, and two implementations using leading SNN frameworks, i.e., snnTorch and SpikingJelly. Hyperparameter optimization was performed using Optuna. Results: Our results demonstrate that the custom-built SNNDeep consistently outperforms framework-based implementations, achieving a maximum validation accuracy of 98.35%, superior adaptability across learning rules, and significantly reduced training overhead. Conclusion:This study provides the first empirical evidence that low-level, highly tunable SNNs can surpass standard frameworks in medical imaging, especially in data-limited, temporally constrained diagnostic settings, thereby opening a new pathway for neuro-inspired AI in precision medicine. 

**Abstract (ZH)**: 目的：脉冲神经网络（SNNs）最近因其在能量效率和生物可塑性方面的优势，被视为传统深度学习模型的有潜力替代方案。它们在高风险生物医学成像中的应用仍然几乎未被探索。方法：本文介绍了SNNDeep，这是第一个专门为基于计算机断层扫描（CT）特征进行肝健康状态二分类设计的定制化SNN，并且优化了这种SNN。为了确保临床相关性及广泛的普适性，该模型基于Medical Segmentation Decathlon（MSD）的任务03\肝脏数据集开发和评估，MSD是一个广泛用于评估多样医学成像任务性能的标准化基准。三种基本不同的学习算法——替代梯度学习、Tempotron规则和生物启发式主动学习——分别在三种架构变体上进行了测试：完全从零开始建立的低级模型，以及使用领先的SNN框架snnTorch和SpikingJelly实现的两种版本。超参数优化使用了Optuna。结果：结果表明，自定义构建的SNNDeep在验证准确性上持续超过基于框架的实现，达到了98.35%的最大验证准确性，并且具有更强的学习规则适应性和显著降低的训练开销。结论：本研究提供了低级、高度可调SNN在医学成像中可以超越标准框架的首个实证证据，特别是在数据有限、时间受限的诊断环境中，从而为精确诊断中的神经启发式AI开辟了一条新途径。 

---
# Towards Better Correctness and Efficiency in Code Generation 

**Title (ZH)**: 向更好的正确性和效率迈进：代码生成的角度 

**Authors**: Yunlong Feng, Yang Xu, Xiao Xu, Binyuan Hui, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.20124)  

**Abstract**: While code large language models have demonstrated remarkable progress in code generation, the generated code often exhibits poor runtime efficiency, limiting its practical application in performance-sensitive scenarios. To address this limitation, we propose an efficiency-oriented reinforcement learning framework guided by a novel performance reward. Based on this framework, we take a deeper dive into the code efficiency problem, identifying then proposing methods to overcome key bottlenecks: (1) Dynamic exploration overcomes the static data constraints of offline fine-tuning, enabling the discovery of more efficient code implementations. (2) The error-insensitive reinforcement learning method and high-contrast efficiency signals are crucial for mitigating systematic errors and achieving effective optimization. (3) Online exploration is most effective when starting from a high-correctness baseline, as this allows for efficiency improvements without sacrificing accuracy. With these discoveries, we finally propose a two-stage tuning method, which achieves high and balanced performance across correctness and efficiency. The results of experiments show the effectiveness of the method, which improves code correctness by 10.18\% and runtime efficiency by 7.75\% on a 7B model, achieving performance comparable to much larger model. 

**Abstract (ZH)**: 尽管代码大型语言模型在代码生成方面取得了显著进展，生成的代码通常运行时效率较差，限制了其在性能敏感场景中的实际应用。为解决这一局限性，我们提出了一种以效率为导向的强化学习框架，并由一种新颖的性能奖励指导。基于此框架，我们深入探讨了代码效率问题，识别并提出了解决关键瓶颈的方法：（1）动态探索克服了离线微调的静态数据限制， enables the discovery of more efficient code implementations.（2）对错误不敏感的强化学习方法和高对比度的效率信号对于减轻系统误差并实现有效的优化至关重要。（3）从高正确性基线开始的在线探索最有效，这允许在不牺牲准确性的前提下提高效率。通过这些发现，我们最终提出了一种两阶段调优方法，该方法在正确性和效率上都实现了高效且平衡的性能。实验结果表明，该方法的有效性，相较7B模型，代码正确性提高了10.18\%，运行时效率提高了7.75\%，并将性能提升至接近更大模型的水平。 

---
# Particle swarm optimization for online sparse streaming feature selection under uncertainty 

**Title (ZH)**: 基于不确定性的在线稀疏流特征选择的粒子群优化方法 

**Authors**: Ruiyang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20123)  

**Abstract**: In real-world applications involving high-dimensional streaming data, online streaming feature selection (OSFS) is widely adopted. Yet, practical deployments frequently face data incompleteness due to sensor failures or technical constraints. While online sparse streaming feature selection (OS2FS) mitigates this issue via latent factor analysis-based imputation, existing methods struggle with uncertain feature-label correlations, leading to inflexible models and degraded performance. To address these gaps, this work proposes POS2FS-an uncertainty-aware online sparse streaming feature selection framework enhanced by particle swarm optimization (PSO). The approach introduces: 1) PSO-driven supervision to reduce uncertainty in feature-label relationships; 2) Three-way decision theory to manage feature fuzziness in supervised learning. Rigorous testing on six real-world datasets confirms POS2FS outperforms conventional OSFS and OS2FS techniques, delivering higher accuracy through more robust feature subset selection. 

**Abstract (ZH)**: 面向高维流数据的不确定性感知在线稀疏流特征选择（POS2FS）框架 

---
# Is Artificial Intelligence Reshaping the Landscape of the International Academic Community of Geosciences? 

**Title (ZH)**: 人工智能正在重塑地球科学国际学术社区的格局吗？ 

**Authors**: Liang Li, Yuntian Li, Wenxin Zhao, Shan Ye, Yun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20117)  

**Abstract**: Through bibliometric analysis and topic modeling, we find that artificial intelligence (AI) is positively transforming geosciences research, with a notable increase in AI-related scientific output in recent years. We are encouraged to observe that earth scientists from developing countries have gained better visibility in the recent AI for Science (AI4S) paradigm and that AI is also improving the landscape of international collaboration in geoscience-related research. 

**Abstract (ZH)**: 通过文献计量分析和主题建模，我们发现人工智能（AI）正积极地变革地质科学研究，近年来与AI相关的科学产出显著增加。我们注意到，来自发展中国家的地球科学家在AI for Science (AI4S)范式中获得了更好的能见度，同时人工智能也改善了地质科学研究领域的国际合作格局。 

---
# A Hierarchical Signal Coordination and Control System Using a Hybrid Model-based and Reinforcement Learning Approach 

**Title (ZH)**: 基于混合模型与强化学习方法的分层信号协调与控制系统 

**Authors**: Xianyue Peng, Shenyang Chen, H. Michael Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20102)  

**Abstract**: Signal control in urban corridors faces the dual challenge of maintaining arterial traffic progression while adapting to demand variations at local intersections. We propose a hierarchical traffic signal coordination and control scheme that integrates model-based optimization with reinforcement learning. The system consists of: (i) a High-Level Coordinator (HLC) that selects coordination strategies based on observed and predicted demand; (ii) a Corridor Coordinator that derives phase constraints from the selected strategy-either Max-Flow Coordination (MFC) or Green-Wave Coordination (GWC); and (iii) Hybrid Signal Agents (HSAs) that determine signal phases via reinforcement learning with action masking to enforce feasibility. Hierarchical reinforcement learning with Proximal Policy Optimization (PPO) is used to train HSA and HLC policies. At the lower level, three HSA policies-MFC-aware, GWC-aware, and pure agent control (PAC) are trained in conjunction with their respective coordination strategies. At the higher level, the HLC is trained to dynamically switch strategies using a multi-objective reward balancing corridor-level and network-wide performance. The proposed scheme was developed and evaluated on a SUMO-RLlib platform. Case results show that hybrid MFC maximizes throughput under heavy demand; hybrid GWC consistently minimizes arterial stops and maintains progression across diverse traffic conditions but can reduce network-wide efficiency; and PAC improves network-wide travel time in moderate demand but is less effective under heavy demand. The hierarchical design enables adaptive strategy selection, achieving robust performance across all demand levels. 

**Abstract (ZH)**: 基于模型优化与强化学习的分级交通信号协调与控制方案 

---
