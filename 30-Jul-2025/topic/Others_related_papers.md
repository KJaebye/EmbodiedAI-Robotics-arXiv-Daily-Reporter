# ODE Methods for Computing One-Dimensional Self-Motion Manifolds 

**Title (ZH)**: ODE方法计算一维自我运动流形 

**Authors**: Dominic Guri, George Kantor  

**Link**: [PDF](https://arxiv.org/pdf/2507.21957)  

**Abstract**: Redundant manipulators are well understood to offer infinite joint configurations for achieving a desired end-effector pose. The multiplicity of inverse kinematics (IK) solutions allows for the simultaneous solving of auxiliary tasks like avoiding joint limits or obstacles. However, the most widely used IK solvers are numerical gradient-based iterative methods that inherently return a locally optimal solution. In this work, we explore the computation of self-motion manifolds (SMMs), which represent the set of all joint configurations that solve the inverse kinematics problem for redundant manipulators. Thus, SMMs are global IK solutions for redundant manipulators. We focus on task redundancies of dimensionality 1, introducing a novel ODE formulation for computing SMMs using standard explicit fixed-step ODE integrators. We also address the challenge of ``inducing'' redundancy in otherwise non-redundant manipulators assigned to tasks naturally described by one degree of freedom less than the non-redundant manipulator. Furthermore, recognizing that SMMs can consist of multiple disconnected components, we propose methods for searching for these separate SMM components. Our formulations and algorithms compute accurate SMM solutions without requiring additional IK refinement, and we extend our methods to prismatic joint systems -- an area not covered in current SMM literature. This manuscript presents the derivation of these methods and several examples that show how the methods work and their limitations. 

**Abstract (ZH)**: 冗余 manipulator 的自运动流形 (SMM) 为实现期望末端执行器姿态提供了无穷多的关节配置。逆运动学 (IK) 解的多样性允许同时解决辅助任务，如避免关节限制或障碍。然而，广泛使用的 IK 求解器本质上是基于数值梯度的迭代方法，只能提供局部最优解。在本文中，我们探讨了自运动流形 (SMMs) 的计算，这代表了所有解决冗余 manipulator 逆运动学问题的关节配置集合。因此，SMMs 是冗余 manipulator 的全局 IK 解。我们专注于维数为 1 的任务冗余性，引入了一种新的 ODE 表述方法，使用标准明确固定步长 ODE 积分器来计算 SMMs。我们还解决了在任务自然描述上少一个自由度的非冗余 manipulator 中引入“冗余性”的挑战。此外，鉴于 SMM 可能由多个不连通的部分组成，我们提出了搜索这些独立 SMM 部分的方法。我们的表述和算法可以在无需额外 IK 精细化的情况下计算准确的 SMM 解，并将方法扩展到柱形关节系统——这是当前 SMM 文献中未涵盖的领域。本文阐述了这些方法的推导以及一些示例，展示了方法的工作原理及其限制。 

---
# Adaptive Prior Scene-Object SLAM for Dynamic Environments 

**Title (ZH)**: 自适应先验场景-对象SLAM技术在动态环境中的应用 

**Authors**: Haolan Zhang, Thanh Nguyen Canh, Chenghao Li, Nak Young Chong  

**Link**: [PDF](https://arxiv.org/pdf/2507.21709)  

**Abstract**: Visual Simultaneous Localization and Mapping (SLAM) plays a vital role in real-time localization for autonomous systems. However, traditional SLAM methods, which assume a static environment, often suffer from significant localization drift in dynamic scenarios. While recent advancements have improved SLAM performance in such environments, these systems still struggle with localization drift, particularly due to abrupt viewpoint changes and poorly characterized moving objects. In this paper, we propose a novel scene-object-based reliability assessment framework that comprehensively evaluates SLAM stability through both current frame quality metrics and scene changes relative to reliable reference frames. Furthermore, to tackle the lack of error correction mechanisms in existing systems when pose estimation becomes unreliable, we employ a pose refinement strategy that leverages information from reliable frames to optimize camera pose estimation, effectively mitigating the adverse effects of dynamic interference. Extensive experiments on the TUM RGB-D datasets demonstrate that our approach achieves substantial improvements in localization accuracy and system robustness under challenging dynamic scenarios. 

**Abstract (ZH)**: 基于场景-对象的可靠性评估框架在动态场景中提升SLAM定位精度和系统鲁棒性 

---
# Model Predictive Adversarial Imitation Learning for Planning from Observation 

**Title (ZH)**: 基于观察的模型预测对抗模仿学习规划 

**Authors**: Tyler Han, Yanda Bao, Bhaumik Mehta, Gabriel Guo, Anubhav Vishwakarma, Emily Kang, Sanghun Jung, Rosario Scalise, Jason Zhou, Bryan Xu, Byron Boots  

**Link**: [PDF](https://arxiv.org/pdf/2507.21533)  

**Abstract**: Human demonstration data is often ambiguous and incomplete, motivating imitation learning approaches that also exhibit reliable planning behavior. A common paradigm to perform planning-from-demonstration involves learning a reward function via Inverse Reinforcement Learning (IRL) then deploying this reward via Model Predictive Control (MPC). Towards unifying these methods, we derive a replacement of the policy in IRL with a planning-based agent. With connections to Adversarial Imitation Learning, this formulation enables end-to-end interactive learning of planners from observation-only demonstrations. In addition to benefits in interpretability, complexity, and safety, we study and observe significant improvements on sample efficiency, out-of-distribution generalization, and robustness. The study includes evaluations in both simulated control benchmarks and real-world navigation experiments using few-to-single observation-only demonstrations. 

**Abstract (ZH)**: 人类示范数据常常模糊且不完整，促使采用也表现出可靠规划行为的模仿学习方法。进行从示范中规划的方法通常涉及通过逆强化学习（IRL）学习奖励函数，然后通过模型预测控制（MPC）部署该奖励。为了统一这些方法，我们推导出IRL中策略的一种基于规划的替代方案。借助对抗模仿学习的联系，这种形式化方法能够通过仅凭观察进行端到端的交互式学习规划器。除了提高可解释性、复杂性和安全性之外，我们研究并观察到在样本效率、域外泛化能力和鲁棒性方面有显著改进。研究包括在模拟控制基准和使用少量至单个仅凭观察的演示进行真实世界导航实验的评估。 

---
# Retrieve-Augmented Generation for Speeding up Diffusion Policy without Additional Training 

**Title (ZH)**: 无需额外训练的检索增强生成方法以加快扩散策略的速度 

**Authors**: Sodtavilan Odonchimed, Tatsuya Matsushima, Simon Holk, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2507.21452)  

**Abstract**: Diffusion Policies (DPs) have attracted attention for their ability to achieve significant accuracy improvements in various imitation learning tasks. However, DPs depend on Diffusion Models, which require multiple noise removal steps to generate a single action, resulting in long generation times. To solve this problem, knowledge distillation-based methods such as Consistency Policy (CP) have been proposed. However, these methods require a significant amount of training time, especially for difficult tasks. In this study, we propose RAGDP (Retrieve-Augmented Generation for Diffusion Policies) as a novel framework that eliminates the need for additional training using a knowledge base to expedite the inference of pre-trained DPs. In concrete, RAGDP encodes observation-action pairs through the DP encoder to construct a vector database of expert demonstrations. During inference, the current observation is embedded, and the most similar expert action is extracted. This extracted action is combined with an intermediate noise removal step to reduce the number of steps required compared to the original diffusion step. We show that by using RAGDP with the base model and existing acceleration methods, we improve the accuracy and speed trade-off with no additional training. Even when accelerating the models 20 times, RAGDP maintains an advantage in accuracy, with a 7% increase over distillation models such as CP. 

**Abstract (ZH)**: 基于检索增强生成的扩散策略（RAGDP）：消除额外训练以加快预训练策略的推理速度 

---
# The Interspeech 2025 Speech Accessibility Project Challenge 

**Title (ZH)**: Interspeech 2025 语音Accessibility项目挑战 

**Authors**: Xiuwen Zheng, Bornali Phukon, Jonghwan Na, Ed Cutrell, Kyu Han, Mark Hasegawa-Johnson, Pan-Pan Jiang, Aadhrik Kuila, Colin Lea, Bob MacDonald, Gautam Mantena, Venkatesh Ravichandran, Leda Sari, Katrin Tomanek, Chang D. Yoo, Chris Zwilling  

**Link**: [PDF](https://arxiv.org/pdf/2507.22047)  

**Abstract**: While the last decade has witnessed significant advancements in Automatic Speech Recognition (ASR) systems, performance of these systems for individuals with speech disabilities remains inadequate, partly due to limited public training data. To bridge this gap, the 2025 Interspeech Speech Accessibility Project (SAP) Challenge was launched, utilizing over 400 hours of SAP data collected and transcribed from more than 500 individuals with diverse speech disabilities. Hosted on EvalAI and leveraging the remote evaluation pipeline, the SAP Challenge evaluates submissions based on Word Error Rate and Semantic Score. Consequently, 12 out of 22 valid teams outperformed the whisper-large-v2 baseline in terms of WER, while 17 teams surpassed the baseline on SemScore. Notably, the top team achieved the lowest WER of 8.11\%, and the highest SemScore of 88.44\% at the same time, setting new benchmarks for future ASR systems in recognizing impaired speech. 

**Abstract (ZH)**: 自动语音识别系统在言语障碍个体中的性能仍不足：2025年Interspeech言语 accessibility项目挑战赛进展 

---
# UI-AGILE: Advancing GUI Agents with Effective Reinforcement Learning and Precise Inference-Time Grounding 

**Title (ZH)**: UI-AGILE：基于有效的强化学习和精确的推理时Grounding提升GUI代理 

**Authors**: Shuquan Lian, Yuhang Wu, Jia Ma, Zihan Song, Bingqi Chen, Xiawu Zheng, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.22025)  

**Abstract**: The emergence of Multimodal Large Language Models (MLLMs) has driven significant advances in Graphical User Interface (GUI) agent capabilities. Nevertheless, existing GUI agent training and inference techniques still suffer from a dilemma for reasoning designs, ineffective reward, and visual noise. To address these issues, we introduce UI-AGILE, a comprehensive framework enhancing GUI agents at both the training and inference stages. For training, we propose a suite of improvements to the Supervised Fine-Tuning (SFT) process: 1) a Continuous Reward function to incentivize high-precision grounding; 2) a "Simple Thinking" reward to balance planning with speed and grounding accuracy; and 3) a Cropping-based Resampling strategy to mitigate the sparse reward problem and improve learning on complex tasks. For inference, we present Decomposed Grounding with Selection, a novel method that dramatically improves grounding accuracy on high-resolution displays by breaking the image into smaller, manageable parts. Experiments show that UI-AGILE achieves the state-of-the-art performance on two benchmarks ScreenSpot-Pro and ScreenSpot-v2. For instance, using both our proposed training and inference enhancement methods brings 23% grounding accuracy improvement over the best baseline on ScreenSpot-Pro. 

**Abstract (ZH)**: 多模态大型语言模型的涌现推动了图形用户界面代理能力的重要进步。尽管如此，现有的图形用户界面代理训练和推理技术仍面临推理设计难题、无效奖励和视觉噪声的困境。为解决这些问题，我们引入了UI-AGILE，这是一个综合框架，旨在提高图形用户界面代理在训练和推理阶段的能力。在训练阶段，我们提出了一系列对监督微调（SFT）过程的改进：1）连续奖励函数，以激励高精度定位；2）“简单思考”奖励，以平衡规划速度与定位准确性；3）基于裁剪的重采样策略，以缓解稀疏奖励问题并提高复杂任务的学习效果。在推理阶段，我们提出了一种分解定位加选择的新方法，通过将图像分解成更小、更易管理的部分，显著提高了高分辨率显示器上的定位准确性。实验结果显示，UI-AGILE在ScreenSpot-Pro和ScreenSpot-v2两个基准测试中达到了最先进的性能。例如，使用我们提出的所有训练和推理增强方法，在ScreenSpot-Pro上实现了23%的定位准确性改进。 

---
# PHAX: A Structured Argumentation Framework for User-Centered Explainable AI in Public Health and Biomedical Sciences 

**Title (ZH)**: PHAX：面向用户的公共卫生和生物医学科学解释型人工智能的结构化论辩框架 

**Authors**: Bahar İlgen, Akshat Dubey, Georges Hattab  

**Link**: [PDF](https://arxiv.org/pdf/2507.22009)  

**Abstract**: Ensuring transparency and trust in AI-driven public health and biomedical sciences systems requires more than accurate predictions-it demands explanations that are clear, contextual, and socially accountable. While explainable AI (XAI) has advanced in areas like feature attribution and model interpretability, most methods still lack the structure and adaptability needed for diverse health stakeholders, including clinicians, policymakers, and the general public. We introduce PHAX-a Public Health Argumentation and eXplainability framework-that leverages structured argumentation to generate human-centered explanations for AI outputs. PHAX is a multi-layer architecture combining defeasible reasoning, adaptive natural language techniques, and user modeling to produce context-aware, audience-specific justifications. More specifically, we show how argumentation enhances explainability by supporting AI-driven decision-making, justifying recommendations, and enabling interactive dialogues across user types. We demonstrate the applicability of PHAX through use cases such as medical term simplification, patient-clinician communication, and policy justification. In particular, we show how simplification decisions can be modeled as argument chains and personalized based on user expertise-enhancing both interpretability and trust. By aligning formal reasoning methods with communicative demands, PHAX contributes to a broader vision of transparent, human-centered AI in public health. 

**Abstract (ZH)**: 确保人工智能驱动的公共卫生和生物医学科学系统中的透明度和信任不仅需要准确的预测，还需要清晰、语境相关且社会责任导向的解释。尽管可解释人工智能（XAI）在特征归因和模型可解释性方面取得了进展，但大多数方法仍缺乏为包括临床医生、政策制定者和普通公众在内的多元化健康利益相关者提供结构化和适应性的能力。我们引入了PHAX——公共卫生论辩与可解释性框架，利用结构化论辩生成以人为中心的AI输出解释。PHAX是一种多层架构，结合了可败论推理、自适应自然语言技术和用户建模，以产生上下文相关的、针对特定受众的理由。具体而言，我们展示了论辩如何通过支持基于AI的决策、对建议进行辩护以及在不同类型用户之间启用互动对话，来增强解释性。我们通过医疗术语简化、患者-临床医生沟通和政策辩护等用例，展示了PHAX的应用性。特别是，我们展示了如何将简化决策建模为论证链，并基于用户专长进行个性化，从而同时增强解释性和信任。通过将形式推理方法与沟通需求相结合，PHAX为公共卫生领域的透明、以人为中心的人工智能贡献了更广泛的观点。 

---
# Efficient Pain Recognition via Respiration Signals: A Single Cross-Attention Transformer Multi-Window Fusion Pipeline 

**Title (ZH)**: 基于呼吸信号的高效疼痛识别：一种单跨注意力变换器多窗融合管道 

**Authors**: Stefanos Gkikas, Ioannis Kyprakis, Manolis Tsiknakis  

**Link**: [PDF](https://arxiv.org/pdf/2507.21886)  

**Abstract**: Pain is a complex condition affecting a large portion of the population. Accurate and consistent evaluation is essential for individuals experiencing pain, and it supports the development of effective and advanced management strategies. Automatic pain assessment systems provide continuous monitoring and support clinical decision-making, aiming to reduce distress and prevent functional decline. This study has been submitted to the \textit{Second Multimodal Sensing Grand Challenge for Next-Gen Pain Assessment (AI4PAIN)}. The proposed method introduces a pipeline that leverages respiration as the input signal and incorporates a highly efficient cross-attention transformer alongside a multi-windowing strategy. Extensive experiments demonstrate that respiration is a valuable physiological modality for pain assessment. Moreover, experiments revealed that compact and efficient models, when properly optimized, can achieve strong performance, often surpassing larger counterparts. The proposed multi-window approach effectively captures both short-term and long-term features, as well as global characteristics, thereby enhancing the model's representational capacity. 

**Abstract (ZH)**: 疼痛是一种影响大量人群的复杂状况。准确且一致的评估对经历疼痛的个体至关重要，有助于开发有效的管理和治疗策略。自动疼痛评估系统提供连续监测并支持临床决策，旨在减轻痛苦并防止功能衰退。本研究已提交至“第二代多模态传感 Grand 挑战赛——下一代疼痛评估（AI4PAIN）”。所提出的方法利用呼吸作为输入信号，并结合了高效率的交叉注意力变换器和多窗口策略。广泛实验表明，呼吸是一种有价值的生理模态用于疼痛评估。此外，实验显示，当适当优化时，紧凑且高效的模型可以达到很强的性能，往往超过较大的模型。所提出的方法窗口策略有效地捕捉了短期、长期特征以及全局特征，从而增强了模型的表征能力。 

---
# The Impact of Foundational Models on Patient-Centric e-Health Systems 

**Title (ZH)**: 基础模型对以患者为中心的电子健康系统的影响 

**Authors**: Elmira Onagh, Alireza Davoodi, Maleknaz Nayebi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21882)  

**Abstract**: As Artificial Intelligence (AI) becomes increasingly embedded in healthcare technologies, understanding the maturity of AI in patient-centric applications is critical for evaluating its trustworthiness, transparency, and real-world impact. In this study, we investigate the integration and maturity of AI feature integration in 116 patient-centric healthcare applications. Using Large Language Models (LLMs), we extracted key functional features, which are then categorized into different stages of the Gartner AI maturity model. Our results show that over 86.21\% of applications remain at the early stages of AI integration, while only 13.79% demonstrate advanced AI integration. 

**Abstract (ZH)**: 随着人工智能（AI）在医疗技术中越来越普及，了解AI在以患者为中心的应用中的成熟度对于评估其可信度、透明度和实际影响至关重要。本研究调查了116个以患者为中心的医疗应用中AI功能集成的整合与成熟度。通过大型语言模型（LLMs），我们提取了关键功能特征，并将其分类为盖特纳人工智能成熟度模型的不同阶段。研究结果显示，超过86.21%的应用仍处于AI集成的早期阶段，仅13.79%的应用展现出先进的AI集成水平。 

---
# Tiny-BioMoE: a Lightweight Embedding Model for Biosignal Analysis 

**Title (ZH)**: Tiny-BioMoE：一种轻量级生物信号分析嵌入模型 

**Authors**: Stefanos Gkikas, Ioannis Kyprakis, Manolis Tsiknakis  

**Link**: [PDF](https://arxiv.org/pdf/2507.21875)  

**Abstract**: Pain is a complex and pervasive condition that affects a significant portion of the population. Accurate and consistent assessment is essential for individuals suffering from pain, as well as for developing effective management strategies in a healthcare system. Automatic pain assessment systems enable continuous monitoring, support clinical decision-making, and help minimize patient distress while mitigating the risk of functional deterioration. Leveraging physiological signals offers objective and precise insights into a person's state, and their integration in a multimodal framework can further enhance system performance. This study has been submitted to the \textit{Second Multimodal Sensing Grand Challenge for Next-Gen Pain Assessment (AI4PAIN)}. The proposed approach introduces \textit{Tiny-BioMoE}, a lightweight pretrained embedding model for biosignal analysis. Trained on $4.4$ million biosignal image representations and consisting of only $7.3$ million parameters, it serves as an effective tool for extracting high-quality embeddings for downstream tasks. Extensive experiments involving electrodermal activity, blood volume pulse, respiratory signals, peripheral oxygen saturation, and their combinations highlight the model's effectiveness across diverse modalities in automatic pain recognition tasks. \textit{\textcolor{blue}{The model's architecture (code) and weights are available at this https URL. 

**Abstract (ZH)**: 疼痛是一种复杂且普遍存在的状况，影响着大量人群。准确且一致的评估对于疼痛患者至关重要，也是在医疗保健系统中制定有效管理策略的基础。自动疼痛评估系统能够实现连续监测，支持临床决策，有助于减轻患者痛苦，同时减小功能衰退的风险。利用生理信号可以提供客观和精确的个体状态洞察，其在多模态框架中的集成可以进一步提升系统性能。本研究已提交至“下一代疼痛评估的第二轮多模态传感大赛（AI4PAIN）”。提出的办法介绍了一种轻量级预训练嵌入模型Tiny-BioMoE。该模型基于440万生理信号图像表示训练，仅包含730万个参数，适合作为提取高质量嵌入用于下游任务的有效工具。广泛涉及电导活性、血容积脉搏、呼吸信号、周围氧饱和度及其组合的实验强调了该模型在多种自动疼痛识别任务中的有效性。模型的架构（代码）和权重可从此链接获取。 

---
# A Neuro-Symbolic Approach for Probabilistic Reasoning on Graph Data 

**Title (ZH)**: 基于神经符号方法的图数据概率推理 

**Authors**: Raffaele Pojer, Andrea Passerini, Kim G. Larsen, Manfred Jaeger  

**Link**: [PDF](https://arxiv.org/pdf/2507.21873)  

**Abstract**: Graph neural networks (GNNs) excel at predictive tasks on graph-structured data but often lack the ability to incorporate symbolic domain knowledge and perform general reasoning. Relational Bayesian Networks (RBNs), in contrast, enable fully generative probabilistic modeling over graph-like structures and support rich symbolic knowledge and probabilistic inference. This paper presents a neuro-symbolic framework that seamlessly integrates GNNs into RBNs, combining the learning strength of GNNs with the flexible reasoning capabilities of RBNs.
We develop two implementations of this integration: one compiles GNNs directly into the native RBN language, while the other maintains the GNN as an external component. Both approaches preserve the semantics and computational properties of GNNs while fully aligning with the RBN modeling paradigm. We also propose a maximum a-posteriori (MAP) inference method for these neuro-symbolic models.
To demonstrate the framework's versatility, we apply it to two distinct problems. First, we transform a GNN for node classification into a collective classification model that explicitly models homo- and heterophilic label patterns, substantially improving accuracy. Second, we introduce a multi-objective network optimization problem in environmental planning, where MAP inference supports complex decision-making. Both applications include new publicly available benchmark datasets.
This work introduces a powerful and coherent neuro-symbolic approach to graph data, bridging learning and reasoning in ways that enable novel applications and improved performance across diverse tasks. 

**Abstract (ZH)**: 基于图神经网络和关系贝叶斯网络的神经符号框架：结合学习与推理的优势以提高图数据的应用性能 

---
# Probabilistic Active Goal Recognition 

**Title (ZH)**: 概率主动目标识别 

**Authors**: Chenyuan Zhang, Cristian Rojas Cardenas, Hamid Rezatofighi, Mor Vered, Buser Say  

**Link**: [PDF](https://arxiv.org/pdf/2507.21846)  

**Abstract**: In multi-agent environments, effective interaction hinges on understanding the beliefs and intentions of other agents. While prior work on goal recognition has largely treated the observer as a passive reasoner, Active Goal Recognition (AGR) focuses on strategically gathering information to reduce uncertainty. We adopt a probabilistic framework for Active Goal Recognition and propose an integrated solution that combines a joint belief update mechanism with a Monte Carlo Tree Search (MCTS) algorithm, allowing the observer to plan efficiently and infer the actor's hidden goal without requiring domain-specific knowledge. Through comprehensive empirical evaluation in a grid-based domain, we show that our joint belief update significantly outperforms passive goal recognition, and that our domain-independent MCTS performs comparably to our strong domain-specific greedy baseline. These results establish our solution as a practical and robust framework for goal inference, advancing the field toward more interactive and adaptive multi-agent systems. 

**Abstract (ZH)**: 在多agent环境中，有效的交互依赖于理解其他agent的信念和意图。与以往主要将观察者视为被动推理者的基于目标识别工作不同，主动目标识别（AGR）侧重于通过战略性的信息收集来减少不确定性。我们采用概率框架进行主动目标识别，并提出了一种结合联合信念更新机制和蒙特卡洛树搜索（MCTS）算法的集成解决方案，使得观察者可以在无需特定领域知识的情况下高效规划并推断出行动者的隐藏目标。通过在网格域中的全面实验评估，我们证明了我们的联合信念更新显著优于被动目标识别，并且我们的领域无关MCTS与我们的强大领域特定贪心基准具有 comparable 性能。这些结果确立了我们的解决方案作为目标推断的实用且稳健框架的地位，推动了该领域向更具交互性和适应性的多agent系统方向发展。 

---
# MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE 

**Title (ZH)**: MixGRPO: 结合ODE-SDE实现基于流的GRPO效率提升 

**Authors**: Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Miles Yang, Zhao Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2507.21802)  

**Abstract**: Although GRPO substantially enhances flow matching models in human preference alignment of image generation, methods such as FlowGRPO still exhibit inefficiency due to the necessity of sampling and optimizing over all denoising steps specified by the Markov Decision Process (MDP). In this paper, we propose $\textbf{MixGRPO}$, a novel framework that leverages the flexibility of mixed sampling strategies through the integration of stochastic differential equations (SDE) and ordinary differential equations (ODE). This streamlines the optimization process within the MDP to improve efficiency and boost performance. Specifically, MixGRPO introduces a sliding window mechanism, using SDE sampling and GRPO-guided optimization only within the window, while applying ODE sampling outside. This design confines sampling randomness to the time-steps within the window, thereby reducing the optimization overhead, and allowing for more focused gradient updates to accelerate convergence. Additionally, as time-steps beyond the sliding window are not involved in optimization, higher-order solvers are supported for sampling. So we present a faster variant, termed $\textbf{MixGRPO-Flash}$, which further improves training efficiency while achieving comparable performance. MixGRPO exhibits substantial gains across multiple dimensions of human preference alignment, outperforming DanceGRPO in both effectiveness and efficiency, with nearly 50% lower training time. Notably, MixGRPO-Flash further reduces training time by 71%. Codes and models are available at $\href{this https URL}{MixGRPO}$. 

**Abstract (ZH)**: MixGRPO：一种利用混合采样策略的新型框架 

---
# Hybrid Causal Identification and Causal Mechanism Clustering 

**Title (ZH)**: 混合因果识别与因果机制聚类 

**Authors**: Saixiong Liu, Yuhua Qian, Jue Li, Honghong Cheng, Feijiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.21792)  

**Abstract**: Bivariate causal direction identification is a fundamental and vital problem in the causal inference field. Among binary causal methods, most methods based on additive noise only use one single causal mechanism to construct a causal model. In the real world, observations are always collected in different environments with heterogeneous causal relationships. Therefore, on observation data, this paper proposes a Mixture Conditional Variational Causal Inference model (MCVCI) to infer heterogeneous causality. Specifically, according to the identifiability of the Hybrid Additive Noise Model (HANM), MCVCI combines the superior fitting capabilities of the Gaussian mixture model and the neural network and elegantly uses the likelihoods obtained from the probabilistic bounds of the mixture conditional variational auto-encoder as causal decision criteria. Moreover, we model the casual heterogeneity into cluster numbers and propose the Mixture Conditional Variational Causal Clustering (MCVCC) method, which can reveal causal mechanism expression. Compared with state-of-the-art methods, the comprehensive best performance demonstrates the effectiveness of the methods proposed in this paper on several simulated and real data. 

**Abstract (ZH)**: 双变量因果方向识别是因果推理领域的一个基础而重要的问题。在二元因果方法中，大多数基于加性噪声的方法仅使用单一因果机制来构建因果模型。在现实世界中，观测数据总是收集自具有异质因果关系的不同环境。因此，本文在观测数据上提出了混合条件变异性因果推理模型（MCVCI）以推断异质因果关系。具体地，根据混合加性噪声模型（HANM）的可识别性，MCVCI 结合了高斯混合模型的优秀拟合能力和神经网络，并巧妙地使用混合条件变分自编码器的概率边界似然作为因果决策标准。此外，我们将因果异质性建模为聚类数目，并提出混合条件变分因果聚类（MCVCC）方法，该方法可以揭示因果机制表达。与现有最佳方法相比，全面的最优性能验证了本文所提出方法在多个模拟和真实数据集上的有效性。 

---
# SAT-Based Bounded Fitting for the Description Logic ALC 

**Title (ZH)**: 基于SAT的描述逻辑ALC的有界拟合方法 

**Authors**: Maurice Funk, Jean Christoph Jung, Tom Voellmer  

**Link**: [PDF](https://arxiv.org/pdf/2507.21752)  

**Abstract**: Bounded fitting is a general paradigm for learning logical formulas from positive and negative data examples, that has received considerable interest recently. We investigate bounded fitting for the description logic ALC and its syntactic fragments. We show that the underlying size-restricted fitting problem is NP-complete for all studied fragments, even in the special case of a single positive and a single negative example. By design, bounded fitting comes with probabilistic guarantees in Valiant's PAC learning framework. In contrast, we show that other classes of algorithms for learning ALC concepts do not provide such guarantees. Finally, we present an implementation of bounded fitting in ALC and its fragments based on a SAT solver. We discuss optimizations and compare our implementation to other concept learning tools. 

**Abstract (ZH)**: 有界拟合是学习正反例数据逻辑公式的一般范式， recently received considerable interest. 我们研究了描述逻辑ALC及其语法片段的有界拟合。我们证明了所有研究片段的基本大小限制的拟合问题是NP完全的，即使在仅有一个正例和一个反例的特殊情况也是如此。由于设计原因，有界拟合在Valiant的PAC学习框架中提供了概率保证。相比之下，我们证明了其他学习ALC概念的算法类没有提供这样的保证。最后，我们基于SAT求解器实现了ALC及其片段的有界拟合，并讨论了优化措施，将我们的实现与其他概念学习工具进行了比较。 

---
# GDAIP: A Graph-Based Domain Adaptive Framework for Individual Brain Parcellation 

**Title (ZH)**: 基于图的领域自适应框架：个体脑区划分 

**Authors**: Jianfei Zhu, Haiqi Zhu, Shaohui Liu, Feng Jiang, Baichun Wei, Chunzhi Yi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21727)  

**Abstract**: Recent deep learning approaches have shown promise in learning such individual brain parcellations from functional magnetic resonance imaging (fMRI). However, most existing methods assume consistent data distributions across domains and struggle with domain shifts inherent to real-world cross-dataset scenarios. To address this challenge, we proposed Graph Domain Adaptation for Individual Parcellation (GDAIP), a novel framework that integrates Graph Attention Networks (GAT) with Minimax Entropy (MME)-based domain adaptation. We construct cross-dataset brain graphs at both the group and individual levels. By leveraging semi-supervised training and adversarial optimization of the prediction entropy on unlabeled vertices from target brain graph, the reference atlas is adapted from the group-level brain graph to the individual brain graph, enabling individual parcellation under cross-dataset settings. We evaluated our method using parcellation visualization, Dice coefficient, and functional homogeneity. Experimental results demonstrate that GDAIP produces individual parcellations with topologically plausible boundaries, strong cross-session consistency, and ability of reflecting functional organization. 

**Abstract (ZH)**: 图域适应个体分叶方法（GDAIP）：基于Graph注意力网络和最小最大熵的域适应框架 

---
# Unrolling Dynamic Programming via Graph Filters 

**Title (ZH)**: 通过图滤波展开动态规划 

**Authors**: Sergio Rozada, Samuel Rey, Gonzalo Mateos, Antonio G. Marques  

**Link**: [PDF](https://arxiv.org/pdf/2507.21705)  

**Abstract**: Dynamic programming (DP) is a fundamental tool used across many engineering fields. The main goal of DP is to solve Bellman's optimality equations for a given Markov decision process (MDP). Standard methods like policy iteration exploit the fixed-point nature of these equations to solve them iteratively. However, these algorithms can be computationally expensive when the state-action space is large or when the problem involves long-term dependencies. Here we propose a new approach that unrolls and truncates policy iterations into a learnable parametric model dubbed BellNet, which we train to minimize the so-termed Bellman error from random value function initializations. Viewing the transition probability matrix of the MDP as the adjacency of a weighted directed graph, we draw insights from graph signal processing to interpret (and compactly re-parameterize) BellNet as a cascade of nonlinear graph filters. This fresh look facilitates a concise, transferable, and unifying representation of policy and value iteration, with an explicit handle on complexity during inference. Preliminary experiments conducted in a grid-like environment demonstrate that BellNet can effectively approximate optimal policies in a fraction of the iterations required by classical methods. 

**Abstract (ZH)**: 动态规划（DP）是许多工程领域中的一项基本工具。DP的主要目标是通过给定的马尔可夫决策过程（MDP）来求解贝尔曼最优方程。标准方法如策略迭代利用这些方程的不动点性质迭代求解。然而，当状态-动作空间巨大或问题涉及长期依赖时，这些算法可能会非常耗时。 herein我们提出了一种新方法，将策略迭代展开并截断为一个可学习的参数模型贝尔网（BellNet），并通过从随机价值函数初始化中最小化贝尔曼误差来进行训练。将MDP的转移概率矩阵视为加权有向图的邻接矩阵，我们从图信号处理中汲取灵感，以解释（并紧凑地重新参数化）贝尔网为级联的非线性图滤波器。这种新的视角为策略和值迭代提供了一种简洁、可迁移且统一的表示，并在推理过程中明确处理了复杂性。初步实验在网格环境中显示，贝尔网可以在经典方法所需迭代次数的很小一部分内有效地逼近最优策略。 

---
# Can the current trends of AI handle a full course of mathematics? 

**Title (ZH)**: 当前的人工智能趋势能否应对完整数学课程？ 

**Authors**: Mariam Alsayyad, Fayadh Kadhem  

**Link**: [PDF](https://arxiv.org/pdf/2507.21664)  

**Abstract**: This paper addresses the question of how able the current trends of Artificial Intelligence (AI) are in managing to take the responsibility of a full course of mathematics at a college level. The study evaluates this ability in four significant aspects, namely, creating a course syllabus, presenting selected material, answering student questions, and creating an assessment. It shows that even though the AI is strong in some important parts like organization and accuracy, there are still some human aspects that are far away from the current abilities of AI. There is still a hidden emotional part, even in science, that cannot be fulfilled by the AI in its current state. This paper suggests some recommendations to integrate the human and AI potentials to create better outcomes in terms of reaching the target of creating a full course of mathematics, at a university level, as best as possible. 

**Abstract (ZH)**: 本文探讨当前人工智能（AI）趋势在承担大学数学完整课程责任方面的能力。研究从四个重要方面评估这种能力，即制定课程大纲、呈现选定材料、回答学生问题和创建评估。研究表明，尽管AI在组织和准确性方面表现出色，但仍有一些人类特质远超当前AI的能力范围。即使在科学领域，AI当前状态下仍无法满足某些隐藏的情感需求。本文提出了一些建议，以整合人类和AI的潜力，尽可能地达成在大学水平上创建完整数学课程的目标，以取得更好的成果。 

---
# "Teammates, Am I Clear?": Analysing Legible Behaviours in Teams 

**Title (ZH)**: “队友，倾听我的声音”：分析团队中的可辨识行为 

**Authors**: Miguel Faria, Francisco S. Melo, Ana Paiva  

**Link**: [PDF](https://arxiv.org/pdf/2507.21631)  

**Abstract**: In this paper we investigate the notion of legibility in sequential decision-making in the context of teams and teamwork. There have been works that extend the notion of legibility to sequential decision making, for deterministic and for stochastic scenarios. However, these works focus on one agent interacting with one human, foregoing the benefits of having legible decision making in teams of agents or in team configurations with humans. In this work we propose an extension of legible decision-making to multi-agent settings that improves the performance of agents working in collaboration. We showcase the performance of legible decision making in team scenarios using our proposed extension in multi-agent benchmark scenarios. We show that a team with a legible agent is able to outperform a team composed solely of agents with standard optimal behaviour. 

**Abstract (ZH)**: 本文探讨了在团队和团队协作背景下序列决策中的可读性问题。已有研究将可读性扩展到确定性和随机情景下的序列决策中。然而，这些研究主要关注单个代理与人类交互的情况，忽视了团队中具有可读性决策或包含人类的团队配置中的优势。本文提出了一种扩展可读性决策的方法，以增强协作中代理的表现。通过在多代理基准场景中应用所提出的方法，展示了在团队场景中可读性决策的性能。我们证明了具有可读性代理的团队能够超越仅由标准最优行为代理组成的团队。 

---
# Finding Uncommon Ground: A Human-Centered Model for Extrospective Explanations 

**Title (ZH)**: 寻求共同点：以人为本的外向型解释模型 

**Authors**: Laura Spillner, Nima Zargham, Mihai Pomarlan, Robert Porzel, Rainer Malaka  

**Link**: [PDF](https://arxiv.org/pdf/2507.21571)  

**Abstract**: The need for explanations in AI has, by and large, been driven by the desire to increase the transparency of black-box machine learning models. However, such explanations, which focus on the internal mechanisms that lead to a specific output, are often unsuitable for non-experts. To facilitate a human-centered perspective on AI explanations, agents need to focus on individuals and their preferences as well as the context in which the explanations are given. This paper proposes a personalized approach to explanation, where the agent tailors the information provided to the user based on what is most likely pertinent to them. We propose a model of the agent's worldview that also serves as a personal and dynamic memory of its previous interactions with the same user, based on which the artificial agent can estimate what part of its knowledge is most likely new information to the user. 

**Abstract (ZH)**: AI解释的个性化方法：基于用户视角的定制化信息提供 

---
# ST-GDance: Long-Term and Collision-Free Group Choreography from Music 

**Title (ZH)**: ST-GDance: 长期且无碰撞的群体 choreography 从音乐生成 

**Authors**: Jing Xu, Weiqiang Wang, Cunjian Chen, Jun Liu, Qiuhong Ke  

**Link**: [PDF](https://arxiv.org/pdf/2507.21518)  

**Abstract**: Group dance generation from music has broad applications in film, gaming, and animation production. However, it requires synchronizing multiple dancers while maintaining spatial coordination. As the number of dancers and sequence length increase, this task faces higher computational complexity and a greater risk of motion collisions. Existing methods often struggle to model dense spatial-temporal interactions, leading to scalability issues and multi-dancer collisions. To address these challenges, we propose ST-GDance, a novel framework that decouples spatial and temporal dependencies to optimize long-term and collision-free group choreography. We employ lightweight graph convolutions for distance-aware spatial modeling and accelerated sparse attention for efficient temporal modeling. This design significantly reduces computational costs while ensuring smooth and collision-free interactions. Experiments on the AIOZ-GDance dataset demonstrate that ST-GDance outperforms state-of-the-art baselines, particularly in generating long and coherent group dance sequences. Project page: this https URL. 

**Abstract (ZH)**: 音乐驱动群舞生成在电影、游戏和动画制作中有广泛的应用。然而，这需要同步多个舞者并保持空间协调。随着舞者数量和序列长度的增加，该任务面临的计算复杂性更高，并且运动碰撞的风险更大。现有方法常常难以建模密集的空间-时间交互，导致可扩展性问题和多舞者碰撞。为此，我们提出了一种名为ST-GDance的新型框架，该框架解耦空间和时间依赖性以优化长期且无碰撞的群舞编排。我们采用轻量级图卷积进行距离感知的空间建模，并采用加速稀疏注意机制实现高效的临时建模。这种设计显著降低了计算成本，同时确保了平滑且无碰撞的交互。实验结果表明，ST-GDance在生成长且连贯的群舞序列方面优于现有的基线方法。项目页面：this https URL。 

---
# Learning to Imitate with Less: Efficient Individual Behavior Modeling in Chess 

**Title (ZH)**: 学习用得更少：象棋中高效个体行为建模 

**Authors**: Zhenwei Tang, Difan Jiao, Eric Xue, Reid McIlroy-Young, Jon Kleinberg, Siddhartha Sen, Ashton Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2507.21488)  

**Abstract**: As humans seek to collaborate with, learn from, and better understand artificial intelligence systems, developing AIs that can accurately emulate individual decision-making becomes increasingly important. Chess, a long-standing AI benchmark with precise skill measurement, offers an ideal testbed for human-AI alignment. However, existing approaches to modeling human behavior require prohibitively large amounts of data from each individual, making them impractical for new or sparsely represented users. In this work, we introduce Maia4All, a framework designed to learn and adapt to individual decision-making styles efficiently, even with limited data. Maia4All achieves this through a two-stage optimization process: (1) an enrichment step, which bridges population and individual-level human behavior modeling with a prototype-enriched model, and (2) a democratization step, which leverages ability levels or user prototypes to initialize and refine individual embeddings with minimal data. Our experimental results show that Maia4All can accurately predict individual moves and profile behavioral patterns with high fidelity, establishing a new standard for personalized human-like AI behavior modeling in chess. Maia4All achieves individual human behavior modeling in chess with only 20 games, compared to the 5,000 games required previously, representing a significant improvement in data efficiency. Our work provides an example of how population AI systems can flexibly adapt to individual users using a prototype-enriched model as a bridge. This approach extends beyond chess, as shown in our case study on idiosyncratic LLMs, highlighting its potential for broader applications in personalized AI adaptation. 

**Abstract (ZH)**: 随着人类寻求与人工智能系统合作、学习和更好地理解这些系统，发展能够准确模拟个体决策的人工智能变得越来越重要。国际象棋作为一种长期的人工智能基准，具有精确的技能衡量标准，为人类-人工智能对齐提供了理想的试验平台。然而，现有的人工行为建模方法需要从每个个体收集大量数据，这使得它们对于新用户或稀疏表示的用户来说不切实际。在本文中，我们介绍了一种名为Maia4All的框架，旨在即使在有限数据的情况下也能高效地学习和适应个体的决策风格。Maia4All通过两阶段优化过程实现这一点：（1）丰富步骤，通过原型增强模型将总体和个体水平的人类行为建模连接起来；（2）民主化步骤，利用能力水平或用户原型以最少的数据初始化和精炼个体嵌入。实验结果表明，Maia4All能够准确预测个体走法并高保真地刻画行为模式，为国际象棋中的个性化类人类人工行为建模设立了新标准。Maia4All仅需20场比赛就能实现个体行为建模，与之前所需的5000场比赛相比，显著提高了数据效率。我们的工作提供了一个例子，展示了如何通过原型增强模型作为一种桥梁使总体人工智能系统灵活适应个别用户。这种方法不仅限于国际象棋，在我们对具有独特性的LLMs的案例研究中得到了证明，突显了其在个性化人工智能适应方面的更广泛应用潜力。 

---
# Optimizing Multi-Tier Supply Chain Ordering with LNN+XGBoost: Mitigating the Bullwhip Effect 

**Title (ZH)**: 基于LNN+XGBoost优化多级供应链订单：减轻牛鞭效应 

**Authors**: Chunan Tong  

**Link**: [PDF](https://arxiv.org/pdf/2507.21383)  

**Abstract**: Supply chain management faces significant challenges, including demand fluctuations, inventory imbalances, and amplified upstream order variability due to the bullwhip effect. Traditional methods, such as simple moving averages, struggle to address dynamic market conditions. Emerging machine learning techniques, including LSTM, reinforcement learning, and XGBoost, offer potential solutions but are limited by computational complexity, training inefficiencies, or constraints in time-series modeling. Liquid Neural Networks, inspired by dynamic biological systems, present a promising alternative due to their adaptability, low computational cost, and robustness to noise, making them suitable for real-time decision-making and edge computing. Despite their success in applications like autonomous vehicles and medical monitoring, their potential in supply chain optimization remains underexplored. This study introduces a hybrid LNN and XGBoost model to optimize ordering strategies in multi-tier supply chains. By leveraging LNN's dynamic feature extraction and XGBoost's global optimization capabilities, the model aims to mitigate the bullwhip effect and enhance cumulative profitability. The research investigates how local and global synergies within the hybrid framework address the dual demands of adaptability and efficiency in SCM. The proposed approach fills a critical gap in existing methodologies, offering an innovative solution for dynamic and efficient supply chain management. 

**Abstract (ZH)**: 基于液态神经网络和XGBoost的 hybrid 模型在多层供应链中的订货策略优化 

---
# Efficacy of AI RAG Tools for Complex Information Extraction and Data Annotation Tasks: A Case Study Using Banks Public Disclosures 

**Title (ZH)**: AI RAG工具在复杂信息提取和数据标注任务中的有效性：以银行公开披露为例的研究 

**Authors**: Nicholas Botti, Flora Haberkorn, Charlotte Hoopes, Shaun Khan  

**Link**: [PDF](https://arxiv.org/pdf/2507.21360)  

**Abstract**: We utilize a within-subjects design with randomized task assignments to understand the effectiveness of using an AI retrieval augmented generation (RAG) tool to assist analysts with an information extraction and data annotation task. We replicate an existing, challenging real-world annotation task with complex multi-part criteria on a set of thousands of pages of public disclosure documents from global systemically important banks (GSIBs) with heterogeneous and incomplete information content. We test two treatment conditions. First, a "naive" AI use condition in which annotators use only the tool and must accept the first answer they are given. And second, an "interactive" AI treatment condition where annotators use the tool interactively, and use their judgement to follow-up with additional information if necessary. Compared to the human-only baseline, the use of the AI tool accelerated task execution by up to a factor of 10 and enhanced task accuracy, particularly in the interactive condition. We find that when extrapolated to the full task, these methods could save up to 268 hours compared to the human-only approach. Additionally, our findings suggest that annotator skill, not just with the subject matter domain, but also with AI tools, is a factor in both the accuracy and speed of task performance. 

**Abstract (ZH)**: 我们利用一种被试内设计并随机分配任务的方法，研究AI检索增强生成（RAG）工具在信息提取和数据标注任务中对分析师的辅助效果。我们复制了一个现有且具有挑战性的实际注释任务，该任务包含复杂的多部分标准，并应用在网络全球系统重要性银行（GSIBs）数千页的公开披露文件上，这些文件包括异质性和不完整的信息内容。我们测试了两种条件。首先，一个简单的AI使用条件，注释者仅使用工具并必须接受最初提供的答案；其次，一个互动的AI治疗条件，注释者可以互动地使用工具，并在必要时利用自己的判断获取额外信息。与仅有人类参与的基线相比，使用AI工具将任务执行速度最多加快10倍，并显著提高任务准确性，尤其是在互动条件下。我们的研究发现，当应用到整个任务时，这些方法相较于仅有人类参与的方法最多可节省268小时。此外，我们的研究结果表明，注释者的技能，不仅在于专业知识领域，也在于对AI工具的掌握，是任务准确性和速度的关键因素。 

---
# Structured Relevance Assessment for Robust Retrieval-Augmented Language Models 

**Title (ZH)**: 结构化相关性评估以提高检索增强语言模型的鲁棒性 

**Authors**: Aryan Raj, Astitva Veer Garg, Anitha D  

**Link**: [PDF](https://arxiv.org/pdf/2507.21287)  

**Abstract**: Retrieval-Augmented Language Models (RALMs) face significant challenges in reducing factual errors, particularly in document relevance evaluation and knowledge integration. We introduce a framework for structured relevance assessment that enhances RALM robustness through improved document evaluation, balanced intrinsic and external knowledge integration, and effective handling of unanswerable queries. Our approach employs a multi-dimensional scoring system that considers both semantic matching and source reliability, utilizing embedding-based relevance scoring and synthetic training data with mixed-quality documents. We implement specialized benchmarking on niche topics, a knowledge integration mechanism, and an "unknown" response protocol for queries with insufficient knowledge coverage. Preliminary evaluations demonstrate significant reductions in hallucination rates and improved transparency in reasoning processes. Our framework advances the development of more reliable question-answering systems capable of operating effectively in dynamic environments with variable data quality. While challenges persist in accurately distinguishing credible information and balancing system latency with thoroughness, this work represents a meaningful step toward enhancing RALM reliability. 

**Abstract (ZH)**: 结构化相关性评估框架：增强检索增强语言模型的鲁棒性 

---
# Ontological Foundations of State Sovereignty 

**Title (ZH)**: 国家主权的本体论基础 

**Authors**: John Beverley, Danielle Limbaugh  

**Link**: [PDF](https://arxiv.org/pdf/2507.21172)  

**Abstract**: This short paper is a primer on the nature of state sovereignty and the importance of claims about it. It also aims to reveal (merely reveal) a strategy for working with vague or contradictory data about which states, in fact, are sovereign. These goals together are intended to set the stage for applied work in ontology about international affairs. 

**Abstract (ZH)**: 这篇简短的论文是对国家主权本质及其主张重要性的基础介绍，同时也旨在揭示（仅仅揭示）处理有关哪些国家实际上享有主权的模糊或矛盾数据的一种策略。这些目标旨在为关于国际事务的本体论应用工作奠定基础。 

---
# An ontological analysis of risk in Basic Formal Ontology 

**Title (ZH)**: 基本形式本体中风险的本体分析 

**Authors**: Federico Donato, Adrien Barton  

**Link**: [PDF](https://arxiv.org/pdf/2507.21171)  

**Abstract**: The paper explores the nature of risk, providing a characterization using the categories of the Basic Formal Ontology (BFO). It argues that the category Risk is a subclass of BFO:Role, contrasting it with a similar view classifying Risk as a subclass of BFO:Disposition. This modeling choice is applied on one example of risk, which represents objects, processes (both physical and mental) and their interrelations, then generalizing from the instances in the example to obtain an overall analysis of risk, making explicit what are the sufficient conditions for being a risk. Plausible necessary conditions are also mentioned for future work. Index Terms: ontology, risk, BFO, role, disposition 

**Abstract (ZH)**: 该论文探讨了风险的本质，使用基本形式本体（BFO）的类别对风险进行刻画。它认为风险类别是BFO:Role的子类别，将其与将风险归类为BFO:Disposition的类似观点区分开来。这种建模选择应用于风险的一个实例，该实例代表物体、过程（包括物理和心理过程）及其相互关系，然后从该实例中的实例中概括，以获得对风险的整体分析，明确说明成为风险所需的充分条件。还提到了一些有说服力的必要条件供未来工作参考。关键词：本体、风险、BFO、角色、倾向。 

---
# Adaptive XAI in High Stakes Environments: Modeling Swift Trust with Multimodal Feedback in Human AI Teams 

**Title (ZH)**: 高风险环境中的自适应XAI：基于多模态反馈建模人类AI团队中的快速信任 

**Authors**: Nishani Fernando, Bahareh Nakisa, Adnan Ahmad, Mohammad Naim Rastgoo  

**Link**: [PDF](https://arxiv.org/pdf/2507.21158)  

**Abstract**: Effective human-AI teaming heavily depends on swift trust, particularly in high-stakes scenarios such as emergency response, where timely and accurate decision-making is critical. In these time-sensitive and cognitively demanding settings, adaptive explainability is essential for fostering trust between human operators and AI systems. However, existing explainable AI (XAI) approaches typically offer uniform explanations and rely heavily on explicit feedback mechanisms, which are often impractical in such high-pressure scenarios. To address this gap, we propose a conceptual framework for adaptive XAI that operates non-intrusively by responding to users' real-time cognitive and emotional states through implicit feedback, thereby enhancing swift trust in high-stakes environments. The proposed adaptive explainability trust framework (AXTF) leverages physiological and behavioral signals, such as EEG, ECG, and eye tracking, to infer user states and support explanation adaptation. At its core is a multi-objective, personalized trust estimation model that maps workload, stress, and emotion to dynamic trust estimates. These estimates guide the modulation of explanation features enabling responsive and personalized support that promotes swift trust in human-AI collaboration. This conceptual framework establishes a foundation for developing adaptive, non-intrusive XAI systems tailored to the rigorous demands of high-pressure, time-sensitive environments. 

**Abstract (ZH)**: 适应性人机团队中的解释性信任框架：在高压力、时间敏感环境中促进快速信任 

---
# Project Patti: Why can You Solve Diabolical Puzzles on one Sudoku Website but not Easy Puzzles on another Sudoku Website? 

**Title (ZH)**: Patti项目：为什么你能在一个数独网站上解决恶棍级谜题却不能在另一个网站上解决简单级谜题？ 

**Authors**: Arman Eisenkolb-Vaithyanathan  

**Link**: [PDF](https://arxiv.org/pdf/2507.21137)  

**Abstract**: In this paper we try to answer the question "What constitutes Sudoku difficulty rating across different Sudoku websites?" Using two distinct methods that can both solve every Sudoku puzzle, I propose two new metrics to characterize Sudoku difficulty. The first method is based on converting a Sudoku puzzle into its corresponding Satisfiability (SAT) problem. The first proposed metric is derived from SAT Clause Length Distribution which captures the structural complexity of a Sudoku puzzle including the number of given digits and the cells they are in. The second method simulates human Sudoku solvers by intertwining four popular Sudoku strategies within a backtracking algorithm called Nishio. The second metric is computed by counting the number of times Sudoku strategies are applied within the backtracking iterations of a randomized Nishio. Using these two metrics, I analyze more than a thousand Sudoku puzzles across five popular websites to characterize every difficulty level in each website. I evaluate the relationship between the proposed metrics and website-labeled difficulty levels using Spearman's rank correlation coefficient, finding strong correlations for 4 out of 5 websites. I construct a universal rating system using a simple, unsupervised classifier based on the two proposed metrics. This rating system is capable of classifying both individual puzzles and entire difficulty levels from the different Sudoku websites into three categories - Universal Easy, Universal Medium, and Universal Hard - thereby enabling consistent difficulty mapping across Sudoku websites. The experimental results show that for 4 out of 5 Sudoku websites, the universal classification aligns well with website-labeled difficulty levels. Finally, I present an algorithm that can be used by early Sudoku practitioners to solve Sudoku puzzles. 

**Abstract (ZH)**: 在这篇文章中，我们尝试回答“不同数独网站上的数独难度评级由什么构成？”的问题。使用两种可以解决所有数独谜题的不同方法，我们提出两种新的指标来表征数独难度。首先，基于将数独谜题转换为其相应的可满足性（SAT）问题的方法。第一个提出的指标源自SAT子句长度分布，它可以捕捉数独谜题的结构性复杂度，包括已给定的数字数量及其所在的单元格。其次，通过在回溯算法Nishio中交织四种流行的数独策略来模拟人类解数独的过程。第二个指标通过计算在一个随机化Nishio的回溯迭代过程中应用数独策略的次数来计算。利用这两个指标，我们分析了五个热门网站上的上千个数独谜题，以表征每个网站上的每一难度级别。我们使用斯皮尔曼等级相关系数评估所提指标与网站标注难度等级之间的关系，发现有4个网站显示出强烈的相关性。我们基于这两个提出的指标构建了一个通用评分系统，该系统能够对来自不同数独网站的单个谜题和整个难度等级进行分类，归为三类：通用简单、通用中等和通用困难，从而在数独网站之间实现一致的难度映射。实验结果显示，对于4个数独网站，通用分类与网站标注难度等级吻合良好。最后，我们提出了一种算法，早期数独实践者可以使用该算法来解决数独谜题。 

---
# NPO: Learning Alignment and Meta-Alignment through Structured Human Feedback 

**Title (ZH)**: NPO: 通过结构化人类反馈学习对齐与元对齐 

**Authors**: Madhava Gaikwad, Ashwini Ramchandra Doke  

**Link**: [PDF](https://arxiv.org/pdf/2507.21131)  

**Abstract**: We present NPO, an alignment-aware learning framework that operationalizes feedback-driven adaptation in human-in-the-loop decision systems. Unlike prior approaches that treat alignment as a static or post-hoc property, NPO introduces a formalization of alignment loss that is measurable, supervisable, and reducible under structured feedback. In parallel, we propose meta-alignment as the fidelity of the monitoring process that governs retraining or override triggers, and show that it is formally reducible to primary alignment via threshold fidelity. Our implementation spans a scalable operational loop involving scenario scoring, threshold tuning, policy validation, and structured feedback ingestion, including "likes", overrides, and abstentions. We provide formal convergence results under stochastic feedback and show that both alignment loss and monitoring fidelity converge additively. Empirically, NPO demonstrates measurable value in hyperscale deployment settings. A simulation-based artifact and ablation studies further illustrate the theoretical principles in action. Together, NPO offers a compact, inspectable architecture for continual alignment monitoring, helping bridge theoretical alignment guarantees with practical reliability in dynamic environments. 

**Abstract (ZH)**: NPO：一种 Awareness-Based 学习框架，用于人类在环决策系统中的对齐适应 

---
# Artificial intelligence for sustainable wine industry: AI-driven management in viticulture, wine production and enotourism 

**Title (ZH)**: 人工智能赋能可持续葡萄酒产业：智能驱动的葡萄种植、葡萄酒生产及葡萄酒旅游业管理 

**Authors**: Marta Sidorkiewicz, Karolina Królikowska, Berenika Dyczek, Edyta Pijet-Migon, Anna Dubel  

**Link**: [PDF](https://arxiv.org/pdf/2507.21098)  

**Abstract**: This study examines the role of Artificial Intelligence (AI) in enhancing sustainability and efficiency within the wine industry. It focuses on AI-driven intelligent management in viticulture, wine production, and enotourism. As the wine industry faces environmental and economic challenges, AI offers innovative solutions to optimize resource use, reduce environmental impact, and improve customer engagement. Understanding AI's potential in sustainable winemaking is crucial for fostering responsible and efficient industry practices. The research is based on a questionnaire survey conducted among Polish winemakers, combined with a comprehensive analysis of AI methods applicable to viticulture, production, and tourism. Key AI technologies, including predictive analytics, machine learning, and computer vision, are explored. The findings indicate that AI enhances vineyard monitoring, optimizes irrigation, and streamlines production processes, contributing to sustainable resource management. In enotourism, AI-powered chatbots, recommendation systems, and virtual tastings personalize consumer experiences. The study highlights AI's impact on economic, environmental, and social sustainability, supporting local wine enterprises and cultural heritage. Keywords: Artificial Intelligence, Sustainable Development, AI-Driven Management, Viticulture, Wine Production, Enotourism, Wine Enterprises, Local Communities 

**Abstract (ZH)**: 本研究探讨了人工智能（AI）在葡萄酒行业中增强可持续性和效率的作用。它重点关注AI驱动的智能管理在葡萄种植、葡萄酒生产和葡萄酒旅游业中的应用。随着葡萄酒行业面临环境和经济挑战，AI提供了优化资源使用、减少环境影响和改善消费者参与的创新解决方案。理解AI在可持续酿酒中的潜力对于促进负责任和高效的行业实践至关重要。该研究基于对波兰酿酒商的问卷调查，并结合了对适用于葡萄种植、生产及旅游业的AI方法的全面分析。研究探讨了包括预测分析、机器学习和计算机视觉在内的关键AI技术。研究结果表明，AI增强了葡萄园的监控，优化了灌溉，并简化了生产流程，从而促进了可持续资源管理。在葡萄酒旅游业中，AI驱动的聊天机器人、推荐系统和虚拟品酒使消费者体验更加个性化。该研究突出了AI对经济、环境和社会可持续性的影响，支持了本地葡萄酒企业和文化遗产。关键词：人工智能，可持续发展，AI驱动管理，葡萄种植，葡萄酒生产，葡萄酒旅游业，本地社区。 

---
# SynLang and Symbiotic Epistemology: A Manifesto for Conscious Human-AI Collaboration 

**Title (ZH)**: SynLang 和共生 epistemology: 人类意识与AI协作的宣言 

**Authors**: Jan Kapusta  

**Link**: [PDF](https://arxiv.org/pdf/2507.21067)  

**Abstract**: Current AI systems rely on opaque reasoning processes that hinder human oversight and collaborative potential. Conventional explainable AI approaches offer post-hoc justifications and often fail to establish genuine symbiotic collaboration. In this paper, the Symbiotic Epistemology is presented as a philosophical foundation for human-AI cognitive partnerships. Unlike frameworks that treat AI as a mere tool or replacement, symbiotic epistemology positions AI as a reasoning partner, fostering calibrated trust by aligning human confidence with AI reliability through explicit reasoning patterns and confidence assessments. SynLang (Symbiotic Syntactic Language) is introduced as a formal protocol for transparent human-AI collaboration. The framework is empirically validated through actual human-AI dialogues demonstrating AI's adaptation to structured reasoning protocols and successful metacognitive intervention. The protocol defines two complementary mechanisms: TRACE for high-level reasoning patterns and TRACE_FE for detailed factor explanations. It also integrates confidence quantification, declarative control over AI behavior, and context inheritance for multi-agent coordination. By structuring communication and embedding confidence-calibrated transparency, SynLang, together with symbiotic epistemology, enables AI systems that enhance human intelligence, preserve human agency, and uphold ethical accountability in collaborative decision-making. Through dual-level transparency, beginning with high-level reasoning patterns and progressing to granular explanations, the protocol facilitates rapid comprehension and supports thorough verification of AI decision-making. 

**Abstract (ZH)**: 当前的AI系统依赖于不透明的推理过程，这阻碍了人类的监督和协作潜力。传统的可解释AI方法通常只提供事后解释，并未能建立真正共生的合作关系。本文提出了共生知识论作为人类与AI认知伙伴关系的哲学基础。与将AI视为 mere 工具或替代品的框架不同，共生知识论将AI定位为推理伙伴，通过明确的推理模式和信任评估与人类信心对齐，从而培养校准的信任。SynLang（共生语义语言）被引入作为一种形式化协议，促进透明的人机合作。该框架通过实际的人机对话经验得到了验证，展示了AI适应结构化推理协议并成功进行元认知干预的能力。该协议定义了两种互补机制：TRACE 用于高级推理模式和 TRACE_FE 用于详细因素解释。该协议还整合了信心量化、声明性控制AI行为和上下文继承以支持多智能体协调。通过结构化沟通并嵌入校准信心的透明性，SynLang 与共生知识论一起，使AI系统能够增强人类智能、保护人类自主权并维护协作决策中的伦理问责制。通过双重水平的透明性，从高级推理模式开始，逐步到细粒度解释，该协议促进快速理解并支持对AI决策的彻底验证。 

---
# Foundation Models for Demand Forecasting via Dual-Strategy Ensembling 

**Title (ZH)**: 基于双重策略集成的Demand Forecasting基础模型研究 

**Authors**: Wei Yang, Defu Cao, Yan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22053)  

**Abstract**: Accurate demand forecasting is critical for supply chain optimization, yet remains difficult in practice due to hierarchical complexity, domain shifts, and evolving external factors. While recent foundation models offer strong potential for time series forecasting, they often suffer from architectural rigidity and limited robustness under distributional change. In this paper, we propose a unified ensemble framework that enhances the performance of foundation models for sales forecasting in real-world supply chains. Our method combines two complementary strategies: (1) Hierarchical Ensemble (HE), which partitions training and inference by semantic levels (e.g., store, category, department) to capture localized patterns; and (2) Architectural Ensemble (AE), which integrates predictions from diverse model backbones to mitigate bias and improve stability. We conduct extensive experiments on the M5 benchmark and three external sales datasets, covering both in-domain and zero-shot forecasting. Results show that our approach consistently outperforms strong baselines, improves accuracy across hierarchical levels, and provides a simple yet effective mechanism for boosting generalization in complex forecasting environments. 

**Abstract (ZH)**: 准确的需求预测对于供应链优化至关重要，但在实践中由于层次复杂性、领域变化和不断演变的外部因素仍然具有挑战性。尽管近期的基础模型在时间序列预测方面表现出强大的潜力，但它们往往缺乏架构柔性和在分布变化下的稳健性。在本文中，我们提出了一种统一的集成框架，以增强基础模型在实际供应链中的销售预测性能。该方法结合了两种互补策略：(1) 层次集成（HE），通过按语义层次（如商店、品类、部门）划分训练和推理来捕获局部模式；(2) 架构集成（AE），通过整合来自不同模型架构的预测来减轻偏差并提高稳定性。我们在M5基准和三个外部销售数据集上进行了广泛的实验，涵盖了领域内和零样本预测。结果表明，我们的方法在各种层次上均能持续优于强基线，并提供了一种简单而有效的机制，以增强复杂预测环境中的泛化能力。 

---
# Supervised Quantum Image Processing 

**Title (ZH)**: 监督量子图像处理 

**Authors**: Marco Parigi, Mehran Khosrojerdi, Filippo Caruso, Leonardo Banchi  

**Link**: [PDF](https://arxiv.org/pdf/2507.22039)  

**Abstract**: In the era of big data and artificial intelligence, the increasing volume of data and the demand to solve more and more complex computational challenges are two driving forces for improving the efficiency of data storage, processing and analysis. Quantum image processing (QIP) is an interdisciplinary field between quantum information science and image processing, which has the potential to alleviate some of these challenges by leveraging the power of quantum computing. In this work, we compare and examine the compression properties of four different Quantum Image Representations (QImRs): namely, Tensor Network Representation (TNR), Flexible Representation of Quantum Image (FRQI), Novel Enhanced Quantum Representation NEQR, and Quantum Probability Image Encoding (QPIE). Our simulations show that FRQI performs a higher compression of image information than TNR, NEQR, and QPIE. Furthermore, we investigate the trade-off between accuracy and memory in binary classification problems, evaluating the performance of quantum kernels based on QImRs compared to the classical linear kernel. Our results indicate that quantum kernels provide comparable classification average accuracy but require exponentially fewer resources for image storage. 

**Abstract (ZH)**: 在大数据和人工智能时代，不断增加的数据量和解决日益复杂的计算挑战的需求，推动了数据存储、处理和分析效率的提高。量子图像处理（QIP）是量子信息科学与图像处理的交叉领域，有可能通过利用量子计算的强大功能来缓解一些这些挑战。在本文中，我们比较和分析了四种不同的量子图像表示（QImRs）的压缩性能：张量网络表示（TNR）、灵活的量子图像表示（FRQI）、新型增强量子表示（NEQR）和量子概率图像编码（QPIE）。我们的模拟结果显示，FRQI在压缩图像信息方面优于TNR、NEQR和QPIE。此外，我们探讨了准确性与内存之间的权衡，在二分类问题中评估基于QImRs的量子核与经典线性核的性能。结果显示，量子核提供了可比拟的分类平均准确率，但需要指数级更少的图像存储资源。 

---
# Bridging Synthetic and Real-World Domains: A Human-in-the-Loop Weakly-Supervised Framework for Industrial Toxic Emission Segmentation 

**Title (ZH)**: 合成与实际领域桥梁构建：基于人类在环的弱监督框架在工业有毒排放分割中的应用 

**Authors**: Yida Tao, Yen-Chia Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22002)  

**Abstract**: Industrial smoke segmentation is critical for air-quality monitoring and environmental protection but is often hampered by the high cost and scarcity of pixel-level annotations in real-world settings. We introduce CEDANet, a human-in-the-loop, class-aware domain adaptation framework that uniquely integrates weak, citizen-provided video-level labels with adversarial feature alignment. Specifically, we refine pseudo-labels generated by a source-trained segmentation model using citizen votes, and employ class-specific domain discriminators to transfer rich source-domain representations to the industrial domain. Comprehensive experiments on SMOKE5K and custom IJmond datasets demonstrate that CEDANet achieves an F1-score of 0.414 and a smoke-class IoU of 0.261 with citizen feedback, vastly outperforming the baseline model, which scored 0.083 and 0.043 respectively. This represents a five-fold increase in F1-score and a six-fold increase in smoke-class IoU. Notably, CEDANet with citizen-constrained pseudo-labels achieves performance comparable to the same architecture trained on limited 100 fully annotated images with F1-score of 0.418 and IoU of 0.264, demonstrating its ability to reach small-sampled fully supervised-level accuracy without target-domain annotations. Our research validates the scalability and cost-efficiency of combining citizen science with weakly supervised domain adaptation, offering a practical solution for complex, data-scarce environmental monitoring applications. 

**Abstract (ZH)**: 基于公民科学的弱监督领域适应框架CEDANet：低成本工业烟雾分割方法 

---
# Teach Me to Trick: Exploring Adversarial Transferability via Knowledge Distillation 

**Title (ZH)**: 教我欺骗：通过知识精炼探索对抗迁移性 

**Authors**: Siddhartha Pradhan, Shikshya Shiwakoti, Neha Bathuri  

**Link**: [PDF](https://arxiv.org/pdf/2507.21992)  

**Abstract**: We investigate whether knowledge distillation (KD) from multiple heterogeneous teacher models can enhance the generation of transferable adversarial examples. A lightweight student model is trained using two KD strategies: curriculum-based switching and joint optimization, with ResNet50 and DenseNet-161 as teachers. The trained student is then used to generate adversarial examples using FG, FGS, and PGD attacks, which are evaluated against a black-box target model (GoogLeNet). Our results show that student models distilled from multiple teachers achieve attack success rates comparable to ensemble-based baselines, while reducing adversarial example generation time by up to a factor of six. An ablation study further reveals that lower temperature settings and the inclusion of hard-label supervision significantly enhance transferability. These findings suggest that KD can serve not only as a model compression technique but also as a powerful tool for improving the efficiency and effectiveness of black-box adversarial attacks. 

**Abstract (ZH)**: 我们调查了多个异构教师模型的知识蒸馏是否能增强可转移对抗样本的生成，并使用ResNet50和DenseNet-161作为教师模型，通过基于课程的学习切换和联合优化两种知识蒸馏策略训练一个轻量级学生模型。训练后，该学生模型使用FG、FGS和PGD攻击生成对抗样本，并与黑色盒目标模型（GoogLeNet）进行评估。结果表明，从多个教师模型蒸馏而来的学生模型在攻击成功率上与基于集成的基线相当，同时对抗样本生成时间最多可减少六倍。进一步的消融研究显示，较低的温度设置和硬标签监督的加入显著提高了可转移性。这些发现表明，知识蒸馏不仅可以作为模型压缩技术，还可以作为提高黑色盒对抗攻击效率和效果的强大工具。 

---
# Fine-Tuning Code Language Models to Detect Cross-Language Bugs 

**Title (ZH)**: 细调代码语言模型以检测跨语言错误 

**Authors**: Zengyang Li, Yimeng Li, Binbin Huang, Peng Liang, Ran Mo, Hui Liu, Yutao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.21954)  

**Abstract**: Multilingual programming, which involves using multiple programming languages (PLs) in a single project, is increasingly common due to its benefits. However, it introduces cross-language bugs (CLBs), which arise from interactions between different PLs and are difficult to detect by single-language bug detection tools. This paper investigates the potential of pre-trained code language models (CodeLMs) in CLB detection. We developed CLCFinder, a cross-language code identification tool, and constructed a CLB dataset involving three PL combinations (Python-C/C++, Java-C/C++, and Python-Java) with nine interaction types. We fine-tuned 13 CodeLMs on this dataset and evaluated their performance, analyzing the effects of dataset size, token sequence length, and code comments. Results show that all CodeLMs performed poorly before fine-tuning, but exhibited varying degrees of performance improvement after fine-tuning, with UniXcoder-base achieving the best F1 score (0.7407). Notably, small fine-tuned CodeLMs tended to performe better than large ones. CodeLMs fine-tuned on single-language bug datasets performed poorly on CLB detection, demonstrating the distinction between CLBs and single-language bugs. Additionally, increasing the fine-tuning dataset size significantly improved performance, while longer token sequences did not necessarily improve the model performance. The impact of code comments varied across models. Some fine-tuned CodeLMs' performance was improved, while others showed degraded performance. 

**Abstract (ZH)**: 多语言编程中的跨语言代码模型在跨语言 bug 检测中的潜力研究 

---
# Enhancing Generalization in Data-free Quantization via Mixup-class Prompting 

**Title (ZH)**: 基于Mixup-class提示的数据无关量化中的泛化增强 

**Authors**: Jiwoong Park, Chaeun Lee, Yongseok Choi, Sein Park, Deokki Hong, Jungwook Choi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21947)  

**Abstract**: Post-training quantization (PTQ) improves efficiency but struggles with limited calibration data, especially under privacy constraints. Data-free quantization (DFQ) mitigates this by generating synthetic images using generative models such as generative adversarial networks (GANs) and text-conditioned latent diffusion models (LDMs), while applying existing PTQ algorithms. However, the relationship between generated synthetic images and the generalizability of the quantized model during PTQ remains underexplored. Without investigating this relationship, synthetic images generated by previous prompt engineering methods based on single-class prompts suffer from issues such as polysemy, leading to performance degradation. We propose \textbf{mixup-class prompt}, a mixup-based text prompting strategy that fuses multiple class labels at the text prompt level to generate diverse, robust synthetic data. This approach enhances generalization, and improves optimization stability in PTQ. We provide quantitative insights through gradient norm and generalization error analysis. Experiments on convolutional neural networks (CNNs) and vision transformers (ViTs) show that our method consistently outperforms state-of-the-art DFQ methods like GenQ. Furthermore, it pushes the performance boundary in extremely low-bit scenarios, achieving new state-of-the-art accuracy in challenging 2-bit weight, 4-bit activation (W2A4) quantization. 

**Abstract (ZH)**: 基于mixup-class提示的数据无开集量化 

---
# Vibe Coding as a Reconfiguration of Intent Mediation in Software Development: Definition, Implications, and Research Agenda 

**Title (ZH)**: 振动编码作为软件开发中意图中介的重新配置：定义、影响与研究议程 

**Authors**: Christian Meske, Tobias Hermanns, Esther von der Weiden, Kai-Uwe Loser, Thorsten Berger  

**Link**: [PDF](https://arxiv.org/pdf/2507.21928)  

**Abstract**: Software development is undergoing a fundamental transformation as vibe coding becomes widespread, with large portions of contemporary codebases now being AI-generated. The disconnect between rapid adoption and limited conceptual understanding highlights the need for an inquiry into this emerging paradigm. Drawing on an intent perspective and historical analysis, we define vibe coding as a software development paradigm where humans and generative AI engage in collaborative flow to co-create software artifacts through natural language dialogue, shifting the mediation of developer intent from deterministic instruction to probabilistic inference. By intent mediation, we refer to the fundamental process through which developers translate their conceptual goals into representations that computational systems can execute. Our results show that vibe coding reconfigures cognitive work by redistributing epistemic labor between humans and machines, shifting the expertise in the software development process away from traditional areas such as design or technical implementation toward collaborative orchestration. We identify key opportunities, including democratization, acceleration, and systemic leverage, alongside risks, such as black box codebases, responsibility gaps, and ecosystem bias. We conclude with a research agenda spanning human-, technology-, and organization-centered directions to guide future investigations of this paradigm. 

**Abstract (ZH)**: 软件开发正经历一场根本性的变革，随着vibe编码的广泛使用，当代代码库中现在有大量部分是由AI生成的。快速采用与有限的概念理解之间的差距突显了对这一新兴范式的探究需求。基于意图视角及历史分析，我们将vibe编码定义为一种软件开发范式，在这种范式中，人类与生成型AI通过自然语言对话协作流动以共同创造软件制品，将开发人员意图的中介从确定性指令转向概率性推理。通过意图中介，我们指的是开发人员将概念性目标转换为计算系统可以执行的表现形式的基本过程。研究结果表明，vibe编码重新配置了认知工作，重新分配了人类与机器之间的本体论劳动，将软件开发过程中的专业性从传统的设计或技术实现领域转向了协作指挥。我们识别出关键机遇，包括普及化、加速和系统性杠杆作用，同时识别出风险，如黑盒代码库、责任缺口和生态系统偏见。我们提出了一项涵盖以人为中心、技术为中心和组织为中心的研究议程，以引导对该范式未来研究的指导。 

---
# Evaluating Deepfake Detectors in the Wild 

**Title (ZH)**: 评估野生环境中的深度伪造检测器 

**Authors**: Viacheslav Pirogov, Maksim Artemev  

**Link**: [PDF](https://arxiv.org/pdf/2507.21905)  

**Abstract**: Deepfakes powered by advanced machine learning models present a significant and evolving threat to identity verification and the authenticity of digital media. Although numerous detectors have been developed to address this problem, their effectiveness has yet to be tested when applied to real-world data. In this work we evaluate modern deepfake detectors, introducing a novel testing procedure designed to mimic real-world scenarios for deepfake detection. Using state-of-the-art deepfake generation methods, we create a comprehensive dataset containing more than 500,000 high-quality deepfake images. Our analysis shows that detecting deepfakes still remains a challenging task. The evaluation shows that in fewer than half of the deepfake detectors tested achieved an AUC score greater than 60%, with the lowest being 50%. We demonstrate that basic image manipulations, such as JPEG compression or image enhancement, can significantly reduce model performance. All code and data are publicly available at this https URL. 

**Abstract (ZH)**: 由先进机器学习模型驱动的Deepfake对身份验证和数字媒体真实性构成了一个重要的且不断演化的威胁。尽管已经开发出许多检测工具来应对这一问题，但它们在实际数据中的有效性尚未得到测试。在本文中，我们评估了现代Deepfake检测器，引入了一种新的测试程序，旨在模拟Deepfake检测的实际场景。借助最先进的Deepfake生成方法，我们创建了一个包含超过500,000张高质量Deepfake图像的综合数据集。我们的分析表明，检测Deepfake仍然是一项具有挑战性的任务。评估结果显示，在测试的Deepfake检测器中，不到一半的检测器实现了AUC评分大于60%，最低得分为50%。我们证明，基本的图像处理操作，如JPEG压缩或图像增强，可以显著降低模型性能。所有代码和数据均可在此网址访问。 

---
# Data-driven quantum Koopman method for simulating nonlinear dynamics 

**Title (ZH)**: 数据驱动的量子Koopman方法用于模拟非线性动力学 

**Authors**: Baoyang Zhang, Zhen Lu, Yaomin Zhao, Yue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21890)  

**Abstract**: Quantum computation offers potential exponential speedups for simulating certain physical systems, but its application to nonlinear dynamics is inherently constrained by the requirement of unitary evolution. We propose the quantum Koopman method (QKM), a data-driven framework that bridges this gap through transforming nonlinear dynamics into linear unitary evolution in higher-dimensional observable spaces. Leveraging the Koopman operator theory to achieve a global linearization, our approach maps system states into a hierarchy of Hilbert spaces using a deep autoencoder. Within the linearized embedding spaces, the state representation is decomposed into modulus and phase components, and the evolution is governed by a set of unitary Koopman operators that act exclusively on the phase. These operators are constructed from diagonal Hamiltonians with coefficients learned from data, a structure designed for efficient implementation on quantum hardware. This architecture enables direct multi-step prediction, and the operator's computational complexity scales logarithmically with the observable space dimension. The QKM is validated across diverse nonlinear systems. Its predictions maintain relative errors below 6% for reaction-diffusion systems and shear flows, and capture key statistics in 2D turbulence. This work establishes a practical pathway for quantum-accelerated simulation of nonlinear phenomena, exploring a framework built on the synergy between deep learning for global linearization and quantum algorithms for unitary dynamics evolution. 

**Abstract (ZH)**: 量子计算为模拟某些物理系统提供了潜在的指数级加速，但在应用于非线性动力学时，其应用受到单位态演化要求的固有约束。我们提出了量子库蒙方法（QKM），这是一种数据驱动的框架，通过将非线性动力学转换为高层可观测空间中的线性单位态演化来弥合这一差距。利用库蒙算子理论实现全局线性化，我们的方法使用深度自编码器将系统状态映射到希尔伯特空间层次结构中。在线性嵌入空间中，状态表示分解为幅度和相位分量，演化由作用于相位的单位态库蒙算子集控制。这些算子由从数据中学习系数的对角哈密顿量构造而成，这一结构设计用于在量子硬件上高效实现。该架构支持直接多步预测，而操作符的计算复杂度与可观测空间维数呈对数关系。QKM已在多种非线性系统中得到验证。其预测反应扩散系统和剪切流的相对误差保持在6%以下，并在二维湍流中捕捉到关键统计量。这项工作为非线性现象的量子加速模拟提供了一条实际途径，探索了深度学习为全局线性化和量子算法为单位态动力学演化之间的协同作用构建的框架。 

---
# Against racing to AGI: Cooperation, deterrence, and catastrophic risks 

**Title (ZH)**: 反对急于迈向AGI：合作、威慑与灾难性风险 

**Authors**: Leonard Dung, Max Hellrigel-Holderbaum  

**Link**: [PDF](https://arxiv.org/pdf/2507.21839)  

**Abstract**: AGI Racing is the view that it is in the self-interest of major actors in AI development, especially powerful nations, to accelerate their frontier AI development to build highly capable AI, especially artificial general intelligence (AGI), before competitors have a chance. We argue against AGI Racing. First, the downsides of racing to AGI are much higher than portrayed by this view. Racing to AGI would substantially increase catastrophic risks from AI, including nuclear instability, and undermine the prospects of technical AI safety research to be effective. Second, the expected benefits of racing may be lower than proponents of AGI Racing hold. In particular, it is questionable whether winning the race enables complete domination over losers. Third, international cooperation and coordination, and perhaps carefully crafted deterrence measures, constitute viable alternatives to racing to AGI which have much smaller risks and promise to deliver most of the benefits that racing to AGI is supposed to provide. Hence, racing to AGI is not in anyone's self-interest as other actions, particularly incentivizing and seeking international cooperation around AI issues, are preferable. 

**Abstract (ZH)**: AGI竞赛的观点认为，在AI发展中的主要参与者，尤其是强大的国家，有动力加速前沿AI开发以建立高度 capable 的AI，尤其是人工通用智能（AGI），并在竞争对手之前占优。我们反对这一观点。首先，竞赛至AGI的负面风险远高于该观点所展现的水平。竞赛至AGI将显著增加AI带来的灾难性风险，包括核不稳定性，并损害技术AI安全性研究的有效性。其次，竞赛所带来的预期益处可能低于AGI竞赛倡导者所认为的水平。特别是，赢得竞赛是否能完全控制失败者存在疑问。第三，国际间合作与协调，以及可能精心设计的威慑措施，是竞赛至AGI的有效替代方案，风险更小，并有望提供竞赛至AGI所承诺的大部分益处。因此，竞赛至AGI并非符合任何一方利益的行为，特别是激励并寻求在AI问题上的国际间合作，更为优选。 

---
# Analysis of Fourier Neural Operators via Effective Field Theory 

**Title (ZH)**: Fourier神经算子的有效场论分析 

**Authors**: Taeyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.21833)  

**Abstract**: Fourier Neural Operators (FNOs) have emerged as leading surrogates for high-dimensional partial-differential equations, yet their stability, generalization and frequency behavior lack a principled explanation. We present the first systematic effective-field-theory analysis of FNOs in an infinite-dimensional function space, deriving closed recursion relations for the layer kernel and four-point vertex and then examining three practically important settings-analytic activations, scale-invariant cases and architectures with residual connections. The theory shows that nonlinear activations inevitably couple frequency inputs to high-frequency modes that are otherwise discarded by spectral truncation, and experiments confirm this frequency transfer. For wide networks we obtain explicit criticality conditions on the weight-initialization ensemble that keep small input perturbations to have uniform scale across depth, and empirical tests validate these predictions. Taken together, our results quantify how nonlinearity enables neural operators to capture non-trivial features, supply criteria for hyper-parameter selection via criticality analysis, and explain why scale-invariant activations and residual connections enhance feature learning in FNOs. 

**Abstract (ZH)**: Fourier神经算子（FNOs）已成为高维偏微分方程的主要代理模型，然而它们的稳定性、泛化能力和频率行为缺乏原理性的解释。我们首次系统地在无限维函数空间中分析FNOs的有效场理论，导出了层核和四点顶点的封闭递归关系，然后探讨了三个实际重要场景：解析激活函数、标度不变情况以及具有残差连接的架构。理论表明，非线性激活函数不可避免地将频率输入耦合到通过光谱截断丢弃的高频率模式中，实验也证实了这种频率转移。对于宽网络，我们得到了保持小输入扰动在深度上均匀尺度的具体临界条件，并实证测试验证了这些预测。综合我们的研究结果，量化了非线性如何使神经算子能够捕捉非平凡特性，提出了通过临界性分析选择超参数的准则，并解释了为何标度不变激活函数和残差连接在FNOs中增强特征学习。 

---
# Unlocking Interpretability for RF Sensing: A Complex-Valued White-Box Transformer 

**Title (ZH)**: 解锁RF传感的可解释性：一种复值白盒Transformer 

**Authors**: Xie Zhang, Yina Wang, Chenshu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21799)  

**Abstract**: The empirical success of deep learning has spurred its application to the radio-frequency (RF) domain, leading to significant advances in Deep Wireless Sensing (DWS). However, most existing DWS models function as black boxes with limited interpretability, which hampers their generalizability and raises concerns in security-sensitive physical applications. In this work, inspired by the remarkable advances of white-box transformers, we present RF-CRATE, the first mathematically interpretable deep network architecture for RF sensing, grounded in the principles of complex sparse rate reduction. To accommodate the unique RF signals, we conduct non-trivial theoretical derivations that extend the original real-valued white-box transformer to the complex domain. By leveraging the CR-Calculus framework, we successfully construct a fully complex-valued white-box transformer with theoretically derived self-attention and residual multi-layer perceptron modules. Furthermore, to improve the model's ability to extract discriminative features from limited wireless data, we introduce Subspace Regularization, a novel regularization strategy that enhances feature diversity, resulting in an average performance improvement of 19.98% across multiple sensing tasks. We extensively evaluate RF-CRATE against seven baselines with multiple public and self-collected datasets involving different RF signals. The results show that RF-CRATE achieves performance on par with thoroughly engineered black-box models, while offering full mathematical interpretability. More importantly, by extending CRATE to the complex domain, RF-CRATE yields substantial improvements, achieving an average classification gain of 5.08% and reducing regression error by 10.34% across diverse sensing tasks compared to CRATE. RF-CRATE is fully open-sourced at: this https URL. 

**Abstract (ZH)**: 射频CRATE：基于复稀疏率降低的可解释深度网络架构 

---
# Learning Kinetic Monte Carlo stochastic dynamics with Deep Generative Adversarial Networks 

**Title (ZH)**: 使用深度生成对抗网络学习动力学蒙特卡洛随机动力学 

**Authors**: Daniele Lanzoni, Olivier Pierre-Louis, Roberto Bergamaschini, Francesco Montalenti  

**Link**: [PDF](https://arxiv.org/pdf/2507.21763)  

**Abstract**: We show that Generative Adversarial Networks (GANs) may be fruitfully exploited to learn stochastic dynamics, surrogating traditional models while capturing thermal fluctuations. Specifically, we showcase the application to a two-dimensional, many-particle system, focusing on surface-step fluctuations and on the related time-dependent roughness. After the construction of a dataset based on Kinetic Monte Carlo simulations, a conditional GAN is trained to propagate stochastically the state of the system in time, allowing the generation of new sequences with a reduced computational cost. Modifications with respect to standard GANs, which facilitate convergence and increase accuracy, are discussed. The trained network is demonstrated to quantitatively reproduce equilibrium and kinetic properties, including scaling laws, with deviations of a few percent from the exact value. Extrapolation limits and future perspectives are critically discussed. 

**Abstract (ZH)**: 我们展示了生成式对抗网络（GANs）可以有效地用于学习随机动力学，替代传统模型并捕捉热波动。具体而言，我们展示了其在二维多粒子系统中的应用，重点关注表面台阶的波动以及相关的时变粗糙度。基于基于动力蒙特卡洛模拟构建的数据集，我们训练了一个条件GAN来随时间推进系统的状态，从而生成新的序列并降低计算成本。关于标准GANs的改进之处，有助于提高收敛性和准确性。训练后的网络被证明能够定量再现平衡和动力学性质，偏差几百分点。我们批判性地讨论了其外推限制和未来展望。 

---
# LiteFat: Lightweight Spatio-Temporal Graph Learning for Real-Time Driver Fatigue Detection 

**Title (ZH)**: LiteFat: 轻量级时空图学习及其在实时驾驶疲劳检测中的应用 

**Authors**: Jing Ren, Suyu Ma, Hong Jia, Xiwei Xu, Ivan Lee, Haytham Fayek, Xiaodong Li, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2507.21756)  

**Abstract**: Detecting driver fatigue is critical for road safety, as drowsy driving remains a leading cause of traffic accidents. Many existing solutions rely on computationally demanding deep learning models, which result in high latency and are unsuitable for embedded robotic devices with limited resources (such as intelligent vehicles/cars) where rapid detection is necessary to prevent accidents. This paper introduces LiteFat, a lightweight spatio-temporal graph learning model designed to detect driver fatigue efficiently while maintaining high accuracy and low computational demands. LiteFat involves converting streaming video data into spatio-temporal graphs (STG) using facial landmark detection, which focuses on key motion patterns and reduces unnecessary data processing. LiteFat uses MobileNet to extract facial features and create a feature matrix for the STG. A lightweight spatio-temporal graph neural network is then employed to identify signs of fatigue with minimal processing and low latency. Experimental results on benchmark datasets show that LiteFat performs competitively while significantly decreasing computational complexity and latency as compared to current state-of-the-art methods. This work enables the development of real-time, resource-efficient human fatigue detection systems that can be implemented upon embedded robotic devices. 

**Abstract (ZH)**: 检测驾驶员疲劳对于道路安全至关重要，因为困倦驾驶仍然是导致交通事故的主要原因之一。许多现有解决方案依赖于计算需求高的深度学习模型，这导致了高延迟，并不适合资源有限（如智能车辆/汽车）的嵌入式机器人设备，而这些设备需要快速检测以防止事故。本文介绍了一种轻量级时空图学习模型LiteFat，旨在高效地检测驾驶员疲劳同时保持高准确性和低计算需求。LiteFat涉及使用面部特征点检测将流式视频数据转换为时空图（STG），并专注于关键运动模式以减少不必要的数据处理。LiteFat使用MobileNet提取面部特征并创建STG的特征矩阵。然后使用轻量级时空图神经网络以最少的处理和低延迟来识别疲劳迹象。基准数据集上的实验结果表明，与当前最先进的方法相比，LiteFat在显著降低计算复杂性和延迟的同时保持了竞争力。此项工作使开发实时、资源高效的人体疲劳检测系统成为可能，并可在嵌入式机器人设备上实现。 

---
# Zero-Shot Machine Unlearning with Proxy Adversarial Data Generation 

**Title (ZH)**: 零样本机器去学习与代理对抗数据生成 

**Authors**: Huiqiang Chen, Tianqing Zhu, Xin Yu, Wanlei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.21738)  

**Abstract**: Machine unlearning aims to remove the influence of specific samples from a trained model. A key challenge in this process is over-unlearning, where the model's performance on the remaining data significantly drops due to the change in the model's parameters. Existing unlearning algorithms depend on the remaining data to prevent this issue. As such, these methods are inapplicable in a more practical scenario, where only the unlearning samples are available (i.e., zero-shot unlearning). This paper presents a novel framework, ZS-PAG, to fill this gap. Our approach offers three key innovations: (1) we approximate the inaccessible remaining data by generating adversarial samples; (2) leveraging the generated samples, we pinpoint a specific subspace to perform the unlearning process, therefore preventing over-unlearning in the challenging zero-shot scenario; and (3) we consider the influence of the unlearning process on the remaining samples and design an influence-based pseudo-labeling strategy. As a result, our method further improves the model's performance after unlearning. The proposed method holds a theoretical guarantee, and experiments on various benchmarks validate the effectiveness and superiority of our proposed method over several baselines. 

**Abstract (ZH)**: 零样本机器遗忘框架ZS-PAG的研究 

---
# Detection Transformers Under the Knife: A Neuroscience-Inspired Approach to Ablations 

**Title (ZH)**: Detection Transformers 在显微刀下：一种受神经科学启发的消融方法 

**Authors**: Nils Hütten, Florian Hölken, Hasan Tercan, Tobias Meisen  

**Link**: [PDF](https://arxiv.org/pdf/2507.21723)  

**Abstract**: In recent years, Explainable AI has gained traction as an approach to enhancing model interpretability and transparency, particularly in complex models such as detection transformers. Despite rapid advancements, a substantial research gap remains in understanding the distinct roles of internal components - knowledge that is essential for improving transparency and efficiency. Inspired by neuroscientific ablation studies, which investigate the functions of brain regions through selective impairment, we systematically analyze the impact of ablating key components in three state-of-the-art detection transformer models: Detection transformer (DETR), deformable detection transformer (DDETR), and DETR with improved denoising anchor boxes (DINO). The ablations target query embeddings, encoder and decoder multi-head self-attentions (MHSA) as well as decoder multi-head cross-attention (MHCA) layers. We evaluate the effects of these ablations on the performance metrics gIoU and F1-score, quantifying effects on both the classification and regression sub-tasks on the COCO dataset. To facilitate reproducibility and future research, we publicly release the DeepDissect library. Our findings reveal model-specific resilience patterns: while DETR is particularly sensitive to ablations in encoder MHSA and decoder MHCA, DDETR's multi-scale deformable attention enhances robustness, and DINO exhibits the greatest resilience due to its look-forward twice update rule, which helps distributing knowledge across blocks. These insights also expose structural redundancies, particularly in DDETR's and DINO's decoder MHCA layers, highlighting opportunities for model simplification without sacrificing performance. This study advances XAI for DETRs by clarifying the contributions of internal components to model performance, offering insights to optimize and improve transparency and efficiency in critical applications. 

**Abstract (ZH)**: 近年来，可解释AI在提高模型可解释性和透明度方面取得了进展，尤其是在检测变换器等复杂模型中。尽管取得了 rapid advancements，但在理解内部组件的独特作用方面仍然存在显著的研究空白，这对于提高透明度和效率至关重要。受神经科学消融研究的启发，这些研究通过选择性损伤来研究大脑区域的功能，我们系统地分析了在三种先进的检测变换器模型（Detection Transformer (DETR)、Deformable Detection Transformer (DDETR) 和 Improved Denoising Anchor Boxes for DETR (DINO)）中消融关键组件的影响。消融实验针对查询嵌入、编码器和解码器的多头自注意力（MHSA）以及解码器的多头交叉注意力（MHCA）层。我们评估了这些消融对gIoU和F1分数等性能指标的影响，量化了分类和回归子任务在COCO数据集上的影响。为了便于再现性和未来研究，我们公开发布了DeepDissect库。我们的研究揭示了模型特定的抗消融模式：虽然DETR对编码器MHSA和解码器MHCA的消融特别敏感，但DDETR的多尺度可变形注意力增强了鲁棒性，而DINO则表现出最大的抗消融性，这归因于其向前更新两次的规则，该规则有助于知识在各块间的分布。这些见解还揭示了结构冗余，特别是在DDETR和DINO的解码器MHCA层中发现，突显了在不牺牲性能的情况下简化模型的机会。该研究通过阐明内部组件对模型性能的贡献，推进了检测变换器的可解释AI，为优化和改进关键应用中的透明度和效率提供了洞察。 

---
# EnTao-GPM: DNA Foundation Model for Predicting the Germline Pathogenic Mutations 

**Title (ZH)**: EnTao-GPM: DNA 基础模型用于预测遗传致病变异 

**Authors**: Zekai Lin, Haoran Sun, Yucheng Guo, Yujie Yang, Yanwen Wang, Bozhen Hu, Chonghang Ye, Qirong Yang, Fan Zhong, Xiaoming Zhang, Lei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21706)  

**Abstract**: Distinguishing pathogenic mutations from benign polymorphisms remains a critical challenge in precision medicine. EnTao-GPM, developed by Fudan University and BioMap, addresses this through three innovations: (1) Cross-species targeted pre-training on disease-relevant mammalian genomes (human, pig, mouse), leveraging evolutionary conservation to enhance interpretation of pathogenic motifs, particularly in non-coding regions; (2) Germline mutation specialization via fine-tuning on ClinVar and HGMD, improving accuracy for both SNVs and non-SNVs; (3) Interpretable clinical framework integrating DNA sequence embeddings with LLM-based statistical explanations to provide actionable insights. Validated against ClinVar, EnTao-GPM demonstrates superior accuracy in mutation classification. It revolutionizes genetic testing by enabling faster, more accurate, and accessible interpretation for clinical diagnostics (e.g., variant assessment, risk identification, personalized treatment) and research, advancing personalized medicine. 

**Abstract (ZH)**: 从良性多态性中区分致病变异仍然是精准医学中的一个关键挑战。复旦大学和BioMap开发的EnTao-GPM通过三项创新解决了这一问题：(1) 跨物种靶向预训练于与疾病相关的哺乳动物基因组（人类、猪、小鼠），利用进化保守性增强对致病变异体的解释，特别是在非编码区域；(2) 通过在ClinVar和HGMD上进行微调专门化于生殖细胞突变，提高对单核苷酸变异和非单核苷酸变异的准确性；(3) 可解释的临床框架，将DNA序列嵌入与基于LLM的统计解释相结合，提供可操作的见解。EnTao-GPM在ClinVar上验证显示了在突变分类中的优越准确性。它通过使遗传测试更快、更准确和更有可及性，革命性地改变了临床诊断（如变异评估、风险识别、个性化治疗）和研究，推进了个性化医学。 

---
# diffSPH: Differentiable Smoothed Particle Hydrodynamics for Adjoint Optimization and Machine Learning 

**Title (ZH)**: diffSPH: 可微光滑粒子动力学在伴随优化和机器学习中的应用 

**Authors**: Rene Winchenbach, Nils Thuerey  

**Link**: [PDF](https://arxiv.org/pdf/2507.21684)  

**Abstract**: We present diffSPH, a novel open-source differentiable Smoothed Particle Hydrodynamics (SPH) framework developed entirely in PyTorch with GPU acceleration. diffSPH is designed centrally around differentiation to facilitate optimization and machine learning (ML) applications in Computational Fluid Dynamics~(CFD), including training neural networks and the development of hybrid models. Its differentiable SPH core, and schemes for compressible (with shock capturing and multi-phase flows), weakly compressible (with boundary handling and free-surface flows), and incompressible physics, enable a broad range of application areas. We demonstrate the framework's unique capabilities through several applications, including addressing particle shifting via a novel, target-oriented approach by minimizing physical and regularization loss terms, a task often intractable in traditional solvers. Further examples include optimizing initial conditions and physical parameters to match target trajectories, shape optimization, implementing a solver-in-the-loop setup to emulate higher-order integration, and demonstrating gradient propagation through hundreds of full simulation steps. Prioritizing readability, usability, and extensibility, this work offers a foundational platform for the CFD community to develop and deploy novel neural networks and adjoint optimization applications. 

**Abstract (ZH)**: 我们介绍diffSPH：一种完全基于PyTorch并利用GPU加速的新型开放源代码可微分光滑粒子流体力学（SPH）框架，及其在计算流体力学（CFD）中的优化和机器学习应用。 

---
# AI Literacy as a Key Driver of User Experience in AI-Powered Assessment: Insights from Socratic Mind 

**Title (ZH)**: AI素养作为AI驱动评估中用户经验的关键驱动因素：苏格拉底思维的洞见 

**Authors**: Meryem Yilmaz Soylu, Jeonghyun Lee, Jui-Tse Hung, Christopher Zhang Cui, David A. Joyner  

**Link**: [PDF](https://arxiv.org/pdf/2507.21654)  

**Abstract**: As Artificial Intelligence (AI) tools become increasingly embedded in higher education, understanding how students interact with these systems is essential to supporting effective learning. This study examines how students' AI literacy and prior exposure to AI technologies shape their perceptions of Socratic Mind, an interactive AI-powered formative assessment tool. Drawing on Self-Determination Theory and user experience research, we analyze relationships among AI literacy, perceived usability, satisfaction, engagement, and perceived learning effectiveness. Data from 309 undergraduates in Computer Science and Business courses were collected through validated surveys. Partial least squares structural equation modeling showed that AI literacy - especially self-efficacy, conceptual understanding, and application skills - significantly predicts usability, satisfaction, and engagement. Usability and satisfaction, in turn, strongly predict perceived learning effectiveness, while prior AI exposure showed no significant effect. These findings highlight that AI literacy, rather than exposure alone, shapes student experiences. Designers should integrate adaptive guidance and user-centered features to support diverse literacy levels, fostering inclusive, motivating, and effective AI-based learning environments. 

**Abstract (ZH)**: 随着人工智能（AI）工具在高等教育中的广泛应用，理解学生与这些系统交互的方式对于支持有效的学习至关重要。本研究探讨了学生的AI素养和对AI技术的先前接触如何影响他们对Socratic Mind的认知，这是一种交互式AI驱动的形式化评估工具。基于自我决定理论和用户经验研究，我们分析了AI素养、感知可用性、满意度、参与度和感知学习有效性之间的关系。通过使用经过验证的问卷收集了309名计算机科学和商学课程学生的数据。部分最小二乘结构方程模型显示，AI素养——尤其是自我效能感、概念理解和应用技能——显著预测了可用性、满意度和参与度。而可用性与满意度又强烈预测了感知学习有效性，而先前的AI接触则没有显著影响。这些发现强调，与单一的接触相比，AI素养塑造了学生的学习体验。设计师应整合自适应指导和用户中心的功能，以支持多样化的素养水平，促进包容性、激励性和有效的基于AI的学习环境。 

---
# GUARD-CAN: Graph-Understanding and Recurrent Architecture for CAN Anomaly Detection 

**Title (ZH)**: GUARD-CAN: 基于图理解的循环架构用于CAN异常检测 

**Authors**: Hyeong Seon Kim, Huy Kang Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.21640)  

**Abstract**: Modern in-vehicle networks face various cyber threats due to the lack of encryption and authentication in the Controller Area Network (CAN). To address this security issue, this paper presents GUARD-CAN, an anomaly detection framework that combines graph-based representation learning with time-series modeling. GUARD-CAN splits CAN messages into fixed-length windows and converts each window into a graph that preserves message order. To detect anomalies in the timeaware and structure-aware context at the same window, GUARD-CAN takes advantage of the overcomplete Autoencoder (AE) and Graph Convolutional Network (GCN) to generate graph embedding vectors. The model groups these vectors into sequences and feeds them into the Gated Recurrent Unit (GRU) to detect temporal anomaly patterns across the graphs. GUARD-CAN performs anomaly detection at both the sequence level and the window level, and this allows multi-perspective performance evaluation. The model also verifies the importance of window size selection through an analysis based on Shannon entropy. As a result, GUARD-CAN shows that the proposed model detects four types of CAN attacks (flooding, fuzzing, replay and spoofing attacks) effectively without relying on complex feature engineering. 

**Abstract (ZH)**: GUARD-CAN：一种结合图表示学习与时间序列建模的异常检测框架 

---
# Hierarchical Graph Neural Network for Compressed Speech Steganalysis 

**Title (ZH)**: 分层图神经网络在压缩语音隐写分析中的应用 

**Authors**: Mustapha Hemis, Hamza Kheddar, Mohamed Chahine Ghanem, Bachir Boudraa  

**Link**: [PDF](https://arxiv.org/pdf/2507.21591)  

**Abstract**: Steganalysis methods based on deep learning (DL) often struggle with computational complexity and challenges in generalizing across different datasets. Incorporating a graph neural network (GNN) into steganalysis schemes enables the leveraging of relational data for improved detection accuracy and adaptability. This paper presents the first application of a Graph Neural Network (GNN), specifically the GraphSAGE architecture, for steganalysis of compressed voice over IP (VoIP) speech streams. The method involves straightforward graph construction from VoIP streams and employs GraphSAGE to capture hierarchical steganalysis information, including both fine grained details and high level patterns, thereby achieving high detection accuracy. Experimental results demonstrate that the developed approach performs well in uncovering quantization index modulation (QIM)-based steganographic patterns in VoIP signals. It achieves detection accuracy exceeding 98 percent even for short 0.5 second samples, and 95.17 percent accuracy under challenging conditions with low embedding rates, representing an improvement of 2.8 percent over the best performing state of the art methods. Furthermore, the model exhibits superior efficiency, with an average detection time as low as 0.016 seconds for 0.5-second samples an improvement of 0.003 seconds. This makes it efficient for online steganalysis tasks, providing a superior balance between detection accuracy and efficiency under the constraint of short samples with low embedding rates. 

**Abstract (ZH)**: 基于深度学习的隐写分析方法往往面临计算复杂度高和跨不同数据集泛化的挑战。将图神经网络（GNN）集成到隐写分析方案中，可以利用关系数据以提高检测准确性和适应性。本文首次将图神经网络（GNN），具体为GraphSAGE架构，应用于压缩VoIP语音流的隐写分析。该方法从VoIP流构建简单图，并利用GraphSAGE捕获分层隐写分析信息，包括细粒度细节和高层次模式，从而实现高检测准确率。实验结果表明，该方法在揭开基于量化指数调制（QIM）的VoIP信号隐写图模式方面表现出色，即使对于0.5秒的短样本，检测准确率也超过98%，在低嵌入率的挑战性条件下，准确率为95.17%，比最佳现有方法高出2.8%。此外，该模型表现出更高的效率，对于0.5秒的样本，平均检测时间为0.016秒，比之前方法快0.003秒，这使其适用于在线隐写分析任务，在短样本和低嵌入率的约束下实现了检测准确率与效率之间的优化平衡。 

---
# Automatic Classification of User Requirements from Online Feedback -- A Replication Study 

**Title (ZH)**: 基于在线反馈的用户需求自动分类——一项复制研究 

**Authors**: Meet Bhatt, Nic Boilard, Muhammad Rehan Chaudhary, Cole Thompson, Jacob Idoko, Aakash Sorathiya, Gouri Ginde  

**Link**: [PDF](https://arxiv.org/pdf/2507.21532)  

**Abstract**: Natural language processing (NLP) techniques have been widely applied in the requirements engineering (RE) field to support tasks such as classification and ambiguity detection. Although RE research is rooted in empirical investigation, it has paid limited attention to replicating NLP for RE (NLP4RE) studies. The rapidly advancing realm of NLP is creating new opportunities for efficient, machine-assisted workflows, which can bring new perspectives and results to the forefront. Thus, we replicate and extend a previous NLP4RE study (baseline), "Classifying User Requirements from Online Feedback in Small Dataset Environments using Deep Learning", which evaluated different deep learning models for requirement classification from user reviews. We reproduced the original results using publicly released source code, thereby helping to strengthen the external validity of the baseline study. We then extended the setup by evaluating model performance on an external dataset and comparing results to a GPT-4o zero-shot classifier. Furthermore, we prepared the replication study ID-card for the baseline study, important for evaluating replication readiness. Results showed diverse reproducibility levels across different models, with Naive Bayes demonstrating perfect reproducibility. In contrast, BERT and other models showed mixed results. Our findings revealed that baseline deep learning models, BERT and ELMo, exhibited good generalization capabilities on an external dataset, and GPT-4o showed performance comparable to traditional baseline machine learning models. Additionally, our assessment confirmed the baseline study's replication readiness; however missing environment setup files would have further enhanced readiness. We include this missing information in our replication package and provide the replication study ID-card for our study to further encourage and support the replication of our study. 

**Abstract (ZH)**: 自然语言处理（NLP）技术在需求工程（RE）领域中的应用：复制和扩展先前的NLP4RE研究 

---
# VN-MTEB: Vietnamese Massive Text Embedding Benchmark 

**Title (ZH)**: VN-MTEB: Vietnamese大规模文本嵌入基准 

**Authors**: Loc Pham, Tung Luu, Thu Vo, Minh Nguyen, Viet Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21500)  

**Abstract**: Vietnam ranks among the top countries in terms of both internet traffic and online toxicity. As a result, implementing embedding models for recommendation and content control duties in applications is crucial. However, a lack of large-scale test datasets, both in volume and task diversity, makes it tricky for scientists to effectively evaluate AI models before deploying them in real-world, large-scale projects. To solve this important problem, we introduce a Vietnamese benchmark, VN-MTEB for embedding models, which we created by translating a large number of English samples from the Massive Text Embedding Benchmark using our new automated framework. We leverage the strengths of large language models (LLMs) and cutting-edge embedding models to conduct translation and filtering processes to retain high-quality samples, guaranteeing a natural flow of language and semantic fidelity while preserving named entity recognition (NER) and code snippets. Our comprehensive benchmark consists of 41 datasets from six tasks specifically designed for Vietnamese text embeddings. In our analysis, we find that bigger and more complex models using Rotary Positional Embedding outperform those using Absolute Positional Embedding in embedding tasks. Datasets are available at HuggingFace: this https URL 

**Abstract (ZH)**: 越南在互联网流量和在线 toxicity 方面名列前茅，因此在应用中实施嵌入模型对于推荐和内容控制至关重要。但由于缺乏大规模的测试数据集，尤其是在数据量和任务多样性方面，科学家在部署这些模型之前难以有效评估 AI 模型。为解决这一重要问题，我们引入了一个越南语基准 VN-MTEB，该基准通过新的自动化框架将大量英语样本翻译而来。我们利用大型语言模型（LLMs）和先进嵌入模型的优势，进行翻译和过滤过程，确保语言自然流畅和语义保真度，同时保留词性识别（NER）和代码片段。我们的基准包括六个任务的 41 个数据集，专门设计用于越南语文本嵌入。我们的分析发现，使用旋转位置嵌入的大而复杂的模型在嵌入任务中优于使用绝对位置嵌入的模型。数据集可在 HuggingFace 获取：this https URL。 

---
# NCCR: to Evaluate the Robustness of Neural Networks and Adversarial Examples 

**Title (ZH)**: NCCR：评估神经网络和对抗样本的鲁棒性 

**Authors**: Pu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21483)  

**Abstract**: Neural networks have received a lot of attention recently, and related security issues have come with it. Many studies have shown that neural networks are vulnerable to adversarial examples that have been artificially perturbed with modification, which is too small to be distinguishable by human perception. Different attacks and defenses have been proposed to solve these problems, but there is little research on evaluating the robustness of neural networks and their inputs. In this work, we propose a metric called the neuron cover change rate (NCCR) to measure the ability of deep learning models to resist attacks and the stability of adversarial examples. NCCR monitors alterations in the output of specifically chosen neurons when the input is perturbed, and networks with a smaller degree of variation are considered to be more robust. The results of the experiment on image recognition and the speaker recognition model show that our metrics can provide a good assessment of the robustness of neural networks or their inputs. It can also be used to detect whether an input is adversarial or not, as adversarial examples are always less robust. 

**Abstract (ZH)**: 神经网络 recently 已经受到了广泛关注，相关的安全问题也随之而来。许多研究表明，神经网络容易受到人工修改且难以被人眼察觉的微小扰动所构造的对抗样本的影响。针对这些问题，已经提出了不同的攻击和防御方法，但鲜有研究评估神经网络及其输入的鲁棒性。在本文中，我们提出了一种称为神经元覆盖变化率（NCCR）的度量标准，用于衡量深度学习模型抵抗攻击的能力和对抗样本的稳定性。NCCR 监控特定选择的神经元在输入被扰动时输出的变化，变化较小的网络被认为更有鲁棒性。图像识别和说话人识别模型的实验结果表明，我们的指标可以提供神经网络或其输入鲁棒性的良好评估。此外，该指标还可以用于检测输入是否为对抗样本，因为对抗样本通常不那么鲁棒。 

---
# Capacity-Constrained Continual Learning 

**Title (ZH)**: 容量约束持续学习 

**Authors**: Zheng Wen, Doina Precup, Benjamin Van Roy, Satinder Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.21479)  

**Abstract**: Any agents we can possibly build are subject to capacity constraints, as memory and compute resources are inherently finite. However, comparatively little attention has been dedicated to understanding how agents with limited capacity should allocate their resources for optimal performance. The goal of this paper is to shed some light on this question by studying a simple yet relevant continual learning problem: the capacity-constrained linear-quadratic-Gaussian (LQG) sequential prediction problem. We derive a solution to this problem under appropriate technical conditions. Moreover, for problems that can be decomposed into a set of sub-problems, we also demonstrate how to optimally allocate capacity across these sub-problems in the steady state. We view the results of this paper as a first step in the systematic theoretical study of learning under capacity constraints. 

**Abstract (ZH)**: 任何我们可能构建的智能体都受到容量约束的限制，因为内存和计算资源是有限的。然而，对于具有有限容量的智能体应如何分配其资源以实现最优性能的研究相对不足。本文旨在通过研究一个简单而相关的一贯学习问题——容量受限的线性二次高斯（LQG）序列预测问题——来回答这个问题。我们在适当的技术条件下推导出该问题的解。此外，对于可以分解为一组子问题的问题，我们还展示了如何在稳态下最优地分配这些子问题的容量。我们将本文的结果视为系统性理论研究学习在容量约束下问题的第一步。 

---
# Hebbian Memory-Augmented Recurrent Networks: Engram Neurons in Deep Learning 

**Title (ZH)**: 基于海BI规则的记忆增强递归网络：深度学习中的记忆神经元 

**Authors**: Daniel Szelogowski  

**Link**: [PDF](https://arxiv.org/pdf/2507.21474)  

**Abstract**: Despite success across diverse tasks, current artificial recurrent network architectures rely primarily on implicit hidden-state memories, limiting their interpretability and ability to model long-range dependencies. In contrast, biological neural systems employ explicit, associative memory traces (i.e., engrams) strengthened through Hebbian synaptic plasticity and activated sparsely during recall. Motivated by these neurobiological insights, we introduce the Engram Neural Network (ENN), a novel recurrent architecture incorporating an explicit, differentiable memory matrix with Hebbian plasticity and sparse, attention-driven retrieval mechanisms. The ENN explicitly models memory formation and recall through dynamic Hebbian traces, improving transparency and interpretability compared to conventional RNN variants. We evaluate the ENN architecture on three canonical benchmarks: MNIST digit classification, CIFAR-10 image sequence modeling, and WikiText-103 language modeling. Our empirical results demonstrate that the ENN achieves accuracy and generalization performance broadly comparable to classical RNN, GRU, and LSTM architectures, with all models converging to similar accuracy and perplexity on the large-scale WikiText-103 task. At the same time, the ENN offers significant enhancements in interpretability through observable memory dynamics. Hebbian trace visualizations further reveal biologically plausible, structured memory formation processes, validating the potential of neuroscience-inspired mechanisms to inform the development of more interpretable and robust deep learning models. 

**Abstract (ZH)**: 尽管当前的人工循环网络架构在多种任务上取得了成功，但它们主要依赖于隐式的隐藏状态记忆，这限制了它们的可解释性和建模长距离依赖的能力。相比之下，生物神经系统通过瞬时记忆痕迹（即记忆回路）来存储信息，这些记忆回路通过Hebbian突触可塑性加强，并在回忆时稀疏激活。受这些神经生物学洞察的启发，我们引入了记忆回路神经网络（ENN），这是一种新颖的循环架构，结合了隐式的、可微的记忆矩阵、Hebbian可塑性以及基于注意力的选择性检索机制。ENN 通过动态的Hebbian痕迹明确建模记忆形成和回忆，从而在透明度和可解释性方面优于传统的循环神经网络变体。我们在三个经典基准上评估了ENN 架构：MNIST 数字分类、CIFAR-10 图像序列建模和 WikiText-103 语言建模。我们的实证结果表明，ENN 在准确率和泛化性能方面与经典的循环神经网络、门控循环单元（GRU）和长短期记忆（LSTM）架构相当，在大规模的 WikiText-103 任务上，所有模型的准确率和困惑度都达到相似水平。同时，ENN 在可解释性方面提供了显著的增强，通过可观察的记忆动态展示了生物上合乎逻辑的、结构化的记忆形成过程，进一步证实了神经科学启发机制在开发更可解释和稳健的深度学习模型方面的潜力。 

---
# Boost Self-Supervised Dataset Distillation via Parameterization, Predefined Augmentation, and Approximation 

**Title (ZH)**: 通过参数化、预定义增强和近似增强自主监督数据集精炼 

**Authors**: Sheng-Feng Yu, Jia-Jiun Yao, Wei-Chen Chiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21455)  

**Abstract**: Although larger datasets are crucial for training large deep models, the rapid growth of dataset size has brought a significant challenge in terms of considerable training costs, which even results in prohibitive computational expenses. Dataset Distillation becomes a popular technique recently to reduce the dataset size via learning a highly compact set of representative exemplars, where the model trained with these exemplars ideally should have comparable performance with respect to the one trained with the full dataset. While most of existing works upon dataset distillation focus on supervised datasets, we instead aim to distill images and their self-supervisedly trained representations into a distilled set. This procedure, named as Self-Supervised Dataset Distillation, effectively extracts rich information from real datasets, yielding the distilled sets with enhanced cross-architecture generalizability. Particularly, in order to preserve the key characteristics of original dataset more faithfully and compactly, several novel techniques are proposed: 1) we introduce an innovative parameterization upon images and representations via distinct low-dimensional bases, where the base selection for parameterization is experimentally shown to play a crucial role; 2) we tackle the instability induced by the randomness of data augmentation -- a key component in self-supervised learning but being underestimated in the prior work of self-supervised dataset distillation -- by utilizing predetermined augmentations; 3) we further leverage a lightweight network to model the connections among the representations of augmented views from the same image, leading to more compact pairs of distillation. Extensive experiments conducted on various datasets validate the superiority of our approach in terms of distillation efficiency, cross-architecture generalization, and transfer learning performance. 

**Abstract (ZH)**: 虽然 Larger Datasets 对于训练大模型至关重要，但数据集规模的快速增长带来了显著的挑战，导致计算成本大幅增加，甚至可能导致计算资源的极大消耗。数据集蒸馏已成为一种流行的技术，通过学习一组高度紧凑的代表样本来减少数据集规模，其中使用这些样本训练的模型理想情况下应与全数据集训练的模型具有可比的性能。尽管现有大多数组织在数据集蒸馏方面的研究主要集中在监督数据集上，但我们旨在将图像及其自监督训练的表示蒸馏到一个蒸馏集中。这一过程被称为自监督数据集蒸馏，有效提取了真实数据集中的丰富信息，生成了具有增强跨体系结构泛化能力的蒸馏集。特别地，为了更忠实且紧凑地保留原始数据集的关键特征，提出了一些新型技术：1) 通过不同的低维度基底对图像和表示进行创新参数化，其中参数化基底的选择在实验中被证明起到了关键作用；2) 通过利用预先确定的增强方法来应对自监督学习中关键组成部分数据增强引发的不稳定性——而在现有的自监督数据集蒸馏工作中对此给予了低估；3) 进一步利用轻量级网络建模同一图像的不同视角表示之间的连接，从而生成更紧凑的蒸馏对。在各种数据集上进行的广泛实验验证了我们在蒸馏效率、跨体系结构泛化能力和迁移学习性能方面的优越性。 

---
# MemShare: Memory Efficient Inference for Large Reasoning Models through KV Cache Reuse 

**Title (ZH)**: MemShare: 通过键值缓存复用的大规模推理模型高效内存推理 

**Authors**: Kaiwen Chen, Xin Tan, Minchen Yu, Hong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21433)  

**Abstract**: Large Reasoning Models (LRMs) have achieved significant advances in mathematical reasoning and formal logic tasks. However, their tendency to generate lengthy chain-of-thought sequences leads to substantial memory overhead during inference. We observe that LRMs frequently produce highly similar intermediate reasoning steps, which correspond to similar KV cache states across layers. Motivated by this observation, we propose MemShare, a novel KV cache management approach that effectively reduces memory overhead. MemShare employs a collaborative filtering algorithm to efficiently identify reusable KV cache blocks and enables zero copy cache reuse to significantly reduce memory overhead, improve throughput while maintaining accuracy. Experimental results demonstrate that MemShare delivers up to 84.79\% improvement in throughput while maintaining better accuracy compared to existing KV cache management methods. 

**Abstract (ZH)**: 大型推理模型（LRMs）在数学推理和形式逻辑任务上取得了显著进展。然而，它们倾向于生成较长的推理链，导致推理过程中内存开销显著增加。我们观察到LRMs经常产生高度相似的中间推理步骤，这些步骤对应于各层中相似的KV缓存状态。受此观察启发，我们提出了MemShare，一种新颖的KV缓存管理方法，有效减少了内存开销。MemShare采用协作过滤算法高效地识别可重复使用的KV缓存块，并通过零拷贝缓存重用显著减少内存开销、提高吞吐量的同时保持准确性。实验结果显示，与现有KV缓存管理方法相比，MemShare在保持更高准确性的基础上，吞吐量最多可提高84.79%。 

---
# Efficient Neural Combinatorial Optimization Solver for the Min-max Heterogeneous Capacitated Vehicle Routing Problem 

**Title (ZH)**: 高效的神经组合优化解决方法：针对最小最大异质容量车辆路由问题 

**Authors**: Xuan Wu, Di Wang, Chunguo Wu, Kaifang Qi, Chunyan Miao, Yubin Xiao, Jian Zhang, You Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.21386)  

**Abstract**: Numerous Neural Combinatorial Optimization (NCO) solvers have been proposed to address Vehicle Routing Problems (VRPs). However, most of these solvers focus exclusively on single-vehicle VRP variants, overlooking the more realistic min-max Heterogeneous Capacitated Vehicle Routing Problem (MMHCVRP), which involves multiple vehicles. Existing MMHCVRP solvers typically select a vehicle and its next node to visit at each decoding step, but often make myopic decoding decisions and overlook key properties of MMHCVRP, including local topological relationships, vehicle permutation invariance, and node symmetry, resulting in suboptimal performance. To better address these limitations, we propose ECHO, an efficient NCO solver. First, ECHO exploits the proposed dual-modality node encoder to capture local topological relationships among nodes. Subsequently, to mitigate myopic decisions, ECHO employs the proposed Parameter-Free Cross-Attention mechanism to prioritize the vehicle selected in the preceding decoding step. Finally, leveraging vehicle permutation invariance and node symmetry, we introduce a tailored data augment strategy for MMHCVRP to stabilize the Reinforcement Learning training process. To assess the performance of ECHO, we conduct extensive experiments. The experimental results demonstrate that ECHO outperforms state-of-the-art NCO solvers across varying numbers of vehicles and nodes, and exhibits well-performing generalization across both scales and distribution patterns. Finally, ablation studies validate the effectiveness of all proposed methods. 

**Abstract (ZH)**: 高效的神经组合优化解器ECHO：应对多元车辆异构容量路由问题 

---
# Deep Reinforcement Learning-based Cell DTX/DRX Configuration for Network Energy Saving 

**Title (ZH)**: 基于深度强化学习的小区DTX/DRX配置方法及其在网络节能中的应用 

**Authors**: Wei Mao, Lili Wei, Omid Semiari, Shu-ping Yeh, Hosein Nikopour  

**Link**: [PDF](https://arxiv.org/pdf/2507.21385)  

**Abstract**: 3GPP Release 18 cell discontinuous transmission and reception (cell DTX/DRX) is an important new network energy saving feature for 5G. As a time-domain technique, it periodically aggregates the user data transmissions in a given duration of time when the traffic load is not heavy, so that the remaining time can be kept silent and advanced sleep modes (ASM) can be enabled to shut down more radio components and save more energy for the cell. However, inevitably the packet delay is increased, as during the silent period no transmission is allowed. In this paper we study how to configure cell DTX/DRX to optimally balance energy saving and packet delay, so that for delay-sensitive traffic maximum energy saving can be achieved while the degradation of quality of service (QoS) is minimized. As the optimal configuration can be different for different network and traffic conditions, the problem is complex and we resort to deep reinforcement learning (DRL) framework to train an AI agent to solve it. Through careful design of 1) the learning algorithm, which implements a deep Q-network (DQN) on a contextual bandit (CB) model, and 2) the reward function, which utilizes a smooth approximation of a theoretically optimal but discontinuous reward function, we are able to train an AI agent that always tries to select the best possible Cell DTX/DRX configuration under any network and traffic conditions. Simulation results show that compared to the case when cell DTX/DRX is not used, our agent can achieve up to ~45% energy saving depending on the traffic load scenario, while always maintaining no more than ~1% QoS degradation. 

**Abstract (ZH)**: 3GPPRelease18小区不连续接收传输(cellDTX/DRX):一种5G网络节能的新技术及其能效与包延迟最优平衡的研究 

---
# MAAD: Automate Software Architecture Design through Knowledge-Driven Multi-Agent Collaboration 

**Title (ZH)**: MAAD：通过知识驱动的多Agent协作实现自动化软件架构设计 

**Authors**: Ruiyin Li, Yiran Zhang, Xiyu Zhou, Peng Liang, Weisong Sun, Jifeng Xuan, Zhi Jin, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21382)  

**Abstract**: Software architecture design is a critical, yet inherently complex and knowledge-intensive phase of software development. It requires deep domain expertise, development experience, architectural knowledge, careful trade-offs among competing quality attributes, and the ability to adapt to evolving requirements. Traditionally, this process is time-consuming and labor-intensive, and relies heavily on architects, often resulting in limited design alternatives, especially under the pressures of agile development. While Large Language Model (LLM)-based agents have shown promising performance across various SE tasks, their application to architecture design remains relatively scarce and requires more exploration, particularly in light of diverse domain knowledge and complex decision-making. To address the challenges, we proposed MAAD (Multi-Agent Architecture Design), an automated framework that employs a knowledge-driven Multi-Agent System (MAS) for architecture design. MAAD orchestrates four specialized agents (i.e., Analyst, Modeler, Designer and Evaluator) to collaboratively interpret requirements specifications and produce architectural blueprints enriched with quality attributes-based evaluation reports. We then evaluated MAAD through a case study and comparative experiments against MetaGPT, a state-of-the-art MAS baseline. Our results show that MAAD's superiority lies in generating comprehensive architectural components and delivering insightful and structured architecture evaluation reports. Feedback from industrial architects across 11 requirements specifications further reinforces MAAD's practical usability. We finally explored the performance of the MAAD framework with three LLMs (GPT-4o, DeepSeek-R1, and Llama 3.3) and found that GPT-4o exhibits better performance in producing architecture design, emphasizing the importance of LLM selection in MAS-driven architecture design. 

**Abstract (ZH)**: 基于多智能体系统的软件架构自动化设计（MAAD） 

---
# ProMemAssist: Exploring Timely Proactive Assistance Through Working Memory Modeling in Multi-Modal Wearable Devices 

**Title (ZH)**: ProMemAssist：通过工作记忆建模在多模态可穿戴设备中探索及时主动协助 

**Authors**: Kevin Pu, Ting Zhang, Naveen Sendhilnathan, Sebastian Freitag, Raj Sodhi, Tanya Jonker  

**Link**: [PDF](https://arxiv.org/pdf/2507.21378)  

**Abstract**: Wearable AI systems aim to provide timely assistance in daily life, but existing approaches often rely on user initiation or predefined task knowledge, neglecting users' current mental states. We introduce ProMemAssist, a smart glasses system that models a user's working memory (WM) in real-time using multi-modal sensor signals. Grounded in cognitive theories of WM, our system represents perceived information as memory items and episodes with encoding mechanisms, such as displacement and interference. This WM model informs a timing predictor that balances the value of assistance with the cost of interruption. In a user study with 12 participants completing cognitively demanding tasks, ProMemAssist delivered more selective assistance and received higher engagement compared to an LLM baseline system. Qualitative feedback highlights the benefits of WM modeling for nuanced, context-sensitive support, offering design implications for more attentive and user-aware proactive agents. 

**Abstract (ZH)**: 可穿戴AI系统旨在提供及时生活协助，但现有方法往往依赖于用户主动触发或预定义的任务知识，忽视了用户当前的心理状态。我们介绍了ProMemAssist，这是一种智能眼镜系统，通过多模态传感器信号实时建模用户的短期记忆（工作记忆）。基于工作记忆的认知理论，我们的系统将感知到的信息表示为具有编码机制（如位移和干扰）的记忆项和事件。该工作记忆模型指导一个时间预测器，平衡协助的价值与中断的成本。在一项涉及12名参与者完成认知密集型任务的用户研究中，ProMemAssist提供了更具选择性的协助，并获得了比LLM基线系统更高的参与度。定性反馈突出了工作记忆建模在提供细致、上下文敏感支持方面的优势，为设计更细心、更用户友好的主动代理提供了设计启示。 

---
# StructText: A Synthetic Table-to-Text Approach for Benchmark Generation with Multi-Dimensional Evaluation 

**Title (ZH)**: 结构文本：一种多维度评估的合成表格到文本基准生成方法 

**Authors**: Satyananda Kashyap, Sola Shirai, Nandana Mihindukulasooriya, Horst Samulowitz  

**Link**: [PDF](https://arxiv.org/pdf/2507.21340)  

**Abstract**: Extracting structured information from text, such as key-value pairs that could augment tabular data, is quite useful in many enterprise use cases. Although large language models (LLMs) have enabled numerous automated pipelines for converting natural language into structured formats, there is still a lack of benchmarks for evaluating their extraction quality, especially in specific domains or focused documents specific to a given organization. Building such benchmarks by manual annotations is labour-intensive and limits the size and scalability of the benchmarks. In this work, we present StructText, an end-to-end framework for automatically generating high-fidelity benchmarks for key-value extraction from text using existing tabular data. It uses available tabular data as structured ground truth, and follows a two-stage ``plan-then-execute'' pipeline to synthetically generate corresponding natural-language text. To ensure alignment between text and structured source, we introduce a multi-dimensional evaluation strategy that combines (a) LLM-based judgments on factuality, hallucination, and coherence and (b) objective extraction metrics measuring numeric and temporal accuracy. We evaluated the proposed method on 71,539 examples across 49 datasets. Results reveal that while LLMs achieve strong factual accuracy and avoid hallucination, they struggle with narrative coherence in producing extractable text. Notably, models presume numerical and temporal information with high fidelity yet this information becomes embedded in narratives that resist automated extraction. We release a framework, including datasets, evaluation tools, and baseline extraction systems, to support continued research. 

**Abstract (ZH)**: 从文本中自动生成高保真结构化基准框架用于键值对提取 

---
# Semantic Numeration Systems as Dynamical Systems 

**Title (ZH)**: 语义数制系统作为动力系统 

**Authors**: Alexander Yu. Chunikhin  

**Link**: [PDF](https://arxiv.org/pdf/2507.21295)  

**Abstract**: The foundational concepts of semantic numeration systems theory are briefly outlined. The action of cardinal semantic operators unfolds over a set of cardinal abstract entities belonging to the cardinal semantic multeity. The cardinal abstract object (CAO) formed by them in a certain connectivity topology is proposed to be considered as a linear discrete dynamical system with nonlinear control. Under the assumption of ideal observability, the CAO state equations are provided for both stationary and non-stationary cases. The fundamental role of the configuration matrix, which combines information about the types of cardinal semantic operators in the CAO, their parameters and topology of connectivity, is demonstrated. 

**Abstract (ZH)**: 语义数制理论的基础概念摘要。卡 guit数义算子的作用在卡 guit抽象实体所构成的卡 Guil数义多元体的集合上展开。由它们在某种连接拓扑下形成的卡 guit抽象对象(CAO)被提出作为一个具有非线性控制的线性离散动态系统。假设理想的可观测性，在稳态和非稳态情况下提供了CAO的状态方程。展示了配置矩阵的基本作用，该矩阵结合了CAO中卡 guit数义算子的类型、参数及其连接拓扑的信息。 

---
# Bubbleformer: Forecasting Boiling with Transformers 

**Title (ZH)**: Bubbleformer：基于变压器的沸腾预测 

**Authors**: Sheikh Md Shakeel Hassan, Xianwei Zou, Akash Dhruv, Vishwanath Ganesan, Aparna Chandramowlishwaran  

**Link**: [PDF](https://arxiv.org/pdf/2507.21244)  

**Abstract**: Modeling boiling (an inherently chaotic, multiphase process central to energy and thermal systems) remains a significant challenge for neural PDE surrogates. Existing models require future input (e.g., bubble positions) during inference because they fail to learn nucleation from past states, limiting their ability to autonomously forecast boiling dynamics. They also fail to model flow boiling velocity fields, where sharp interface-momentum coupling demands long-range and directional inductive biases. We introduce Bubbleformer, a transformer-based spatiotemporal model that forecasts stable and long-range boiling dynamics including nucleation, interface evolution, and heat transfer without dependence on simulation data during inference. Bubbleformer integrates factorized axial attention, frequency-aware scaling, and conditions on thermophysical parameters to generalize across fluids, geometries, and operating conditions. To evaluate physical fidelity in chaotic systems, we propose interpretable physics-based metrics that evaluate heat-flux consistency, interface geometry, and mass conservation. We also release BubbleML 2.0, a high-fidelity dataset that spans diverse working fluids (cryogens, refrigerants, dielectrics), boiling configurations (pool and flow boiling), flow regimes (bubbly, slug, annular), and boundary conditions. Bubbleformer sets new benchmark results in both prediction and forecasting of two-phase boiling flows. 

**Abstract (ZH)**: 基于变压器的时空模型Bubbleformer在预测沸腾动态方面的突破：无需仿真数据实现自主预报多相沸腾流 

---
# Learning from Limited and Imperfect Data 

**Title (ZH)**: 从有限且不完美数据中学习 

**Authors**: Harsh Rangwani  

**Link**: [PDF](https://arxiv.org/pdf/2507.21205)  

**Abstract**: The distribution of data in the world (eg, internet, etc.) significantly differs from the well-curated datasets and is often over-populated with samples from common categories. The algorithms designed for well-curated datasets perform suboptimally when used for learning from imperfect datasets with long-tailed imbalances and distribution shifts. To expand the use of deep models, it is essential to overcome the labor-intensive curation process by developing robust algorithms that can learn from diverse, real-world data distributions. Toward this goal, we develop practical algorithms for Deep Neural Networks which can learn from limited and imperfect data present in the real world. This thesis is divided into four segments, each covering a scenario of learning from limited or imperfect data. The first part of the thesis focuses on Learning Generative Models from Long-Tail Data, where we mitigate the mode-collapse and enable diverse aesthetic image generations for tail (minority) classes. In the second part, we enable effective generalization on tail classes through Inductive Regularization schemes, which allow tail classes to generalize as effectively as the head classes without requiring explicit generation of images. In the third part, we develop algorithms for Optimizing Relevant Metrics for learning from long-tailed data with limited annotation (semi-supervised), followed by the fourth part, which focuses on the Efficient Domain Adaptation of the model to various domains with very few to zero labeled samples. 

**Abstract (ZH)**: 世界范围内（如互联网等）的数据分布与精心整理的数据集显著不同，通常包含过多的常见类别样本。为精心整理的数据集设计的算法在用于学习不完美数据集（这些数据集具有长尾不平衡和分布偏移）时表现不佳。为了扩大深度模型的应用范围，必须通过开发能够从多样化的实际数据分布中学习的稳健算法来克服劳动密集型的数据整理过程。为此，我们开发了适用于深度神经网络的实用算法，使其能够从现实世界中有限且不完美的数据中进行学习。本论文分为四个部分，每一部分覆盖一种从有限或不完美数据中学习的场景。第一部分专注于从长尾数据中学习生成模型，其中我们缓解了模式坍缩并为尾部（少数）类别生成多样化的美学图像。第二部分通过归纳正则化方案使尾部类别能够有效泛化，无需显式生成图像即可实现与头部类别相同的泛化效果。第三部分开发了适用于具有有限标注（半监督）长尾数据学习的优化相关度量算法。第四部分专注于高效地将模型迁移到各种领域，即使只有少量或没有标注样本也能实现适应。 

---
# EdgeAgentX-DT: Integrating Digital Twins and Generative AI for Resilient Edge Intelligence in Tactical Networks 

**Title (ZH)**: EdgeAgentX-DT：将数字孪生与生成式AI集成以在战术网络中实现灵活边缘智能 

**Authors**: Abir Ray  

**Link**: [PDF](https://arxiv.org/pdf/2507.21196)  

**Abstract**: We introduce EdgeAgentX-DT, an advanced extension of the EdgeAgentX framework that integrates digital twin simulations and generative AI-driven scenario training to significantly enhance edge intelligence in military networks. EdgeAgentX-DT utilizes network digital twins, virtual replicas synchronized with real-world edge devices, to provide a secure, realistic environment for training and validation. Leveraging generative AI methods, such as diffusion models and transformers, the system creates diverse and adversarial scenarios for robust simulation-based agent training. Our multi-layer architecture includes: (1) on-device edge intelligence; (2) digital twin synchronization; and (3) generative scenario training. Experimental simulations demonstrate notable improvements over EdgeAgentX, including faster learning convergence, higher network throughput, reduced latency, and improved resilience against jamming and node failures. A case study involving a complex tactical scenario with simultaneous jamming attacks, agent failures, and increased network loads illustrates how EdgeAgentX-DT sustains operational performance, whereas baseline methods fail. These results highlight the potential of digital-twin-enabled generative training to strengthen edge AI deployments in contested environments. 

**Abstract (ZH)**: EdgeAgentX-DT：一种集成数字孪生仿真和生成AI驱动场景训练的先进边缘代理框架，显著增强军事网络边缘智能 

---
# MaXsive: High-Capacity and Robust Training-Free Generative Image Watermarking in Diffusion Models 

**Title (ZH)**: MaXsive：高容量且鲁棒的无需训练生成式图像水印在扩散模型中的应用 

**Authors**: Po-Yuan Mao, Cheng-Chang Tsai, Chun-Shien Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21195)  

**Abstract**: The great success of the diffusion model in image synthesis led to the release of gigantic commercial models, raising the issue of copyright protection and inappropriate content generation. Training-free diffusion watermarking provides a low-cost solution for these issues. However, the prior works remain vulnerable to rotation, scaling, and translation (RST) attacks. Although some methods employ meticulously designed patterns to mitigate this issue, they often reduce watermark capacity, which can result in identity (ID) collusion. To address these problems, we propose MaXsive, a training-free diffusion model generative watermarking technique that has high capacity and robustness. MaXsive best utilizes the initial noise to watermark the diffusion model. Moreover, instead of using a meticulously repetitive ring pattern, we propose injecting the X-shape template to recover the RST distortions. This design significantly increases robustness without losing any capacity, making ID collusion less likely to happen. The effectiveness of MaXsive has been verified on two well-known watermarking benchmarks under the scenarios of verification and identification. 

**Abstract (ZH)**: 无训练扩散模型生成水印技术MaXsive：高容量与鲁棒性兼备 

---
# Contrast-CAT: Contrasting Activations for Enhanced Interpretability in Transformer-based Text Classifiers 

**Title (ZH)**: Contrast-CAT：对比激活以增强基于变换器的文字分类器的可解释性 

**Authors**: Sungmin Han, Jeonghyun Lee, Sangkyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.21186)  

**Abstract**: Transformers have profoundly influenced AI research, but explaining their decisions remains challenging -- even for relatively simpler tasks such as classification -- which hinders trust and safe deployment in real-world applications. Although activation-based attribution methods effectively explain transformer-based text classification models, our findings reveal that these methods can be undermined by class-irrelevant features within activations, leading to less reliable interpretations. To address this limitation, we propose Contrast-CAT, a novel activation contrast-based attribution method that refines token-level attributions by filtering out class-irrelevant features. By contrasting the activations of an input sequence with reference activations, Contrast-CAT generates clearer and more faithful attribution maps. Experimental results across various datasets and models confirm that Contrast-CAT consistently outperforms state-of-the-art methods. Notably, under the MoRF setting, it achieves average improvements of x1.30 in AOPC and x2.25 in LOdds over the most competing methods, demonstrating its effectiveness in enhancing interpretability for transformer-based text classification. 

**Abstract (ZH)**: Transformer模型在AI研究中产生了深远的影响，但解释其决策依然具有挑战性——即使是在相对简单的分类任务上也是如此，这阻碍了实际应用中的信任和安全部署。尽管基于激活的归因方法能够有效解释基于Transformer的文本分类模型，但我们发现这些方法可能会受到激活中无关类别的特征的影响，导致解释不够可靠。为解决这一局限性，我们提出了Contrast-CAT，一种新颖的基于激活对比的归因方法，通过过滤掉无关类别的特征来细化词元级归因。通过将输入序列的激活与参考激活进行对比，Contrast-CAT生成了更为清晰和忠实的归因图。实验结果表明，Contrast-CAT在多个数据集和模型上一致优于现有最佳方法。特别是在MoRF设置下，Contrast-CAT在AOPC上平均提高了1.30倍，在LOdds上提高了2.25倍，证明了其在增强Transformer基文本分类模型可解释性方面的有效性。 

---
# Trustworthy AI: UK Air Traffic Control Revisited 

**Title (ZH)**: 可信人工智能：英国空中交通管制再审视 

**Authors**: Rob Procter, Mark Rouncefield  

**Link**: [PDF](https://arxiv.org/pdf/2507.21169)  

**Abstract**: Exploring the socio-technical challenges confronting the adoption of AI in organisational settings is something that has so far been largely absent from the related literature. In particular, research into requirements for trustworthy AI typically overlooks how people deal with the problems of trust in the tools that they use as part of their everyday work practices. This article presents some findings from an ongoing ethnographic study of how current tools are used in air traffic control work and what it reveals about requirements for trustworthy AI in air traffic control and other safety-critical application domains. 

**Abstract (ZH)**: 探索组织环境中人工智能应用所面临的社会技术挑战相关文献目前仍 largely absent。特别是，有关可信人工智能的研究通常忽略了人们在其日常工作中使用工具时如何处理信任问题。本文呈现了对空中交通控制工作中当前工具使用情况的一些民族志研究发现，以及这些发现揭示了空中交通控制和其他安全性关键应用领域中可信人工智能的要求。 

---
# OCSVM-Guided Representation Learning for Unsupervised Anomaly Detection 

**Title (ZH)**: OCSVM引导的表示学习在无监督异常检测中的应用 

**Authors**: Nicolas Pinon, Carole Lartizien  

**Link**: [PDF](https://arxiv.org/pdf/2507.21164)  

**Abstract**: Unsupervised anomaly detection (UAD) aims to detect anomalies without labeled data, a necessity in many machine learning applications where anomalous samples are rare or not available. Most state-of-the-art methods fall into two categories: reconstruction-based approaches, which often reconstruct anomalies too well, and decoupled representation learning with density estimators, which can suffer from suboptimal feature spaces. While some recent methods attempt to couple feature learning and anomaly detection, they often rely on surrogate objectives, restrict kernel choices, or introduce approximations that limit their expressiveness and robustness. To address this challenge, we propose a novel method that tightly couples representation learning with an analytically solvable one-class SVM (OCSVM), through a custom loss formulation that directly aligns latent features with the OCSVM decision boundary. The model is evaluated on two tasks: a new benchmark based on MNIST-C, and a challenging brain MRI subtle lesion detection task. Unlike most methods that focus on large, hyperintense lesions at the image level, our approach succeeds to target small, non-hyperintense lesions, while we evaluate voxel-wise metrics, addressing a more clinically relevant scenario. Both experiments evaluate a form of robustness to domain shifts, including corruption types in MNIST-C and scanner/age variations in MRI. Results demonstrate performance and robustness of our proposed mode,highlighting its potential for general UAD and real-world medical imaging applications. The source code is available at this https URL 

**Abstract (ZH)**: 无监督异常检测（UAD）旨在无需标记数据的情况下检测异常，这在许多机器学习应用中是必需的，尤其是在异常样本罕见或不可用的情况下。大多数最先进的方法分为两类：基于重建的方法，这些方法常常能够很好地重建异常；以及解耦的表示学习与密度估计方法，这些方法可能会遭受次优特征空间的问题。虽然一些最新方法试图将特征学习与异常检测结合，但它们往往依赖于代理目标、限制内核选择或引入近似方法，从而限制了它们的表达能力和鲁棒性。为了解决这一挑战，我们提出了一种新颖的方法，通过自定义损失函数将表示学习与解析可解的一类支持向量机（OCSVM）紧密耦合。该模型在两个任务上进行评估：基于MNIST-C的新基准测试，以及一个具有挑战性的脑部MRI微小病灶检测任务。与大多数方法专注于图像水平的大、高信号病灶不同，我们的方法能够成功地针对小、非高信号病灶，同时我们通过体素级指标进行评估，解决了更具有临床相关性的场景。两个实验评估了模型对领域转移的鲁棒性，包括MNIST-C中的损坏类型以及MRI中的扫描器/年龄变化。结果展示了我们提出模型的性能和鲁棒性，强调了其在通用UAD和现实世界医学成像应用中的潜在价值。源代码可在以下网址获取。 

---
# Handling Out-of-Distribution Data: A Survey 

**Title (ZH)**: 处理分布外数据：一个综述 

**Authors**: Lakpa Tamang, Mohamed Reda Bouadjenek, Richard Dazeley, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2507.21160)  

**Abstract**: In the field of Machine Learning (ML) and data-driven applications, one of the significant challenge is the change in data distribution between the training and deployment stages, commonly known as distribution shift. This paper outlines different mechanisms for handling two main types of distribution shifts: (i) Covariate shift: where the value of features or covariates change between train and test data, and (ii) Concept/Semantic-shift: where model experiences shift in the concept learned during training due to emergence of novel classes in the test phase. We sum up our contributions in three folds. First, we formalize distribution shifts, recite on how the conventional method fails to handle them adequately and urge for a model that can simultaneously perform better in all types of distribution shifts. Second, we discuss why handling distribution shifts is important and provide an extensive review of the methods and techniques that have been developed to detect, measure, and mitigate the effects of these shifts. Third, we discuss the current state of distribution shift handling mechanisms and propose future research directions in this area. Overall, we provide a retrospective synopsis of the literature in the distribution shift, focusing on OOD data that had been overlooked in the existing surveys. 

**Abstract (ZH)**: 在机器学习（ML）和数据驱动应用领域，训练阶段与部署阶段数据分布的变化，即分布偏移，是一个重要的挑战。本文概述了处理两种主要类型分布偏移的不同机制：（i）自变量偏移：特征或自变量值在训练数据和测试数据之间发生变化；（ii）概念/语义偏移：由于测试阶段出现新类导致模型在训练中学习到的概念发生变化。我们从三个方面总结了我们的贡献。首先，我们形式化了分布偏移，说明了传统方法在处理它们时的不足，并呼吁一个能够同时在各种类型分布偏移中表现更佳的模型。其次，我们讨论了为什么处理分布偏移很重要，并提供了对已开发用于检测、衡量和缓解这些偏移影响的方法和技术的全面回顾。第三，我们讨论了当前处理分布偏移机制的状态，并提出了该领域的未来研究方向。总体而言，我们为分布偏移文献提供了一个回顾性综述，重点关注现有综述中被忽视的OOD数据。 

---
# Deep Reinforcement Learning for Real-Time Green Energy Integration in Data Centers 

**Title (ZH)**: 实时绿色能源集成在数据中心中的深度强化学习方法 

**Authors**: Abderaouf Bahi, Amel Ourici  

**Link**: [PDF](https://arxiv.org/pdf/2507.21153)  

**Abstract**: This paper explores the implementation of a Deep Reinforcement Learning (DRL)-optimized energy management system for e-commerce data centers, aimed at enhancing energy efficiency, cost-effectiveness, and environmental sustainability. The proposed system leverages DRL algorithms to dynamically manage the integration of renewable energy sources, energy storage, and grid power, adapting to fluctuating energy availability in real time. The study demonstrates that the DRL-optimized system achieves a 38\% reduction in energy costs, significantly outperforming traditional Reinforcement Learning (RL) methods (28\%) and heuristic approaches (22\%). Additionally, it maintains a low SLA violation rate of 1.5\%, compared to 3.0\% for RL and 4.8\% for heuristic methods. The DRL-optimized approach also results in an 82\% improvement in energy efficiency, surpassing other methods, and a 45\% reduction in carbon emissions, making it the most environmentally friendly solution. The system's cumulative reward of 950 reflects its superior performance in balancing multiple objectives. Through rigorous testing and ablation studies, the paper validates the effectiveness of the DRL model's architecture and parameters, offering a robust solution for energy management in data centers. The findings highlight the potential of DRL in advancing energy optimization strategies and addressing sustainability challenges. 

**Abstract (ZH)**: 基于深度强化学习优化的电子商务数据中心能源管理系统及其应用 

---
# Deep Unfolding for MIMO Signal Detection 

**Title (ZH)**: MIMO信号检测的深度解折叠方法 

**Authors**: Hangli Ge, Noboru Koshizuka  

**Link**: [PDF](https://arxiv.org/pdf/2507.21152)  

**Abstract**: In this paper, we propose a deep unfolding neural network-based MIMO detector that incorporates complex-valued computations using Wirtinger calculus. The method, referred as Dynamic Partially Shrinkage Thresholding (DPST), enables efficient, interpretable, and low-complexity MIMO signal detection. Unlike prior approaches that rely on real-valued approximations, our method operates natively in the complex domain, aligning with the fundamental nature of signal processing tasks. The proposed algorithm requires only a small number of trainable parameters, allowing for simplified training. Numerical results demonstrate that the proposed method achieves superior detection performance with fewer iterations and lower computational complexity, making it a practical solution for next-generation massive MIMO systems. 

**Abstract (ZH)**: 基于Wirtinger微积分的复值计算动态部分收缩阈值MIMO检测器 

---
# Advancing Wildfire Risk Prediction via Morphology-Aware Curriculum Contrastive Learning 

**Title (ZH)**: 基于形态意识层次对比学习的 wildfire 风险预测提升 

**Authors**: Fabrizio Lo Scudo, Alessio De Rango, Luca Furnari, Alfonso Senatore, Donato D'Ambrosio, Giuseppe Mendicino, Gianluigi Greco  

**Link**: [PDF](https://arxiv.org/pdf/2507.21147)  

**Abstract**: Wildfires significantly impact natural ecosystems and human health, leading to biodiversity loss, increased hydrogeological risks, and elevated emissions of toxic substances. Climate change exacerbates these effects, particularly in regions with rising temperatures and prolonged dry periods, such as the Mediterranean. This requires the development of advanced risk management strategies that utilize state-of-the-art technologies. However, in this context, the data show a bias toward an imbalanced setting, where the incidence of wildfire events is significantly lower than typical situations. This imbalance, coupled with the inherent complexity of high-dimensional spatio-temporal data, poses significant challenges for training deep learning architectures. Moreover, since precise wildfire predictions depend mainly on weather data, finding a way to reduce computational costs to enable more frequent updates using the latest weather forecasts would be beneficial. This paper investigates how adopting a contrastive framework can address these challenges through enhanced latent representations for the patch's dynamic features. We thus introduce a new morphology-based curriculum contrastive learning that mitigates issues associated with diverse regional characteristics and enables the use of smaller patch sizes without compromising performance. An experimental analysis is performed to validate the effectiveness of the proposed modeling strategies. 

**Abstract (ZH)**: 野火显著影响自然生态系统和人类健康，导致生物多样性丧失、水文地质风险增加以及有毒物质排放加剧。气候变化加剧了这些影响，尤其是在温度上升和干旱期延长的地区，如地中海地区。这需要开发先进的风险管理策略，利用最先进的技术。然而，在这种背景下，数据表明野火事件的发生频率严重失衡，远低于典型情况。这种不平衡，加之高维度时空数据的固有复杂性，对训练深度学习架构提出了重大挑战。此外，由于精确的野火预测主要依赖于气象数据，找到一种方法降低计算成本，以利用最新的天气预报数据更频繁地更新预测，将是有益的。本文探讨了如何通过增强局部动态特征的潜在表示来采用对比框架来应对这些挑战。我们因此引入了一种基于形态学的分阶对比学习，以缓解与区域特征多样性相关的问题，并能够在不牺牲性能的情况下使用更小的局部尺寸。进行了实验分析以验证所提建模策略的有效性。 

---
# Towards Unifying Quantitative Security Benchmarking for Multi Agent Systems 

**Title (ZH)**: 面向多agent系统的定量安全基准统合 

**Authors**: Gauri Sharma, Vidhi Kulkarni, Miles King, Ken Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21146)  

**Abstract**: Evolving AI systems increasingly deploy multi-agent architectures where autonomous agents collaborate, share information, and delegate tasks through developing protocols. This connectivity, while powerful, introduces novel security risks. One such risk is a cascading risk: a breach in one agent can cascade through the system, compromising others by exploiting inter-agent trust. In tandem with OWASP's initiative for an Agentic AI Vulnerability Scoring System we define an attack vector, Agent Cascading Injection, analogous to Agent Impact Chain and Blast Radius, operating across networks of agents. In an ACI attack, a malicious input or tool exploit injected at one agent leads to cascading compromises and amplified downstream effects across agents that trust its outputs. We formalize this attack with an adversarial goal equation and key variables (compromised agent, injected exploit, polluted observations, etc.), capturing how a localized vulnerability can escalate into system-wide failure. We then analyze ACI's properties -- propagation chains, amplification factors, and inter-agent compound effects -- and map these to OWASP's emerging Agentic AI risk categories (e.g. Impact Chain and Orchestration Exploits). Finally, we argue that ACI highlights a critical need for quantitative benchmarking frameworks to evaluate the security of agent-to-agent communication protocols. We outline a methodology for stress-testing multi-agent systems (using architectures such as Google's A2A and Anthropic's MCP) against cascading trust failures, developing upon groundwork for measurable, standardized agent-to-agent security evaluation. Our work provides the necessary apparatus for engineers to benchmark system resilience, make data-driven architectural trade-offs, and develop robust defenses against a new generation of agentic threats. 

**Abstract (ZH)**: evolving AI系统中多代理架构的演进促使自主代理合作、共享信息并通过发展协议委派任务。这种连接性虽然强大，但也引入了新型安全风险。其中一种风险是连锁风险：一个代理的漏洞可以导致系统中其他代理被利用其间的信任关系而受到影响。我们与OWASP的Agent AI漏洞评分系统倡议相呼应，定义了一个攻击向量——代理连锁注入，类似于代理影响链和破坏半径，这一攻击向量在代理网络中发挥作用。在连锁注入攻击中，恶意输入或工具漏洞注入到一个代理，导致其信任的其他代理出现连锁性的破坏和下游效应的放大。我们通过敌手目标方程和关键变量（受损代理、注入漏洞、污染观测等）形式化这个攻击，捕捉局部漏洞如何升级为系统级故障。接着，我们分析连锁注入攻击的特性——传播链、放大因子和代理间的复合效应，并将这些映射到OWASP正在发展的Agent AI风险类别（例如影响链和编排性攻击）。最终，我们认为连锁注入攻击突显了定量基准测试框架对评估代理间通信协议安全性的迫切需求。我们概述了一种方法，用于压力测试多代理系统（如Google的A2A和Anthropic的MCP架构）以抵御连锁信任失败，从而在可测量和标准化的代理间安全评估的基础上构建更坚实的理论基础。我们的工作提供了工程师所需的工具，以评估系统韧性、进行数据驱动的架构权衡，并开发出对抗新一代代理威胁的稳健防御措施。 

---
# Privacy Artifact ConnecTor (PACT): Embedding Enterprise Artifacts for Compliance AI Agents 

**Title (ZH)**: 隐私特征连接器（PACT）：嵌入企业特征以合规AI代理 

**Authors**: Chenhao Fang, Yanqing Peng, Rajeev Rao, Matt Sarmiento, Wendy Summer, Arya Pudota, Alex Goncalves, Jordi Mola, Hervé Robert  

**Link**: [PDF](https://arxiv.org/pdf/2507.21142)  

**Abstract**: Enterprise environments contain a heterogeneous, rapidly growing collection of internal artifacts related to code, data, and many different tools. Critical information for assessing privacy risk and ensuring regulatory compliance is often embedded across these varied resources, each with their own arcane discovery and extraction techniques. Therefore, large-scale privacy compliance in adherence to governmental regulations requires systems to discern the interconnected nature of diverse artifacts in a common, shared universe.
We present Privacy Artifact ConnecT or (PACT), an embeddings-driven graph that links millions of artifacts spanning multiple artifact types generated by a variety of teams and projects. Powered by the state-of-the-art DRAGON embedding model, PACT uses a contrastive learning objective with light fine-tuning to link artifacts via their textual components such as raw metadata, ownership specifics, and compliance context. Experimental results show that PACT's fine-tuned model improves recall@1 from 18% to 53%, the query match rate from 9.6% to 69.7% when paired with a baseline AI agent, and the hitrate@1 from 25.7% to 44.9% for candidate selection in a standard recommender system. 

**Abstract (ZH)**: 企业环境中包含大量异质且快速增长的代码、数据和各种工具的相关内部资料。这些资料中的关键信息往往散嵌于多种资源之中，每种资源又具有各自的独特的发现和提取方法。因此，为了遵循政府法规进行大规模的隐私合规性要求，需要系统能够识别多样资料在共同共享环境中的相互关联性。
我们提出了一种基于嵌入式的图结构——Privacy Artifact Connect or（PACT），它可以连接多种类型的不同团队和项目生成的数百万种资料。PACT 由最先进的 DRAGON 嵌入式模型驱动，通过对比学习目标和轻量级的微调，利用文本组件（如原始元数据、所有权具体信息和合规背景）来连接这些资料。实验结果显示，与基准AI代理配合使用时，PACT 微调后的模型将召回率@1 提高至 53%，查询匹配率提高至 69.7%，标准推荐系统中候选选择的命中率@1 提高至 44.9%。 

---
# TTS-1 Technical Report 

**Title (ZH)**: TTS-1 技术报告 

**Authors**: Oleg Atamanenko, Anna Chalova, Joseph Coombes, Nikki Cope, Phillip Dang, Zhifeng Deng, Jimmy Du, Michael Ermolenko, Feifan Fan, Yufei Feng, Cheryl Fichter, Pavel Filimonov, Louis Fischer, Kylan Gibbs, Valeria Gusarova, Pavel Karpik, Andreas Assad Kottner, Ian Lee, Oliver Louie, Jasmine Mai, Mikhail Mamontov, Suri Mao, Nurullah Morshed, Igor Poletaev, Florin Radu, Dmytro Semernia, Evgenii Shingarev, Vikram Sivaraja, Peter Skirko, Rinat Takhautdinov, Robert Villahermosa, Jean Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21138)  

**Abstract**: We introduce Inworld TTS-1, a set of two Transformer-based autoregressive text-to-speech (TTS) models. Our largest model, TTS-1-Max, has 8.8B parameters and is designed for utmost quality and expressiveness in demanding applications. TTS-1 is our most efficient model, with 1.6B parameters, built for real-time speech synthesis and on-device use cases. By scaling train-time compute and applying a sequential process of pre-training, fine-tuning, and RL-alignment of the speech-language model (SpeechLM) component, both models achieve state-of-the-art performance on a variety of benchmarks, demonstrating exceptional quality relying purely on in-context learning of the speaker's voice. Inworld TTS-1 and TTS-1-Max can generate high-resolution 48 kHz speech with low latency, and support 11 languages with fine-grained emotional control and non-verbal vocalizations through audio markups. We additionally open-source our training and modeling code under an MIT license. 

**Abstract (ZH)**: Inworld TTS-1：基于Transformer的自动回归文本转语音模型集 

---
# A Study on Variants of Conventional, Fuzzy, and Nullspace-Based Independence Criteria for Improving Supervised and Unsupervised Learning 

**Title (ZH)**: 基于传统、模糊及 Null 空间独立性准则变体的监督与无监督学习改进研究 

**Authors**: Mojtaba Moattari  

**Link**: [PDF](https://arxiv.org/pdf/2507.21136)  

**Abstract**: Unsupervised and supervised learning methods conventionally use kernels to capture nonlinearities inherent in data structure. However experts have to ensure their proposed nonlinearity maximizes variability and capture inherent diversity of data. We reviewed all independence criteria to design unsupervised learners. Then we proposed 3 independence criteria and used them to design unsupervised and supervised dimensionality reduction methods. We evaluated contrast, accuracy and interpretability of these methods in both linear and neural nonlinear settings. The results show that the methods have outperformed the baseline (tSNE, PCA, regularized LDA, VAE with (un)supervised learner and layer sharing) and opened a new line of interpretable machine learning (ML) for the researchers. 

**Abstract (ZH)**: 无监督和监督学习方法通常使用核函数来捕捉数据结构中的非线性特征，但在提出非线性特征时，专家需要确保其最大化数据的变异性并捕捉数据的内在多样性。我们回顾了所有独立性准则以设计无监督学习者。然后我们提出了三种独立性准则，并使用这些准则设计了无监督和监督降维方法。我们在线性和神经非线性设置下评估了这些方法的对比度、准确性和可解释性。结果表明，这些方法优于基线方法（tSNE、PCA、正则化LDA、VAE与无监督学习者和层共享），并且为研究人员开辟了一条新的可解释机器学习路径。 

---
# Affect-aware Cross-Domain Recommendation for Art Therapy via Music Preference Elicitation 

**Title (ZH)**: 基于音乐偏好激发的情感aware跨域推荐的艺术疗法 

**Authors**: Bereket A. Yilma, Luis A. Leiva  

**Link**: [PDF](https://arxiv.org/pdf/2507.21120)  

**Abstract**: Art Therapy (AT) is an established practice that facilitates emotional processing and recovery through creative expression. Recently, Visual Art Recommender Systems (VA RecSys) have emerged to support AT, demonstrating their potential by personalizing therapeutic artwork recommendations. Nonetheless, current VA RecSys rely on visual stimuli for user modeling, limiting their ability to capture the full spectrum of emotional responses during preference elicitation. Previous studies have shown that music stimuli elicit unique affective reflections, presenting an opportunity for cross-domain recommendation (CDR) to enhance personalization in AT. Since CDR has not yet been explored in this context, we propose a family of CDR methods for AT based on music-driven preference elicitation. A large-scale study with 200 users demonstrates the efficacy of music-driven preference elicitation, outperforming the classic visual-only elicitation approach. Our source code, data, and models are available at this https URL 

**Abstract (ZH)**: 基于音乐驱动偏好评价的艺术疗法跨域推荐方法 

---
# Failure Risk Prediction in a MOOC: A Multivariate Time Series Analysis Approach 

**Title (ZH)**: MOOC中失败风险预测：一种多变量时间序列分析方法 

**Authors**: Anass El Ayady, Maxime Devanne, Germain Forestier, Nour El Mawas  

**Link**: [PDF](https://arxiv.org/pdf/2507.21118)  

**Abstract**: MOOCs offer free and open access to a wide audience, but completion rates remain low, often due to a lack of personalized content. To address this issue, it is essential to predict learner performance in order to provide tailored feedback. Behavioral traces-such as clicks and events-can be analyzed as time series to anticipate learners' outcomes. This work compares multivariate time series classification methods to identify at-risk learners at different stages of the course (after 5, 10 weeks, etc.). The experimental evaluation, conducted on the Open University Learning Analytics Dataset (OULAD), focuses on three courses: two in STEM and one in SHS. Preliminary results show that the evaluated approaches are promising for predicting learner failure in MOOCs. The analysis also suggests that prediction accuracy is influenced by the amount of recorded interactions, highlighting the importance of rich and diverse behavioral data. 

**Abstract (ZH)**: MOOCs提供免费开放的学习资源给广泛受众，但完成率低下，常常由于缺乏个性化内容。为解决这一问题，预测学习者表现以提供个性化反馈至关重要。行为轨迹，如点击和事件，可以作为时间序列进行分析以预估学习者的学业成果。本研究将比较多元时间序列分类方法，以识别不同课程阶段（5周、10周等）可能出现学业风险的学习者。实验评估基于Open University Learning Analytics Dataset (OULAD)，重点分析三门课程：两门STEM课程和一门社会科学与健康科学课程。初步结果表明，评估的方法在预测MOOC中学习者辍学方面具有潜力。分析还表明，预测准确性受记录交互数量的影响，突显了丰富多样行为数据的重要性。 

---
# FedFlex: Federated Learning for Diverse Netflix Recommendations 

**Title (ZH)**: FedFlex: 联邦学习下的 diverse Netflix 推荐 

**Authors**: Sven Lankester, Manel Slokom, Gustavo de Carvalho Bertoli, Matias Vizcaino, Emmanuelle Beauxis Aussalet, Laura Hollink  

**Link**: [PDF](https://arxiv.org/pdf/2507.21115)  

**Abstract**: Federated learning is a decentralized approach that enables collaborative model training across multiple devices while preserving data privacy. It has shown significant potential in various domains, including healthcare and personalized recommendation systems. However, most existing work on federated recommendation systems has focused primarily on improving accuracy, with limited attention to fairness and diversity. In this paper, we introduce FedFlex, a federated recommender system for Netflix-style TV series recommendations. FedFlex integrates two state-of-the-art matrix factorization algorithms for personalized fine-tuning. FedFlex also applies Maximal Marginal Relevance (MMR) to re-rank items and enhance diversity. We conduct extensive experiments comparing recommendations generated by SVD and BPR algorithms. In a live two-week user study, participants received two recommendation lists: List A, based on SVD or BPR, and List B, a re-ranked version emphasizing diversity. Participants were asked to click on the movies they were interested in watching. Our findings demonstrate that FedFlex effectively introduces diverse content, such as new genres, into recommendations without necessarily compromising user satisfaction. 

**Abstract (ZH)**: 联邦学习是一种去中心化的方法，能够在保护数据隐私的前提下，在多个设备之间实现模型的协同训练。它在包括医疗保健和个性化推荐系统在内的各个领域展示了显著的潜力。然而，现有的大多数联邦推荐系统工作主要集中在提高准确性上，对公平性和多样性关注较少。在本文中，我们提出了一种名为FedFlex的联邦推荐系统，适用于Netflix风格的电视剧推荐。FedFlex结合了两种最先进的矩阵分解算法以实现个性化调整，并应用最大相关性度量（MMR）重新排序项目以增强多样性。我们进行了广泛的实验，对比了SVD和BPR算法生成的推荐效果。在一个为期两周的实时用户研究中，参与者分别获得了基于SVD或BPR的推荐列表A和强调多样性的重新排序列表B，被要求点击他们感兴趣的电影。我们的研究结果表明，FedFlex能够在不必要牺牲用户满意度的情况下，有效引入新的内容类型，如新类型电影。 

---
# Page image classification for content-specific data processing 

**Title (ZH)**: 特定内容图像分类 

**Authors**: Kateryna Lutsai, Pavel Straňák  

**Link**: [PDF](https://arxiv.org/pdf/2507.21114)  

**Abstract**: Digitization projects in humanities often generate vast quantities of page images from historical documents, presenting significant challenges for manual sorting and analysis. These archives contain diverse content, including various text types (handwritten, typed, printed), graphical elements (drawings, maps, photos), and layouts (plain text, tables, forms). Efficiently processing this heterogeneous data requires automated methods to categorize pages based on their content, enabling tailored downstream analysis pipelines. This project addresses this need by developing and evaluating an image classification system specifically designed for historical document pages, leveraging advancements in artificial intelligence and machine learning. The set of categories was chosen to facilitate content-specific processing workflows, separating pages requiring different analysis techniques (e.g., OCR for text, image analysis for graphics) 

**Abstract (ZH)**: 人文领域的数字化项目常常生成大量的历史文献页面图像，这为手动分类和分析带来了巨大挑战。这些档案包含多样化的内容，包括各种文本类型（手写、打印、印刷）、图形元素（绘制、地图、照片）以及布局（纯文本、表格、表单）。有效地处理这些异构数据需要能够根据内容对页面进行自动分类的方法，从而为下游分析流水线提供定制化的处理流程。该项目通过开发和评估一种专门针对历史文献页面的图像分类系统来满足这一需求，利用了人工智能和机器学习的最新进展。所选的类别集旨在促进内容特定的处理工作流程，将需要不同分析技术的页面分开（例如，OCR用于文本、图像分析用于图形）。 

---
# A Formal Rebuttal of "The Blockchain Trilemma: A Formal Proof of the Inherent Trade-Offs Among Decentralization, Security, and Scalability" 

**Title (ZH)**: 对“区块链三难问题：去中心化、安全性和可扩展性内在权衡的正式证明”的正式反驳 

**Authors**: Craig Wright  

**Link**: [PDF](https://arxiv.org/pdf/2507.21111)  

**Abstract**: This paper presents a comprehensive refutation of the so-called "blockchain trilemma," a widely cited but formally ungrounded claim asserting an inherent trade-off between decentralisation, security, and scalability in blockchain protocols. Through formal analysis, empirical evidence, and detailed critique of both methodology and terminology, we demonstrate that the trilemma rests on semantic equivocation, misuse of distributed systems theory, and a failure to define operational metrics. Particular focus is placed on the conflation of topological network analogies with protocol-level architecture, the mischaracterisation of Bitcoin's design--including the role of miners, SPV clients, and header-based verification--and the failure to ground claims in complexity-theoretic or adversarial models. By reconstructing Bitcoin as a deterministic, stateless distribution protocol governed by evidentiary trust, we show that scalability is not a trade-off but an engineering outcome. The paper concludes by identifying systemic issues in academic discourse and peer review that have allowed such fallacies to persist, and offers formal criteria for evaluating future claims in blockchain research. 

**Abstract (ZH)**: 本文全面反驳了所谓的“区块链三难问题”这一广泛引用但缺乏正式依据的说法，该说法声称在区块链协议中去中心化、安全性和可扩展性之间存在着固有的权衡。通过形式分析、实证证据以及对方法论和术语的详细批评，我们证明“三难问题”基于语义含糊、分布式系统理论的误用以及缺乏操作性衡量指标。特别关注的是将拓扑网络类比与协议级架构混淆、对Bitcoin设计的误character化，包括矿工、SPV客户端和基于区块头验证的作用，以及缺乏复杂性理论或对抗性模型的支持。通过对Bitcoin重新构建为一个由证据信任支配的确定性、无状态分布协议，我们证明可扩展性并非权衡而是工程结果。文章最后指出了学术话语和同行评审中的系统性问题，这些问题导致了此类谬误的持续存在，并提出了评估未来区块链研究中主张的形式标准。 

---
# Task-Focused Consolidation with Spaced Recall: Making Neural Networks learn like college students 

**Title (ZH)**: 基于间隔回忆的任务导向性 consolidation：使神经网络像大学生一样学习 

**Authors**: Prital Bamnodkar  

**Link**: [PDF](https://arxiv.org/pdf/2507.21109)  

**Abstract**: Deep Neural Networks often suffer from a critical limitation known as Catastrophic Forgetting, where performance on past tasks degrades after learning new ones. This paper introduces a novel continual learning approach inspired by human learning strategies like Active Recall, Deliberate Practice and Spaced Repetition, named Task Focused Consolidation with Spaced Recall (TFC-SR). TFC-SR enhances the standard experience replay with a mechanism we termed the Active Recall Probe. It is a periodic, task-aware evaluation of the model's memory that stabilizes the representations of past knowledge. We test TFC-SR on the Split MNIST and Split CIFAR-100 benchmarks against leading regularization-based and replay-based baselines. Our results show that TFC-SR performs significantly better than these methods. For instance, on the Split CIFAR-100, it achieves a final accuracy of 13.17% compared to standard replay's 7.40%. We demonstrate that this advantage comes from the stabilizing effect of the probe itself, and not from the difference in replay volume. Additionally, we analyze the trade-off between memory size and performance and show that while TFC-SR performs better in memory-constrained environments, higher replay volume is still more effective when available memory is abundant. We conclude that TFC-SR is a robust and efficient approach, highlighting the importance of integrating active memory retrieval mechanisms into continual learning systems. 

**Abstract (ZH)**: 深度神经网络往往受到一种名为灾难性遗忘的关键限制，即在学习新任务后，对过去任务的表现会下降。本文介绍了一种受人类学习策略（如主动回忆、刻意练习和分散复习）启发的新连续学习方法，名为基于分散回忆的任务聚焦巩固（TFC-SR）。TFC-SR通过我们称之为主动回忆探针的机制增强标准的经验回放，这是一种周期性的、任务感知的记忆评估，能够稳定过去知识的表示。我们使用Split MNIST和Split CIFAR-100基准测试TFC-SR，与基于正则化和回放缓冲的领先基准方法进行比较。结果显示，TFC-SR的表现显著优于这些方法。例如，在Split CIFAR-100中，TFC-SR的最终准确率为13.17%，而标准回放缓冲仅为7.40%。我们展示了这一优势来源于探针本身的稳定作用，而不是回放容量的不同。此外，我们分析了记忆大小与性能之间的权衡，并表明虽然在资源受限的环境中TFC-SR表现更好，但当可用内存充足时，更大的回放容量仍然更为有效。我们得出结论，TFC-SR是一种稳健而高效的策略，强调了将主动记忆检索机制整合到连续学习系统中的重要性。 

---
# A Survey of Classification Tasks and Approaches for Legal Contracts 

**Title (ZH)**: 法律合同分类任务及方法综述 

**Authors**: Amrita Singh, Aditya Joshi, Jiaojiao Jiang, Hye-young Paik  

**Link**: [PDF](https://arxiv.org/pdf/2507.21108)  

**Abstract**: Given the large size and volumes of contracts and their underlying inherent complexity, manual reviews become inefficient and prone to errors, creating a clear need for automation. Automatic Legal Contract Classification (LCC) revolutionizes the way legal contracts are analyzed, offering substantial improvements in speed, accuracy, and accessibility. This survey delves into the challenges of automatic LCC and a detailed examination of key tasks, datasets, and methodologies. We identify seven classification tasks within LCC, and review fourteen datasets related to English-language contracts, including public, proprietary, and non-public sources. We also introduce a methodology taxonomy for LCC, categorized into Traditional Machine Learning, Deep Learning, and Transformer-based approaches. Additionally, the survey discusses evaluation techniques and highlights the best-performing results from the reviewed studies. By providing a thorough overview of current methods and their limitations, this survey suggests future research directions to improve the efficiency, accuracy, and scalability of LCC. As the first comprehensive survey on LCC, it aims to support legal NLP researchers and practitioners in improving legal processes, making legal information more accessible, and promoting a more informed and equitable society. 

**Abstract (ZH)**: 基于自动法律合同分类的挑战与方法综述：提高效率、准确性和可扩展性的未来研究方向 

---
# iLSU-T: an Open Dataset for Uruguayan Sign Language Translation 

**Title (ZH)**: iLSU-T：乌拉圭手语翻译的开放数据集 

**Authors**: Ariel E. Stassi, Yanina Boria, J. Matías Di Martino, Gregory Randall  

**Link**: [PDF](https://arxiv.org/pdf/2507.21104)  

**Abstract**: Automatic sign language translation has gained particular interest in the computer vision and computational linguistics communities in recent years. Given each sign language country particularities, machine translation requires local data to develop new techniques and adapt existing ones. This work presents iLSU T, an open dataset of interpreted Uruguayan Sign Language RGB videos with audio and text transcriptions. This type of multimodal and curated data is paramount for developing novel approaches to understand or generate tools for sign language processing. iLSU T comprises more than 185 hours of interpreted sign language videos from public TV broadcasting. It covers diverse topics and includes the participation of 18 professional interpreters of sign language. A series of experiments using three state of the art translation algorithms is presented. The aim is to establish a baseline for this dataset and evaluate its usefulness and the proposed pipeline for data processing. The experiments highlight the need for more localized datasets for sign language translation and understanding, which are critical for developing novel tools to improve accessibility and inclusion of all individuals. Our data and code can be accessed. 

**Abstract (ZH)**: 自动手语翻译近年来在计算机视觉和计算语言学领域引起了特别的兴趣。鉴于每个手语国家的独特性，机器翻译需要当地数据来开发新技术和适应现有技术。本文介绍了iLSU T，一个包含解释的乌拉圭手语RGB视频及其音频和文本转录的开源数据集。此类多模态和精心整理的数据对于开发新的手语处理方法至关重要。iLSU T 包含超过185小时的公共电视台转译手语视频，涵盖了多个主题，并包括18名专业手语翻译者的参与。本文展示了用三个最先进的翻译算法进行的一系列实验，旨在为该数据集建立基准并评估其用途及其提出的处理流程。实验强调了需要更多局部化的手语翻译和理解数据集的重要性，这对于开发新的工具以提高所有人都能获得的无障碍和包容性至关重要。我们的数据和代码可以访问。 

---
# Assessing the Ecological Impact of AI 

**Title (ZH)**: 评估人工智能的生态影响 

**Authors**: Sylvia Wenmackers  

**Link**: [PDF](https://arxiv.org/pdf/2507.21102)  

**Abstract**: Philosophers of technology have recently started paying more attention to the environmental impacts of AI, in particular of large language models (LLMs) and generative AI (genAI) applications. Meanwhile, few developers of AI give concrete estimates of the ecological impact of their models and products, and even when they do so, their analysis is often limited to green house gas emissions of certain stages of AI development or use. The current proposal encourages practically viable analyses of the sustainability aspects of genAI informed by philosophical ideas. 

**Abstract (ZH)**: 技术哲学家最近开始更加关注人工智能的环境影响，特别是大型语言模型（LLMs）和生成人工智能（genAI）应用的影响。然而，很少有AI开发者提供其模型和产品的具体生态影响估计，即使有所估计，分析通常也仅限于某些阶段的AI开发或使用所产生的温室气体排放。当前的提议鼓励基于哲学观念来进行可持续性方面的实际可行分析。 

---
# Thinking Like a Scientist: Can Interactive Simulations Foster Critical AI Literacy? 

**Title (ZH)**: 科学家般思考：交互式模拟能否培养批判性人工智能素养？ 

**Authors**: Yiling Zhao, Audrey Michal, Nithum Thain, Hari Subramonyam  

**Link**: [PDF](https://arxiv.org/pdf/2507.21090)  

**Abstract**: As AI systems shape individual and societal decisions, fostering critical AI literacy is essential. Traditional approaches, such as blog articles, static lessons, and social media discussions, often fail to support deep conceptual understanding and critical engagement. This study examines whether interactive simulations can help learners think like a scientist by engaging them in hypothesis testing, experimentation, and direct observation of AI behavior. In a controlled study with 605 participants, we assess how interactive AI tutorials impact learning of key concepts such as fairness, dataset representativeness, and bias in language models. Results show that interactive simulations effectively enhance AI literacy across topics, supporting greater knowledge transfer and self-reported confidence, though engagement alone does not predict learning. This work contributes to the growing field of AI literacy education, highlighting how interactive, inquiry-driven methodologies can better equip individuals to critically engage with AI in their daily lives. 

**Abstract (ZH)**: 随着AI系统影响个人和社会决策，培养批判性AI literacy至关重要。传统的教学方法，如博客文章、静态课程和社交媒体讨论，往往无法支持深刻的概念理解与批判性参与。本研究探讨交互式模拟是否能通过让学习者参与假设测试、实验和直接观察AI行为，帮助他们像科学家一样思考。在一项包括605名参与者的受控研究中，我们评估交互式AI教程对公平性、数据集代表性以及语言模型偏见等关键概念学习的影响。结果表明，交互式模拟有效地提升了跨学科的AI literacy，促进了知识迁移和自我报告的自信心，但单纯的参与并不能预测学习效果。本研究为不断发展的AI literacy教育领域做出了贡献，突出了以探究为导向的交互式方法如何更好地帮助个人在日常生活中批判性地应对AI。 

---
# Empathy in Explanation 

**Title (ZH)**: 共情在解释中的作用 

**Authors**: Katherine M. Collins, Kartik Chandra, Adrian Weller, Jonathan Ragan-Kelley, Joshua B. Tenenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2507.21081)  

**Abstract**: Why do we give the explanations we do? Recent work has suggested that we should think of explanation as a kind of cooperative social interaction, between a why-question-asker and an explainer. Here, we apply this perspective to consider the role that emotion plays in this social interaction. We develop a computational framework for modeling explainers who consider the emotional impact an explanation might have on a listener. We test our framework by using it to model human intuitions about how a doctor might explain to a patient why they have a disease, taking into account the patient's propensity for regret. Our model predicts human intuitions well, better than emotion-agnostic ablations, suggesting that people do indeed reason about emotion when giving explanations. 

**Abstract (ZH)**: 我们为何给出这样的解释？近期研究表明，我们应该将解释视为一种合作关系，发生在为何提问者与解释者之间。在此基础上，我们探讨情感在这类社会互动中的作用。我们发展了一个计算框架，用于建模解释者考虑解释可能给听众带来的情感影响。我们通过使用该框架来模拟医生向患解释其患病原因时的情感影响，考虑到患者后悔的可能性。我们的模型很好地预测了人们的情感直觉，比不知情的情感模型表现更好，这表明人们在给出解释时确实会考虑情感因素。 

---
# Which symbol grounding problem should we try to solve? 

**Title (ZH)**: 我们应该尝试解决哪种符号 grounding 问题？ 

**Authors**: Vincent C. Müller  

**Link**: [PDF](https://arxiv.org/pdf/2507.21080)  

**Abstract**: Floridi and Taddeo propose a condition of "zero semantic commitment" for solutions to the grounding problem, and a solution to it. I argue briefly that their condition cannot be fulfilled, not even by their own solution. After a look at Luc Steels' very different competing suggestion, I suggest that we need to re-think what the problem is and what role the 'goals' in a system play in formulating the problem. On the basis of a proper understanding of computing, I come to the conclusion that the only sensible grounding problem is how we can explain and re-produce the behavioral ability and function of meaning in artificial computational agents 

**Abstract (ZH)**: 弗罗里迪和塔德多提出“零语义承诺”条件以解决基底问题，并提出了一个解决方案。我认为他们的条件无法实现，甚至他们自己的解决方案也无法满足。在审视卢克·斯蒂尔斯非常不同的竞争建议后，我建议我们需要重新思考问题本身以及系统中的“目标”在界定问题中的作用。基于对计算的正确理解，我认为唯一合乎情理的基底问题是：我们如何解释和重现人工计算代理的的行为能力和意义功能。 

---
# Data-Driven and Participatory Approaches toward Neuro-Inclusive AI 

**Title (ZH)**: 数据驱动与参与式面向神经包容的AI 

**Authors**: Naba Rizvi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21077)  

**Abstract**: Biased data representation in AI marginalizes up to 75 million autistic people worldwide through medical applications viewing autism as a deficit of neurotypical social skills rather than an aspect of human diversity, and this perspective is grounded in research questioning the humanity of autistic people. Turing defined artificial intelligence as the ability to mimic human communication, and as AI development increasingly focuses on human-like agents, this benchmark remains popular. In contrast, we define Neuro-Inclusive AI as datasets and systems that move away from mimicking humanness as a benchmark for machine intelligence. Then, we explore the origins, prevalence, and impact of anti-autistic biases in current research. Our work finds that 90% of human-like AI agents exclude autistic perspectives, and AI creators continue to believe ethical considerations are beyond the scope of their work. To improve the autistic representation in data, we conduct empirical experiments with annotators and LLMs, finding that binary labeling schemes sufficiently capture the nuances of labeling anti-autistic hate speech. Our benchmark, AUTALIC, can be used to evaluate or fine-tune models, and was developed to serve as a foundation for more neuro-inclusive future work. 

**Abstract (ZH)**: AI中偏向性的数据表示通过医疗应用边缘化多达7500万自闭症患者：以自闭症缺乏神经典型社会技能为缺陷而非人类多样性的 aspects 视角为基础，这种观点根植于质疑自闭症患者人性的研究。图灵定义人工智能为模仿人类交流的能力，随着AI开发越来越多地关注类人代理，这一标准依然流行。相比之下，我们定义神经包容性AI为远离将类人性作为机器智能标准的数据集和系统。然后，我们探讨了当前研究中反自闭症偏见的起源、普遍存在性和影响。我们的工作发现，90%的类人AI代理排斥自闭症视角，AI创作者仍认为伦理考量超出了他们的工作范围。为了改善数据中的自闭症代表性，我们进行了一项以注释员和大模型为基础的实证实验，发现二元标签方案能够充分捕捉反自闭症仇恨言论的细微差别。我们的基准AUTALIC可以用于评估或微调模型，旨在作为未来更加神经包容性工作的基础。 

---
# GAITEX: Human motion dataset from impaired gait and rehabilitation exercises of inertial and optical sensor data 

**Title (ZH)**: GAITEX：基于惯性和光学传感器数据的异常步态及康复锻炼的人体动作数据集 

**Authors**: Andreas Spilz, Heiko Oppel, Jochen Werner, Kathrin Stucke-Straub, Felix Capanni, Michael Munz  

**Link**: [PDF](https://arxiv.org/pdf/2507.21069)  

**Abstract**: Wearable inertial measurement units (IMUs) offer a cost-effective and scalable means to assess human movement quality in clinical and everyday settings. However, the development of robust sensor-based classification models for physiotherapeutic exercises and gait analysis requires large, diverse datasets, which are costly and time-consuming to collect. Here, we present a multimodal dataset of physiotherapeutic exercises - including correct and clinically relevant variants - and gait-related exercises - including both normal and impaired gait patterns - recorded from 19 participants using synchronized IMUs and marker-based motion capture (MoCap). The dataset includes raw data from nine IMUs and thirty-five optical markers capturing full-body kinematics. Each IMU is additionally equipped with four optical markers, enabling precise comparison between IMU-derived orientation estimates and reference values from the MoCap system. To support further analysis, we also provide processed IMU orientations aligned with common segment coordinate systems, subject-specific OpenSim models, inverse kinematics results, and tools for visualizing IMU orientations in the musculoskeletal context. Detailed annotations of movement execution quality and time-stamped segmentations support diverse analysis goals. This dataset supports the development and benchmarking of machine learning models for tasks such as automatic exercise evaluation, gait analysis, temporal activity segmentation, and biomechanical parameter estimation. To facilitate reproducibility, we provide code for postprocessing, sensor-to-segment alignment, inverse kinematics computation, and technical validation. This resource is intended to accelerate research in machine learning-driven human movement analysis. 

**Abstract (ZH)**: 可穿戴惯性测量单元（IMU）为在临床和日常生活环境中评估人类运动质量提供了低成本且可扩展的方法。然而，为了开发适用于物理治疗锻炼和步态分析的稳健传感器基分类模型，需要收集大量多样化的数据集，而这在成本和时间上都极具挑战性。在此，我们呈现了一个包含物理治疗锻炼（包括正确的和临床相关的变体）和步态相关锻炼（包括正常和受损步态模式）的多模态数据集，这些数据集是在19名参与者同步使用惯性测量单元（IMU）和标记基动捕（MoCap）系统记录下来的。数据集包括九个IMU的原始数据和35个光学标记捕捉到的全身运动学。每个IMU还配备了四个光学标记，使得IMU获取的姿态估计值与基于MoCap系统的参考值之间能够进行精确比较。此外，我们还提供了与常用节段坐标系统对齐的IMU姿态、个体特定的OpenSim模型、逆运动学结果以及在运动学背景下可视化IMU姿态的工具。详细的运动执行质量标注和时间戳分割支持多样化的分析目标。该数据集支持自动锻炼评估、步态分析、时间活动分割和生物力学参数估计等机器学习模型的开发与基准测试。为了确保可再现性，我们提供了后处理代码、传感器到节段对齐、逆运动学计算以及技术验证的工具。该资源旨在推动基于机器学习的人类运动分析研究。 

---
# Privacy-Preserving AI for Encrypted Medical Imaging: A Framework for Secure Diagnosis and Learning 

**Title (ZH)**: 加密医疗影像的隐私保护AI框架：安全诊断与学习体系 

**Authors**: Abdullah Al Siam, Sadequzzaman Shohan  

**Link**: [PDF](https://arxiv.org/pdf/2507.21060)  

**Abstract**: The rapid integration of Artificial Intelligence (AI) into medical diagnostics has raised pressing concerns about patient privacy, especially when sensitive imaging data must be transferred, stored, or processed. In this paper, we propose a novel framework for privacy-preserving diagnostic inference on encrypted medical images using a modified convolutional neural network (Masked-CNN) capable of operating on transformed or ciphered image formats. Our approach leverages AES-CBC encryption coupled with JPEG2000 compression to protect medical images while maintaining their suitability for AI inference. We evaluate the system using public DICOM datasets (NIH ChestX-ray14 and LIDC-IDRI), focusing on diagnostic accuracy, inference latency, storage efficiency, and privacy leakage resistance. Experimental results show that the encrypted inference model achieves performance comparable to its unencrypted counterpart, with only marginal trade-offs in accuracy and latency. The proposed framework bridges the gap between data privacy and clinical utility, offering a practical, scalable solution for secure AI-driven diagnostics. 

**Abstract (ZH)**: 人工智能在医疗诊断中的快速集成引发了对患者隐私的迫切关注，尤其是当需要传输、存储或处理敏感影像数据时。本文提出了一种新型框架，使用修改后的卷积神经网络（Masked-CNN）进行加密医疗影像的隐私保护诊断推理，该框架能够在变换或密文格式的影像上运行。我们的方法结合使用AES-CBC加密和JPEG2000压缩，以保护医疗影像同时保持其适用于AI推理的适用性。我们使用公开的DICOM数据集（NIH ChestX-ray14和LIDC-IDRI）进行系统评估，重点关注诊断准确性、推理延迟、存储效率和隐私泄露抵抗力。实验结果表明，加密推理模型在性能上与非加密版本相当，仅在准确性与延迟方面有微小的权衡。所提出的框架在数据隐私与临床应用之间架起了桥梁，提供了一种实用且可扩展的安全AI驱动诊断解决方案。 

---
# Categorical Classification of Book Summaries Using Word Embedding Techniques 

**Title (ZH)**: 使用词嵌入技术对图书摘要进行分类学分类 

**Authors**: Kerem Keskin, Mümine Kaya Keleş  

**Link**: [PDF](https://arxiv.org/pdf/2507.21058)  

**Abstract**: In this study, book summaries and categories taken from book sites were classified using word embedding methods, natural language processing techniques and machine learning algorithms. In addition, one hot encoding, Word2Vec and Term Frequency - Inverse Document Frequency (TF-IDF) methods, which are frequently used word embedding methods were used in this study and their success was compared. Additionally, the combination table of the pre-processing methods used is shown and added to the table. Looking at the results, it was observed that Support Vector Machine, Naive Bayes and Logistic Regression Models and TF-IDF and One-Hot Encoder word embedding techniques gave more successful results for Turkish texts. 

**Abstract (ZH)**: 在本书研究中，来自书籍网站的书摘要和类别被使用词嵌入方法、自然语言处理技术和机器学习算法进行分类。此外，本研究中使用了一热编码、Word2Vec 和词频-逆文档频数（TF-IDF）等常用词嵌入方法，并对其成功率进行了比较。同时展示了所使用预处理方法的组合表。结果表明，支持向量机、朴素贝叶斯和支持回归模型以及 TF-IDF 和一热编码词嵌入技术在土耳其文本上取得了更成功的结果。 

---
# AI-Driven Generation of Data Contracts in Modern Data Engineering Systems 

**Title (ZH)**: 基于AI的数据合同生成在现代数据工程系统中的应用 

**Authors**: Harshraj Bhoite  

**Link**: [PDF](https://arxiv.org/pdf/2507.21056)  

**Abstract**: Data contracts formalize agreements between data producers and consumers regarding schema, semantics, and quality expectations. As data pipelines grow in complexity, manual authoring and maintenance of contracts becomes error-prone and labor-intensive. We present an AI-driven framework for automatic data contract generation using large language models (LLMs). Our system leverages parameter-efficient fine-tuning methods, including LoRA and PEFT, to adapt LLMs to structured data domains. The models take sample data or schema descriptions and output validated contract definitions in formats such as JSON Schema and Avro. We integrate this framework into modern data platforms (e.g., Databricks, Snowflake) to automate contract enforcement at scale. Experimental results on synthetic and real-world datasets demonstrate that the fine-tuned LLMs achieve high accuracy in generating valid contracts and reduce manual workload by over 70%. We also discuss key challenges such as hallucination, version control, and the need for continuous learning. This work demonstrates that generative AI can enable scalable, agile data governance by bridging the gap between intent and implementation in enterprise data management. 

**Abstract (ZH)**: 基于大规模语言模型的自动数据合同生成框架 

---
# High hopes for "Deep Medicine"? AI, economics, and the future of care 

**Title (ZH)**: “深 Medicine”的高期望？AI、经济学与照护的未来 

**Authors**: Robert Sparrow, Joshua Hatherley  

**Link**: [PDF](https://arxiv.org/pdf/2507.21054)  

**Abstract**: In the much-celebrated book Deep Medicine, Eric Topol argues that the development of artificial intelligence for health care will lead to a dramatic shift in the culture and practice of medicine. In the next several decades, he suggests, AI will become sophisticated enough that many of the everyday tasks of physicians could be delegated to it. Topol is perhaps the most articulate advocate of the benefits of AI in medicine, but he is hardly alone in spruiking its potential to allow physicians to dedicate more of their time and attention to providing empathetic care for their patients in the future. Unfortunately, several factors suggest a radically different picture for the future of health care. Far from facilitating a return to a time of closer doctor-patient relationships, the use of medical AI seems likely to further erode therapeutic relationships and threaten professional and patient satisfaction. 

**Abstract (ZH)**: 在备受赞誉的著作《深度医学》中，埃里克·托波尔 argue了人工智能在医疗领域的开发将导致医学文化与实践的重大转变。在未来几十年里，他建议，人工智能将变得足够先进，以至于许多医生的日常工作可以委托给它处理。托波尔可能是人工智能在医学领域最大声的支持者之一，但并非唯一一个认为人工智能可能使医生能够将更多时间和注意力集中在对患者的同理心护理上的倡导者。不幸的是，一些因素预示着医疗保健未来的急剧不同图景。实际上，医疗人工智能的使用似乎更有可能进一步削弱治疗关系，并威胁专业人员和患者的满意度。 

---
# Online hierarchical partitioning of the output space in extreme multi-label data stream 

**Title (ZH)**: 极端多标签数据流中在线层次化输出空间分区 

**Authors**: Lara Neves, Afonso Lourenço, Alberto Cano, Goreti Marreiros  

**Link**: [PDF](https://arxiv.org/pdf/2507.20894)  

**Abstract**: Mining data streams with multi-label outputs poses significant challenges due to evolving distributions, high-dimensional label spaces, sparse label occurrences, and complex label dependencies. Moreover, concept drift affects not only input distributions but also label correlations and imbalance ratios over time, complicating model adaptation. To address these challenges, structured learners are categorized into local and global methods. Local methods break down the task into simpler components, while global methods adapt the algorithm to the full output space, potentially yielding better predictions by exploiting label correlations. This work introduces iHOMER (Incremental Hierarchy Of Multi-label Classifiers), an online multi-label learning framework that incrementally partitions the label space into disjoint, correlated clusters without relying on predefined hierarchies. iHOMER leverages online divisive-agglomerative clustering based on \textit{Jaccard} similarity and a global tree-based learner driven by a multivariate \textit{Bernoulli} process to guide instance partitioning. To address non-stationarity, it integrates drift detection mechanisms at both global and local levels, enabling dynamic restructuring of label partitions and subtrees. Experiments across 23 real-world datasets show iHOMER outperforms 5 state-of-the-art global baselines, such as MLHAT, MLHT of Pruned Sets and iSOUPT, by 23\%, and 12 local baselines, such as binary relevance transformations of kNN, EFDT, ARF, and ADWIN bagging/boosting ensembles, by 32\%, establishing its robustness for online multi-label classification. 

**Abstract (ZH)**: Mining 多标签数据流中的结构化学习面临显著挑战，由于分布演变、高维标签空间、稀疏标签出现和复杂的标签依赖关系。此外，概念漂移不仅影响输入分布，还影响标签相关性和不平衡比率，增加了模型适应的复杂性。为应对这些挑战，结构化学习方法被划分为局部和全局方法。局部方法将任务分解为更简单的组件，而全局方法适应整个输出空间，可能通过利用标签相关性获得更好的预测。本研究提出了 iHOMER（增量多标签分类器层次结构），这是一种在线多标签学习框架，无需依赖预定义的层次结构，即可逐步将标签空间划分为互不相交且相关的聚类。iHOMER 利用基于 Jaccard 相似性的在线分裂-合并聚类，并通过基于多元伯努利过程的全局树型学习器指导实例划分。为应对非平稳性，它在全局和局部层面都集成了漂移检测机制，能够动态重新构建标签分区和子树。实验结果表明，iHOMER 在 23 个真实数据集上优于 5 个最先进的全局基线（如 MLHAT、Pruned Sets 的 MLHT 和 iSOUPT），性能高出 23%，并且在 12 个局部基线（如 kNN 的二元相关性转换、EFDT、ARF 和 ADWIN 袋装/提升集成）上表现高出 32%，证明了其在在线多标签分类中的 robustness。 

---
