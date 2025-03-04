# Dynamically Local-Enhancement Planner for Large-Scale Autonomous Driving 

**Title (ZH)**: 大规模自主驾驶的动态局部增强规划器 

**Authors**: Nanshan Deng, Weitao Zhou, Bo Zhang, Junze Wen, Kun Jiang, Zhong Cao, Diange Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.21134)  

**Abstract**: Current autonomous vehicles operate primarily within limited regions, but there is increasing demand for broader applications. However, as models scale, their limited capacity becomes a significant challenge for adapting to novel scenarios. It is increasingly difficult to improve models for new situations using a single monolithic model. To address this issue, we introduce the concept of dynamically enhancing a basic driving planner with local driving data, without permanently modifying the planner itself. This approach, termed the Dynamically Local-Enhancement (DLE) Planner, aims to improve the scalability of autonomous driving systems without significantly expanding the planner's size. Our approach introduces a position-varying Markov Decision Process formulation coupled with a graph neural network that extracts region-specific driving features from local observation data. The learned features describe the local behavior of the surrounding objects, which is then leveraged to enhance a basic reinforcement learning-based policy. We evaluated our approach in multiple scenarios and compared it with a one-for-all driving model. The results show that our method outperforms the baseline policy in both safety (collision rate) and average reward, while maintaining a lighter scale. This approach has the potential to benefit large-scale autonomous vehicles without the need for largely expanding on-device driving models. 

**Abstract (ZH)**: 动态局部增强驱动规划器：无需大幅扩展规划器规模即提高自动驾驶系统的可扩展性 

---
# Jointly Assigning Processes to Machines and Generating Plans for Autonomous Mobile Robots in a Smart Factory 

**Title (ZH)**: 智能工厂中联合分配过程和自主移动机器人生成计划的研究 

**Authors**: Christopher Leet, Aidan Sciortino, Sven Koenig  

**Link**: [PDF](https://arxiv.org/pdf/2502.21101)  

**Abstract**: A modern smart factory runs a manufacturing procedure using a collection of programmable machines. Typically, materials are ferried between these machines using a team of mobile robots. To embed a manufacturing procedure in a smart factory, a factory operator must a) assign its processes to the smart factory's machines and b) determine how agents should carry materials between machines. A good embedding maximizes the smart factory's throughput; the rate at which it outputs products. Existing smart factory management systems solve the aforementioned problems sequentially, limiting the throughput that they can achieve. In this paper we introduce ACES, the Anytime Cyclic Embedding Solver, the first solver which jointly optimizes the assignment of processes to machines and the assignment of paths to agents. We evaluate ACES and show that it can scale to real industrial scenarios. 

**Abstract (ZH)**: 一种现代智能工厂使用可编程机器运行制造过程。通常，材料在这些机器之间由一组移动机器人传递。要将制造过程嵌入智能工厂，工厂操作员必须完成以下两项任务：a) 将其过程分配给智能工厂的机器，b) 确定代理在机器之间携带材料的路径。良好的嵌入可以最大化智能工厂的 throughput；即其产出产品速率。现有的智能工厂管理系统按顺序解决上述问题，限制了它们能达到的 throughput。本文介绍了一种即席循环嵌入求解器 ACES，这是第一个同时优化过程分配给机器和路径分配给代理的求解器。我们评估了 ACES，并展示了它能够适应实际工业场景。 

---
# AuthSim: Towards Authentic and Effective Safety-critical Scenario Generation for Autonomous Driving Tests 

**Title (ZH)**: AuthSim: 向往真实有效的自动驾驶安全关键场景生成 

**Authors**: Yukuan Yang, Xucheng Lu, Zhili Zhang, Zepeng Wu, Guoqi Li, Lingzhong Meng, Yunzhi Xue  

**Link**: [PDF](https://arxiv.org/pdf/2502.21100)  

**Abstract**: Generating adversarial safety-critical scenarios is a pivotal method for testing autonomous driving systems, as it identifies potential weaknesses and enhances system robustness and reliability. However, existing approaches predominantly emphasize unrestricted collision scenarios, prompting non-player character (NPC) vehicles to attack the ego vehicle indiscriminately. These works overlook these scenarios' authenticity, rationality, and relevance, resulting in numerous extreme, contrived, and largely unrealistic collision events involving aggressive NPC vehicles. To rectify this issue, we propose a three-layer relative safety region model, which partitions the area based on danger levels and increases the likelihood of NPC vehicles entering relative boundary regions. This model directs NPC vehicles to engage in adversarial actions within relatively safe boundary regions, thereby augmenting the scenarios' authenticity. We introduce AuthSim, a comprehensive platform for generating authentic and effective safety-critical scenarios by integrating the three-layer relative safety region model with reinforcement learning. To our knowledge, this is the first attempt to address the authenticity and effectiveness of autonomous driving system test scenarios comprehensively. Extensive experiments demonstrate that AuthSim outperforms existing methods in generating effective safety-critical scenarios. Notably, AuthSim achieves a 5.25% improvement in average cut-in distance and a 27.12% enhancement in average collision interval time, while maintaining higher efficiency in generating effective safety-critical scenarios compared to existing methods. This underscores its significant advantage in producing authentic scenarios over current methodologies. 

**Abstract (ZH)**: 生成具有相对安全区域的可信关键安全场景用于自动驾驶系统测试 

---
# Characteristics Analysis of Autonomous Vehicle Pre-crash Scenarios 

**Title (ZH)**: 自主车辆预碰撞场景特征分析 

**Authors**: Yixuan Li, Xuesong Wang, Tianyi Wang, Qian Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20789)  

**Abstract**: To date, hundreds of crashes have occurred in open road testing of automated vehicles (AVs), highlighting the need for improving AV reliability and safety. Pre-crash scenario typology classifies crashes based on vehicle dynamics and kinematics features. Building on this, characteristics analysis can identify similar features under comparable crashes, offering a more effective reflection of general crash patterns and providing more targeted recommendations for enhancing AV performance. However, current studies primarily concentrated on crashes among conventional human-driven vehicles, leaving a gap in research dedicated to in-depth AV crash analyses. In this paper, we analyzed the latest California AV collision reports and used the newly revised pre-crash scenario typology to identify pre-crash scenarios. We proposed a set of mapping rules for automatically extracting these AV pre-crash scenarios, successfully identifying 24 types with a 98.1% accuracy rate, and obtaining two key scenarios of AV crashes (i.e., rear-end scenarios and intersection scenarios) through detailed analysis. Association analyses of rear-end scenarios showed that the significant environmental influencing factors were traffic control type, location type, light, etc. For intersection scenarios prone to severe crashes with detailed descriptions, we employed causal analyses to obtain the significant causal factors: habitual violations and expectations of certain behavior. Optimization recommendations were then formulated, addressing both governmental oversight and AV manufacturers' potential improvements. The findings of this paper could guide government authorities to develop related regulations, help manufacturers design AV test scenarios, and identify potential shortcomings in control algorithms specific to various real-world scenarios, thereby optimizing AV systems effectively. 

**Abstract (ZH)**: 到目前为止，开放道路上自动驾驶车辆（AVs）的撞车事故已经发生数百起，凸显了提高AV可靠性和安全性的必要性。基于车辆动力学和运动特征的预撞场景分类能够根据过往的撞车事件识别出相似特征，更有效地反映一般撞车模式，并为提升AV性能提出更具针对性的建议。然而，当前的研究主要集中在传统由人类驾驶的车辆上，缺乏对AV撞车事件的深入分析。本文分析了最新发布的加利福尼亚自动驾驶车辆碰撞报告，并利用新的预撞场景分类体系识别预撞场景。我们提出了一套自动提取AV预撞场景的映射规则，成功识别了24种类型，准确率为98.1%，并通过详细分析获得了两种关键AV撞车场景（即追尾场景和交叉口场景）。追尾场景的关联分析表明，显著的环境影响因素包括交通控制类型、位置类型、光线等。对于容易发生严重碰撞的交叉口场景，我们采用因果分析获得显著的因果因素：习惯性违规和对某些行为的预期。然后制定优化建议，涵盖了政府监管和AV制造商可能的改进措施。本文的研究成果可以引导政府部门制定相关法规，帮助制造商设计AV测试场景，并识别特定现实场景中控制算法的具体缺陷，从而有效地优化AV系统。 

---
# Delayed-Decision Motion Planning in the Presence of Multiple Predictions 

**Title (ZH)**: 带有多个预测的延迟决策运动规划 

**Authors**: David Isele, Alexandre Miranda Anon, Faizan M. Tariq, Goro Yeh, Avinash Singh, Sangjae Bae  

**Link**: [PDF](https://arxiv.org/pdf/2502.20636)  

**Abstract**: Reliable automated driving technology is challenged by various sources of uncertainties, in particular, behavioral uncertainties of traffic agents. It is common for traffic agents to have intentions that are unknown to others, leaving an automated driving car to reason over multiple possible behaviors. This paper formalizes a behavior planning scheme in the presence of multiple possible futures with corresponding probabilities. We present a maximum entropy formulation and show how, under certain assumptions, this allows delayed decision-making to improve safety. The general formulation is then turned into a model predictive control formulation, which is solved as a quadratic program or a set of quadratic programs. We discuss implementation details for improving computation and verify operation in simulation and on a mobile robot. 

**Abstract (ZH)**: 可靠的自动驾驶技术受到多种不确定性挑战，特别是在交通代理行为不确定性方面的挑战。交通代理常常有他人未知的意图，使得自动驾驶车辆需要推理多种可能的行为。本文在存在多种可能未来的背景下形式化了一种行为规划方案，并提出了最大熵表述，展示了在某些假设下，这如何推迟决策以提高安全性。接着，将通用表述转换为模型预测控制表述，该表述可以作为二次规划或多组二次规划求解。讨论了改进计算的实现细节，并在仿真和移动机器人上验证了操作。 

---
# Modeling Human Beliefs about AI Behavior for Scalable Oversight 

**Title (ZH)**: 基于人类对AI行为信念的建模以实现 scalable oversight 

**Authors**: Leon Lang, Patrick Forré  

**Link**: [PDF](https://arxiv.org/pdf/2502.21262)  

**Abstract**: Contemporary work in AI alignment often relies on human feedback to teach AI systems human preferences and values. Yet as AI systems grow more capable, human feedback becomes increasingly unreliable. This raises the problem of scalable oversight: How can we supervise AI systems that exceed human capabilities? In this work, we propose to model the human evaluator's beliefs about the AI system's behavior to better interpret the human's feedback. We formalize human belief models and theoretically analyze their role in inferring human values. We then characterize the remaining ambiguity in this inference and conditions for which the ambiguity disappears. To mitigate reliance on exact belief models, we then introduce the relaxation of human belief model covering. Finally, we propose using foundation models to construct covering belief models, providing a new potential approach to scalable oversight. 

**Abstract (ZH)**: 当代AI对齐研究常依赖人类反馈来教导AI系统人类的偏好和价值观。然而，随着AI系统能力的增强，人类反馈的可靠性逐渐降低。这引发了可扩展监督的问题：我们如何监督超越人类能力的AI系统？在这项工作中，我们建议通过建模人类评估者对AI系统行为的信念来更好地解释人类的反馈。我们形式化了人类信念模型，并从理论上分析了其在推断人类价值观中的作用。随后，我们描述了这一推断中剩余的模糊性以及消除模糊性的条件。为了减少对精确信念模型的依赖，我们引入了人类信念模型覆盖的放宽。最后，我们提出使用基础模型来构建覆盖信念模型，提供了一种新的可扩展监督的新潜在方法。 

---
# Towards Developing Ethical Reasoners: Integrating Probabilistic Reasoning and Decision-Making for Complex AI Systems 

**Title (ZH)**: 开发伦理推理者：将概率推理与决策结合应用于复杂AI系统 

**Authors**: Nijesh Upreti, Jessica Ciupa, Vaishak Belle  

**Link**: [PDF](https://arxiv.org/pdf/2502.21250)  

**Abstract**: A computational ethics framework is essential for AI and autonomous systems operating in complex, real-world environments. Existing approaches often lack the adaptability needed to integrate ethical principles into dynamic and ambiguous contexts, limiting their effectiveness across diverse scenarios. To address these challenges, we outline the necessary ingredients for building a holistic, meta-level framework that combines intermediate representations, probabilistic reasoning, and knowledge representation. The specifications therein emphasize scalability, supporting ethical reasoning at both individual decision-making levels and within the collective dynamics of multi-agent systems. By integrating theoretical principles with contextual factors, it facilitates structured and context-aware decision-making, ensuring alignment with overarching ethical standards. We further explore proposed theorems outlining how ethical reasoners should operate, offering a foundation for practical implementation. These constructs aim to support the development of robust and ethically reliable AI systems capable of navigating the complexities of real-world moral decision-making scenarios. 

**Abstract (ZH)**: 计算伦理框架对于在复杂实际环境中共存的AI和自主系统是必不可少的。现有的方法往往缺乏将伦理原则整合到动态和模糊情境中的适应性，限制了其在多种场景中的有效性。为解决这些挑战，我们概述了构建一个综合的、元水平框架的必要要素，该框架结合了中间表示、概率推理和知识表示。其中的规范强调了可扩展性，支持在个体决策层面和多智能体系统集体动态中的伦理推理。通过结合理论原则与情境因素，它促进结构化的、情境意识的决策，确保与总体伦理标准一致。我们进一步探讨了提出的定理，说明了伦理推理器应如何运作，为其实际应用奠定了基础。这些构建旨在支持开发稳健且伦理可靠的AI系统，使其能够应对真实世界道德决策场景的复杂性。 

---
# An Algebraic Framework for Hierarchical Probabilistic Abstraction 

**Title (ZH)**: 一种分层概率抽象的代数框架 

**Authors**: Nijesh Upreti, Vaishak Belle  

**Link**: [PDF](https://arxiv.org/pdf/2502.21216)  

**Abstract**: Abstraction is essential for reducing the complexity of systems across diverse fields, yet designing effective abstraction methodology for probabilistic models is inherently challenging due to stochastic behaviors and uncertainties. Current approaches often distill detailed probabilistic data into higher-level summaries to support tractable and interpretable analyses, though they typically struggle to fully represent the relational and probabilistic hierarchies through single-layered abstractions. We introduce a hierarchical probabilistic abstraction framework aimed at addressing these challenges by extending a measure-theoretic foundation for hierarchical abstraction. The framework enables modular problem-solving via layered mappings, facilitating both detailed layer-specific analysis and a cohesive system-wide understanding. This approach bridges high-level conceptualization with low-level perceptual data, enhancing interpretability and allowing layered analysis. Our framework provides a robust foundation for abstraction analysis across AI subfields, particularly in aligning System 1 and System 2 thinking, thereby supporting the development of diverse abstraction methodologies. 

**Abstract (ZH)**: 面向概率模型的有效抽象方法：一种层级抽象框架 

---
# A Survey of Link Prediction in Temporal Networks 

**Title (ZH)**: temporal 网络中链接预测综述 

**Authors**: Jiafeng Xiong, Ahmad Zareie, Rizos Sakellariou  

**Link**: [PDF](https://arxiv.org/pdf/2502.21185)  

**Abstract**: Temporal networks have gained significant prominence in the past decade for modelling dynamic interactions within complex systems. A key challenge in this domain is Temporal Link Prediction (TLP), which aims to forecast future connections by analysing historical network structures across various applications including social network analysis. While existing surveys have addressed specific aspects of TLP, they typically lack a comprehensive framework that distinguishes between representation and inference methods. This survey bridges this gap by introducing a novel taxonomy that explicitly examines representation and inference from existing methods, providing a novel classification of approaches for TLP. We analyse how different representation techniques capture temporal and structural dynamics, examining their compatibility with various inference methods for both transductive and inductive prediction tasks. Our taxonomy not only clarifies the methodological landscape but also reveals promising unexplored combinations of existing techniques. This taxonomy provides a systematic foundation for emerging challenges in TLP, including model explainability and scalable architectures for complex temporal networks. 

**Abstract (ZH)**: 时间网络在过去十年中因其在建模复杂系统中动态相互作用方面的显著优势而获得了广泛关注。该领域的一个关键挑战是时间链接预测（TLP），其目标是通过分析不同应用领域的历史网络结构来预测未来连接。现有综述虽然涵盖了TLP的特定方面，但通常缺乏区分表示和推理方法的全面框架。本文通过引入一个新的分类体系，明确地从现有方法的角度审视表示和推理，提供了TLP方法的一种新颖分类。我们分析了不同的表示技术如何捕获时间和结构动态，并检查它们与各类推理方法的兼容性，特别是针对归巢和归纳预测任务的情况。该分类体系不仅澄清了方法论的景观，还揭示了现有技术组合的有希望但尚未探索的组合。该分类体系为TLP中新兴挑战，包括模型可解释性和复杂时间网络的大规模架构，提供了一个系统的基础。 

---
# Are foundation models useful feature extractors for electroencephalography analysis? 

**Title (ZH)**: 深度学习基础模型在 electroencephalography 分析中有效提取特征吗？ 

**Authors**: Özgün Turgut, Felix S. Bott, Markus Ploner, Daniel Rueckert  

**Link**: [PDF](https://arxiv.org/pdf/2502.21086)  

**Abstract**: The success of foundation models in natural language processing and computer vision has motivated similar approaches for general time series analysis. While these models are effective for a variety of tasks, their applicability in medical domains with limited data remains largely unexplored. To address this, we investigate the effectiveness of foundation models in medical time series analysis involving electroencephalography (EEG). Through extensive experiments on tasks such as age prediction, seizure detection, and the classification of clinically relevant EEG events, we compare their diagnostic accuracy with that of specialised EEG models. Our analysis shows that foundation models extract meaningful EEG features, outperform specialised models even without domain adaptation, and localise task-specific biomarkers. Moreover, we demonstrate that diagnostic accuracy is substantially influenced by architectural choices such as context length. Overall, our study reveals that foundation models with general time series understanding eliminate the dependency on large domain-specific datasets, making them valuable tools for clinical practice. 

**Abstract (ZH)**: 基础模型在自然语言处理和计算机视觉中的成功促使人们将其应用于通用时间序列分析。尽管这些模型在多种任务上表现有效，但在医疗领域数据有限的情况下其应用潜力仍未得到充分探索。为解决这一问题，我们研究了基础模型在涉及脑电图(EEG)的医疗时间序列分析中的有效性。通过在年龄预测、癫痫发作检测以及临床相关EEG事件分类等多种任务中进行广泛实验，我们将基础模型的诊断准确性与专门的EEG模型进行了对比。我们的分析表明，基础模型能够提取有意义的EEG特征，在无需领域适应的情况下甚至优于专门模型，并能定位任务特定的生物标志物。此外，我们展示了架构选择，如上下文长度，对诊断准确性有显著影响。总体而言，我们的研究揭示了具有通用时间序列理解能力的基础模型可以消除对大量领域特定数据集的依赖，使它们成为临床实践中的有价值工具。 

---
# A Pilot Empirical Study on When and How to Use Knowledge Graphs as Retrieval Augmented Generation 

**Title (ZH)**: 一项关于何时及如何使用知识图谱增强生成的试点实证研究 

**Authors**: Xujie Yuan, Yongxu Liu, Shimin Di, Shiwen Wu, Libin Zheng, Rui Meng, Xiaofang Zhou, Lei Chen, Jian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2502.20854)  

**Abstract**: The integration of Knowledge Graphs (KGs) into the Retrieval Augmented Generation (RAG) framework has attracted significant interest, with early studies showing promise in mitigating hallucinations and improving model accuracy. However, a systematic understanding and comparative analysis of the rapidly emerging KG-RAG methods are still lacking. This paper seeks to lay the foundation for systematically answering the question of when and how to use KG-RAG by analyzing their performance in various application scenarios associated with different technical configurations. After outlining the mind map using KG-RAG framework and summarizing its popular pipeline, we conduct a pilot empirical study of KG-RAG works to reimplement and evaluate 6 KG-RAG methods across 7 datasets in diverse scenarios, analyzing the impact of 9 KG-RAG configurations in combination with 17 LLMs. Our results underscore the critical role of appropriate application conditions and optimal configurations of KG-RAG components. 

**Abstract (ZH)**: 知识图谱（KGs）融入检索增强生成（RAG）框架的研究及其应用分析仍有待系统性理解与比较：一项关于KG-RAG方法性能的试点实证研究 

---
# MedHallTune: An Instruction-Tuning Benchmark for Mitigating Medical Hallucination in Vision-Language Models 

**Title (ZH)**: MedHallTune: 一个减轻视觉语言模型医疗幻觉的指令调优基准测试 

**Authors**: Qiao Yan, Yuchen Yuan, Xiaowei Hu, Yihan Wang, Jiaqi Xu, Jinpeng Li, Chi-Wing Fu, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2502.20780)  

**Abstract**: The increasing use of vision-language models (VLMs) in healthcare applications presents great challenges related to hallucinations, in which the models may generate seemingly plausible results that are in fact incorrect. Such hallucinations can jeopardize clinical decision making, potentially harming the diagnosis and treatments. In this work, we propose MedHallTune, a large-scale benchmark designed specifically to evaluate and mitigate hallucinations in medical VLMs. Comprising over 100,000 images and 1,000,000 instruction pairs, MedHallTune includes both hallucination and non-hallucination samples, each with ground-truth annotations. We conduct a comprehensive evaluation of current medical and general VLMs using MedHallTune, assessing their performance across key metrics, including clinical accuracy, relevance, detail level, and risk level. The experimental results show that fine-tuning with MedHallTune successfully improves the ability of several existing models to manage hallucinations and boost their zero-shot performance on downstream visual-question-answering (VQA) tasks, making them more reliable for practical medical applications. Our work contributes to the development of more trustworthy VLMs. Codes and dataset will be available at \href{this https URL}{MedHallTune}. 

**Abstract (ZH)**: 视觉-语言模型在医疗应用中日益增多，带来了与幻觉相关的重要挑战，模型可能生成看似合理但实际上错误的结果，这可能危及临床决策，影响诊断和治疗。本文提出MedHallTune，这是一个大规模基准，专门用于评估和缓解医疗视觉-语言模型中的幻觉问题。MedHallTune包含超过100,000张图像和1,000,000个指令对，包括幻觉和非幻觉样本，并附有ground-truth注解。我们使用MedHallTune对当前的医疗和通用视觉-语言模型进行全面评估，评估其在临床准确性、相关性、细节水平和风险水平等方面的表现。实验结果表明，使用MedHallTunefine-tuning显著提高了几种现有模型处理幻觉的能力，并增强了其在下游视觉问答(VQA)任务中的零样本性能，使它们更适合实际的医疗应用。我们的工作有助于开发更可信赖的视觉-语言模型。相关代码和数据集将在MedHallTune获得。 

---
# Damper-B-PINN: Damper Characteristics-Based Bayesian Physics-Informed Neural Network for Vehicle State Estimation 

**Title (ZH)**: 基于阻尼特性 Bayesian 物理相关神经网络的车辆状态估计 

**Authors**: Tianyi Zeng, Tianyi Wang, Junfeng Jiao, Xinbo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20772)  

**Abstract**: State estimation for Multi-Input Multi-Output (MIMO) systems with noise, such as vehicle chassis systems, presents a significant challenge due to the imperfect and complex relationship between inputs and outputs. To solve this problem, we design a Damper characteristics-based Bayesian Physics-Informed Neural Network (Damper-B-PINN). First, we introduce a neuron forward process inspired by the mechanical properties of dampers, which limits abrupt jumps in neuron values between epochs while maintaining search capability. Additionally, we apply an optimized Bayesian dropout layer to the MIMO system to enhance robustness against noise and prevent non-convergence issues. Physical information is incorporated into the loss function to serve as a physical prior for the neural network. The effectiveness of our Damper-B-PINN architecture is then validated across ten datasets and fourteen vehicle types, demonstrating superior accuracy, computational efficiency, and convergence in vehicle state estimation (i.e., dynamic wheel load) compared to other state-of-the-art benchmarks. 

**Abstract (ZH)**: 基于阻尼特性的时间物理感知神经网络（Damper-B-PINN）在多输入多输出系统状态估计中的应用 

---
# DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking 

**Title (ZH)**: DeepSolution：通过基于树的探索和双向思考提升复杂工程解决方案设计 

**Authors**: Zhuoqun Li, Haiyang Yu, Xuanang Chen, Hongyu Lin, Yaojie Lu, Fei Huang, Xianpei Han, Yongbin Li, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.20730)  

**Abstract**: Designing solutions for complex engineering challenges is crucial in human production activities. However, previous research in the retrieval-augmented generation (RAG) field has not sufficiently addressed tasks related to the design of complex engineering solutions. To fill this gap, we introduce a new benchmark, SolutionBench, to evaluate a system's ability to generate complete and feasible solutions for engineering problems with multiple complex constraints. To further advance the design of complex engineering solutions, we propose a novel system, SolutionRAG, that leverages the tree-based exploration and bi-point thinking mechanism to generate reliable solutions. Extensive experimental results demonstrate that SolutionRAG achieves state-of-the-art (SOTA) performance on the SolutionBench, highlighting its potential to enhance the automation and reliability of complex engineering solution design in real-world applications. 

**Abstract (ZH)**: 设计复杂工程解决方案是人类生产活动中的关键。然而，先前在检索增强生成（RAG）领域中的研究尚未充分解决与复杂工程解决方案设计相关的问题。为了填补这一空白，我们引入了一个新的基准——SolutionBench，以评估系统生成多约束条件下完整且可行的工程问题解决方案的能力。为进一步推动复杂工程解决方案的设计，我们提出了一种新颖的系统——SolutionRAG，该系统利用基于树的探索和双点思维机制生成可靠解决方案。大量实验证明，SolutionRAG在SolutionBench上的性能达到了最先进的水平，突显了其在实际应用中增强复杂工程解决方案设计的自动化和可靠性的潜力。 

---
# Fuzzy Speculative Decoding for a Tunable Accuracy-Runtime Tradeoff 

**Title (ZH)**: 可调准确率-时延 TRADEOFF 的模糊猜测译码 

**Authors**: Maximilian Holsman, Yukun Huang, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2502.20704)  

**Abstract**: Speculative Decoding (SD) enforces strict distributional equivalence to the target model, limiting potential speed ups as distributions of near-equivalence achieve comparable outcomes in many cases. Furthermore, enforcing distributional equivalence means that users are unable to trade deviations from the target model distribution for further inference speed gains. To address these limitations, we introduce Fuzzy Speculative Decoding (FSD) - a decoding algorithm that generalizes SD by accepting candidate tokens purely based on the divergences between the target and draft model distributions. By allowing for controlled divergence from the target model, FSD enables users to flexibly trade generation quality for inference speed. Across several benchmarks, our method is able to achieve significant runtime improvements of over 5 tokens per second faster than SD at only an approximate 2% absolute reduction in benchmark accuracy. In many cases, FSD is even able to match SD benchmark accuracy at over 2 tokens per second faster, demonstrating that distributional equivalence is not necessary to maintain target model performance. 

**Abstract (ZH)**: 模糊推测解码（FSD）：一种通过接受基于目标模型和草稿模型分布偏差的候选项来泛化推测解码的解码算法 

---
# Why Trust in AI May Be Inevitable 

**Title (ZH)**: 为什么对AI的信任可能是不可避免的 

**Authors**: Nghi Truong, Phanish Puranam, Ilia Testlin  

**Link**: [PDF](https://arxiv.org/pdf/2502.20701)  

**Abstract**: In human-AI interactions, explanation is widely seen as necessary for enabling trust in AI systems. We argue that trust, however, may be a pre-requisite because explanation is sometimes impossible. We derive this result from a formalization of explanation as a search process through knowledge networks, where explainers must find paths between shared concepts and the concept to be explained, within finite time. Our model reveals that explanation can fail even under theoretically ideal conditions - when actors are rational, honest, motivated, can communicate perfectly, and possess overlapping knowledge. This is because successful explanation requires not just the existence of shared knowledge but also finding the connection path within time constraints, and it can therefore be rational to cease attempts at explanation before the shared knowledge is discovered. This result has important implications for human-AI interaction: as AI systems, particularly Large Language Models, become more sophisticated and able to generate superficially compelling but spurious explanations, humans may default to trust rather than demand genuine explanations. This creates risks of both misplaced trust and imperfect knowledge integration. 

**Abstract (ZH)**: 在人-AI交互中，信任可能是启用对AI系统信任的前提，而非必要条件。 

---
# Automatic database description generation for Text-to-SQL 

**Title (ZH)**: 文本到SQL的自动数据库描述生成 

**Authors**: Yingqi Gao, Zhiling Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.20657)  

**Abstract**: In the context of the Text-to-SQL task, table and column descriptions are crucial for bridging the gap between natural language and database schema. This report proposes a method for automatically generating effective database descriptions when explicit descriptions are unavailable. The proposed method employs a dual-process approach: a coarse-to-fine process, followed by a fine-to-coarse process. The coarse-to-fine approach leverages the inherent knowledge of LLM to guide the understanding process from databases to tables and finally to columns. This approach provides a holistic understanding of the database structure and ensures contextual alignment. Conversely, the fine-to-coarse approach starts at the column level, offering a more accurate and nuanced understanding when stepping back to the table level. Experimental results on the Bird benchmark indicate that using descriptions generated by the proposed improves SQL generation accuracy by 0.93\% compared to not using descriptions, and achieves 37\% of human-level performance. The source code is publicly available at this https URL. 

**Abstract (ZH)**: 在Text-to-SQL任务中，表格和列描述对于自然语言与数据库模式之间的桥梁构建至关重要。本报告提出了一种在缺乏显式描述时自动生成有效数据库描述的方法。该方法采用双过程方法：粗到细过程，接着是细到粗过程。粗到细的过程利用LLM的固有知识，从数据库到表再到列逐步引导理解过程，提供一个全面的数据库结构理解，并确保上下文对齐。相反，细到粗的过程从列级开始，逐步回退到表级，提供更准确和细腻的理解。实验结果表明，与不使用描述相比，使用所提出方法生成的描述可以将SQL生成准确性提高0.93%，并达到人类水平性能的37%。源代码在此公开链接处获取。 

---
# PersonaBench: Evaluating AI Models on Understanding Personal Information through Accessing (Synthetic) Private User Data 

**Title (ZH)**: PersonaBench: 评估AI模型在访问（合成）私人用户数据时理解个人信息的能力 

**Authors**: Juntao Tan, Liangwei Yang, Zuxin Liu, Zhiwei Liu, Rithesh Murthy, Tulika Manoj Awalgaonkar, Jianguo Zhang, Weiran Yao, Ming Zhu, Shirley Kokane, Silvio Savarese, Huan Wang, Caiming Xiong, Shelby Heinecke  

**Link**: [PDF](https://arxiv.org/pdf/2502.20616)  

**Abstract**: Personalization is critical in AI assistants, particularly in the context of private AI models that work with individual users. A key scenario in this domain involves enabling AI models to access and interpret a user's private data (e.g., conversation history, user-AI interactions, app usage) to understand personal details such as biographical information, preferences, and social connections. However, due to the sensitive nature of such data, there are no publicly available datasets that allow us to assess an AI model's ability to understand users through direct access to personal information.
To address this gap, we introduce a synthetic data generation pipeline that creates diverse, realistic user profiles and private documents simulating human activities. Leveraging this synthetic data, we present PersonaBench, a benchmark designed to evaluate AI models' performance in understanding personal information derived from simulated private user data.
We evaluate Retrieval-Augmented Generation (RAG) pipelines using questions directly related to a user's personal information, supported by the relevant private documents provided to the models. Our results reveal that current retrieval-augmented AI models struggle to answer private questions by extracting personal information from user documents, highlighting the need for improved methodologies to enhance personalization capabilities in AI. 

**Abstract (ZH)**: 个性化对于AI助手至关重要，尤其是在涉及个人用户private AI模型的情况下。在这种背景下，一个关键场景是让AI模型访问和解读用户的私人数据（例如对话历史、用户-AI交互、应用使用情况），以理解个人细节，如个人资料信息、偏好和社会联系。但由于这些数据的敏感性，目前没有公有数据集可以评估AI模型通过直接访问个人数据理解用户的能力。
为了填补这一空白，我们引入了一个合成数据生成管道，创建多样且现实的用户资料和模拟人类活动的私人文件。利用这些合成数据，我们提出了PersonaBench，一个用于评估AI模型在理解来自模拟私人用户数据的个人信息方面的表现的基准。
我们使用与用户个人信息相关的问题评估检索增强生成（RAG）管道，并提供相关私人文档支持模型。我们的结果显示，当前的检索增强AI模型在从用户文件中提取个人信息以回答私人问题方面存在挑战，这突显了提高AI个性化能力的方法学改进需求。 

---
# On Benchmarking Human-Like Intelligence in Machines 

**Title (ZH)**: 在机器中benchmark人类水平的智能 

**Authors**: Lance Ying, Katherine M. Collins, Lionel Wong, Ilia Sucholutsky, Ryan Liu, Adrian Weller, Tianmin Shu, Thomas L. Griffiths, Joshua B. Tenenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2502.20502)  

**Abstract**: Recent benchmark studies have claimed that AI has approached or even surpassed human-level performances on various cognitive tasks. However, this position paper argues that current AI evaluation paradigms are insufficient for assessing human-like cognitive capabilities. We identify a set of key shortcomings: a lack of human-validated labels, inadequate representation of human response variability and uncertainty, and reliance on simplified and ecologically-invalid tasks. We support our claims by conducting a human evaluation study on ten existing AI benchmarks, suggesting significant biases and flaws in task and label designs. To address these limitations, we propose five concrete recommendations for developing future benchmarks that will enable more rigorous and meaningful evaluations of human-like cognitive capacities in AI with various implications for such AI applications. 

**Abstract (ZH)**: 近年来，基准研究声称AI已在各种认知任务上达到了甚至超越了人类水平。然而，本文认为当前的AI评估范式不足以评估类人类的认知能力。我们确定了一系列关键缺陷：缺乏经人类验证的标签、人类反应变异性和不确定性表示不足，以及依赖于简化且生态无效的任务。我们通过在十个现有AI基准上进行的人类评估研究支持这些观点，指出任务和标签设计中存在重大偏差和缺陷。为了应对这些限制，我们提出了五项具体建议，以开发未来能够更严谨和有意义地评估具有各种应用含义的人类类认知能力的基准。 

---
# R-ParVI: Particle-based variational inference through lens of rewards 

**Title (ZH)**: R-ParVI: 基于奖励视角的粒子变分推断 

**Authors**: Yongchao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20482)  

**Abstract**: A reward-guided, gradient-free ParVI method, \textit{R-ParVI}, is proposed for sampling partially known densities (e.g. up to a constant). R-ParVI formulates the sampling problem as particle flow driven by rewards: particles are drawn from a prior distribution, navigate through parameter space with movements determined by a reward mechanism blending assessments from the target density, with the steady state particle configuration approximating the target geometry. Particle-environment interactions are simulated by stochastic perturbations and the reward mechanism, which drive particles towards high density regions while maintaining diversity (e.g. preventing from collapsing into clusters). R-ParVI offers fast, flexible, scalable and stochastic sampling and inference for a class of probabilistic models such as those encountered in Bayesian inference and generative modelling. 

**Abstract (ZH)**: 基于奖励导向的无梯度ParVI方法R-ParVI用于采样部分已知密度（例如，相差常数的情况） 

---
# Beyond transparency: computational reliabilism as an externalist epistemology of algorithms 

**Title (ZH)**: 超越透明性：计算可靠主义作为算法的外部主义认识论 

**Authors**: Juan Manuel Durán  

**Link**: [PDF](https://arxiv.org/pdf/2502.20402)  

**Abstract**: This chapter is interested in the epistemology of algorithms. As I intend to approach the topic, this is an issue about epistemic justification. Current approaches to justification emphasize the transparency of algorithms, which entails elucidating their internal mechanisms -- such as functions and variables -- and demonstrating how (or that) these produce outputs. Thus, the mode of justification through transparency is contingent on what can be shown about the algorithm and, in this sense, is internal to the algorithm. In contrast, I advocate for an externalist epistemology of algorithms that I term computational reliabilism (CR). While I have previously introduced and examined CR in the field of computer simulations ([42, 53, 4]), this chapter extends this reliabilist epistemology to encompass a broader spectrum of algorithms utilized in various scientific disciplines, with a particular emphasis on machine learning applications. At its core, CR posits that an algorithm's output is justified if it is produced by a reliable algorithm. A reliable algorithm is one that has been specified, coded, used, and maintained utilizing reliability indicators. These reliability indicators stem from formal methods, algorithmic metrics, expert competencies, cultures of research, and other scientific endeavors. The primary aim of this chapter is to delineate the foundations of CR, explicate its operational mechanisms, and outline its potential as an externalist epistemology of algorithms. 

**Abstract (ZH)**: 本章关注算法的认识论。在探讨这一主题时，这是一个关于认知正当性的议题。当前的认知正当性方法强调算法的透明性，这要求阐明其内部机制——如函数和变量——并证明这些机制是如何（或者为什么）产生输出的。因此，通过透明性进行的认知正当性依赖于可以展示的关于算法的内容，这种正当性在这一意义上是内部的。相反，我提倡一种外部主义的算法认识论，我称之为计算可靠论（Computational Reliability, CR）。虽然在此之前已在我对计算机模拟领域的研究中引入和探讨了CR（[42, 53, 4]），但本章将这一可靠论的认识论扩展到涵盖各种科学学科中使用的广泛算法，特别是机器学习应用。CR的核心观点是，如果一个算法的输出是由一个可靠的算法生成的，则这一输出是正当的。一个可靠的算法是指通过使用可靠性指标来指定、编码、使用和维护的算法。这些可靠性指标来源于形式方法、算法度量、专家能力、研究文化以及其他科学研究。本章的主要目的是阐明CR的基础，解释其运作机制，并概述其作为算法外部主义认识论的潜力。 

---
# Clustering Context in Off-Policy Evaluation 

**Title (ZH)**: 离策评估中的上下文聚类 

**Authors**: Daniel Guzman-Olivares, Philipp Schmidt, Jacek Golebiowski, Artur Bekasov  

**Link**: [PDF](https://arxiv.org/pdf/2502.21304)  

**Abstract**: Off-policy evaluation can leverage logged data to estimate the effectiveness of new policies in e-commerce, search engines, media streaming services, or automatic diagnostic tools in healthcare. However, the performance of baseline off-policy estimators like IPS deteriorates when the logging policy significantly differs from the evaluation policy. Recent work proposes sharing information across similar actions to mitigate this problem. In this work, we propose an alternative estimator that shares information across similar contexts using clustering. We study the theoretical properties of the proposed estimator, characterizing its bias and variance under different conditions. We also compare the performance of the proposed estimator and existing approaches in various synthetic problems, as well as a real-world recommendation dataset. Our experimental results confirm that clustering contexts improves estimation accuracy, especially in deficient information settings. 

**Abstract (ZH)**: 离策略评估可以通过利用日志数据来估算新的政策在电子商务、搜索引擎、媒体流服务或医疗中的诊断工具中的有效性。然而，基线离策略估计器如IPS在记录策略与评估策略有显著差异时性能会下降。近期工作提出了共享相似动作间的信息以缓解这一问题的方法。在这项工作中，我们提出了一种新的估计器，该估计器通过聚类共享相似上下文间的信息。我们研究了所提出估计器的理论性质，分别在不同条件下对其偏差和方差进行了刻画。我们还在各种合成问题以及一个实际推荐数据集中比较了所提出估计器与现有方法的性能。我们的实验结果证实，在信息不足的环境中，聚类上下文可以提高估计准确性。 

---
# BAnG: Bidirectional Anchored Generation for Conditional RNA Design 

**Title (ZH)**: BAnG: 双向锚定生成方法在条件RNA设计中的应用 

**Authors**: Roman Klypa, Alberto Bietti, Sergei Grudinin  

**Link**: [PDF](https://arxiv.org/pdf/2502.21274)  

**Abstract**: Designing RNA molecules that interact with specific proteins is a critical challenge in experimental and computational biology. Existing computational approaches require a substantial amount of experimentally determined RNA sequences for each specific protein or a detailed knowledge of RNA structure, restricting their utility in practice. To address this limitation, we develop RNA-BAnG, a deep learning-based model designed to generate RNA sequences for protein interactions without these requirements. Central to our approach is a novel generative method, Bidirectional Anchored Generation (BAnG), which leverages the observation that protein-binding RNA sequences often contain functional binding motifs embedded within broader sequence contexts. We first validate our method on generic synthetic tasks involving similar localized motifs to those appearing in RNAs, demonstrating its benefits over existing generative approaches. We then evaluate our model on biological sequences, showing its effectiveness for conditional RNA sequence design given a binding protein. 

**Abstract (ZH)**: 基于深度学习的设计 RNA 分子与特定蛋白质交互的研究：RNA-BAnG模型开发及其应用 

---
# Adaptive Keyframe Sampling for Long Video Understanding 

**Title (ZH)**: 长视频理解中的自适应关键帧采样 

**Authors**: Xi Tang, Jihao Qiu, Lingxi Xie, Yunjie Tian, Jianbin Jiao, Qixiang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2502.21271)  

**Abstract**: Multimodal large language models (MLLMs) have enabled open-world visual understanding by injecting visual input as extra tokens into large language models (LLMs) as contexts. However, when the visual input changes from a single image to a long video, the above paradigm encounters difficulty because the vast amount of video tokens has significantly exceeded the maximal capacity of MLLMs. Therefore, existing video-based MLLMs are mostly established upon sampling a small portion of tokens from input data, which can cause key information to be lost and thus produce incorrect answers. This paper presents a simple yet effective algorithm named Adaptive Keyframe Sampling (AKS). It inserts a plug-and-play module known as keyframe selection, which aims to maximize the useful information with a fixed number of video tokens. We formulate keyframe selection as an optimization involving (1) the relevance between the keyframes and the prompt, and (2) the coverage of the keyframes over the video, and present an adaptive algorithm to approximate the best solution. Experiments on two long video understanding benchmarks validate that Adaptive Keyframe Sampling improves video QA accuracy (beyond strong baselines) upon selecting informative keyframes. Our study reveals the importance of information pre-filtering in video-based MLLMs. Code is available at this https URL. 

**Abstract (ZH)**: 多模态大规模语言模型通过将视觉输入作为额外标记注入大型语言模型中，实现了开放世界的视觉理解。然而，当视觉输入从单张图像变为长视频时，上述范式遇到困难，因为大量的视频标记远远超过了多模态大规模语言模型的最大容量。因此，现有的基于视频的多模态大规模语言模型大多基于从输入数据中抽样一小部分标记建立，这可能导致关键信息丢失并产生错误的答案。本文提出了一种简单而有效的算法，称为自适应关键帧采样（Adaptive Keyframe Sampling, AKS）。该算法插入了一个插拔式的模块，即关键帧选择，旨在用固定数量的视频标记最大化有用信息。我们将关键帧选择形式化为涉及（1）关键帧与提示的相关性，以及（2）关键帧在视频中的覆盖范围的优化，并提出了一种自适应算法来近似最佳解。在两个长视频理解基准上的实验验证了自适应关键帧采样通过选择具有信息量的关键帧提高了视频问答准确性（超越了强大基线）。我们的研究揭示了视频基础的多模态大规模语言模型中信息预过滤的重要性。代码可在以下链接获取。 

---
# Supporting the development of Machine Learning for fundamental science in a federated Cloud with the AI_INFN platform 

**Title (ZH)**: 在AI_INFN平台支持下的联邦云中推进基础科学的机器学习发展 

**Authors**: Lucio Anderlini, Matteo Barbetti, Giulio Bianchini, Diego Ciangottini, Stefano Dal Pra, Diego Michelotto, Carmelo Pellegrino, Rosa Petrini, Alessandro Pascolini, Daniele Spiga  

**Link**: [PDF](https://arxiv.org/pdf/2502.21266)  

**Abstract**: Machine Learning (ML) is driving a revolution in the way scientists design, develop, and deploy data-intensive software. However, the adoption of ML presents new challenges for the computing infrastructure, particularly in terms of provisioning and orchestrating access to hardware accelerators for development, testing, and production. The INFN-funded project AI_INFN ("Artificial Intelligence at INFN") aims at fostering the adoption of ML techniques within INFN use cases by providing support on multiple aspects, including the provision of AI-tailored computing resources. It leverages cloud-native solutions in the context of INFN Cloud, to share hardware accelerators as effectively as possible, ensuring the diversity of the Institute's research activities is not compromised. In this contribution, we provide an update on the commissioning of a Kubernetes platform designed to ease the development of GPU-powered data analysis workflows and their scalability on heterogeneous, distributed computing resources, possibly federated as Virtual Kubelets with the interLink provider. 

**Abstract (ZH)**: 机器学习（ML）正在推动科学家在设计、开发和部署数据密集型软件方式上的革命。然而，ML的采用为计算基础设施带来了新的挑战，特别是在开发、测试和生产中提供和编排硬件加速器访问方面。由INFN资助的AI_INFN项目（“INFN人工智能”）旨在通过在多个方面提供支持，促进ML技术在INFN使用案例中的应用，包括提供人工智能定制计算资源。该项目利用INFN云中的原生云解决方案，尽可能有效地共享硬件加速器，确保研究所的研究活动多样性不受影响。在此贡献中，我们提供了为简化基于GPU的数据分析工作流开发及其在异构分布式计算资源上的扩展性而设计的Kubernetes平台的最新情况，这些资源可能通过interLink提供商作为虚拟kubelet进行联邦化。 

---
# Foundation Models -- A Panacea for Artificial Intelligence in Pathology? 

**Title (ZH)**: 基础模型—— pathology 中人工智能的万能药？ 

**Authors**: Nita Mulliqi, Anders Blilie, Xiaoyi Ji, Kelvin Szolnoky, Henrik Olsson, Sol Erika Boman, Matteo Titus, Geraldine Martinez Gonzalez, Julia Anna Mielcarz, Masi Valkonen, Einar Gudlaugsson, Svein R. Kjosavik, José Asenjo, Marcello Gambacorta, Paolo Libretti, Marcin Braun, Radzislaw Kordek, Roman Łowicki, Kristina Hotakainen, Päivi Väre, Bodil Ginnerup Pedersen, Karina Dalsgaard Sørensen, Benedicte Parm Ulhøi, Pekka Ruusuvuori, Brett Delahunt, Hemamali Samaratunga, Toyonori Tsuzuki, Emilius A.M. Janssen, Lars Egevad, Martin Eklund, Kimmo Kartasalo  

**Link**: [PDF](https://arxiv.org/pdf/2502.21264)  

**Abstract**: The role of artificial intelligence (AI) in pathology has evolved from aiding diagnostics to uncovering predictive morphological patterns in whole slide images (WSIs). Recently, foundation models (FMs) leveraging self-supervised pre-training have been widely advocated as a universal solution for diverse downstream tasks. However, open questions remain about their clinical applicability and generalization advantages over end-to-end learning using task-specific (TS) models. Here, we focused on AI with clinical-grade performance for prostate cancer diagnosis and Gleason grading. We present the largest validation of AI for this task, using over 100,000 core needle biopsies from 7,342 patients across 15 sites in 11 countries. We compared two FMs with a fully end-to-end TS model in a multiple instance learning framework. Our findings challenge assumptions that FMs universally outperform TS models. While FMs demonstrated utility in data-scarce scenarios, their performance converged with - and was in some cases surpassed by - TS models when sufficient labeled training data were available. Notably, extensive task-specific training markedly reduced clinically significant misgrading, misdiagnosis of challenging morphologies, and variability across different WSI scanners. Additionally, FMs used up to 35 times more energy than the TS model, raising concerns about their sustainability. Our results underscore that while FMs offer clear advantages for rapid prototyping and research, their role as a universal solution for clinically applicable medical AI remains uncertain. For high-stakes clinical applications, rigorous validation and consideration of task-specific training remain critically important. We advocate for integrating the strengths of FMs and end-to-end learning to achieve robust and resource-efficient AI pathology solutions fit for clinical use. 

**Abstract (ZH)**: 人工智能在病理学中的作用从辅助诊断演变为空整个组织切片图像（WSI）中发现预测形态学模式。最近，利用自我监督预训练的基础模型（FMs）被广泛推荐为解决多样下游任务的通用解决方案。然而，其在临床应用中的适用性和与针对特定任务（TS）模型的端到端学习相比的泛化优势仍存在疑问。在此，我们专注于具有临床级别性能的前列腺癌诊断和Gleason分级的AI。我们使用来自11个国家15个中心的7,342名患者的超过100,000个芯针活检样本对该任务进行了最大的验证。我们在多项实例学习框架中比较了两种FMs和一个完全端到端的TS模型。我们的发现挑战了FMs普遍优于TS模型的假设。尽管FMs在数据稀缺场景中显示出实用性，但在有足够的标注训练数据时，其性能与TS模型一致，并在某些情况下被TS模型超越。此外，FMs的能耗是TS模型的35倍以上，引起了对其可持续性的担忧。我们的结果表明，虽然FMs在快速原型设计和研究中具有明显优势，但它们作为临床适用的医疗AI的通用解决方案的角色仍不确定。对于高风险的临床应用，严格验证和考虑针对特定任务的训练仍然是至关重要的。我们倡导结合FMs和端到端学习的优势，以实现符合临床使用要求的稳健且资源高效的AI病理解决方案。 

---
# RuCCoD: Towards Automated ICD Coding in Russian 

**Title (ZH)**: RuCCoD: 向自动化俄罗斯ICD编码迈进 

**Authors**: Aleksandr Nesterov, Andrey Sakhovskiy, Ivan Sviridov, Airat Valiev, Vladimir Makharev, Petr Anokhin, Galina Zubkova, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2502.21263)  

**Abstract**: This study investigates the feasibility of automating clinical coding in Russian, a language with limited biomedical resources. We present a new dataset for ICD coding, which includes diagnosis fields from electronic health records (EHRs) annotated with over 10,000 entities and more than 1,500 unique ICD codes. This dataset serves as a benchmark for several state-of-the-art models, including BERT, LLaMA with LoRA, and RAG, with additional experiments examining transfer learning across domains (from PubMed abstracts to medical diagnosis) and terminologies (from UMLS concepts to ICD codes). We then apply the best-performing model to label an in-house EHR dataset containing patient histories from 2017 to 2021. Our experiments, conducted on a carefully curated test set, demonstrate that training with the automated predicted codes leads to a significant improvement in accuracy compared to manually annotated data from physicians. We believe our findings offer valuable insights into the potential for automating clinical coding in resource-limited languages like Russian, which could enhance clinical efficiency and data accuracy in these contexts. 

**Abstract (ZH)**: 本研究探讨了在资源有限的俄语环境下自动化临床编码的可行性。我们提出了一项新的ICD编码数据集，其中包括来自电子健康记录（EHRs）的超过10,000个实体和1,500多个唯一ICD代码的诊断字段标注。该数据集被用于多种先进模型（包括BERT、带有LoRA的LLaMA和RAG）的基准测试，并进行跨领域（从PubMed摘要到医疗诊断）和术语（从UMLS概念到ICD代码）的迁移学习实验。随后，我们将表现最佳的模型应用于一个包含2017年至2021年患者历史记录的内部EHR数据集进行标注。通过在精心选择的测试集上进行的实验表明，使用自动化预测的编码进行训练可以显著提高准确性，优于医生手动标注的数据。我们认为我们的研究结果为资源有限的语言（如俄语）的临床编码自动化提供了有价值的见解，这可能在这些环境中提高临床效率和数据准确性。 

---
# AMPLE: Event-Driven Accelerator for Mixed-Precision Inference of Graph Neural Networks 

**Title (ZH)**: AMPLE：事件驱动混合精度图神经网络推理加速器 

**Authors**: Pedro Gimenes, Yiren Zhao, George Constantinides  

**Link**: [PDF](https://arxiv.org/pdf/2502.21196)  

**Abstract**: Graph Neural Networks (GNNs) have recently gained attention due to their performance on non-Euclidean data. The use of custom hardware architectures proves particularly beneficial for GNNs due to their irregular memory access patterns, resulting from the sparse structure of graphs. However, existing FPGA accelerators are limited by their double buffering mechanism, which doesn't account for the irregular node distribution in typical graph datasets. To address this, we introduce \textbf{AMPLE} (Accelerated Message Passing Logic Engine), an FPGA accelerator leveraging a new event-driven programming flow. We develop a mixed-arithmetic architecture, enabling GNN inference to be quantized at a node-level granularity. Finally, prefetcher for data and instructions is implemented to optimize off-chip memory access and maximize node parallelism. Evaluation on citation and social media graph datasets ranging from $2$K to $700$K nodes showed a mean speedup of $243\times$ and $7.2\times$ against CPU and GPU counterparts, respectively. 

**Abstract (ZH)**: 基于FPGA的加速器AMPLE：图神经网络的事件驱动计算架构 

---
# Predicting clinical outcomes from patient care pathways represented with temporal knowledge graphs 

**Title (ZH)**: 基于时间知识图表示的患者护理路径预测临床结果 

**Authors**: Jong Ho Jhee, Alberto Megina, Pacôme Constant Dit Beaufils, Matilde Karakachoff, Richard Redon, Alban Gaignard, Adrien Coulet  

**Link**: [PDF](https://arxiv.org/pdf/2502.21138)  

**Abstract**: Background: With the increasing availability of healthcare data, predictive modeling finds many applications in the biomedical domain, such as the evaluation of the level of risk for various conditions, which in turn can guide clinical decision making. However, it is unclear how knowledge graph data representations and their embedding, which are competitive in some settings, could be of interest in biomedical predictive modeling. Method: We simulated synthetic but realistic data of patients with intracranial aneurysm and experimented on the task of predicting their clinical outcome. We compared the performance of various classification approaches on tabular data versus a graph-based representation of the same data. Next, we investigated how the adopted schema for representing first individual data and second temporal data impacts predictive performances. Results: Our study illustrates that in our case, a graph representation and Graph Convolutional Network (GCN) embeddings reach the best performance for a predictive task from observational data. We emphasize the importance of the adopted schema and of the consideration of literal values in the representation of individual data. Our study also moderates the relative impact of various time encoding on GCN performance. 

**Abstract (ZH)**: 背景：随着医疗健康数据的不断增加，预测建模在生物医学领域得到了广泛应用，例如评估各种条件的风险水平，进而指导临床决策。然而，在某些情况下，知识图谱数据表示及其嵌入是否对生物医学预测建模具有研究价值尚不清楚。方法：我们模拟了颅内动脉瘤患者的合成但现实数据，并对预测其临床结果的任务进行了实验。我们比较了各种分类方法在表格数据与基于图的数据表示之间的性能。接着，我们研究了表示个体数据的模式和表示时间数据的模式如何影响预测性能。结果：我们的研究表明，在我们的案例中，图表示和图卷积网络（GCN）嵌入最适合从观察数据中进行预测任务。我们强调了所采用模式和个体数据表示中literal值的重要性。此外，我们的研究还减弱了各种时间编码对GCN性能的相对影响。 

---
# Einleitung [Introduction] 

**Title (ZH)**: 简介 

**Authors**: Vincent C. Müller  

**Link**: [PDF](https://arxiv.org/pdf/2502.21131)  

**Abstract**: Hilary Putnam's biography and philosophical development reflect the history of Anglo-Saxon philosophy over the last 40 years. Putnam has influenced this history significantly for almost as long. In this introduction, the main aim is to present the context in which Putnam stands and from which his philosophical contributions can be understood. In the context of a sketch of Putnam's philosophical development, a preliminary historical classification of his work will also be attempted, even if this is not the place for a comprehensive critique or presentation: The introduction must remain at a fairly elementary level and of course cannot replace a reading of the texts. Since Putnam's work is certainly part of a rapprochement between 'analytic' and 'continental' philosophy, the introduction to the texts translated here should finally make clear what Putnam has to offer non-analytically oriented readers.
Hilary Putnams Biographie und philosophische Entwicklung spiegeln die Geschichte der angelsächsischen Philosophie in den letzten 40 Jahren. Beinahe ebenso lange hat Putnam diese Geschichte wesentlich beeinflußt. In der vorliegenden Einleitung soll vor allem der Kontext dargestellt werden, in dem Putnam steht und aus dem heraus verständlich wird, was er philosophisch zu sagen hat. Im Rahmen einer Skizze von Putnams philosophischer Entwicklung soll zudem eine vorläufige philosophiehistorische Einordnung versucht werden, auch wenn hier nicht der Ort für eine umfassende Kritik oder Darstellung sein kann: Die Einleitung muß auf recht elementarem Niveau bleiben und kann eine Lektüre der Texte natürlich nicht ersetzen. Da Putnams Werk sicherlich Teil einer Annäherung von 'analytischer' und 'kontinentaler' Philosophie ist, soll bei der Einführung in die hier übersetzten Texte schließlich deutlich werden, was Putnam nicht analytisch orientierten Lesern zu bieten hat. 

**Abstract (ZH)**: 希尔ary·普特南的生平与哲学发展反映了近40年英美哲学的历史。普特南几乎同样长时间显著影响了这一历史。在本文献的引言中，主要目的是呈现普特南所处的背景环境，从这一环境中可以理解他的哲学贡献。在普特南哲学发展概览的背景下，还将尝试进行初步的哲学历史分类，尽管这里不是进行全面批判或展示的合适场所：引言必须保持在相当基础的水平上，当然无法替代对文本的阅读。由于普特南的工作无疑是‘分析’哲学与‘大陆’哲学接近的一部分，因此在介绍这里翻译的文本时，最终应清楚说明普特南对非分析哲学导向读者的价值所在。 

---
# Causality Is Key to Understand and Balance Multiple Goals in Trustworthy ML and Foundation Models 

**Title (ZH)**: 因果关系是理解并 balancing 多目标在可信赖机器学习和基础模型中的关键 

**Authors**: Ruta Binkyte, Ivaxi Sheth, Zhijing Jin, Muhammad Havaei, Bernhardt Schölkopf, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2502.21123)  

**Abstract**: Ensuring trustworthiness in machine learning (ML) systems is crucial as they become increasingly embedded in high-stakes domains. This paper advocates for the integration of causal methods into machine learning to navigate the trade-offs among key principles of trustworthy ML, including fairness, privacy, robustness, accuracy, and explainability. While these objectives should ideally be satisfied simultaneously, they are often addressed in isolation, leading to conflicts and suboptimal solutions. Drawing on existing applications of causality in ML that successfully align goals such as fairness and accuracy or privacy and robustness, this paper argues that a causal approach is essential for balancing multiple competing objectives in both trustworthy ML and foundation models. Beyond highlighting these trade-offs, we examine how causality can be practically integrated into ML and foundation models, offering solutions to enhance their reliability and interpretability. Finally, we discuss the challenges, limitations, and opportunities in adopting causal frameworks, paving the way for more accountable and ethically sound AI systems. 

**Abstract (ZH)**: 确保机器学习系统可信性至关重要，随着它们在高风险领域中的应用越来越广泛。本文提倡将因果方法集成到机器学习中，以平衡可信机器学习的关键原则之间的权衡，包括公平性、隐私、稳健性、准确性和可解释性。虽然这些目标理想情况下应同时满足，但它们通常被孤立地处理，导致冲突和次优解决方案。借鉴现有在机器学习中成功实现公平性和准确性或隐私与稳健性目标相结合的应用，本文认为因果方法对于在可信机器学习和基础模型中平衡多个竞争性目标是必不可少的。除了指出这些权衡之外，我们还探讨了如何在机器学习和基础模型中实际集成因果方法，提出了提高其可靠性和可解释性的解决方案。最后，我们讨论采用因果框架面临的挑战、限制和机遇，为更具可问责性和伦理性的AI系统铺平道路。 

---
# PASemiQA: Plan-Assisted Agent for Question Answering on Semi-Structured Data with Text and Relational Information 

**Title (ZH)**: PASemiQA: 基于计划辅助的半结构化数据文本与关系信息问答代理 

**Authors**: Hansi Yang, Qi Zhang, Wei Jiang, Jianguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.21087)  

**Abstract**: Large language models (LLMs) have shown impressive abilities in answering questions across various domains, but they often encounter hallucination issues on questions that require professional and up-to-date knowledge. To address this limitation, retrieval-augmented generation (RAG) techniques have been proposed, which retrieve relevant information from external sources to inform their responses. However, existing RAG methods typically focus on a single type of external data, such as vectorized text database or knowledge graphs, and cannot well handle real-world questions on semi-structured data containing both text and relational information. To bridge this gap, we introduce PASemiQA, a novel approach that jointly leverages text and relational information in semi-structured data to answer questions. PASemiQA first generates a plan to identify relevant text and relational information to answer the question in semi-structured data, and then uses an LLM agent to traverse the semi-structured data and extract necessary information. Our empirical results demonstrate the effectiveness of PASemiQA across different semi-structured datasets from various domains, showcasing its potential to improve the accuracy and reliability of question answering systems on semi-structured data. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各个领域回答问题方面展现了出色的能力，但在需要专业和最新知识的问题上常常遇到幻觉问题。为解决这一限制，检索增强生成（RAG）技术被提出，通过从外部来源检索相关的信息来指导它们的回答。然而，现有的RAG方法通常仅侧重于一种类型的外部数据，如向量化的文本数据库或知识图谱，并不能很好地处理包含文本和关系信息的半结构化数据中的实际问题。为了弥合这一差距，我们介绍了PASemiQA，这是一种新的方法，能够联合利用半结构化数据中的文本和关系信息来回答问题。PASemiQA 首先生成一个计划来识别半结构化数据中与问题相关的文本和关系信息，然后使用LLM代理遍历半结构化数据并提取必要的信息。我们的实验证明了PASemiQA在不同领域半结构化数据集上的有效性，展示了其在半结构化数据中的问题回答系统上提高准确性和可靠性的潜力。 

---
# Enhancing deep neural networks through complex-valued representations and Kuramoto synchronization dynamics 

**Title (ZH)**: 通过复值表示和Kuramoto同步动力学增强深度神经网络 

**Authors**: Sabine Muzellec, Andrea Alamia, Thomas Serre, Rufin VanRullen  

**Link**: [PDF](https://arxiv.org/pdf/2502.21077)  

**Abstract**: Neural synchrony is hypothesized to play a crucial role in how the brain organizes visual scenes into structured representations, enabling the robust encoding of multiple objects within a scene. However, current deep learning models often struggle with object binding, limiting their ability to represent multiple objects effectively. Inspired by neuroscience, we investigate whether synchrony-based mechanisms can enhance object encoding in artificial models trained for visual categorization. Specifically, we combine complex-valued representations with Kuramoto dynamics to promote phase alignment, facilitating the grouping of features belonging to the same object. We evaluate two architectures employing synchrony: a feedforward model and a recurrent model with feedback connections to refine phase synchronization using top-down information. Both models outperform their real-valued counterparts and complex-valued models without Kuramoto synchronization on tasks involving multi-object images, such as overlapping handwritten digits, noisy inputs, and out-of-distribution transformations. Our findings highlight the potential of synchrony-driven mechanisms to enhance deep learning models, improving their performance, robustness, and generalization in complex visual categorization tasks. 

**Abstract (ZH)**: 神经同步性在视觉场景组织中的作用被假设为关键，它使大脑能够将视觉场景编码为结构化的表示，从而实现对场景中多个对象的稳健编码。然而，当前的深度学习模型在对象捆绑方面常常表现不佳，限制了它们有效表示多个对象的能力。受神经科学的启发，我们研究了基于同步性的机制是否能够增强用于视觉分类的人工模型中的对象编码。具体而言，我们结合复值表示与库拉莫托动力学来促进相位对齐，有利于将属于同一对象的特征分组。我们评估了两种采用同步性的架构：前馈模型和具有反馈连接的递归模型，该递归模型利用自上而下的信息来精细调节相位同步。两种模型在涉及多个对象的图像的任务上，如重叠的手写数字、噪声输入和离域变换，均优于其对应的实值模型和不采用库拉莫托同步的复值模型。我们的研究结果突显了由同步性驱动的机制在增强深度学习模型方面的潜力，提高了这些模型在复杂视觉分类任务中的性能、稳健性和泛化能力。 

---
# Fast Adversarial Training against Sparse Attacks Requires Loss Smoothing 

**Title (ZH)**: 快 adversarial training 对抗稀疏攻击需损失平滑 

**Authors**: Xuyang Zhong, Yixiao Huang, Chen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.21041)  

**Abstract**: This paper studies fast adversarial training against sparse adversarial perturbations bounded by $l_0$ norm. We demonstrate the challenges of employing $1$-step attacks on $l_0$ bounded perturbations for fast adversarial training, including degraded performance and the occurrence of catastrophic overfitting (CO). We highlight that CO in $l_0$ adversarial training is caused by sub-optimal perturbation locations of $1$-step attack. Theoretical and empirical analyses reveal that the loss landscape of $l_0$ adversarial training is more craggy compared to its $l_\infty$, $l_2$ and $l_1$ counterparts. Moreover, we corroborate that the craggy loss landscape can aggravate CO. To address these issues, we propose Fast-LS-$l_0$ that incorporates soft labels and the trade-off loss function to smooth the adversarial loss landscape. Extensive experiments demonstrate our method can overcome the challenge of catastrophic overfitting, achieve state-of-the-art performance, and narrow down the performance gap between $1$-step and multi-step adversarial training against sparse attacks. 

**Abstract (ZH)**: 本文研究了针对稀疏范数受限的对抗扰动的快速对抗训练方法，探讨了一步攻击在零范数扰动下的快速对抗训练所面临的挑战，包括性能下降和灾难性过拟合问题。我们表明，灾难性过拟合在零范数对抗训练中是由一步攻击的次优扰动位置引起的。理论和实证分析表明，零范数对抗训练的损失景观比其∞范数、2范数和1范数对应物更具崎岖不平。此外，我们证实崎岖不平的损失景观会加剧灾难性过拟合。为了解决这些问题，我们提出了Fast-LS-$l_0$方法，该方法结合了软标签和权衡损失函数以平滑对抗损失景观。大量实验表明，该方法能够克服灾难性过拟合问题，实现最先进的性能，并减少单步与多步对抗训练在稀疏攻击下的性能差距。 

---
# Reward Learning from Multiple Feedback Types 

**Title (ZH)**: 多反馈类型奖励学习 

**Authors**: Yannick Metz, András Geiszl, Raphaël Baur, Mennatallah El-Assady  

**Link**: [PDF](https://arxiv.org/pdf/2502.21038)  

**Abstract**: Learning rewards from preference feedback has become an important tool in the alignment of agentic models. Preference-based feedback, often implemented as a binary comparison between multiple completions, is an established method to acquire large-scale human feedback. However, human feedback in other contexts is often much more diverse. Such diverse feedback can better support the goals of a human annotator, and the simultaneous use of multiple sources might be mutually informative for the learning process or carry type-dependent biases for the reward learning process. Despite these potential benefits, learning from different feedback types has yet to be explored extensively. In this paper, we bridge this gap by enabling experimentation and evaluating multi-type feedback in a broad set of environments. We present a process to generate high-quality simulated feedback of six different types. Then, we implement reward models and downstream RL training for all six feedback types. Based on the simulated feedback, we investigate the use of types of feedback across ten RL environments and compare them to pure preference-based baselines. We show empirically that diverse types of feedback can be utilized and lead to strong reward modeling performance. This work is the first strong indicator of the potential of multi-type feedback for RLHF. 

**Abstract (ZH)**: 基于多种反馈类型的奖励学习在强化学习人类反馈中的应用探索 

---
# Synthesizing Tabular Data Using Selectivity Enhanced Generative Adversarial Networks 

**Title (ZH)**: 使用选择性增强生成对抗网络合成表格数据 

**Authors**: Youran Zhou, Jianzhong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2502.21034)  

**Abstract**: As E-commerce platforms face surging transactions during major shopping events like Black Friday, stress testing with synthesized data is crucial for resource planning. Most recent studies use Generative Adversarial Networks (GANs) to generate tabular data while ensuring privacy and machine learning utility. However, these methods overlook the computational demands of processing GAN-generated data, making them unsuitable for E-commerce stress testing.
This thesis introduces a novel GAN-based approach incorporating query selectivity constraints, a key factor in database transaction processing. We integrate a pre-trained deep neural network to maintain selectivity consistency between real and synthetic data. Our method, tested on five real-world datasets, outperforms three state-of-the-art GANs and a VAE model, improving selectivity estimation accuracy by up to 20pct and machine learning utility by up to 6 pct. 

**Abstract (ZH)**: 电子商务平台在黑五等重大购物活动期间面临交易激增，合成数据的压力测试对于资源规划至关重要。现有的研究表明，使用生成式对抗网络（GANs）生成表数据以确保隐私和机器学习效用是关键，然而这些方法忽视了处理GAN生成数据的计算需求，使其不适合用于电子商务压力测试。本文提出了一种结合查询选择性约束的新型GAN方法，这是一种数据库事务处理中的关键因素。我们集成了一个预训练的深度神经网络以在实际和合成数据之间保持选择性一致性。该方法在五个真实数据集上的测试结果表明，它优于三种最先进的GAN和一个VAE模型，在选择性估计准确性上提高了最多20%，在机器学习效用上提高了最多6%。 

---
# Improving Open-world Continual Learning under the Constraints of Scarce Labeled Data 

**Title (ZH)**: 在稀缺标注数据约束下的开放世界连续学习改进 

**Authors**: Yujie Li, Xiangkun Wang, Xin Yang, Marcello Bonsangue, Junbo Zhang, Tianrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.20974)  

**Abstract**: Open-world continual learning (OWCL) adapts to sequential tasks with open samples, learning knowledge incrementally while preventing forgetting. However, existing OWCL still requires a large amount of labeled data for training, which is often impractical in real-world applications. Given that new categories/entities typically come with limited annotations and are in small quantities, a more realistic situation is OWCL with scarce labeled data, i.e., few-shot training samples. Hence, this paper investigates the problem of open-world few-shot continual learning (OFCL), challenging in (i) learning unbounded tasks without forgetting previous knowledge and avoiding overfitting, (ii) constructing compact decision boundaries for open detection with limited labeled data, and (iii) transferring knowledge about knowns and unknowns and even update the unknowns to knowns once the labels of open samples are learned. In response, we propose a novel OFCL framework that integrates three key components: (1) an instance-wise token augmentation (ITA) that represents and enriches sample representations with additional knowledge, (2) a margin-based open boundary (MOB) that supports open detection with new tasks emerge over time, and (3) an adaptive knowledge space (AKS) that endows unknowns with knowledge for the updating from unknowns to knowns. Finally, extensive experiments show the proposed OFCL framework outperforms all baselines remarkably with practical importance and reproducibility. The source code is released at this https URL. 

**Abstract (ZH)**: 开放世界少量标注持续学习（OFCL）：学习未定义任务的同时防止遗忘和过度拟合，构建紧凑的决策边界以进行开放检测，以及转移已知和未知的知识，并在开放样本标签学习后更新未知为已知。 

---
# Concealed Adversarial attacks on neural networks for sequential data 

**Title (ZH)**: 隐藏在序列数据神经网络中的对抗攻击 

**Authors**: Petr Sokerin, Dmitry Anikin, Sofia Krehova, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2502.20948)  

**Abstract**: The emergence of deep learning led to the broad usage of neural networks in the time series domain for various applications, including finance and medicine. While powerful, these models are prone to adversarial attacks: a benign targeted perturbation of input data leads to significant changes in a classifier's output. However, formally small attacks in the time series domain become easily detected by the human eye or a simple detector model.
We develop a concealed adversarial attack for different time-series models: it provides more realistic perturbations, being hard to detect by a human or model discriminator. To achieve this goal, the proposed adversarial attack maximizes an aggregation of a classifier and a trained discriminator loss. To make the attack stronger, we also propose a training procedure for a discriminator that provides broader coverage of possible attacks. Extensive benchmarking on six UCR time series datasets across four diverse architectures - including recurrent, convolutional, state-space, and transformer-based models - demonstrates the superiority of our attack for a concealability-efficiency trade-off. Our findings highlight the growing challenge of designing robust time series models, emphasizing the need for improved defenses against realistic and effective attacks. 

**Abstract (ZH)**: 深度学习的兴起导致了神经网络在时间序列领域的广泛使用，应用于金融和医学等多个领域。尽管这些模型非常强大，但它们容易遭受对抗攻击：对输入数据进行良性目标扰动会导致分类器输出发生显著变化。然而，在时间序列领域，即使是细微的攻击也容易被人类眼睛或简单的检测模型发现。

我们为不同时间序列模型开发了一种隐蔽的对抗攻击：它提供了更具现实性的扰动，难以被人类或模型鉴别器发现。为了实现这一目标，所提出的对抗攻击最大化了分类器和训练好的鉴别器损失的聚合。为了使攻击更强，我们还提出了一种训练鉴别器的方法，以提供更广泛的攻击覆盖范围。在四个不同的架构（包括循环模型、卷积模型、状态空间模型和基于变换器的模型）和六个UCR时间序列数据集上的广泛基准测试显示出我们在隐蔽性和效率权衡中的优势。我们的研究结果突显了设计健壮的时间序列模型的越来越大的挑战，强调了对现实和有效的攻击采取改进防御措施的必要性。 

---
# Generative Uncertainty in Diffusion Models 

**Title (ZH)**: 生成不确定性在扩散模型中的应用 

**Authors**: Metod Jazbec, Eliot Wong-Toi, Guoxuan Xia, Dan Zhang, Eric Nalisnick, Stephan Mandt  

**Link**: [PDF](https://arxiv.org/pdf/2502.20946)  

**Abstract**: Diffusion models have recently driven significant breakthroughs in generative modeling. While state-of-the-art models produce high-quality samples on average, individual samples can still be low quality. Detecting such samples without human inspection remains a challenging task. To address this, we propose a Bayesian framework for estimating generative uncertainty of synthetic samples. We outline how to make Bayesian inference practical for large, modern generative models and introduce a new semantic likelihood (evaluated in the latent space of a feature extractor) to address the challenges posed by high-dimensional sample spaces. Through our experiments, we demonstrate that the proposed generative uncertainty effectively identifies poor-quality samples and significantly outperforms existing uncertainty-based methods. Notably, our Bayesian framework can be applied post-hoc to any pretrained diffusion or flow matching model (via the Laplace approximation), and we propose simple yet effective techniques to minimize its computational overhead during sampling. 

**Abstract (ZH)**: 基于扩散模型的生成不确定性 Bayesian 框架：低质量样本检测与效率优化 

---
# WebFAQ: A Multilingual Collection of Natural Q&A Datasets for Dense Retrieval 

**Title (ZH)**: WebFAQ：用于密集检索的多语言自然问答数据集集合 

**Authors**: Michael Dinzinger, Laura Caspari, Kanishka Ghosh Dastidar, Jelena Mitrović, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2502.20936)  

**Abstract**: We present WebFAQ, a large-scale collection of open-domain question answering datasets derived from FAQ-style this http URL annotations. In total, the data collection consists of 96 million natural question-answer (QA) pairs across 75 languages, including 47 million (49%) non-English samples. WebFAQ further serves as the foundation for 20 monolingual retrieval benchmarks with a total size of 11.2 million QA pairs (5.9 million non-English). These datasets are carefully curated through refined filtering and near-duplicate detection, yielding high-quality resources for training and evaluating multilingual dense retrieval models. To empirically confirm WebFAQ's efficacy, we use the collected QAs to fine-tune an in-domain pretrained XLM-RoBERTa model. Through this process of dataset-specific fine-tuning, the model achieves significant retrieval performance gains, which generalize - beyond WebFAQ - to other multilingual retrieval benchmarks evaluated in zero-shot setting. Last but not least, we utilize WebFAQ to construct a set of QA-aligned bilingual corpora spanning over 1000 language pairs using state-of-the-art bitext mining and automated LLM-assessed translation evaluation. Due to our advanced, automated method of bitext dataset generation, the resulting bilingual corpora demonstrate higher translation quality compared to similar datasets. WebFAQ and all associated resources are publicly available on GitHub and HuggingFace. 

**Abstract (ZH)**: WebFAQ：大规模多语言开放领域问答数据集及其应用 

---
# A Fused Gromov-Wasserstein Approach to Subgraph Contrastive Learning 

**Title (ZH)**: 融合格瓦.addComponenttoclass()-茨威格距离的方法在子图对比学习中的应用 

**Authors**: Amadou S. Sangare, Nicolas Dunou, Jhony H. Giraldo, Fragkiskos D. Malliaros  

**Link**: [PDF](https://arxiv.org/pdf/2502.20885)  

**Abstract**: Self-supervised learning has become a key method for training deep learning models when labeled data is scarce or unavailable. While graph machine learning holds great promise across various domains, the design of effective pretext tasks for self-supervised graph representation learning remains challenging. Contrastive learning, a popular approach in graph self-supervised learning, leverages positive and negative pairs to compute a contrastive loss function. However, current graph contrastive learning methods often struggle to fully use structural patterns and node similarities. To address these issues, we present a new method called Fused Gromov Wasserstein Subgraph Contrastive Learning (FOSSIL). Our model integrates node-level and subgraph-level contrastive learning, seamlessly combining a standard node-level contrastive loss with the Fused Gromov-Wasserstein distance. This combination helps our method capture both node features and graph structure together. Importantly, our approach works well with both homophilic and heterophilic graphs and can dynamically create views for generating positive and negative pairs. Through extensive experiments on benchmark graph datasets, we show that FOSSIL outperforms or achieves competitive performance compared to current state-of-the-art methods. 

**Abstract (ZH)**: 无监督学习在标注数据稀缺或不可用时已成为训练深度学习模型的关键方法。尽管图机器学习在各个领域表现出巨大潜力，但对于有效 pretext 任务的设计以进行自监督图表示学习仍具有挑战性。对比学习，图自监督学习中的一种流行方法，利用正负配对计算对比损失函数。然而，当前的图对比学习方法往往难以充分利用结构模式和节点相似性。为解决这些问题，我们提出了一种新的方法，名为融合佐默夫 Wasserstein 子图对比学习（FOSSIL）。该模型将节点级和子图级对比学习相结合，无缝地将标准的节点级对比损失与融合佐默夫 Wasserstein 距离结合在一起。这种结合有助于我们的方法同时捕获节点特征和图结构。重要的是，我们的方法适用于同质图和异质图，并且能够动态创建视图以生成正负配对。通过在基准图数据集上的广泛实验，我们展示了 FOSSIL 在当前最先进的方法中表现出色或可与其竞争。 

---
# Neuro-Symbolic Learning for Galois Groups: Unveiling Probabilistic Trends in Polynomials 

**Title (ZH)**: 神经符号学习在伽罗瓦群中的应用：探索多项式的概率趋势 

**Authors**: Elira Shaska, Tony Shaska  

**Link**: [PDF](https://arxiv.org/pdf/2502.20844)  

**Abstract**: This paper presents a neurosymbolic approach to classifying Galois groups of polynomials, integrating classical Galois theory with machine learning to address challenges in algebraic computation. By combining neural networks with symbolic reasoning we develop a model that outperforms purely numerical methods in accuracy and interpretability. Focusing on sextic polynomials with height $\leq 6$, we analyze a database of 53,972 irreducible examples, uncovering novel distributional trends, such as the 20 sextic polynomials with Galois group $C_6$ spanning just seven invariant-defined equivalence classes. These findings offer the first empirical insights into Galois group probabilities under height constraints and lay the groundwork for exploring solvability by radicals. Demonstrating AI's potential to reveal patterns beyond traditional symbolic techniques, this work paves the way for future research in computational algebra, with implications for probabilistic conjectures and higher degree classifications. 

**Abstract (ZH)**: 一种神经符号方法用于分类多项式的伽罗瓦群：结合经典伽罗瓦理论与机器学习以应对代数计算挑战 

---
# Weakly Supervised Multiple Instance Learning for Whale Call Detection and Localization in Long-Duration Passive Acoustic Monitoring 

**Title (ZH)**: 弱监督多重实例学习在长时段被动声学监测中虎鲸叫声检测与定位 

**Authors**: Ragib Amin Nihal, Benjamin Yen, Runwu Shi, Kazuhiro Nakadai  

**Link**: [PDF](https://arxiv.org/pdf/2502.20838)  

**Abstract**: Marine ecosystem monitoring via Passive Acoustic Monitoring (PAM) generates vast data, but deep learning often requires precise annotations and short segments. We introduce DSMIL-LocNet, a Multiple Instance Learning framework for whale call detection and localization using only bag-level labels. Our dual-stream model processes 2-30 minute audio segments, leveraging spectral and temporal features with attention-based instance selection. Tests on Antarctic whale data show longer contexts improve classification (F1: 0.8-0.9) while medium instances ensure localization precision (0.65-0.70). This suggests MIL can enhance scalable marine monitoring. Code: this https URL 

**Abstract (ZH)**: 通过被动声学监测（PAM）进行海洋生态系统的监控产生了大量数据，但深度学习通常需要精确的标注和短段音频。我们引入了一种仅使用类别级标签进行鲸鱼叫声检测和定位的多实例学习框架DSMIL-LocNet。该双流模型处理2-30分钟的音频段，利用基于注意力的实例选择处理频谱和时间特征。在南极鲸鱼数据上的测试表明，更长的上下文可以提高分类性能（F1: 0.8-0.9），而中等大小的实例可以确保定位精度（0.65-0.70）。这表明多实例学习可以增强可扩展的海洋监控。代码：此链接 URL。 

---
# Flattening Supply Chains: When do Technology Improvements lead to Disintermediation? 

**Title (ZH)**: 扁平化供应链：何时技术改进会导致中间商被排除？ 

**Authors**: S. Nageeb Ali, Nicole Immorlica, Meena Jagadeesan, Brendan Lucier  

**Link**: [PDF](https://arxiv.org/pdf/2502.20783)  

**Abstract**: In the digital economy, technological innovations make it cheaper to produce high-quality content. For example, generative AI tools reduce costs for creators who develop content to be distributed online, but can also reduce production costs for the users who consume that content. These innovations can thus lead to disintermediation, since consumers may choose to use these technologies directly, bypassing intermediaries. To investigate when technological improvements lead to disintermediation, we study a game with an intermediary, suppliers of a production technology, and consumers. First, we show disintermediation occurs whenever production costs are too high or too low. We then investigate the consequences of disintermediation for welfare and content quality at equilibrium. While the intermediary is welfare-improving, the intermediary extracts all gains to social welfare and its presence can raise or lower content quality. We further analyze how disintermediation is affected by the level of competition between suppliers and the intermediary's fee structure. More broadly, our results take a step towards assessing how production technology innovations affect the survival of intermediaries and impact the digital economy. 

**Abstract (ZH)**: 在数字经济中，技术创新使生产高质量内容的成本更低。例如，生成式AI工具降低了内容创作者的生产成本，同时也降低了内容消费者的内容生产成本。这些创新可能导致中间商被绕过，因为消费者可能直接使用这些技术，而不通过中间商。为了探究技术进步何时导致中间商被绕过的现象，我们研究了一个涉及中间商、生产技术供应商和消费者的博弈模型。首先，我们表明生产成本过高或过低时都会发生中间商被绕过的情况。然后，我们探讨了平衡状态下中间商被绕过对福利和内容质量的影响。虽然中间商提高了社会福利，但中间商能够提取所有社会福利增益，其存在可能会提高或降低内容质量。此外，我们分析了供应商之间的竞争程度和中间商的收费结构如何影响中间商被绕过的现象。更广泛地说，我们的结果向前迈进了一步，评估了生产技术的创新如何影响中间商的生存，并影响数字经济。 

---
# NeuroMorse: A Temporally Structured Dataset For Neuromorphic Computing 

**Title (ZH)**: NeuroMorse: 一种用于神经形态计算的时间结构化数据集 

**Authors**: Ben Walters, Yeshwanth Bethi, Taylor Kergan, Binh Nguyen, Amirali Amirsoleimani, Jason K. Eshraghian, Saeed Afshar, Mostafa Rahimi Azghadi  

**Link**: [PDF](https://arxiv.org/pdf/2502.20729)  

**Abstract**: Neuromorphic engineering aims to advance computing by mimicking the brain's efficient processing, where data is encoded as asynchronous temporal events. This eliminates the need for a synchronisation clock and minimises power consumption when no data is present. However, many benchmarks for neuromorphic algorithms primarily focus on spatial features, neglecting the temporal dynamics that are inherent to most sequence-based tasks. This gap may lead to evaluations that fail to fully capture the unique strengths and characteristics of neuromorphic systems. In this paper, we present NeuroMorse, a temporally structured dataset designed for benchmarking neuromorphic learning systems. NeuroMorse converts the top 50 words in the English language into temporal Morse code spike sequences. Despite using only two input spike channels for Morse dots and dashes, complex information is encoded through temporal patterns in the data. The proposed benchmark contains feature hierarchy at multiple temporal scales that test the capacity of neuromorphic algorithms to decompose input patterns into spatial and temporal hierarchies. We demonstrate that our training set is challenging to categorise using a linear classifier and that identifying keywords in the test set is difficult using conventional methods. The NeuroMorse dataset is available at Zenodo, with our accompanying code on GitHub at this https URL. 

**Abstract (ZH)**: 仿生工程旨在通过模仿大脑高效的处理方式来推进计算技术，其中数据以异步时间事件的形式进行编码。这消除了需要同步时钟的需求，并在没有数据时最小化功耗。然而，许多神经形态算法的基准测试主要侧重于空间特征，忽视了大多数基于序列的任务中固有的时间动态性。这一差距可能导致对神经形态系统独特优势和特性未能充分捕捉的评估。在本文中，我们提出了一种名为NeuroMorse的时间结构化数据集，用于 benchmark 神经形态学习系统。NeuroMorse 将英文中最常用的50个单词转换为时间莫尔斯电码突触序列。尽管仅使用两个输入突触通道来编码莫尔斯点和划，复杂信息仍然是通过数据中的时间模式进行编码的。所提出的 benchmark 包含跨多个时间尺度的特征层次结构，以测试神经形态算法将输入模式分解为空间和时间层次结构的能力。我们展示了使用线性分类器对训练集进行分类具有挑战性，并且使用常规方法在测试集中识别关键词也具有挑战性。NeuroMorse 数据集可在 Zenodo 获取，相关的代码可在 GitHub 上找到：this https URL。 

---
# Generating Clinically Realistic EHR Data via a Hierarchy- and Semantics-Guided Transformer 

**Title (ZH)**: 基于层次结构和语义引导的变压器生成临床现实的EHR数据 

**Authors**: Guanglin Zhou, Sebastiano Barbieri  

**Link**: [PDF](https://arxiv.org/pdf/2502.20719)  

**Abstract**: Generating realistic synthetic electronic health records (EHRs) holds tremendous promise for accelerating healthcare research, facilitating AI model development and enhancing patient privacy. However, existing generative methods typically treat EHRs as flat sequences of discrete medical codes. This approach overlooks two critical aspects: the inherent hierarchical organization of clinical coding systems and the rich semantic context provided by code descriptions. Consequently, synthetic patient sequences often lack high clinical fidelity and have limited utility in downstream clinical tasks. In this paper, we propose the Hierarchy- and Semantics-Guided Transformer (HiSGT), a novel framework that leverages both hierarchical and semantic information for the generative process. HiSGT constructs a hierarchical graph to encode parent-child and sibling relationships among clinical codes and employs a graph neural network to derive hierarchy-aware embeddings. These are then fused with semantic embeddings extracted from a pre-trained clinical language model (e.g., ClinicalBERT), enabling the Transformer-based generator to more accurately model the nuanced clinical patterns inherent in real EHRs. Extensive experiments on the MIMIC-III and MIMIC-IV datasets demonstrate that HiSGT significantly improves the statistical alignment of synthetic data with real patient records, as well as supports robust downstream applications such as chronic disease classification. By addressing the limitations of conventional raw code-based generative models, HiSGT represents a significant step toward clinically high-fidelity synthetic data generation and a general paradigm suitable for interpretable medical code representation, offering valuable applications in data augmentation and privacy-preserving healthcare analytics. 

**Abstract (ZH)**: 基于层次和语义指导的变压器（HiSGT）生成真实化的合成电子健康记录 

---
# Unleashing the Potential of Two-Tower Models: Diffusion-Based Cross-Interaction for Large-Scale Matching 

**Title (ZH)**: 释放双塔模型的潜力：基于扩散的跨交互匹配 

**Authors**: Yihan Wang, Fei Xiong, Zhexin Han, Qi Song, Kaiqiao Zhan, Ben Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20687)  

**Abstract**: Two-tower models are widely adopted in the industrial-scale matching stage across a broad range of application domains, such as content recommendations, advertisement systems, and search engines. This model efficiently handles large-scale candidate item screening by separating user and item representations. However, the decoupling network also leads to a neglect of potential information interaction between the user and item representations. Current state-of-the-art (SOTA) approaches include adding a shallow fully connected layer(i.e., COLD), which is limited by performance and can only be used in the ranking stage. For performance considerations, another approach attempts to capture historical positive interaction information from the other tower by regarding them as the input features(i.e., DAT). Later research showed that the gains achieved by this method are still limited because of lacking the guidance on the next user intent. To address the aforementioned challenges, we propose a "cross-interaction decoupling architecture" within our matching paradigm. This user-tower architecture leverages a diffusion module to reconstruct the next positive intention representation and employs a mixed-attention module to facilitate comprehensive cross-interaction. During the next positive intention generation, we further enhance the accuracy of its reconstruction by explicitly extracting the temporal drift within user behavior sequences. Experiments on two real-world datasets and one industrial dataset demonstrate that our method outperforms the SOTA two-tower models significantly, and our diffusion approach outperforms other generative models in reconstructing item representations. 

**Abstract (ZH)**: 双塔模型在内容推荐、广告系统和搜索引擎等多个应用领域的大规模匹配阶段广泛采用。这种模型通过分离用户和项目表示有效地处理大规模候选项目筛选，但这也导致了用户和项目表示之间潜在信息交互的忽视。当前最先进（SOTA）的方法包括添加一个浅层全连接层（如COLD），这在性能上受到限制，只能用于排序阶段。为考虑性能，另一种方法试图通过将另一个塔的历史正面交互信息作为输入特征来捕捉历史正面交互信息（如DAT），但后来的研究显示，这种方法的改进仍然有限，因为缺乏对未来用户意图的指导。为解决上述挑战，我们提出了一种“交叉交互分离架构”作为匹配范式的一部分。这种用户塔架构利用扩散模块重建下一正面意图表示，并采用混合注意力模块促进全面的交叉交互。在生成下一正面意图时，我们进一步通过显式提取用户行为序列中的时间漂移来提高其重建的准确性。在两个真实世界数据集和一个工业数据集上的实验表明，我们的方法显著优于现有的SOTA双塔模型，而我们的扩散方法在重建项目表示方面也优于其他生成模型。 

---
# Fine-tuning BERT with Bidirectional LSTM for Fine-grained Movie Reviews Sentiment Analysis 

**Title (ZH)**: 使用双向LSTM Fine-tuning BERT进行细粒度电影评论情感分析 

**Authors**: Gibson Nkhata, Susan Gauch, Usman Anjum, Justin Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2502.20682)  

**Abstract**: Sentiment Analysis (SA) is instrumental in understanding peoples viewpoints facilitating social media monitoring recognizing products and brands and gauging customer satisfaction. Consequently SA has evolved into an active research domain within Natural Language Processing (NLP). Many approaches outlined in the literature devise intricate frameworks aimed at achieving high accuracy, focusing exclusively on either binary sentiment classification or fine-grained sentiment classification. In this paper our objective is to fine-tune the pre-trained BERT model with Bidirectional LSTM (BiLSTM) to enhance both binary and fine-grained SA specifically for movie reviews. Our approach involves conducting sentiment classification for each review followed by computing the overall sentiment polarity across all reviews. We present our findings on binary classification as well as fine-grained classification utilizing benchmark datasets. Additionally we implement and assess two accuracy improvement techniques Synthetic Minority Oversampling Technique (SMOTE) and NLP Augmenter (NLPAUG) to bolster the models generalization in fine-grained sentiment classification. Finally a heuristic algorithm is employed to calculate the overall polarity of predicted reviews from the BERT+BiLSTM output vector. Our approach performs comparably with state-of-the-art (SOTA) techniques in both classifications. For instance in binary classification we achieve 97.67% accuracy surpassing the leading SOTA model NB-weighted-BON+dv-cosine by 0.27% on the renowned IMDb dataset. Conversely for five-class classification on SST-5 while the top SOTA model RoBERTa+large+Self-explaining attains 55.5% accuracy our model achieves 59.48% accuracy surpassing the BERT-large baseline by 3.6%. 

**Abstract (ZH)**: 情感分析（SA）对于理解人们的观点、促进社交媒体监控、识别产品和品牌及评估顾客满意度至关重要。因此，SA已成为自然语言处理（NLP）领域的一项活跃研究方向。文献中概述的许多方法设计了复杂框架，旨在实现高准确度，专注于二元情感分类或细粒度情感分类之一。本文旨在通过结合预训练的BERT模型与双向LSTM（BiLSTM）来增强二元和细粒度情感分析，特别适用于电影评论。我们的方法包括对每个评论进行情感分类，然后计算所有评论的情感极性。我们将使用基准数据集呈现二元分类和细粒度分类的发现。此外，我们实现并评估了两种提高准确性的技术，即合成少数类过采样技术（SMOTE）和NLP增强器（NLPAUG），以增强模型在细粒度情感分类中的泛化能力。最后，我们采用启发式算法计算从BERT+BiLSTM输出向量中预测评论的整体极性。我们的方法在两种分类中均与当前最佳技术（SOTA）具有可比性。例如，在二元分类中，我们在著名的IMDb数据集上达到了97.67%的准确率，超越了排名第一的SOTA模型NB-weighted-BON+dv-cosine 0.27%。相反，在SST-5的五类分类中，尽管最佳SOTA模型RoBERTa+large+Self-explaining 的准确率为55.5%，我们的模型达到了59.48%的准确率，比BERT-large基线高出了3.6%。 

---
# Disentangling Feature Structure: A Mathematically Provable Two-Stage Training Dynamics in Transformers 

**Title (ZH)**: 拆解特征结构：Transformer中的可数学证明的二阶段训练动力学 

**Authors**: Zixuan Gong, Jiaye Teng, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20681)  

**Abstract**: Transformers may exhibit two-stage training dynamics during the real-world training process. For instance, when training GPT-2 on the Counterfact dataset, the answers progress from syntactically incorrect to syntactically correct to semantically correct. However, existing theoretical analyses hardly account for this two-stage phenomenon. In this paper, we theoretically demonstrate how such two-stage training dynamics occur in transformers. Specifically, we analyze the dynamics of transformers using feature learning techniques under in-context learning regimes, based on a disentangled two-type feature structure. Such disentanglement of feature structure is general in practice, e.g., natural languages contain syntax and semantics, and proteins contain primary and secondary structures. To our best known, this is the first rigorous result regarding a two-stage optimization process in transformers. Additionally, a corollary indicates that such a two-stage process is closely related to the spectral properties of the attention weights, which accords well with empirical findings. 

**Abstract (ZH)**: 变压器在实际训练过程中可能表现出两阶段训练动力学。我们理论上展示了这种两阶段训练动力学如何在变压器中发生。具体而言，我们基于解耦的两种类型特征结构，在上下文学习框架下使用特征学习技术分析了变压器的动力学。据我们所知，这是首次关于变压器两阶段优化过程的严格结果。此外，一个推论表明，这种两阶段过程与注意力权重的谱性质密切相关，这与实证发现相符。 

---
# OpenEarthSensing: Large-Scale Fine-Grained Benchmark for Open-World Remote Sensing 

**Title (ZH)**: 开放地球感知：开放世界遥感大规模细粒度基准 

**Authors**: Xiang Xiang, Zhuo Xu, Yao Deng, Qinhao Zhou, Yifan Liang, Ke Chen, Qingfang Zheng, Yaowei Wang, Xilin Chen, Wen Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.20668)  

**Abstract**: In open-world remote sensing, deployed models must continuously adapt to a steady influx of new data, which often exhibits various shifts compared to what the model encountered during the training phase. To effectively handle the new data, models are required to detect semantic shifts, adapt to covariate shifts, and continuously update themselves. These challenges give rise to a variety of open-world tasks. However, existing open-world remote sensing studies typically train and test within a single dataset to simulate open-world conditions. Currently, there is a lack of large-scale benchmarks capable of evaluating multiple open-world tasks. In this paper, we introduce OpenEarthSensing, a large-scale fine-grained benchmark for open-world remote sensing. OpenEarthSensing includes 189 scene and objects categories, covering the vast majority of potential semantic shifts that may occur in the real world. Additionally, OpenEarthSensing encompasses five data domains with significant covariate shifts, including two RGB satellite domians, one RGB aerial domian, one MS RGB domian, and one infrared domian. The various domains provide a more comprehensive testbed for evaluating the generalization performance of open-world models. We conduct the baseline evaluation of current mainstream open-world tasks and methods on OpenEarthSensing, demonstrating that it serves as a challenging benchmark for open-world remote sensing. 

**Abstract (ZH)**: 开放世界遥感中大规模细粒度基准OpenEarthSensing及其挑战性评估 

---
# Dataset Distillation with Neural Characteristic Function: A Minmax Perspective 

**Title (ZH)**: 基于神经特征函数的Dataset Distillation：一个最小最大视角 

**Authors**: Shaobo Wang, Yicun Yang, Zhiyuan Liu, Chenghao Sun, Xuming Hu, Conghui He, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20653)  

**Abstract**: Dataset distillation has emerged as a powerful approach for reducing data requirements in deep learning. Among various methods, distribution matching-based approaches stand out for their balance of computational efficiency and strong performance. However, existing distance metrics used in distribution matching often fail to accurately capture distributional differences, leading to unreliable measures of discrepancy. In this paper, we reformulate dataset distillation as a minmax optimization problem and introduce Neural Characteristic Function Discrepancy (NCFD), a comprehensive and theoretically grounded metric for measuring distributional differences. NCFD leverages the Characteristic Function (CF) to encapsulate full distributional information, employing a neural network to optimize the sampling strategy for the CF's frequency arguments, thereby maximizing the discrepancy to enhance distance estimation. Simultaneously, we minimize the difference between real and synthetic data under this optimized NCFD measure. Our approach, termed Neural Characteristic Function Matching (\mymethod{}), inherently aligns the phase and amplitude of neural features in the complex plane for both real and synthetic data, achieving a balance between realism and diversity in synthetic samples. Experiments demonstrate that our method achieves significant performance gains over state-of-the-art methods on both low- and high-resolution datasets. Notably, we achieve a 20.5\% accuracy boost on ImageSquawk. Our method also reduces GPU memory usage by over 300$\times$ and achieves 20$\times$ faster processing speeds compared to state-of-the-art methods. To the best of our knowledge, this is the first work to achieve lossless compression of CIFAR-100 on a single NVIDIA 2080 Ti GPU using only 2.3 GB of memory. 

**Abstract (ZH)**: 基于分布匹配的神经特征函数差异（NCFD）在数据集蒸馏中的应用：一种全面且理论上 grounded 的度量方法 

---
# FedConv: A Learning-on-Model Paradigm for Heterogeneous Federated Clients 

**Title (ZH)**: FedConv：一种针对异构联邦客户端的模型学习范式 

**Authors**: Leming Shen, Qiang Yang, Kaiyan Cui, Yuanqing Zheng, Xiao-Yong Wei, Jianwei Liu, Jinsong Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.20639)  

**Abstract**: Federated Learning (FL) facilitates collaborative training of a shared global model without exposing clients' private data. In practical FL systems, clients (e.g., edge servers, smartphones, and wearables) typically have disparate system resources. Conventional FL, however, adopts a one-size-fits-all solution, where a homogeneous large global model is transmitted to and trained on each client, resulting in an overwhelming workload for less capable clients and starvation for other clients. To address this issue, we propose FedConv, a client-friendly FL framework, which minimizes the computation and memory burden on resource-constrained clients by providing heterogeneous customized sub-models. FedConv features a novel learning-on-model paradigm that learns the parameters of the heterogeneous sub-models via convolutional compression. Unlike traditional compression methods, the compressed models in FedConv can be directly trained on clients without decompression. To aggregate the heterogeneous sub-models, we propose transposed convolutional dilation to convert them back to large models with a unified size while retaining personalized information from clients. The compression and dilation processes, transparent to clients, are optimized on the server leveraging a small public dataset. Extensive experiments on six datasets demonstrate that FedConv outperforms state-of-the-art FL systems in terms of model accuracy (by more than 35% on average), computation and communication overhead (with 33% and 25% reduction, respectively). 

**Abstract (ZH)**: 联邦学习（FL）促进共享全局模型的协作训练，同时不暴露客户端的私有数据。在实际的FL系统中，客户端（如边缘服务器、智能手机和可穿戴设备）通常具有不同的系统资源。然而，传统的FL采用一刀切的解决方案，将一个同质的大规模全局模型传输给并训练在每个客户端上，这给能力较弱的客户端带来了过重的工作负担，而其他客户端则被闲置。为了解决这一问题，我们提出了一种客户端友好的联邦学习框架FedConv，通过提供异质定制子模型来最小化资源受限客户端的计算和内存负担。FedConv具备一种新颖的基于模型的学习范式，通过卷积压缩学习异质子模型的参数。与传统的压缩方法不同，FedConv中的压缩模型可以直接在客户端上进行训练，无需解压缩。为了聚合异质子模型，我们提出了转置卷积扩张，将其转换回具有统一大小的大模型，同时保留来自客户端的个性化信息。压缩和扩张过程透明地在服务器上通过一个小规模公共数据集进行优化。在六个数据集上的 extensive 实验表明，FedConv 在模型准确度（平均提升超过35%）、计算和通信开销（分别减少33%和25%）方面优于最新的FL系统。 

---
# A Compact Model for Large-Scale Time Series Forecasting 

**Title (ZH)**: 大规模时间序列预报的紧凑模型 

**Authors**: Chin-Chia Michael Yeh, Xiran Fan, Zhimeng Jiang, Yujie Fan, Huiyuan Chen, Uday Singh Saini, Vivian Lai, Xin Dai, Junpeng Wang, Zhongfang Zhuang, Liang Wang, Yan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.20634)  

**Abstract**: Spatio-temporal data, which commonly arise in real-world applications such as traffic monitoring, financial transactions, and ride-share demands, represent a special category of multivariate time series. They exhibit two distinct characteristics: high dimensionality and commensurability across spatial locations. These attributes call for computationally efficient modeling approaches and facilitate the use of univariate forecasting models in a channel-independent fashion. SparseTSF, a recently introduced competitive univariate forecasting model, harnesses periodicity to achieve compactness by concentrating on cross-period dynamics, thereby extending the Pareto frontier with respect to model size and predictive performance. Nonetheless, it underperforms on spatio-temporal data due to an inadequate capture of intra-period temporal dependencies. To address this shortcoming, we propose UltraSTF, which integrates a cross-period forecasting module with an ultra-compact shape bank component. Our model effectively detects recurring patterns in time series through the attention mechanism of the shape bank component, thereby strengthening its ability to learn intra-period dynamics. UltraSTF achieves state-of-the-art performance on the LargeST benchmark while employing fewer than 0.2% of the parameters required by the second-best approaches, thus further extending the Pareto frontier of existing methods. 

**Abstract (ZH)**: 基于时空数据的超紧凑时空预测模型：UltraSTF 

---
# Lattice Protein Folding with Variational Annealing 

**Title (ZH)**: 变分退火蛋白质折叠 lattice protein folding with variational annealing 

**Authors**: Shoummo Ahsan Khandoker, Estelle M. Inack, Mohamed Hibat-Allah  

**Link**: [PDF](https://arxiv.org/pdf/2502.20632)  

**Abstract**: Understanding the principles of protein folding is a cornerstone of computational biology, with implications for drug design, bioengineering, and the understanding of fundamental biological processes. Lattice protein folding models offer a simplified yet powerful framework for studying the complexities of protein folding, enabling the exploration of energetically optimal folds under constrained conditions. However, finding these optimal folds is a computationally challenging combinatorial optimization problem. In this work, we introduce a novel upper-bound training scheme that employs masking to identify the lowest-energy folds in two-dimensional Hydrophobic-Polar (HP) lattice protein folding. By leveraging Dilated Recurrent Neural Networks (RNNs) integrated with an annealing process driven by temperature-like fluctuations, our method accurately predicts optimal folds for benchmark systems of up to 60 beads. Our approach also effectively masks invalid folds from being sampled without compromising the autoregressive sampling properties of RNNs. This scheme is generalizable to three spatial dimensions and can be extended to lattice protein models with larger alphabets. Our findings emphasize the potential of advanced machine learning techniques in tackling complex protein folding problems and a broader class of constrained combinatorial optimization challenges. 

**Abstract (ZH)**: 理解蛋白质折叠原理是计算生物学的基石，对于药物设计、生物工程以及对基本生物过程的理解具有重要意义。格子蛋白质折叠模型提供了一种简化但强大的框架来研究蛋白质折叠的复杂性，使得在受约束条件下探索能量最优折叠成为可能。然而，找到这些最优折叠是一个计算上具有挑战性的组合优化问题。在本文中，我们介绍了一种新颖的上限训练方案，通过掩码技术在二维疏水-极性（HP）格子蛋白质折叠中识别最低能量折叠。通过利用集成退火过程的扩孔递归神经网络（RNNs），我们的方法能够准确预测多达60个珠子的标准系统中的最优折叠。我们的方法还有效地阻止无效折叠被采样，同时不损害RNNs的自回归采样特性。该方案可推广到三维空间，并可以扩展到具有更大字母表的格子蛋白质模型。我们的研究强调了高级机器学习技术在解决复杂蛋白质折叠问题和更广泛的受约束组合优化挑战中的潜在价值。 

---
# Continuous Adversarial Text Representation Learning for Affective Recognition 

**Title (ZH)**: 连续对抗性文本表示学习在情感识别中的应用 

**Authors**: Seungah Son, Andrez Saurez, Dongsoo Har  

**Link**: [PDF](https://arxiv.org/pdf/2502.20613)  

**Abstract**: While pre-trained language models excel at semantic understanding, they often struggle to capture nuanced affective information critical for affective recognition tasks. To address these limitations, we propose a novel framework for enhancing emotion-aware embeddings in transformer-based models. Our approach introduces a continuous valence-arousal labeling system to guide contrastive learning, which captures subtle and multi-dimensional emotional nuances more effectively. Furthermore, we employ a dynamic token perturbation mechanism, using gradient-based saliency to focus on sentiment-relevant tokens, improving model sensitivity to emotional cues. The experimental results demonstrate that the proposed framework outperforms existing methods, achieving up to 15.5% improvement in the emotion classification benchmark, highlighting the importance of employing continuous labels. This improvement demonstrates that the proposed framework is effective in affective representation learning and enables precise and contextually relevant emotional understanding. 

**Abstract (ZH)**: 预训练语言模型在 semantics 理解方面表现出色，但在情感识别任务中往往难以捕捉关键的细微情感信息。为解决这些问题，我们提出了一种增强变压器模型中情感意识嵌入的新框架。该方法引入了连续的正负面情感标注系统，以指导对比学习，从而更有效地捕捉细微和多维的情感 nuance。此外，我们采用了一种动态标记扰动机制，使用梯度基于的显著性聚焦于与情感相关的关键标记，提高模型对情感暗示的敏感性。实验结果表明，所提出的框架优于现有方法，在情感分类基准测试中最高可实现 15.5% 的性能提升，突显了使用连续标签的重要性。这一改进证明了所提出框架在情感表示学习中的有效性，并能实现精确且上下文相关的情感理解。 

---
# Exploring the Impact of Temperature Scaling in Softmax for Classification and Adversarial Robustness 

**Title (ZH)**: 探索温度标定在分类和对抗鲁棒性中的影响 

**Authors**: Hao Xuan, Bokai Yang, Xingyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.20604)  

**Abstract**: The softmax function is a fundamental component in deep learning. This study delves into the often-overlooked parameter within the softmax function, known as "temperature," providing novel insights into the practical and theoretical aspects of temperature scaling for image classification. Our empirical studies, adopting convolutional neural networks and transformers on multiple benchmark datasets, reveal that moderate temperatures generally introduce better overall performance. Through extensive experiments and rigorous theoretical analysis, we explore the role of temperature scaling in model training and unveil that temperature not only influences learning step size but also shapes the model's optimization direction. Moreover, for the first time, we discover a surprising benefit of elevated temperatures: enhanced model robustness against common corruption, natural perturbation, and non-targeted adversarial attacks like Projected Gradient Descent. We extend our discoveries to adversarial training, demonstrating that, compared to the standard softmax function with the default temperature value, higher temperatures have the potential to enhance adversarial training. The insights of this work open new avenues for improving model performance and security in deep learning applications. 

**Abstract (ZH)**: softmax函数是深度学习中的一个基本组件。本研究探讨了softmax函数中常常被忽视的参数“温度”，并从理论和实践两方面提供了关于温度缩放对图像分类影响的新见解。通过在多个基准数据集上采用卷积神经网络和变换器进行实证研究，我们发现适度的温度通常能改善整体性能。通过广泛的实验和严格的理论分析，我们探索了温度缩放在模型训练中的作用，并揭示了温度不仅影响学习步长，还塑造了模型的优化方向。此外，我们首次发现高温的一个意外优势：提高了模型对常见 corruption、自然扰动和非目标对抗攻击（如投影梯度下降）的鲁棒性。我们将这些发现扩展到对抗训练，表明与默认温度值的标准softmax函数相比，较高的温度有可能增强对抗训练。本研究的洞察为改进深度学习应用中的模型性能和安全打开了新的途径。 

---
# LiteASR: Efficient Automatic Speech Recognition with Low-Rank Approximation 

**Title (ZH)**: LiteASR: 低秩逼近下的高效自动语音识别 

**Authors**: Keisuke Kamahori, Jungo Kasai, Noriyuki Kojima, Baris Kasikci  

**Link**: [PDF](https://arxiv.org/pdf/2502.20583)  

**Abstract**: Modern automatic speech recognition (ASR) models, such as OpenAI's Whisper, rely on deep encoder-decoder architectures, and their encoders are a critical bottleneck for efficient deployment due to high computational intensity. We introduce LiteASR, a low-rank compression scheme for ASR encoders that significantly reduces inference costs while maintaining transcription accuracy. Our approach leverages the strong low-rank properties observed in intermediate activations: by applying principal component analysis (PCA) with a small calibration dataset, we approximate linear transformations with a chain of low-rank matrix multiplications, and further optimize self-attention to work in the reduced dimension. Evaluation results show that our method can compress Whisper large-v3's encoder size by over 50%, matching Whisper medium's size with better transcription accuracy, thereby establishing a new Pareto-optimal frontier of efficiency and performance. The code of LiteASR is available at this https URL. 

**Abstract (ZH)**: 现代自动语音识别（ASR）模型，如OpenAI的Whisper，依赖于深度编码-解码架构，而其编码器由于计算强度高成为高效部署的关键瓶颈。我们引入LiteASR，这是一种针对ASR编码器的低秩压缩方案，显著降低了推理成本同时保持了转写精度。我们的方法利用了中间激活显示出的强烈低秩特性：通过使用小型校准数据集的主成分分析（PCA），我们通过低秩矩阵乘法链来近似线性变换，并进一步优化自我注意力机制使其在降低的维度下工作。评估结果表明，我们的方法可以将Whisper large-v3的编码器大小压缩超过50%，在保持更好转写精度的情况下达到与Whisper medium相当的大小，从而建立了效率和性能的新帕累托最优前沿。LiteASR的代码可在以下链接获取。 

---
# Interpreting CLIP with Hierarchical Sparse Autoencoders 

**Title (ZH)**: 使用层次稀疏自编码器解释CLIP 

**Authors**: Vladimir Zaigrajew, Hubert Baniecki, Przemyslaw Biecek  

**Link**: [PDF](https://arxiv.org/pdf/2502.20578)  

**Abstract**: Sparse autoencoders (SAEs) are useful for detecting and steering interpretable features in neural networks, with particular potential for understanding complex multimodal representations. Given their ability to uncover interpretable features, SAEs are particularly valuable for analyzing large-scale vision-language models (e.g., CLIP and SigLIP), which are fundamental building blocks in modern systems yet remain challenging to interpret and control. However, current SAE methods are limited by optimizing both reconstruction quality and sparsity simultaneously, as they rely on either activation suppression or rigid sparsity constraints. To this end, we introduce Matryoshka SAE (MSAE), a new architecture that learns hierarchical representations at multiple granularities simultaneously, enabling a direct optimization of both metrics without compromise. MSAE establishes a new state-of-the-art Pareto frontier between reconstruction quality and sparsity for CLIP, achieving 0.99 cosine similarity and less than 0.1 fraction of variance unexplained while maintaining ~80% sparsity. Finally, we demonstrate the utility of MSAE as a tool for interpreting and controlling CLIP by extracting over 120 semantic concepts from its representation to perform concept-based similarity search and bias analysis in downstream tasks like CelebA. 

**Abstract (ZH)**: Matryoshka 稀疏自编码器：一种同时学习多粒度层次表示的新架构及其在 CLIP 解释与控制中的应用 

---
# PFformer: A Position-Free Transformer Variant for Extreme-Adaptive Multivariate Time Series Forecasting 

**Title (ZH)**: PFformer: 一种极端自适应多变量时间序列预测的无位置Transformer变体 

**Authors**: Yanhong Li, David C. Anastasiu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20571)  

**Abstract**: Multivariate time series (MTS) forecasting is vital in fields like weather, energy, and finance. However, despite deep learning advancements, traditional Transformer-based models often diminish the effect of crucial inter-variable relationships by singular token embedding and struggle to effectively capture complex dependencies among variables, especially in datasets with rare or extreme events. These events create significant imbalances and lead to high skewness, complicating accurate prediction efforts. This study introduces PFformer, a position-free Transformer-based model designed for single-target MTS forecasting, specifically for challenging datasets characterized by extreme variability. PFformer integrates two novel embedding strategies: Enhanced Feature-based Embedding (EFE) and Auto-Encoder-based Embedding (AEE). EFE effectively encodes inter-variable dependencies by mapping related sequence subsets to high-dimensional spaces without positional constraints, enhancing the encoder's functionality. PFformer shows superior forecasting accuracy without the traditional limitations of positional encoding in MTS modeling. We evaluated PFformer across four challenging datasets, focusing on two key forecasting scenarios: long sequence prediction for 3 days ahead and rolling predictions every four hours to reflect real-time decision-making processes in water management. PFformer demonstrated remarkable improvements, from 20% to 60%, compared with state-of-the-art models. 

**Abstract (ZH)**: 多变量时间序列（MTS）预测在气象、能源和金融等领域至关重要。然而，尽管深度学习取得了进展，传统的基于Transformer的模型常常通过单一的令牌嵌入减弱关键变量间关系的影响，并且在捕捉变量间的复杂依赖关系方面存在困难，尤其是在包含罕见或极端事件的数据集中表现不佳。这些事件造成了显著的数据不平衡和高偏度，使得准确预测变得更加复杂。本文介绍了一种名为PFformer的无位置Transformer模型，专门针对具有极端变异性等挑战的数据集进行单目标MTS预测。PFformer结合了两种新的嵌入策略：增强特征嵌入（EFE）和基于自编码器的嵌入（AEE）。EFE通过将相关的序列子集映射到高维空间中，而不受位置约束，有效地编码了变量间的依赖关系，增强了编码器的功能。PFformer在不依赖传统位置编码的情况下展现了卓越的预测准确性。我们在这四个具有挑战性的数据集上评估了PFformer，重点关注两种关键的预测场景：3天后的长序列预测以及每四小时滚动预测以反映水资源管理中的实时决策过程。与最先进的模型相比，PFformer在预测准确性上实现了20%到60%的显著改善。 

---
# DPZV: Resource Efficient ZO Optimization For Differentially Private VFL 

**Title (ZH)**: DPZV: 资源高效差异隐私联合学习中的零梯度优化 

**Authors**: Jianing Zhang, Evan Chen, Chaoyue Liu, Christopher G. Brinton  

**Link**: [PDF](https://arxiv.org/pdf/2502.20565)  

**Abstract**: Vertical Federated Learning (VFL) enables collaborative model training across feature-partitioned data, yet faces significant privacy risks and inefficiencies when scaling to large models. We propose DPZV, a memory-efficient Zeroth-Order(ZO) optimization framework that integrates differential privacy (DP) with vertical federated learning, addressing three critical challenges: (1) privacy vulnerabilities from gradient leakage, (2) high computation/communication costs of first-order methods, and (3) excessive memory footprint in conventional zeroth-order approaches. Our framework eliminates backpropagation through two-point gradient estimation, reducing client memory usage by 90\% compared to first-order counterparts while enabling asynchronous communication. By strategically injecting Gaussian noise on the server, DPZV achieves rigorous $(\epsilon, \delta)$-DP guarantees without third-party trust assumptions. Theoretical analysis establishes a convergence rate matching centralized case under non-convex objectives. Extensive experiments on image and NLP benchmarks demonstrate that DPZV outperforms all baselines in accuracy while providing strong privacy assurances ($\epsilon \leq 10$) and requiring far fewer computation resources, establishing new state-of-the-art privacy-utility tradeoffs for resource-constrained VFL deployments. 

**Abstract (ZH)**: DPZV：一种集成差分隐私的高效零阶优化框架以应对垂直联邦学习中的隐私与效率挑战 

---
# Revisiting Kernel Attention with Correlated Gaussian Process Representation 

**Title (ZH)**: 重新审视核注意力与相关高斯过程表示方法 

**Authors**: Long Minh Bui, Tho Tran Huu, Duy Dinh, Tan Minh Nguyen, Trong Nghia Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20525)  

**Abstract**: Transformers have increasingly become the de facto method to model sequential data with state-of-the-art performance. Due to its widespread use, being able to estimate and calibrate its modeling uncertainty is important to understand and design robust transformer models. To achieve this, previous works have used Gaussian processes (GPs) to perform uncertainty calibration for the attention units of transformers and attained notable successes. However, such approaches have to confine the transformers to the space of symmetric attention to ensure the necessary symmetric requirement of their GP's kernel specification, which reduces the representation capacity of the model. To mitigate this restriction, we propose the Correlated Gaussian Process Transformer (CGPT), a new class of transformers whose self-attention units are modeled as cross-covariance between two correlated GPs (CGPs). This allows asymmetries in attention and can enhance the representation capacity of GP-based transformers. We also derive a sparse approximation for CGP to make it scale better. Our empirical studies show that both CGP-based and sparse CGP-based transformers achieve better performance than state-of-the-art GP-based transformers on a variety of benchmark tasks. The code for our experiments is available at this https URL. 

**Abstract (ZH)**: Transformer已逐渐成为广泛用于建模序列数据的实际上乘方法，并取得了最先进的性能。由于其广泛应用，估算和校准其建模不确定性以理解并设计稳健的Transformer模型变得尤为重要。为了实现这一点，先前的工作使用高斯过程（GPs）对Transformer的注意力单元进行不确定性校准并取得了显著的成功。然而，这些方法需要将Transformer限制在对称注意力的空间中，以确保其GP核规范所需的对称性要求，这降低了模型的表示能力。为缓解这一限制，我们提出了相关高斯过程Transformer（CGPT），这是一种新的Transformer类，其自注意力单元被建模为两个相关高斯过程（CGPs）之间的交叉协方差。这允许注意力不对称，并能增强基于GP的Transformer的表示能力。我们还推导出CGP的稀疏近似，使其更加具有扩展性。我们的实证研究表明，基于CGP和稀疏CGP的Transformer在多种基准任务上都优于最先进的基于GP的Transformer。我们的实验代码可在以下链接获得：this https URL。 

---
# EgoNormia: Benchmarking Physical Social Norm Understanding 

**Title (ZH)**: EgoNormia：物理社会规范理解的基准测试 

**Authors**: MohammadHossein Rezaei, Yicheng Fu, Phil Cuvin, Caleb Ziems, Yanzhe Zhang, Hao Zhu, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20490)  

**Abstract**: Human activity is moderated by norms. When performing actions in the real world, humans not only follow norms, but also consider the trade-off between different norms However, machines are often trained without explicit supervision on norm understanding and reasoning, especially when the norms are grounded in a physical and social context. To improve and evaluate the normative reasoning capability of vision-language models (VLMs), we present EgoNormia $\|\epsilon\|$, consisting of 1,853 ego-centric videos of human interactions, each of which has two related questions evaluating both the prediction and justification of normative actions. The normative actions encompass seven categories: safety, privacy, proxemics, politeness, cooperation, coordination/proactivity, and communication/legibility. To compile this dataset at scale, we propose a novel pipeline leveraging video sampling, automatic answer generation, filtering, and human validation. Our work demonstrates that current state-of-the-art vision-language models lack robust norm understanding, scoring a maximum of 45% on EgoNormia (versus a human bench of 92%). Our analysis of performance in each dimension highlights the significant risks of safety, privacy, and the lack of collaboration and communication capability when applied to real-world agents. We additionally show that through a retrieval-based generation method, it is possible to use EgoNomia to enhance normative reasoning in VLMs. 

**Abstract (ZH)**: 人类行为受规范调节。在进行实际动作时，人类不仅遵守规范，还会权衡不同规范之间的取舍。然而，机器往往没有在明确监督下学习规范理解与推理，尤其是在规范根植于物理和社会情境时更是如此。为提升并评估视觉-语言模型（VLMs）的规范推理能力，我们提出了EgoNormia $\|\epsilon\|$，包含1,853个以人类互动为中心的视频，每个视频包含两个相关问题，分别评估规范动作的预测与理由。这些规范动作涵盖了七个类别：安全、隐私、人际距离、礼貌、合作、协作/积极性以及沟通/可读性。为了大规模编纂此数据集，我们提出了一种新的工作流程，利用视频采样、自动答案生成、过滤与人工验证。我们的研究显示，当前最先进的视觉-语言模型在EgoNormia上的规范理解能力较为脆弱，最高得分仅为45%（相比之下，人类基准得分为92%）。我们对每个维度的性能分析指出，当应用于真实世界代理时，其在安全、隐私方面存在显著风险，且缺乏合作和沟通能力。我们还展示了通过检索为基础的生成方法，可以利用EgoNormia提升视觉-语言模型的规范推理能力。 

---
# Will AI replace Software Engineers? Hold your Breath 

**Title (ZH)**: AI会取代软件工程师吗？屏住呼吸。 

**Authors**: Abhik Roychoudhury, Andreas Zeller  

**Link**: [PDF](https://arxiv.org/pdf/2502.20429)  

**Abstract**: Artificial Intelligence (AI) technology such as Large Language Models (LLMs) have become extremely popular in creating code. This has led to the conjecture that future software jobs will be exclusively conducted by LLMs, and the software industry will cease to exist. But software engineering is much more than producing code -- notably, \emph{maintaining} large software and keeping it reliable is a major part of software engineering, which LLMs are not yet capable of. 

**Abstract (ZH)**: 人工智能（AI）技术，如大型语言模型（LLMs），在生成代码方面变得极为流行。这导致有人猜测未来所有的软件工作都将由LLMs完成，软件产业也将不复存在。然而，软件工程远不止编写代码——特别是维护大型软件并确保其可靠运行是软件工程的重要组成部分，而这一点目前LLMs尚无法做到。 

---
# DeePen: Penetration Testing for Audio Deepfake Detection 

**Title (ZH)**: DeePen: 音频深度伪造检测的渗透测试 

**Authors**: Nicolas Müller, Piotr Kawa, Adriana Stan, Thien-Phuc Doan, Souhwan Jung, Wei Herng Choong, Philip Sperl, Konstantin Böttinger  

**Link**: [PDF](https://arxiv.org/pdf/2502.20427)  

**Abstract**: Deepfakes - manipulated or forged audio and video media - pose significant security risks to individuals, organizations, and society at large. To address these challenges, machine learning-based classifiers are commonly employed to detect deepfake content. In this paper, we assess the robustness of such classifiers through a systematic penetration testing methodology, which we introduce as DeePen. Our approach operates without prior knowledge of or access to the target deepfake detection models. Instead, it leverages a set of carefully selected signal processing modifications - referred to as attacks - to evaluate model vulnerabilities. Using DeePen, we analyze both real-world production systems and publicly available academic model checkpoints, demonstrating that all tested systems exhibit weaknesses and can be reliably deceived by simple manipulations such as time-stretching or echo addition. Furthermore, our findings reveal that while some attacks can be mitigated by retraining detection systems with knowledge of the specific attack, others remain persistently effective. We release all associated code. 

**Abstract (ZH)**: Deepfakes - 遭受个体、组织和社会的重大安全风险的操控或伪造音频和视频媒体。为应对这些挑战，通常采用基于机器学习的分类器来检测深伪内容。在本文中，我们通过一种系统性的渗透测试方法 DeePen 评估此类分类器的鲁棒性。我们的方法不需要事先了解或访问目标深伪检测模型，而是利用一组精心选择的信号处理修改（称为攻击）来评估模型的脆弱性。使用 DeePen，我们分析了现实世界生产的系统和公开可用的学术模型检查点，发现所有测试系统都存在弱点，并且可以通过简单操作如时间拉伸或回声添加可靠地欺骗。此外，我们的研究发现，虽然一些攻击可以通过利用特定攻击知识重新训练检测系统来缓解，但其他攻击仍具有持续效果。我们发布了所有相关代码。 

---
# Efficient Risk-sensitive Planning via Entropic Risk Measures 

**Title (ZH)**: 通过熵风险度量进行高效的风险敏感规划 

**Authors**: Alexandre Marthe, Samuel Bounan, Aurélien Garivier, Claire Vernade  

**Link**: [PDF](https://arxiv.org/pdf/2502.20423)  

**Abstract**: Risk-sensitive planning aims to identify policies maximizing some tail-focused metrics in Markov Decision Processes (MDPs). Such an optimization task can be very costly for the most widely used and interpretable metrics such as threshold probabilities or (Conditional) Values at Risk. Indeed, previous work showed that only Entropic Risk Measures (EntRM) can be efficiently optimized  through dynamic programming, leaving a hard-to-interpret parameter to choose.     We show that the computation of the full set of optimal policies for EntRM across parameter values leads to tight approximations for the metrics of interest. We prove that this optimality front can be computed effectively thanks to a novel structural analysis and smoothness properties of entropic risks.     Empirical results demonstrate that our approach achieves strong performance in a variety of decision-making scenarios. 

**Abstract (ZH)**: 基于熵的风险敏感规划旨在识别最大化马尔可夫决策过程(MDP)中极端指标的策略。对于阈值概率或(条件)风险价值等最常用且可解释的指标，这种优化任务通常是代价高昂的。事实上，先前的工作表明，只有熵风险度量(Entropic Risk Measures, EntRM)可以通过动态规划高效优化，但选择一个难以解释的参数。我们表明，计算EntRM在整个参数值范围内的全部最优策略可导出对目标指标的紧密近似。我们证明，由于熵风险的独特结构分析和平滑性质，这种最优前沿可以通过有效的方式计算。实证结果表明，我们的方法在多种决策场景中表现出强大的性能。 

---
# Backpropagation-free Spiking Neural Networks with the Forward-Forward Algorithm 

**Title (ZH)**: 无需反向传播的脉冲神经网络：前向前向算法 

**Authors**: Mohammadnavid Ghader, Saeed Reza Kheradpisheh, Bahar Farahani, Mahmood Fazlali  

**Link**: [PDF](https://arxiv.org/pdf/2502.20411)  

**Abstract**: Spiking Neural Networks (SNNs) offer a biologically inspired computational paradigm that emulates neuronal activity through discrete spike-based processing. Despite their advantages, training SNNs with traditional backpropagation (BP) remains challenging due to computational inefficiencies and a lack of biological plausibility. This study explores the Forward-Forward (FF) algorithm as an alternative learning framework for SNNs. Unlike backpropagation, which relies on forward and backward passes, the FF algorithm employs two forward passes, enabling localized learning, enhanced computational efficiency, and improved compatibility with neuromorphic hardware. We introduce an FF-based SNN training framework and evaluate its performance across both non-spiking (MNIST, Fashion-MNIST, CIFAR-10) and spiking (Neuro-MNIST, SHD) datasets. Experimental results demonstrate that our model surpasses existing FF-based SNNs by over 5% on MNIST and Fashion-MNIST while achieving accuracy comparable to state-of-the-art backpropagation-trained SNNs. On more complex tasks such as CIFAR-10 and SHD, our approach outperforms other SNN models by up to 6% and remains competitive with leading backpropagation-trained SNNs. These findings highlight the FF algorithm's potential to advance SNN training methodologies and neuromorphic computing by addressing key limitations of backpropagation. 

**Abstract (ZH)**: 基于前向传播的Spiking神经网络训练方法：Forward-Forward算法在SNNs中的应用与性能评估 

---
# Adversarial Robustness of Partitioned Quantum Classifiers 

**Title (ZH)**: 分拆量子分类器的对抗鲁棒性 

**Authors**: Pouya Kananian, Hans-Arno Jacobsen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20403)  

**Abstract**: Adversarial robustness in quantum classifiers is a critical area of study, providing insights into their performance compared to classical models and uncovering potential advantages inherent to quantum machine learning. In the NISQ era of quantum computing, circuit cutting is a notable technique for simulating circuits that exceed the qubit limitations of current devices, enabling the distribution of a quantum circuit's execution across multiple quantum processing units through classical communication. We examine how partitioning quantum classifiers through circuit cutting increase their susceptibility to adversarial attacks, establishing a link between attacking the state preparation channels in wire cutting and implementing adversarial gates within intermediate layers of a quantum classifier. We then proceed to study the latter problem from both a theoretical and experimental perspective. 

**Abstract (ZH)**: 量子分类器的对抗鲁棒性是研究的关键领域，提供了与经典模型性能对比的见解，并揭示了量子机器学习潜在的优势。在量子计算的NISQ时代，电路切割是一种显著的技术，用于模拟超越当前设备量子位限制的电路，通过经典通信将量子电路的执行分配到多个量子处理单元。我们探讨了通过电路切割分区量子分类器如何增加其对对抗攻击的易感性，建立了一个链接，即攻击线切割中的量子态准备通道与在量子分类器中间层中实施对抗门之间的联系。随后，我们从理论和实验两个角度研究了后者的问题。 

---
