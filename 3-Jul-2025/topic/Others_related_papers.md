# An RRT* algorithm based on Riemannian metric model for optimal path planning 

**Title (ZH)**: 基于黎曼度量模型的RRT*最优路径规划算法 

**Authors**: Yu Zhang, Qi Zhou, Xiao-Song Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.01697)  

**Abstract**: This paper presents a Riemannian metric-based model to solve the optimal path planning problem on two-dimensional smooth submanifolds in high-dimensional space. Our model is based on constructing a new Riemannian metric on a two-dimensional projection plane, which is induced by the high-dimensional Euclidean metric on two-dimensional smooth submanifold and reflects the environmental information of the robot. The optimal path planning problem in high-dimensional space is therefore transformed into a geometric problem on the two-dimensional plane with new Riemannian metric. Based on the new Riemannian metric, we proposed an incremental algorithm RRT*-R on the projection plane. The experimental results show that the proposed algorithm is suitable for scenarios with uneven fields in multiple dimensions. The proposed algorithm can help the robot to effectively avoid areas with drastic changes in height, ground resistance and other environmental factors. More importantly, the RRT*-R algorithm shows better smoothness and optimization properties compared with the original RRT* algorithm using Euclidean distance in high-dimensional workspace. The length of the entire path by RRT*-R is a good approximation of the theoretical minimum geodesic distance on projection plane. 

**Abstract (ZH)**: 基于黎曼度量的二维光滑子流形上最优路径规划模型 

---
# A Review on Sound Source Localization in Robotics: Focusing on Deep Learning Methods 

**Title (ZH)**: 机器人领域声音源定位综述：聚焦深度学习方法 

**Authors**: Reza Jalayer, Masoud Jalayer, Amirali Baniasadi  

**Link**: [PDF](https://arxiv.org/pdf/2507.01143)  

**Abstract**: Sound source localization (SSL) adds a spatial dimension to auditory perception, allowing a system to pinpoint the origin of speech, machinery noise, warning tones, or other acoustic events, capabilities that facilitate robot navigation, human-machine dialogue, and condition monitoring. While existing surveys provide valuable historical context, they typically address general audio applications and do not fully account for robotic constraints or the latest advancements in deep learning. This review addresses these gaps by offering a robotics-focused synthesis, emphasizing recent progress in deep learning methodologies. We start by reviewing classical methods such as Time Difference of Arrival (TDOA), beamforming, Steered-Response Power (SRP), and subspace analysis. Subsequently, we delve into modern machine learning (ML) and deep learning (DL) approaches, discussing traditional ML and neural networks (NNs), convolutional neural networks (CNNs), convolutional recurrent neural networks (CRNNs), and emerging attention-based architectures. The data and training strategy that are the two cornerstones of DL-based SSL are explored. Studies are further categorized by robot types and application domains to facilitate researchers in identifying relevant work for their specific contexts. Finally, we highlight the current challenges in SSL works in general, regarding environmental robustness, sound source multiplicity, and specific implementation constraints in robotics, as well as data and learning strategies in DL-based SSL. Also, we sketch promising directions to offer an actionable roadmap toward robust, adaptable, efficient, and explainable DL-based SSL for next-generation robots. 

**Abstract (ZH)**: 声源定位（SSL）为听觉感知增加了空间维度，使系统能够准确定位语音、机械噪声、警告音或其他声学事件的来源，这些能力有助于机器人的导航、人机对话和状态监控。虽然现有的综述提供了宝贵的历史背景，但它们通常关注一般音频应用，未充分考虑到机器人特有的限制或深度学习的最新进展。本文通过提供以机器人为重点的综合概述来弥补这些空白，强调了深度学习方法的最新进展。我们首先审查了经典方法，如时间到达差（TDOA）、波束成形、定向响应功率（SRP）和子空间分析。随后，我们探讨了现代机器学习（ML）和深度学习（DL）方法，讨论了传统机器学习和神经网络（NNs）、卷积神经网络（CNNs）、卷积递归神经网络（CRNNs）以及新兴的基于注意力的架构。我们在文章中探讨了基于DL的SSL的两大基石：数据和训练策略。我们进一步按机器人类型和应用领域分类研究，以帮助研究人员识别与其特定背景相关的工作。最后，我们指出了基于DL的SSL研究中普遍存在的挑战，包括环境鲁棒性、声源多样性以及在机器人中实施的具体限制，以及基于DL的SSL中的数据和学习策略。我们还勾勒出有前途的研究方向，为下一代机器人提供稳健、适应性强、高效和可解释的DL基础声源定位的实际路线图。 

---
# Time-Varying Coverage Control: A Distributed Tracker-Planner MPC Framework 

**Title (ZH)**: 时变覆盖控制：分布式跟踪-规划MPC框架 

**Authors**: Patrick Benito Eberhard, Johannes Köhler, Oliver Hüsser, Melanie N. Zeilinger, Andrea Carron  

**Link**: [PDF](https://arxiv.org/pdf/2507.01567)  

**Abstract**: Time-varying coverage control addresses the challenge of coordinating multiple agents covering an environment where regions of interest change over time. This problem has broad applications, including the deployment of autonomous taxis and coordination in search and rescue operations. The achievement of effective coverage is complicated by the presence of time-varying density functions, nonlinear agent dynamics, and stringent system and safety constraints. In this paper, we present a distributed multi-agent control framework for time-varying coverage under nonlinear constrained dynamics. Our approach integrates a reference trajectory planner and a tracking model predictive control (MPC) scheme, which operate at different frequencies within a multi-rate framework. For periodic density functions, we demonstrate closed-loop convergence to an optimal configuration of trajectories and provide formal guarantees regarding constraint satisfaction, collision avoidance, and recursive feasibility. Additionally, we propose an efficient algorithm capable of handling nonperiodic density functions, making the approach suitable for practical applications. Finally, we validate our method through hardware experiments using a fleet of four miniature race cars. 

**Abstract (ZH)**: 时间varying覆盖控制解决了环境兴趣区域随时间变化时协调多个代理的问题。这一问题有着广泛的应用，包括自主出租车的部署和搜救操作中的协调。有效覆盖的实现受到非线性代理动力学、时间varying密度函数以及严格系统和安全约束的复杂影响。在本文中，我们提出了一种适用于非线性约束动力学的时间varying覆盖分布式多代理控制框架。我们的方法在多速率框架中结合了参考轨迹规划器和跟踪模型预测控制（MPC）方案。对于周期性密度函数，我们证明了闭环收敛到最优轨迹配置，并提供了关于约束满足、碰撞避免和递归可行性形式上的保证。此外，我们提出了一种高效算法，能够处理非周期性密度函数，使该方法适用于实际应用。最后，我们通过使用四辆微型赛车的硬件实验验证了该方法。 

---
# Cooperative Target Capture in 3D Engagements over Switched Dynamic Graphs 

**Title (ZH)**: 基于切换动态图的三维交战中协同目标捕获 

**Authors**: Abhinav Sinha, Shashi Ranjan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.01350)  

**Abstract**: This paper presents a leaderless cooperative guidance strategy for simultaneous time-constrained interception of a stationary target when the interceptors exchange information over switched dynamic graphs. We specifically focus on scenarios when the interceptors lack radial acceleration capabilities, relying solely on their lateral acceleration components. This consideration aligns with their inherent kinematic turn constraints. The proposed strategy explicitly addresses the complexities of coupled 3D engagements, thereby mitigating performance degradation that typically arises when the pitch and yaw channels are decoupled into two separate, mutually orthogonal planar engagements. Moreover, our formulation incorporates modeling uncertainties associated with the time-to-go estimation into the derivation of cooperative guidance commands to ensure robustness against inaccuracies in dynamic engagement scenarios. To optimize control efficiency, we analytically derive the lateral acceleration components in the orthogonal pitch and yaw channels by solving an instantaneous optimization problem, subject to an affine constraint. We show that the proposed cooperative guidance commands guarantee consensus in time-to-go values within a predefined time, which can be prescribed as a design parameter, regardless of the interceptors' initial configurations. We provide simulations to attest to the efficacy of the proposed method. 

**Abstract (ZH)**: 一种在切换动态图上交换信息且缺乏径向加速度能力的拦截器同时时间约束拦截静止目标的无需领导者协同引导策略 

---
# Optimal Dispersion Under Asynchrony 

**Title (ZH)**: 异步环境下最优分散化 

**Authors**: Debasish Pattanayak, Ajay D. Kshemkalyani, Manish Kumar, Anisur Rahaman Molla, Gokarna Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2507.01298)  

**Abstract**: We study the dispersion problem in anonymous port-labeled graphs: $k \leq n$ mobile agents, each with a unique ID and initially located arbitrarily on the nodes of an $n$-node graph with maximum degree $\Delta$, must autonomously relocate so that no node hosts more than one agent. Dispersion serves as a fundamental task in distributed computing of mobile agents, and its complexity stems from key challenges in local coordination under anonymity and limited memory.
The goal is to minimize both the time to achieve dispersion and the memory required per agent. It is known that any algorithm requires $\Omega(k)$ time in the worst case, and $\Omega(\log k)$ bits of memory per agent. A recent result [SPAA'25] gives an optimal $O(k)$-time algorithm in the synchronous setting and an $O(k \log k)$-time algorithm in the asynchronous setting, both using $O(\log(k+\Delta))$ bits.
In this paper, we close the complexity gap in the asynchronous setting by presenting the first dispersion algorithm that runs in optimal $O(k)$ time using $O(\log(k+\Delta))$ bits of memory per agent. Our solution is based on a novel technique we develop in this paper that constructs a port-one tree in anonymous graphs, which may be of independent interest. 

**Abstract (ZH)**: 匿名端标图中的分散问题研究：在具有最大度为Δ的n个节点的图上，k（≤n）个具有唯一ID的移动代理初始位于图的节点上，自主重新定位以确保每个节点不超过一个代理。我们在分布式移动代理计算中将分散作为基本任务，其复杂性源于匿名性和有限内存下局部协调的关键挑战。本研究旨在最小化实现分散所需时间和每个代理所需的内存。已知任何算法在最坏情况下需要Ω(k)时间，并且每个代理需要Ω(log k)位内存。最近的结果在同步环境中给出了最优的O(k)时间算法，在异步环境中给出了O(k log k)时间算法，两者都使用O(log(k+Δ))位内存。在本文中，我们通过提出第一个在最优O(k)时间且每代理使用O(log(k+Δ))位内存的分散算法，填补了异步环境中的复杂性缺口。我们的解决方案基于本文中开发的一种新技巧，用于构建匿名图上的端一树，这可能具有独立的研究兴趣。 

---
# Learning to Segment for Vehicle Routing Problems 

**Title (ZH)**: 学习分割方法解决车辆路线问题 

**Authors**: Wenbin Ouyang, Sirui Li, Yining Ma, Cathy Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.01037)  

**Abstract**: Iterative search heuristics are widely recognized as state-of-the-art for solving Vehicle Routing Problems (VRPs). In this work, we identify and exploit a critical observation: within these solvers, a large portion of the solution remains stable, i.e., unchanged across search iterations, causing redundant computations, especially for large-scale VRPs with long subtours. To address this, we pioneer the formal study of the First-Segment-Then-Aggregate (FSTA) decomposition technique to accelerate iterative solvers. Specifically, FSTA preserves stable solution segments during the search, aggregates nodes within each segment into fixed hypernodes, and focuses the search only on unstable portions. Yet, a key challenge lies in identifying which segments should be aggregated by FSTA. To this end, we then introduce Learning-to-Segment (L2Seg), a novel neural framework to intelligently differentiate potentially stable and unstable portions for FSTA decomposition. We present three L2Seg variants: non-autoregressive (globally comprehensive but locally indiscriminate), autoregressive (locally refined but globally deficient), and their synergy, with bespoke training and inference strategies. Empirical results on CVRP and VRPTW suggest that L2Seg accelerates state-of-the-art iterative solvers by up to 7x. Additionally, we provide in-depth analysis showing NAR and AR synergy achieves best performance by combining their complementary strengths. Notably, L2Seg is a flexible framework that is compatible with traditional, learning-based, and hybrid solvers, while supporting a broad class of VRPs. 

**Abstract (ZH)**: 迭代搜索启发式方法被认为是解决车辆 routing 问题（VRPs）的前沿方法。在此工作中，我们识别并利用一个重要观察：在这些求解器中，大部分解决方案在搜索迭代过程中保持稳定，即在多次迭代中不发生变化，尤其是在大规模具有长子环的 VRPs 中导致冗余计算。为解决这一问题，我们开创性地研究了 First-Segment-Then-Aggregate (FSTA) 分解技术以加速迭代求解器。具体而言，FSTA 在搜索过程中保留稳定的解决方案段，将每个段内的节点聚合为固定超节点，并仅对不稳定的部分进行搜索。然而，在 FSTA 中聚合哪些段是关键挑战。为解决这一问题，我们随后引入了 Learning-to-Segment (L2Seg)，这是一种新颖的神经网络框架，用于智能地区分 FSTA 分解中潜在的稳定和不稳定部分。我们提出了三种 L2Seg 变体：非自回归（全局综合但局部不分辨）、自回归（局部细腻但全局不足）及其结合，配有专门的训练和推理策略。实验结果表明，L2Seg 可以将最先进的迭代求解器加速多达 7 倍。此外，我们提供了深入分析，表明非自回归和自回归结合利用其互补优势可实现最佳性能。值得注意的是，L2Seg 是一个灵活的框架，兼容传统、基于学习以及混合求解器，并支持广泛的 VRPs 类型。 

---
# Refining Gelfond Rationality Principle Towards More Comprehensive Foundational Principles for Answer Set Semantics 

**Title (ZH)**: 面向更全面的答案集语义基础原则细化格尔丰德有理性原则 

**Authors**: Yi-Dong Shen, Thomas Eiter  

**Link**: [PDF](https://arxiv.org/pdf/2507.01833)  

**Abstract**: Non-monotonic logic programming is the basis for a declarative problem solving paradigm known as answer set programming (ASP). Departing from the seminal definition by Gelfond and Lifschitz in 1988 for simple normal logic programs, various answer set semantics have been proposed for extensions. We consider two important questions: (1) Should the minimal model property, constraint monotonicity and foundedness as defined in the literature be mandatory conditions for an answer set semantics in general? (2) If not, what other properties could be considered as general principles for answer set semantics? We address the two questions. First, it seems that the three aforementioned conditions may sometimes be too strong, and we illustrate with examples that enforcing them may exclude expected answer sets. Second, we evolve the Gelfond answer set (GAS) principles for answer set construction by refining the Gelfond's rationality principle to well-supportedness, minimality w.r.t. negation by default and minimality w.r.t. epistemic negation. The principle of well-supportedness guarantees that every answer set is constructible from if-then rules obeying a level mapping and is thus free of circular justification, while the two minimality principles ensure that the formalism minimizes knowledge both at the level of answer sets and of world views. Third, to embody the refined GAS principles, we extend the notion of well-supportedness substantially to answer sets and world views, respectively. Fourth, we define new answer set semantics in terms of the refined GAS principles. Fifth, we use the refined GAS principles as an alternative baseline to intuitively assess the existing answer set semantics. Finally, we analyze the computational complexity. 

**Abstract (ZH)**: 非单调逻辑编程是回答集编程（ASP）这一声明式问题求解范式的基础。从Gelfond和Lifschitz在1988年对简单正常逻辑程序的开创性定义出发，已提出了各种回答集语义扩展。我们考虑了两个重要问题：（1）文献中定义的最小模型性质、约束单调性和奠基性是否应作为一般回答集语义的基本条件？（2）如果不应如此，可以考虑哪些其他属性作为回答集语义的一般原则？我们解答了这两个问题。首先，看来上述三个条件有时可能过于严格，我们通过例子说明强制执行它们可能会排除预期的回答集。其次，我们通过将Gelfond的合理性原则细化为支持性、基于默认否定的最小性和基于知识否定的最小性，来改进Gelfond回答集（GAS）原则，以回答集构造为出发点。支持性原则保证每个回答集可通过遵循等级映射的if-then规则构造，从而避免循环证明，而两个最小性原则则确保该形式化方法在回答集和世界观层面都尽量减少知识。第三，为了体现改进后的GAS原则，我们大幅扩展了支持性的概念分别应用于回答集和世界观。第四，我们根据改进后的GAS原则定义新的回答集语义。第五，我们使用改进后的GAS原则作为替代基准，以直观方式评估现有的回答集语义。最后，我们分析了计算复杂性。 

---
# Joint Matching and Pricing for Crowd-shipping with In-store Customers 

**Title (ZH)**: 基于商店内顾客的 crowds-shipping 匹配与定价联合优化 

**Authors**: Arash Dehghan, Mucahit Cevik, Merve Bodur, Bissan Ghaddar  

**Link**: [PDF](https://arxiv.org/pdf/2507.01749)  

**Abstract**: This paper examines the use of in-store customers as delivery couriers in a centralized crowd-shipping system, targeting the growing need for efficient last-mile delivery in urban areas. We consider a brick-and-mortar retail setting where shoppers are offered compensation to deliver time-sensitive online orders. To manage this process, we propose a Markov Decision Process (MDP) model that captures key uncertainties, including the stochastic arrival of orders and crowd-shippers, and the probabilistic acceptance of delivery offers. Our solution approach integrates Neural Approximate Dynamic Programming (NeurADP) for adaptive order-to-shopper assignment with a Deep Double Q-Network (DDQN) for dynamic pricing. This joint optimization strategy enables multi-drop routing and accounts for offer acceptance uncertainty, aligning more closely with real-world operations. Experimental results demonstrate that the integrated NeurADP + DDQN policy achieves notable improvements in delivery cost efficiency, with up to 6.7\% savings over NeurADP with fixed pricing and approximately 18\% over myopic baselines. We also show that allowing flexible delivery delays and enabling multi-destination routing further reduces operational costs by 8\% and 17\%, respectively. These findings underscore the advantages of dynamic, forward-looking policies in crowd-shipping systems and offer practical guidance for urban logistics operators. 

**Abstract (ZH)**: 基于集中众包配送系统的店内顾客送货研究：面向城市地区的高效最后公里配送需求 

---
# T3DM: Test-Time Training-Guided Distribution Shift Modelling for Temporal Knowledge Graph Reasoning 

**Title (ZH)**: T3DM：测试时训练引导的分布偏移建模以进行时间知识图推理 

**Authors**: Yuehang Si, Zefan Zeng, Jincai Huang, Qing Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.01597)  

**Abstract**: Temporal Knowledge Graph (TKG) is an efficient method for describing the dynamic development of facts along a timeline. Most research on TKG reasoning (TKGR) focuses on modelling the repetition of global facts and designing patterns of local historical facts. However, they face two significant challenges: inadequate modeling of the event distribution shift between training and test samples, and reliance on random entity substitution for generating negative samples, which often results in low-quality sampling. To this end, we propose a novel distributional feature modeling approach for training TKGR models, Test-Time Training-guided Distribution shift Modelling (T3DM), to adjust the model based on distribution shift and ensure the global consistency of model reasoning. In addition, we design a negative-sampling strategy to generate higher-quality negative quadruples based on adversarial training. Extensive experiments show that T3DM provides better and more robust results than the state-of-the-art baselines in most cases. 

**Abstract (ZH)**: 基于时间的知识图谱推理的训练时训练指导分佈迁移建模（T3DM） 

---
# A Fuzzy Approach to the Specification, Verification and Validation of Risk-Based Ethical Decision Making Models 

**Title (ZH)**: 基于风险的伦理决策模型的规范、验证与验证的模糊方法 

**Authors**: Abeer Dyoub, Francesca A. Lisi  

**Link**: [PDF](https://arxiv.org/pdf/2507.01410)  

**Abstract**: The ontological and epistemic complexities inherent in the moral domain make it challenging to establish clear standards for evaluating the performance of a moral machine. In this paper, we present a formal method to describe Ethical Decision Making models based on ethical risk assessment. Then, we show how these models that are specified as fuzzy rules can be verified and validated using fuzzy Petri nets. A case study from the medical field is considered to illustrate the proposed approach. 

**Abstract (ZH)**: 道德领域内在的本体论和认识论复杂性使得确立评价道德机器性能的明确标准具有挑战性。本文提出了一种基于伦理风险评估描述伦理决策模型的形式化方法，展示了如何使用模糊Petri网验证和验证用模糊规则指定的模型。考虑了医疗领域的案例研究以说明所提出的方法。 

---
# Beyond Black-Box AI: Interpretable Hybrid Systems for Dementia Care 

**Title (ZH)**: 超越黑箱AI：可解释的混合系统在痴呆症护理中的应用 

**Authors**: Matthew JY Kang, Wenli Yang, Monica R Roberts, Byeong Ho Kang, Charles B Malpas  

**Link**: [PDF](https://arxiv.org/pdf/2507.01282)  

**Abstract**: The recent boom of large language models (LLMs) has re-ignited the hope that artificial intelligence (AI) systems could aid medical diagnosis. Yet despite dazzling benchmark scores, LLM assistants have yet to deliver measurable improvements at the bedside. This scoping review aims to highlight the areas where AI is limited to make practical contributions in the clinical setting, specifically in dementia diagnosis and care.
Standalone machine-learning models excel at pattern recognition but seldom provide actionable, interpretable guidance, eroding clinician trust. Adjacent use of LLMs by physicians did not result in better diagnostic accuracy or speed. Key limitations trace to the data-driven paradigm: black-box outputs which lack transparency, vulnerability to hallucinations, and weak causal reasoning. Hybrid approaches that combine statistical learning with expert rule-based knowledge, and involve clinicians throughout the process help bring back interpretability. They also fit better with existing clinical workflows, as seen in examples like PEIRS and ATHENA-CDS.
Future decision-support should prioritise explanatory coherence by linking predictions to clinically meaningful causes. This can be done through neuro-symbolic or hybrid AI that combines the language ability of LLMs with human causal expertise. AI researchers have addressed this direction, with explainable AI and neuro-symbolic AI being the next logical steps in further advancement in AI. However, they are still based on data-driven knowledge integration instead of human-in-the-loop approaches. Future research should measure success not only by accuracy but by improvements in clinician understanding, workflow fit, and patient outcomes. A better understanding of what helps improve human-computer interactions is greatly needed for AI systems to become part of clinical practice. 

**Abstract (ZH)**: 近期大型语言模型的兴起重新点燃了AI辅助医学诊断的希望。然而，尽管基准测试成绩令人耀眼，AI助手仍未在临床环境中带来可量化的改进。本综述旨在highlight AI在临床环境中进行实际贡献的局限性，特别是在痴呆症诊断和护理中的应用。独立的机器学习模型在模式识别方面表现出色，但很少能提供可操作的、可解释的指导，削弱了临床医生的信任。医生与大型语言模型的相邻使用并未提高诊断准确性和速度。关键限制源自数据驱动的范式：不透明的黑箱输出、易产生幻觉以及因果推理能力较弱。结合统计学习与基于专家规则的知识，并在整个过程中涉及临床医生的方法有助于恢复可解释性。这种方法也更符合现有的临床工作流程，如PEIRS和ATHENA-CDS等示例所示。未来的决策支持应优先考虑解释一致性，通过将LLM的自然语言能力与人类因果专业知识相结合的神经符号AI或混合AI来实现这一目标。AI研究者已经朝着这一方向努力，可解释AI和神经符号AI是进一步推进AI逻辑步骤。然而，它们仍然基于数据驱动的知识整合，而非人工在环的方法。未来的研究应该不仅以准确性来衡量成功，还应关注临床医生的理解、工作流程的适应性和患者的结局。更好地理解如何提高人机交互将有助于AI系统成为临床实践的一部分。 

---
# Adaptability of ASR Models on Low-Resource Language: A Comparative Study of Whisper and Wav2Vec-BERT on Bangla 

**Title (ZH)**: 低资源语言环境下ASR模型的适应性：Whisper与Wav2Vec-BERT在孟加拉语中的比较研究 

**Authors**: Md Sazzadul Islam Ridoy, Sumi Akter, Md. Aminur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2507.01931)  

**Abstract**: In recent years, neural models trained on large multilingual text and speech datasets have shown great potential for supporting low-resource languages. This study investigates the performances of two state-of-the-art Automatic Speech Recognition (ASR) models, OpenAI's Whisper (Small & Large-V2) and Facebook's Wav2Vec-BERT on Bangla, a low-resource language. We have conducted experiments using two publicly available datasets: Mozilla Common Voice-17 and OpenSLR to evaluate model performances. Through systematic fine-tuning and hyperparameter optimization, including learning rate, epochs, and model checkpoint selection, we have compared the models based on Word Error Rate (WER), Character Error Rate (CER), Training Time, and Computational Efficiency. The Wav2Vec-BERT model outperformed Whisper across all key evaluation metrics, demonstrated superior performance while requiring fewer computational resources, and offered valuable insights to develop robust speech recognition systems in low-resource linguistic settings. 

**Abstract (ZH)**: 最近几年，用于大规模多语言文本和语音数据训练的神经模型在支持低资源语言方面展现了巨大潜力。本研究考察了两种最先进的自动语音识别（ASR）模型，OpenAI的Whisper（Small & Large-V2）和Facebook的Wav2Vec-BERT在孟加拉语低资源语言上的性能。我们使用两个公开可用的数据集Mozilla Common Voice-17和OpenSLR来评估模型性能。通过系统的微调和超参数优化，包括学习率、epoch次数和模型检查点的选择，我们基于词错误率（WER）、字符错误率（CER）、训练时间以及计算效率对模型进行了比较。Wav2Vec-BERT模型在所有关键评价指标上均优于Whisper，表现出更优秀的性能并需要更少的计算资源，为在低资源语言背景下开发稳健的语音识别系统提供了宝贵见解。 

---
# Exploring a Hybrid Deep Learning Approach for Anomaly Detection in Mental Healthcare Provider Billing: Addressing Label Scarcity through Semi-Supervised Anomaly Detection 

**Title (ZH)**: 基于半监督异常检测解决标签稀缺性的一种混合深度学习方法在精神健康医疗提供商账单异常检测中的探索 

**Authors**: Samirah Bakker, Yao Ma, Seyed Sahand Mohammadi Ziabari  

**Link**: [PDF](https://arxiv.org/pdf/2507.01924)  

**Abstract**: The complexity of mental healthcare billing enables anomalies, including fraud. While machine learning methods have been applied to anomaly detection, they often struggle with class imbalance, label scarcity, and complex sequential patterns. This study explores a hybrid deep learning approach combining Long Short-Term Memory (LSTM) networks and Transformers, with pseudo-labeling via Isolation Forests (iForest) and Autoencoders (AE). Prior work has not evaluated such hybrid models trained on pseudo-labeled data in the context of healthcare billing. The approach is evaluated on two real-world billing datasets related to mental healthcare. The iForest LSTM baseline achieves the highest recall (0.963) on declaration-level data. On the operation-level data, the hybrid iForest-based model achieves the highest recall (0.744), though at the cost of lower precision. These findings highlight the potential of combining pseudo-labeling with hybrid deep learning in complex, imbalanced anomaly detection settings. 

**Abstract (ZH)**: 医疗保健billing复杂性导致异常，包括欺诈。虽然已经应用了机器学习方法进行异常检测，但它们往往难以处理类别不平衡、标签稀缺和复杂的序列模式。本研究探索结合长短期记忆（LSTM）网络和变换器的混合深度学习方法，并通过孤立森林（iForest）和自动编码器（AE）进行伪标签化。以往的研究尚未在医疗保健billing上下文中评估基于伪标签数据训练的此类混合模型。该方法在两个与医疗保健billing相关的实际billing数据集上进行评估。iForest LSTM基线在声明级数据上获得了最高的召回率（0.963）。在操作级数据上，基于iForest的混合模型获得了最高的召回率（0.744），但精确度较低。这些发现突显了在复杂、不平衡的异常检测设置中结合伪标签与混合深度学习的潜力。 

---
# End-to-End Large Portfolio Optimization for Variance Minimization with Neural Networks through Covariance Cleaning 

**Title (ZH)**: 通过协方差清洁利用神经网络实现端到端大型投资组合优化以最小化方差 

**Authors**: Christian Bongiorno, Efstratios Manolakis, Rosario Nunzio Mantegna  

**Link**: [PDF](https://arxiv.org/pdf/2507.01918)  

**Abstract**: We develop a rotation-invariant neural network that provides the global minimum-variance portfolio by jointly learning how to lag-transform historical returns and how to regularise both the eigenvalues and the marginal volatilities of large equity covariance matrices. This explicit mathematical mapping offers clear interpretability of each module's role, so the model cannot be regarded as a pure black-box. The architecture mirrors the analytical form of the global minimum-variance solution yet remains agnostic to dimension, so a single model can be calibrated on panels of a few hundred stocks and applied, without retraining, to one thousand US equities-a cross-sectional jump that demonstrates robust out-of-sample generalisation. The loss function is the future realized minimum portfolio variance and is optimized end-to-end on real daily returns. In out-of-sample tests from January 2000 to December 2024 the estimator delivers systematically lower realised volatility, smaller maximum drawdowns, and higher Sharpe ratios than the best analytical competitors, including state-of-the-art non-linear shrinkage. Furthermore, although the model is trained end-to-end to produce an unconstrained (long-short) minimum-variance portfolio, we show that its learned covariance representation can be used in general optimizers under long-only constraints with virtually no loss in its performance advantage over competing estimators. These gains persist when the strategy is executed under a highly realistic implementation framework that models market orders at the auctions, empirical slippage, exchange fees, and financing charges for leverage, and they remain stable during episodes of acute market stress. 

**Abstract (ZH)**: 一种旋转不变神经网络及其在全局最小方差组合中的应用 

---
# AI4Research: A Survey of Artificial Intelligence for Scientific Research 

**Title (ZH)**: AI4Research：人工智能在科学研究中的应用综述 

**Authors**: Qiguang Chen, Mingda Yang, Libo Qin, Jinhao Liu, Zheng Yan, Jiannan Guan, Dengyun Peng, Yiyan Ji, Hanjing Li, Mengkang Hu, Yimeng Zhang, Yihao Liang, Yuhang Zhou, Jiaqi Wang, Zhi Chen, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2507.01903)  

**Abstract**: Recent advancements in artificial intelligence (AI), particularly in large language models (LLMs) such as OpenAI-o1 and DeepSeek-R1, have demonstrated remarkable capabilities in complex domains such as logical reasoning and experimental coding. Motivated by these advancements, numerous studies have explored the application of AI in the innovation process, particularly in the context of scientific research. These AI technologies primarily aim to develop systems that can autonomously conduct research processes across a wide range of scientific disciplines. Despite these significant strides, a comprehensive survey on AI for Research (AI4Research) remains absent, which hampers our understanding and impedes further development in this field. To address this gap, we present a comprehensive survey and offer a unified perspective on AI4Research. Specifically, the main contributions of our work are as follows: (1) Systematic taxonomy: We first introduce a systematic taxonomy to classify five mainstream tasks in AI4Research. (2) New frontiers: Then, we identify key research gaps and highlight promising future directions, focusing on the rigor and scalability of automated experiments, as well as the societal impact. (3) Abundant applications and resources: Finally, we compile a wealth of resources, including relevant multidisciplinary applications, data corpora, and tools. We hope our work will provide the research community with quick access to these resources and stimulate innovative breakthroughs in AI4Research. 

**Abstract (ZH)**: Recent advancements in artificial intelligence (AI) and large language models (LLMs) in logical reasoning and experimental coding have demonstrated remarkable capabilities in complex domains. Motivated by these advancements, numerous studies have explored the application of AI in scientific research innovation. To address the absence of a comprehensive survey on AI for Research (AI4Research), we present a comprehensive survey and offer a unified perspective on AI4Research. 

---
# Towards Foundation Auto-Encoders for Time-Series Anomaly Detection 

**Title (ZH)**: 面向时间序列异常检测的foundation自动编码器研究 

**Authors**: Gastón García González, Pedro Casas, Emilio Martínez, Alicia Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2507.01875)  

**Abstract**: We investigate a novel approach to time-series modeling, inspired by the successes of large pretrained foundation models. We introduce FAE (Foundation Auto-Encoders), a foundation generative-AI model for anomaly detection in time-series data, based on Variational Auto-Encoders (VAEs). By foundation, we mean a model pretrained on massive amounts of time-series data which can learn complex temporal patterns useful for accurate modeling, forecasting, and detection of anomalies on previously unseen datasets. FAE leverages VAEs and Dilated Convolutional Neural Networks (DCNNs) to build a generic model for univariate time-series modeling, which could eventually perform properly in out-of-the-box, zero-shot anomaly detection applications. We introduce the main concepts of FAE, and present preliminary results in different multi-dimensional time-series datasets from various domains, including a real dataset from an operational mobile ISP, and the well known KDD 2021 Anomaly Detection dataset. 

**Abstract (ZH)**: 我们提出了一种受大规模预训练基础模型成功启发的时间序列建模新方法。我们引入了FAE（基础自编码器），这是一种基于变异自编码器（VAE）的时间序列异常检测生成AI模型。通过基础这一概念，我们指的是在一个庞大时间序列数据集上预训练的模型，能够学习复杂的时序模式，从而有助于准确建模、预测和检测未见过数据集的异常。FAE利用了变异自编码器和扩张卷积神经网络（DCNN）来构建一种通用的单变量时间序列建模模型，最终可在开箱即用和零样本异常检测应用中表现良好。我们介绍了FAE的核心概念，并在多个跨领域的多维时间序列数据集上展示了初步结果，这些数据集包括一个实际操作中的移动ISP数据集和著名的KDD 2021异常检测数据集。 

---
# mGRADE: Minimal Recurrent Gating Meets Delay Convolutions for Lightweight Sequence Modeling 

**Title (ZH)**: mGRADE: 最小循环门控结合延迟卷积的轻量级序列建模方法 

**Authors**: Tristan Torchet, Christian Metzner, Laura Kriener, Melika Payvand  

**Link**: [PDF](https://arxiv.org/pdf/2507.01829)  

**Abstract**: Edge devices for temporal processing demand models that capture both short- and long- range dynamics under tight memory constraints. While Transformers excel at sequence modeling, their quadratic memory scaling with sequence length makes them impractical for such settings. Recurrent Neural Networks (RNNs) offer constant memory but train sequentially, and Temporal Convolutional Networks (TCNs), though efficient, scale memory with kernel size. To address this, we propose mGRADE (mininally Gated Recurrent Architecture with Delay Embedding), a hybrid-memory system that integrates a temporal 1D-convolution with learnable spacings followed by a minimal gated recurrent unit (minGRU). This design allows the convolutional layer to realize a flexible delay embedding that captures rapid temporal variations, while the recurrent module efficiently maintains global context with minimal memory overhead. We validate our approach on two synthetic tasks, demonstrating that mGRADE effectively separates and preserves multi-scale temporal features. Furthermore, on challenging pixel-by-pixel image classification benchmarks, mGRADE consistently outperforms both pure convolutional and pure recurrent counterparts using approximately 20% less memory footprint, highlighting its suitability for memory-constrained temporal processing at the edge. This highlights mGRADE's promise as an efficient solution for memory-constrained multi-scale temporal processing at the edge. 

**Abstract (ZH)**: 边缘设备对时序处理的需求模型应在紧凑的内存约束下捕捉短程和长程动态。为应对这一挑战，我们提出了mGRADE（最小门控循环架构与延迟嵌入），这是一种结合时序一维卷积和可学习间隔，随后是minimal gated recurrent unit (minGRU)的混合内存系统。此设计允许卷积层实现灵活的延迟嵌入，捕捉快速时序变化，同时循环模块以最小的内存开销高效保持全局上下文。我们在两个合成任务上验证了我们的方法，证明mGRADE能够有效分离并保留多尺度时序特征。此外，在具有挑战性的逐像素图像分类基准测试中，mGRADE在相同内存占用下比纯粹卷积和纯粹循环对应的模型表现更好，突显了其在边缘设备上进行多尺度时序处理的高效性。 

---
# MILP-SAT-GNN: Yet Another Neural SAT Solver 

**Title (ZH)**: MILP-SAT-GNN：另一个神经SAT求解器 

**Authors**: Franco Alberto Cardillo, Hamza Khyari, Umberto Straccia  

**Link**: [PDF](https://arxiv.org/pdf/2507.01825)  

**Abstract**: We proposes a novel method that enables Graph Neural Networks (GNNs) to solve SAT problems by leveraging a technique developed for applying GNNs to Mixed Integer Linear Programming (MILP). Specifically, k-CNF formulae are mapped into MILP problems, which are then encoded as weighted bipartite graphs and subsequently fed into a GNN for training and testing. From a theoretical perspective: (i) we establish permutation and equivalence invariance results, demonstrating that the method produces outputs that are stable under reordering of clauses and variables; (ii) we identify a theoretical limitation, showing that for a class of formulae called foldable formulae, standard GNNs cannot always distinguish satisfiable from unsatisfiable instances; (iii) we prove a universal approximation theorem, establishing that with Random Node Initialization (RNI), the method can approximate SAT solving to arbitrary precision on finite datasets, that is, the GNN becomes approximately sound and complete on such datasets. Furthermore, we show that for unfoldable formulae, the same approximation guarantee can be achieved without the need for RNI. Finally, we conduct an experimental evaluation of our approach, which show that, despite the simplicity of the neural architecture, the method achieves promising results. 

**Abstract (ZH)**: 我们提出了一种新颖的方法，通过利用将图形神经网络（GNNs）应用于混合整数线性规划（MILP）的技术来解决可满足性（SAT）问题。具体而言，k-CNF公式被映射到MILP问题，然后编码为加权二分图，并随后输入GNN进行训练和测试。从理论角度来看：（i）我们建立了置换和等价不变性结果，表明该方法在重排序子句和变量时生成的输出是稳定的；（ii）我们识别了一个理论上的限制，表明对于一类称为可折叠公式的问题，标准的GNN不能总是在判别可满足实例和不可满足实例时发挥作用；（iii）我们证明了一个通用逼近定理，表明通过随机节点初始化（RNI），该方法可以针对有限数据集以任意精度逼近SAT求解，即GNN在这些数据集上变得近似地有效且完备。此外，我们展示了对于非可展开公式，可以在无需RNI的情况下达到相同的逼近保证。最后，我们对本方法进行了实验评估，结果显示，尽管神经网络架构的简单性，该方法取得了令人鼓舞的结果。 

---
# Empowering Manufacturers with Privacy-Preserving AI Tools: A Case Study in Privacy-Preserving Machine Learning to Solve Real-World Problems 

**Title (ZH)**: 保护制造商的隐私保护人工智能工具：基于隐私保护机器学习解决实际问题的案例研究 

**Authors**: Xiaoyu Ji, Jessica Shorland, Joshua Shank, Pascal Delpe-Brice, Latanya Sweeney, Jan Allebach, Ali Shakouri  

**Link**: [PDF](https://arxiv.org/pdf/2507.01808)  

**Abstract**: Small- and medium-sized manufacturers need innovative data tools but, because of competition and privacy concerns, often do not want to share their proprietary data with researchers who might be interested in helping. This paper introduces a privacy-preserving platform by which manufacturers may safely share their data with researchers through secure methods, so that those researchers then create innovative tools to solve the manufacturers' real-world problems, and then provide tools that execute solutions back onto the platform for others to use with privacy and confidentiality guarantees. We illustrate this problem through a particular use case which addresses an important problem in the large-scale manufacturing of food crystals, which is that quality control relies on image analysis tools. Previous to our research, food crystals in the images were manually counted, which required substantial and time-consuming human efforts, but we have developed and deployed a crystal analysis tool which makes this process both more rapid and accurate. The tool enables automatic characterization of the crystal size distribution and numbers from microscope images while the natural imperfections from the sample preparation are automatically removed; a machine learning model to count high resolution translucent crystals and agglomeration of crystals was also developed to aid in these efforts. The resulting algorithm was then packaged for real-world use on the factory floor via a web-based app secured through the originating privacy-preserving platform, allowing manufacturers to use it while keeping their proprietary data secure. After demonstrating this full process, future directions are also explored. 

**Abstract (ZH)**: 小型和中型制造商需要创新的数据工具，但由于竞争和隐私 concerns，他们通常不愿将其专有数据与可能愿意帮助他们解决问题的研究人员分享。本文介绍了一个隐私保护平台，通过安全方法使制造商可以安全地与研究人员共享其数据，从而让研究人员开发出创新工具来解决制造商的实际问题，并将执行解决方案的工具通过平台提供给他人使用，同时保证隐私和保密性。我们通过一个特定的用例来阐述这一问题，该用例针对大规模制造食品晶体中的一个重要问题，即质量控制依赖于图像分析工具。在我们之前的研究所之前，食品晶体在图像中的数量都是通过人工手动计数，这需要大量的时间和人力，但我们开发并部署了一个晶体分析工具，使这一过程既更快又更准确。该工具能够自动分析显微镜图像中的晶体尺寸分布和数量，并自动去除样本准备过程中自然产生的不完善；我们还开发了一个用于统计高分辨率透明晶体及其结晶聚集的机器学习模型，以辅助这些工作。所得到的算法随后通过起源于隐私保护平台的基于Web的应用程序在工厂现场进行打包，使制造商能够在保护其专有数据的同时使用它。在演示了整个过程后，还探讨了未来的发展方向。 

---
# BranchNet: A Neuro-Symbolic Learning Framework for Structured Multi-Class Classification 

**Title (ZH)**: BranchNet: 一种用于结构化多类别分类的神经符号学习框架 

**Authors**: Dalia Rodríguez-Salas, Christian Riess  

**Link**: [PDF](https://arxiv.org/pdf/2507.01781)  

**Abstract**: We introduce BranchNet, a neuro-symbolic learning framework that transforms decision tree ensembles into sparse, partially connected neural networks. Each branch, defined as a decision path from root to a parent of leaves, is mapped to a hidden neuron, preserving symbolic structure while enabling gradient-based optimization. The resulting models are compact, interpretable, and require no manual architecture tuning. Evaluated on a suite of structured multi-class classification benchmarks, BranchNet consistently outperforms XGBoost in accuracy, with statistically significant gains. We detail the architecture, training procedure, and sparsity dynamics, and discuss the model's strengths in symbolic interpretability as well as its current limitations, particularly on binary tasks where further adaptive calibration may be beneficial. 

**Abstract (ZH)**: BranchNet：一种将决策树ensemble转换为稀疏部分连接神经网络的神经符号学习框架 

---
# GPU-based complete search for nonlinear minimization subject to bounds 

**Title (ZH)**: 基于GPU的区间约束非线性最小化完全搜索 

**Authors**: Guanglu Zhang, Qihang Shan, Jonathan Cagan  

**Link**: [PDF](https://arxiv.org/pdf/2507.01770)  

**Abstract**: This paper introduces a GPU-based complete search method to enclose the global minimum of a nonlinear function subject to simple bounds on the variables. Using interval analysis, coupled with the computational power and architecture of GPU, the method iteratively rules out the regions in the search domain where the global minimum cannot exist and leaves a finite set of regions where the global minimum must exist. For effectiveness, because of the rigor of interval analysis, the method is guaranteed to enclose the global minimum of the nonlinear function even in the presence of rounding errors. For efficiency, the method employs a novel GPU-based single program, single data parallel programming style to circumvent major GPU performance bottlenecks, and a variable cycling technique is also integrated into the method to reduce computational cost when minimizing large-scale nonlinear functions. The method is validated by minimizing 10 multimodal benchmark test functions with scalable dimensions, including the well-known Ackley function, Griewank function, Levy function, and Rastrigin function. These benchmark test functions represent grand challenges of global optimization, and enclosing the guaranteed global minimum of these benchmark test functions with more than 80 dimensions has not been reported in the literature. Our method completely searches the feasible domain and successfully encloses the guaranteed global minimum of these 10 benchmark test functions with up to 10,000 dimensions using only one GPU in a reasonable computation time, far exceeding the reported results in the literature due to the unique method design and implementation based on GPU architecture. 

**Abstract (ZH)**: 基于GPU的区间分析全面搜索方法用于在变量简单约束下包围非线性函数的全局最小值 

---
# Enhanced Generative Model Evaluation with Clipped Density and Coverage 

**Title (ZH)**: 增强生成模型评估：裁剪密度与覆盖范围方法 

**Authors**: Nicolas Salvy, Hugues Talbot, Bertrand Thirion  

**Link**: [PDF](https://arxiv.org/pdf/2507.01761)  

**Abstract**: Although generative models have made remarkable progress in recent years, their use in critical applications has been hindered by their incapacity to reliably evaluate sample quality. Quality refers to at least two complementary concepts: fidelity and coverage. Current quality metrics often lack reliable, interpretable values due to an absence of calibration or insufficient robustness to outliers. To address these shortcomings, we introduce two novel metrics, Clipped Density and Clipped Coverage. By clipping individual sample contributions and, for fidelity, the radii of nearest neighbor balls, our metrics prevent out-of-distribution samples from biasing the aggregated values. Through analytical and empirical calibration, these metrics exhibit linear score degradation as the proportion of poor samples increases. Thus, they can be straightforwardly interpreted as equivalent proportions of good samples. Extensive experiments on synthetic and real-world datasets demonstrate that Clipped Density and Clipped Coverage outperform existing methods in terms of robustness, sensitivity, and interpretability for evaluating generative models. 

**Abstract (ZH)**: 尽管生成模型在近年来取得了显著进步，但在关键应用中的使用受到了其无法可靠评估样本质量的限制。质量至少包括两个互补的概念：保真度和覆盖度。当前的质量度量往往由于缺乏校准或对离群值的鲁棒性不足而缺乏可靠的、可解释的值。为了应对这些缺陷，我们引入了两个新的度量标准：剪裁密度和剪裁覆盖度。通过剪裁个体样本贡献和，对于保真度，最近邻球体的半径，这些度量标准可以防止分布外样本偏倚汇总值。通过分析和经验校准，这些度量标准显示出线性得分下降，当不良样本的比例增加时。因此，它们可以简单地解释为良好样本的比例。在合成和实际数据集上的广泛实验表明，剪裁密度和剪裁覆盖度在鲁棒性、敏感性和可解释性方面优于现有方法，用于评估生成模型。 

---
# Relational Causal Discovery with Latent Confounders 

**Title (ZH)**: 潜混杂因素下的关系因果发现 

**Authors**: Andrea Piras, Matteo Negro, Ragib Ahsan, David Arbour, Elena Zheleva  

**Link**: [PDF](https://arxiv.org/pdf/2507.01700)  

**Abstract**: Estimating causal effects from real-world relational data can be challenging when the underlying causal model and potential confounders are unknown. While several causal discovery algorithms exist for learning causal models with latent confounders from data, they assume that the data is independent and identically distributed (i.i.d.) and are not well-suited for learning from relational data. Similarly, existing relational causal discovery algorithms assume causal sufficiency, which is unrealistic for many real-world datasets. To address this gap, we propose RelFCI, a sound and complete causal discovery algorithm for relational data with latent confounders. Our work builds upon the Fast Causal Inference (FCI) and Relational Causal Discovery (RCD) algorithms and it defines new graphical models, necessary to support causal discovery in relational domains. We also establish soundness and completeness guarantees for relational d-separation with latent confounders. We present experimental results demonstrating the effectiveness of RelFCI in identifying the correct causal structure in relational causal models with latent confounders. 

**Abstract (ZH)**: 基于潜在混杂因素的关联数据因果发现：RelFCI算法 

---
# Deep Recommender Models Inference: Automatic Asymmetric Data Flow Optimization 

**Title (ZH)**: 深度推荐模型推理：自动非对称数据流优化 

**Authors**: Giuseppe Ruggeri, Renzo Andri, Daniele Jahier Pagliari, Lukas Cavigelli  

**Link**: [PDF](https://arxiv.org/pdf/2507.01676)  

**Abstract**: Deep Recommender Models (DLRMs) inference is a fundamental AI workload accounting for more than 79% of the total AI workload in Meta's data centers. DLRMs' performance bottleneck is found in the embedding layers, which perform many random memory accesses to retrieve small embedding vectors from tables of various sizes. We propose the design of tailored data flows to speedup embedding look-ups. Namely, we propose four strategies to look up an embedding table effectively on one core, and a framework to automatically map the tables asymmetrically to the multiple cores of a SoC. We assess the effectiveness of our method using the Huawei Ascend AI accelerators, comparing it with the default Ascend compiler, and we perform high-level comparisons with Nvidia A100. Results show a speed-up varying from 1.5x up to 6.5x for real workload distributions, and more than 20x for extremely unbalanced distributions. Furthermore, the method proves to be much more independent of the query distribution than the baseline. 

**Abstract (ZH)**: DLRMs嵌入查找加速设计：在Meta数据中心占总AI工作负载超过79%的DLRMs推理中，性能瓶颈在于嵌入层进行的许多随机内存访问以检索来自不同大小表中的小嵌入向量。我们提出了定制数据流设计以加速嵌入查找。具体地，我们提出了四种策略在单个内核上有效地查找嵌入表，并提出了一种框架以自动将表非对称地映射到SoC的多个内核。我们使用华为Ascend AI加速器评估了该方法的有效性，将其与默认Ascend编译器进行比较，并与Nvidia A100进行了高级比较。结果显示，在真实工作负载分布下加速比从1.5倍到6.5倍不等，而在极端不平衡分布下超过20倍。此外，该方法在查询分布方面的独立性远高于基线。 

---
# Comparing Optimization Algorithms Through the Lens of Search Behavior Analysis 

**Title (ZH)**: 通过搜索行为分析比较优化算法 

**Authors**: Gjorgjina Cenikj, Gašper Petelin, Tome Eftimov  

**Link**: [PDF](https://arxiv.org/pdf/2507.01668)  

**Abstract**: The field of numerical optimization has recently seen a surge in the development of "novel" metaheuristic algorithms, inspired by metaphors derived from natural or human-made processes, which have been widely criticized for obscuring meaningful innovations and failing to distinguish themselves from existing approaches. Aiming to address these concerns, we investigate the applicability of statistical tests for comparing algorithms based on their search behavior. We utilize the cross-match statistical test to compare multivariate distributions and assess the solutions produced by 114 algorithms from the MEALPY library. These findings are incorporated into an empirical analysis aiming to identify algorithms with similar search behaviors. 

**Abstract (ZH)**: 数值优化领域最近见证了“新颖”的元启发式算法的发展 surge，这些算法受到自然或人造过程的比喻启发，但常被批评模糊了有意义的创新并难以区分自己与现有方法。为应对这些担忧，我们研究了基于搜索行为比较算法的应用统计检验方法。我们使用交叉匹配统计检验比较多元分布，并评估MEALPY库中114个算法产生的解决方案。这些发现被纳入一项实证分析，旨在识别具有类似搜索行为的算法。 

---
# GradMetaNet: An Equivariant Architecture for Learning on Gradients 

**Title (ZH)**: GradMetaNet: 一种基于梯度的学习同胚架构 

**Authors**: Yoav Gelberg, Yam Eitan, Aviv Navon, Aviv Shamsian, Theo, Putterman, Michael Bronstein, Haggai Maron  

**Link**: [PDF](https://arxiv.org/pdf/2507.01649)  

**Abstract**: Gradients of neural networks encode valuable information for optimization, editing, and analysis of models. Therefore, practitioners often treat gradients as inputs to task-specific algorithms, e.g. for pruning or optimization. Recent works explore learning algorithms that operate directly on gradients but use architectures that are not specifically designed for gradient processing, limiting their applicability. In this paper, we present a principled approach for designing architectures that process gradients. Our approach is guided by three principles: (1) equivariant design that preserves neuron permutation symmetries, (2) processing sets of gradients across multiple data points to capture curvature information, and (3) efficient gradient representation through rank-1 decomposition. Based on these principles, we introduce GradMetaNet, a novel architecture for learning on gradients, constructed from simple equivariant blocks. We prove universality results for GradMetaNet, and show that previous approaches cannot approximate natural gradient-based functions that GradMetaNet can. We then demonstrate GradMetaNet's effectiveness on a diverse set of gradient-based tasks on MLPs and transformers, such as learned optimization, INR editing, and estimating loss landscape curvature. 

**Abstract (ZH)**: 神经网络梯度中编码的信息用于模型优化、编辑和分析具有重要价值。因此，实践者常将梯度作为特定任务算法的输入，例如剪枝或优化。近期的研究探索直接在梯度上操作的学习算法，但这些算法的架构并未专门设计用于梯度处理，限制了其应用范围。在本文中，我们提出了一种严谨的方法来设计处理梯度的架构。我们的方法遵循三个原则：（1）保持神经元置换对称性的协变设计；（2）处理多个数据点上的梯度集以捕获曲率信息；（3）通过秩1分解高效表示梯度。基于这些原则，我们引入了GradMetaNet，这是一种用于在梯度上学习的新架构，由简单的协变块构建而成。我们证明了GradMetaNet的通用性结果，并展示了先前的方法无法逼近GradMetaNet可以逼近的自然梯度基函数。然后，我们通过在MLP和变换器上的多种梯度基任务中证明GradMetaNet的有效性，如学习优化、INR编辑和估计损失景观曲率。 

---
# Customized Exploration of Landscape Features Driving Multi-Objective Combinatorial Optimization Performance 

**Title (ZH)**: 定制化探索驱动多目标组合优化性能的景观特征分析 

**Authors**: Ana Nikolikj, Gabriela Ochoa, Tome Eftimov  

**Link**: [PDF](https://arxiv.org/pdf/2507.01638)  

**Abstract**: We present an analysis of landscape features for predicting the performance of multi-objective combinatorial optimization algorithms. We consider features from the recently proposed compressed Pareto Local Optimal Solutions Networks (C-PLOS-net) model of combinatorial landscapes. The benchmark instances are a set of rmnk-landscapes with 2 and 3 objectives and various levels of ruggedness and objective correlation. We consider the performance of three algorithms -- Pareto Local Search (PLS), Global Simple EMO Optimizer (GSEMO), and Non-dominated Sorting Genetic Algorithm (NSGA-II) - using the resolution and hypervolume metrics. Our tailored analysis reveals feature combinations that influence algorithm performance specific to certain landscapes. This study provides deeper insights into feature importance, tailored to specific rmnk-landscapes and algorithms. 

**Abstract (ZH)**: 我们提出了一种关于 landscapes 特征在预测多目标组合优化算法性能方面的分析。我们考虑了最近提出的压缩帕雷托局部最优解网络（C-PLOS-net）模型中的组合 landscapes 特征。基准实例为包含 2 和 3 个目标以及不同复杂度和目标相关性的 rmnk-landscapes。我们使用分辨率和 hypervolume 指标分析了 Pareto 局部搜索（PLS）、全局简单 EMO 优化器（GSEMO）和非支配排序遗传算法（NSGA-II）的性能。我们的定制分析揭示了特定 landscapes 下影响算法性能的特征组合。本研究提供了对特征重要性更深入的理解，针对特定的 rmnk-landscapes 和算法进行了定制。 

---
# Enhanced Influence-aware Group Recommendation for Online Media Propagation 

**Title (ZH)**: 增强影响力感知的群体推荐以优化在线媒体传播 

**Authors**: Chengkun He, Xiangmin Zhou, Chen Wang, Longbing Cao, Jie Shao, Xiaodong Li, Guang Xu, Carrie Jinqiu Hu, Zahir Tari  

**Link**: [PDF](https://arxiv.org/pdf/2507.01616)  

**Abstract**: Group recommendation over social media streams has attracted significant attention due to its wide applications in domains such as e-commerce, entertainment, and online news broadcasting. By leveraging social connections and group behaviours, group recommendation (GR) aims to provide more accurate and engaging content to a set of users rather than individuals. Recently, influence-aware GR has emerged as a promising direction, as it considers the impact of social influence on group decision-making. In earlier work, we proposed Influence-aware Group Recommendation (IGR) to solve this task. However, this task remains challenging due to three key factors: the large and ever-growing scale of social graphs, the inherently dynamic nature of influence propagation within user groups, and the high computational overhead of real-time group-item matching.
To tackle these issues, we propose an Enhanced Influence-aware Group Recommendation (EIGR) framework. First, we introduce a Graph Extraction-based Sampling (GES) strategy to minimise redundancy across multiple temporal social graphs and effectively capture the evolving dynamics of both groups and items. Second, we design a novel DYnamic Independent Cascade (DYIC) model to predict how influence propagates over time across social items and user groups. Finally, we develop a two-level hash-based User Group Index (UG-Index) to efficiently organise user groups and enable real-time recommendation generation. Extensive experiments on real-world datasets demonstrate that our proposed framework, EIGR, consistently outperforms state-of-the-art baselines in both effectiveness and efficiency. 

**Abstract (ZH)**: 社交媒体流上的群体推荐因在电子商务、娱乐和在线新闻广播等领域广泛的应用而引起了广泛关注。基于社交联系和群体行为，群体推荐（GR）旨在为一组用户提供更准确和吸引人的内容，而非单个用户。最近，考虑到社会影响力对群体决策的影响，影响力意识下的群体推荐成为了一个有前景的方向。在早期的工作中，我们提出了影响力意识下的群体推荐（IGR）来解决这一问题。然而，由于三个关键因素的影响——社会图形的庞大且不断增长的规模、影响力在用户群体中的固有动态传播特性以及实时群体-项目匹配的高计算开销——这一任务仍然具有挑战性。

为了应对这些挑战，我们提出了一种增强的影响力意识下的群体推荐（EIGR）框架。首先，我们引入了一种基于图提取的采样（GES）策略，以最大程度地减少多个时间社交图的冗余，并有效地捕捉群体和项目随时间的演变动态。其次，我们设计了一种新颖的动力独立级联（DYIC）模型，用于预测社会项目和用户群体中影响力随时间的传播情况。最后，我们开发了一种基于两级哈希的用户群体索引（UG-Index），以高效组织用户群体并实现实时推荐生成。在真实世界数据集上的广泛实验表明，我们提出的方法EIGR在效果和效率方面均优于最先进的基线方法。 

---
# Survivability of Backdoor Attacks on Unconstrained Face Recognition Systems 

**Title (ZH)**: 不受约束条件下的后门攻击在人脸识别系统中的生存能力 

**Authors**: Quentin Le Roux, Yannick Teglia, Teddy Furon, Philippe Loubet-Moundi, Eric Bourbao  

**Link**: [PDF](https://arxiv.org/pdf/2507.01607)  

**Abstract**: The widespread use of deep learning face recognition raises several security concerns. Although prior works point at existing vulnerabilities, DNN backdoor attacks against real-life, unconstrained systems dealing with images captured in the wild remain a blind spot of the literature. This paper conducts the first system-level study of backdoors in deep learning-based face recognition systems. This paper yields four contributions by exploring the feasibility of DNN backdoors on these pipelines in a holistic fashion. We demonstrate for the first time two backdoor attacks on the face detection task: face generation and face landmark shift attacks. We then show that face feature extractors trained with large margin losses also fall victim to backdoor attacks. Combining our models, we then show using 20 possible pipeline configurations and 15 attack cases that a single backdoor enables an attacker to bypass the entire function of a system. Finally, we provide stakeholders with several best practices and countermeasures. 

**Abstract (ZH)**: 深度学习面部识别的广泛应用引发了若干安全关切。尽管先前工作指出了现有漏洞，但针对真实生活中未加约束系统的图像捕获进行的DNN后门攻击仍属于文献盲区。本文首次在系统层面研究了基于深度学习的面部识别系统的后门攻击。通过全面探索这些管道中DNN后门的可行性，本文提出了四项贡献。我们首次展示了两种面部检测任务中的后门攻击：面部生成攻击和面部关键点偏移攻击。随后，我们证明了使用大 margin 损失训练的面部特征提取器也受到后门攻击的影响。结合我们的模型，我们展示了在20种可能的管道配置和15种攻击情况下，单一后门能够使攻击者绕过整个系统的功能。最后，我们为相关方提供了若干最佳实践和对策。 

---
# Exploring Classical Piano Performance Generation with Expressive Music Variational AutoEncoder 

**Title (ZH)**: 探究具表现力的音乐变分自动编码器在古典钢琴表演生成中的应用 

**Authors**: Jing Luo, Xinyu Yang, Jie Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.01582)  

**Abstract**: The creativity of classical music arises not only from composers who craft the musical sheets but also from performers who interpret the static notations with expressive nuances. This paper addresses the challenge of generating classical piano performances from scratch, aiming to emulate the dual roles of composer and pianist in the creative process. We introduce the Expressive Compound Word (ECP) representation, which effectively captures both the metrical structure and expressive nuances of classical performances. Building on this, we propose the Expressive Music Variational AutoEncoder (XMVAE), a model featuring two branches: a Vector Quantized Variational AutoEncoder (VQ-VAE) branch that generates score-related content, representing the Composer, and a vanilla VAE branch that produces expressive details, fulfilling the role of Pianist. These branches are jointly trained with similar Seq2Seq architectures, leveraging a multiscale encoder to capture beat-level contextual information and an orthogonal Transformer decoder for efficient compound tokens decoding. Both objective and subjective evaluations demonstrate that XMVAE generates classical performances with superior musical quality compared to state-of-the-art models. Furthermore, pretraining the Composer branch on extra musical score datasets contribute to a significant performance gain. 

**Abstract (ZH)**: 古典音乐的创造力不仅来源于创作乐谱的作曲家，还来源于以富有表达性的微妙细节诠释静态符号的表演者。本文旨在从头生成古典钢琴演奏，以模拟作曲家和演奏者在创作过程中的双重角色。我们引入了表达性复合词（ECP）表示，有效地捕捉了古典演奏的节奏结构和表达性微妙之处。在此基础上，我们提出了一种模型——表达性音乐变分自编码器（XMVAE），该模型具有两个分支：一个向量量化变分自编码器（VQ-VAE）分支用于生成与乐谱相关的内容，代表作曲家；另一个基本的VAE分支用于生成表达性细节，履行演奏者角色。这两个分支通过类似的Seq2Seq架构并利用多尺度编码器捕获节拍级别的上下文信息以及正交Transformer解码器高效地解码复合标记进行联合训练。客观和主观评估均证明，与现有模型相比，XMVAE生成的古典演奏具有更高的音乐质量。此外，对额外的音乐乐谱数据集进行预训练，可显著提升模型性能。 

---
# Real-Time Emergency Vehicle Siren Detection with Efficient CNNs on Embedded Hardware 

**Title (ZH)**: 基于高效CNN在嵌入式硬件上的实时紧急车辆警报检测 

**Authors**: Marco Giordano, Stefano Giacomelli, Claudia Rinaldi, Fabio Graziosi  

**Link**: [PDF](https://arxiv.org/pdf/2507.01563)  

**Abstract**: We present a full-stack emergency vehicle (EV) siren detection system designed for real-time deployment on embedded hardware. The proposed approach is based on E2PANNs, a fine-tuned convolutional neural network derived from EPANNs, and optimized for binary sound event detection under urban acoustic conditions. A key contribution is the creation of curated and semantically structured datasets - AudioSet-EV, AudioSet-EV Augmented, and Unified-EV - developed using a custom AudioSet-Tools framework to overcome the low reliability of standard AudioSet annotations. The system is deployed on a Raspberry Pi 5 equipped with a high-fidelity DAC+microphone board, implementing a multithreaded inference engine with adaptive frame sizing, probability smoothing, and a decision-state machine to control false positive activations. A remote WebSocket interface provides real-time monitoring and facilitates live demonstration capabilities. Performance is evaluated using both framewise and event-based metrics across multiple configurations. Results show the system achieves low-latency detection with improved robustness under realistic audio conditions. This work demonstrates the feasibility of deploying IoS-compatible SED solutions that can form distributed acoustic monitoring networks, enabling collaborative emergency vehicle tracking across smart city infrastructures through WebSocket connectivity on low-cost edge devices. 

**Abstract (ZH)**: 一种基于E2PANNs的实际部署嵌入式硬件的全栈应急车辆警报检测系统 

---
# AI and Remote Sensing for Resilient and Sustainable Built Environments: A Review of Current Methods, Open Data and Future Directions 

**Title (ZH)**: AI和遥感在韧性可持续建成环境中的应用：现有方法、开放数据及未来方向 

**Authors**: Ubada El Joulani, Tatiana Kalganova, Stergios-Aristoteles Mitoulis, Sotirios Argyroudis  

**Link**: [PDF](https://arxiv.org/pdf/2507.01547)  

**Abstract**: Critical infrastructure, such as transport networks, underpins economic growth by enabling mobility and trade. However, ageing assets, climate change impacts (e.g., extreme weather, rising sea levels), and hybrid threats ranging from natural disasters to cyber attacks and conflicts pose growing risks to their resilience and functionality. This review paper explores how emerging digital technologies, specifically Artificial Intelligence (AI), can enhance damage assessment and monitoring of transport infrastructure. A systematic literature review examines existing AI models and datasets for assessing damage in roads, bridges, and other critical infrastructure impacted by natural disasters. Special focus is given to the unique challenges and opportunities associated with bridge damage detection due to their structural complexity and critical role in connectivity. The integration of SAR (Synthetic Aperture Radar) data with AI models is also discussed, with the review revealing a critical research gap: a scarcity of studies applying AI models to SAR data for comprehensive bridge damage assessment. Therefore, this review aims to identify the research gaps and provide foundations for AI-driven solutions for assessing and monitoring critical transport infrastructures. 

**Abstract (ZH)**: 新兴数字技术，特别是人工智能（AI），如何增强运输基础设施的损害评估与监测：以桥梁损害检测为例及合成孔径雷达（SAR）数据的整合探讨 

---
# Chargax: A JAX Accelerated EV Charging Simulator 

**Title (ZH)**: Chargax: 一个使用JAX加速的电动汽车充电模拟器 

**Authors**: Koen Ponse, Jan Felix Kleuker, Aske Plaat, Thomas Moerland  

**Link**: [PDF](https://arxiv.org/pdf/2507.01522)  

**Abstract**: Deep Reinforcement Learning can play a key role in addressing sustainable energy challenges. For instance, many grid systems are heavily congested, highlighting the urgent need to enhance operational efficiency. However, reinforcement learning approaches have traditionally been slow due to the high sample complexity and expensive simulation requirements. While recent works have effectively used GPUs to accelerate data generation by converting environments to JAX, these works have largely focussed on classical toy problems. This paper introduces Chargax, a JAX-based environment for realistic simulation of electric vehicle charging stations designed for accelerated training of RL agents. We validate our environment in a variety of scenarios based on real data, comparing reinforcement learning agents against baselines. Chargax delivers substantial computational performance improvements of over 100x-1000x over existing environments. Additionally, Chargax' modular architecture enables the representation of diverse real-world charging station configurations. 

**Abstract (ZH)**: 深度强化学习在应对可持续能源挑战中可以发挥关键作用。例如，许多电网系统严重拥堵，突显了提升运行效率的迫切需求。然而，传统的强化学习方法由于较高的样本复杂度和昂贵的仿真要求而动作缓慢。尽管最近的工作通过将环境转换为JAX并利用GPU有效加速了数据生成，但这些工作主要集中在经典的玩具问题上。本文介绍了Chargax，这是一种基于JAX的环境，用于现实模拟电动汽车充电站，旨在加速RL代理的训练。我们在基于真实数据的多种场景中验证了我们的环境，并将强化学习代理与基线进行了比较。Chargax在现有环境中提供了超过100-1000倍的计算性能改进。此外，Chargax的模块化架构能够表示多种多样的实际充电站配置。 

---
# Epistemic Scarcity: The Economics of Unresolvable Unknowns 

**Title (ZH)**: 知识稀缺：不可解未知的经济学 

**Authors**: Craig S Wright  

**Link**: [PDF](https://arxiv.org/pdf/2507.01483)  

**Abstract**: This paper presents a praxeological analysis of artificial intelligence and algorithmic governance, challenging assumptions about the capacity of machine systems to sustain economic and epistemic order. Drawing on Misesian a priori reasoning and Austrian theories of entrepreneurship, we argue that AI systems are incapable of performing the core functions of economic coordination: interpreting ends, discovering means, and communicating subjective value through prices. Where neoclassical and behavioural models treat decisions as optimisation under constraint, we frame them as purposive actions under uncertainty.
We critique dominant ethical AI frameworks such as Fairness, Accountability, and Transparency (FAT) as extensions of constructivist rationalism, which conflict with a liberal order grounded in voluntary action and property rights. Attempts to encode moral reasoning in algorithms reflect a misunderstanding of ethics and economics. However complex, AI systems cannot originate norms, interpret institutions, or bear responsibility. They remain opaque, misaligned, and inert.
Using the concept of epistemic scarcity, we explore how information abundance degrades truth discernment, enabling both entrepreneurial insight and soft totalitarianism. Our analysis ends with a civilisational claim: the debate over AI concerns the future of human autonomy, institutional evolution, and reasoned choice. The Austrian tradition, focused on action, subjectivity, and spontaneous order, offers the only coherent alternative to rising computational social control. 

**Abstract (ZH)**: 人工智能与算法治理的实践分析：挑战机器系统维持经济与知识秩序的能力 

---
# Zero-Incentive Dynamics: a look at reward sparsity through the lens of unrewarded subgoals 

**Title (ZH)**: 零激励动力学：通过未奖励子目标视角考察奖励稀疏性 

**Authors**: Yannick Molinghen, Tom Lenaerts  

**Link**: [PDF](https://arxiv.org/pdf/2507.01470)  

**Abstract**: This work re-examines the commonly held assumption that the frequency of rewards is a reliable measure of task difficulty in reinforcement learning. We identify and formalize a structural challenge that undermines the effectiveness of current policy learning methods: when essential subgoals do not directly yield rewards. We characterize such settings as exhibiting zero-incentive dynamics, where transitions critical to success remain unrewarded. We show that state-of-the-art deep subgoal-based algorithms fail to leverage these dynamics and that learning performance is highly sensitive to the temporal proximity between subgoal completion and eventual reward. These findings reveal a fundamental limitation in current approaches and point to the need for mechanisms that can infer latent task structure without relying on immediate incentives. 

**Abstract (ZH)**: 本研究重新审视了强化学习中奖励频率是任务难度可靠指标的普遍假设，指出并正式化了一个结构上的挑战，该挑战削弱了当前策略学习方法的有效性：当关键子目标未直接产生奖励时。我们把这种设置描述为零激励动力学，其中对成功至关重要的状态转换未获奖励。我们证明，最先进的基于子目标的深度算法无法利用这些动力学，且学习性能高度依赖于子目标完成与最终奖励之间的时间接近程度。这些发现揭示了当前方法的根本局限性，并指出了需要能够不依赖即时激励来推断潜在任务结构的机制。 

---
# Tensor Program Optimization for the RISC-V Vector Extension Using Probabilistic Programs 

**Title (ZH)**: 基于概率程序的RISC-V向量扩展张量程序优化 

**Authors**: Federico Nicolas Peccia, Frederik Haxel, Oliver Bringmann  

**Link**: [PDF](https://arxiv.org/pdf/2507.01457)  

**Abstract**: RISC-V provides a flexible and scalable platform for applications ranging from embedded devices to high-performance computing clusters. Particularly, its RISC-V Vector Extension (RVV) becomes of interest for the acceleration of AI workloads. But writing software that efficiently utilizes the vector units of RISC-V CPUs without expert knowledge requires the programmer to rely on the autovectorization features of compilers or hand-crafted libraries like muRISCV-NN. Smarter approaches, like autotuning frameworks, have been missing the integration with the RISC-V RVV extension, thus heavily limiting the efficient deployment of complex AI workloads. In this paper, we present a workflow based on the TVM compiler to efficiently map AI workloads onto RISC-V vector units. Instead of relying on hand-crafted libraries, we integrated the RVV extension into TVM's MetaSchedule framework, a probabilistic program framework for tensor operation tuning. We implemented different RISC-V SoCs on an FPGA and tuned a wide range of AI workloads on them. We found that our proposal shows a mean improvement of 46% in execution latency when compared against the autovectorization feature of GCC, and 29% against muRISCV-NN. Moreover, the binary resulting from our proposal has a smaller code memory footprint, making it more suitable for embedded devices. Finally, we also evaluated our solution on a commercially available RISC-V SoC implementing the RVV 1.0 Vector Extension and found our solution is able to find mappings that are 35% faster on average than the ones proposed by LLVM. We open-sourced our proposal for the community to expand it to target other RISC-V extensions. 

**Abstract (ZH)**: RISC-V为从嵌入式设备到高性能计算集群的应用提供了一个灵活且可扩展的平台。特别是，其RISC-V向量扩展（RVV）引起了对加速AI工作负载的兴趣。但在没有专家知识的情况下，编写能够高效利用RISC-V CPU向量单元的软件需要依赖编译器的自动向量化功能或像muRISCV-NN这样的手工构建库。现有的更智能的方法，如自动调优框架，尚未将RISC-V RVV扩展集成进来，从而极大地限制了复杂AI工作负载的高效部署。在本文中，我们基于TVM编译器提出了一种工作流，以高效地将AI工作负载映射到RISC-V向量单元上。我们没有依赖手工构建的库，而是将RVV扩展集成到TVM的MetaSchedule框架中，这是一个用于张量操作调优的概率程序框架。我们使用FPGA实现了不同的RISC-V SOC，并在它们上调优了广泛的AI工作负载。我们发现，与GCC的自动向量化功能相比，我们的提案在执行延迟上平均改进了46%，与muRISCV-NN相比，平均改进了29%。此外，由我们提案生成的二进制文件具有较小的代码内存占用，使其更适合嵌入式设备。最后，我们还在一个采用RVV 1.0向量扩展的商用RISC-V SOC上评估了我们的解决方案，发现我们的解决方案在平均速度上比LLVM提出的方法快35%。我们将我们的提案开源给社区，以便将其扩展为目标其他RISC-V扩展。 

---
# Hardware-software co-exploration with racetrack memory based in-memory computing for CNN inference in embedded systems 

**Title (ZH)**: 基于赛马存储器的存内计算在嵌入式系统中实现CNN推理的硬件-软件协同探索 

**Authors**: Benjamin Chen Ming Choong, Tao Luo, Cheng Liu, Bingsheng He, Wei Zhang, Joey Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.01429)  

**Abstract**: Deep neural networks generate and process large volumes of data, posing challenges for low-resource embedded systems. In-memory computing has been demonstrated as an efficient computing infrastructure and shows promise for embedded AI applications. Among newly-researched memory technologies, racetrack memory is a non-volatile technology that allows high data density fabrication, making it a good fit for in-memory computing. However, integrating in-memory arithmetic circuits with memory cells affects both the memory density and power efficiency. It remains challenging to build efficient in-memory arithmetic circuits on racetrack memory within area and energy constraints. To this end, we present an efficient in-memory convolutional neural network (CNN) accelerator optimized for use with racetrack memory. We design a series of fundamental arithmetic circuits as in-memory computing cells suited for multiply-and-accumulate operations. Moreover, we explore the design space of racetrack memory based systems and CNN model architectures, employing co-design to improve the efficiency and performance of performing CNN inference in racetrack memory while maintaining model accuracy. Our designed circuits and model-system co-optimization strategies achieve a small memory bank area with significant improvements in energy and performance for racetrack memory based embedded systems. 

**Abstract (ZH)**: 基于赛道存储器的高效卷积神经网络加速器设计 

---
# Penalizing Transparency? How AI Disclosure and Author Demographics Shape Human and AI Judgments About Writing 

**Title (ZH)**: 惩罚透明性？AI披露和作者 demographic 特征如何影响人类和AI对写作的判断 

**Authors**: Inyoung Cheong, Alicia Guo, Mina Lee, Zhehui Liao, Kowe Kadoma, Dongyoung Go, Joseph Chee Chang, Peter Henderson, Mor Naaman, Amy X. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.01418)  

**Abstract**: As AI integrates in various types of human writing, calls for transparency around AI assistance are growing. However, if transparency operates on uneven ground and certain identity groups bear a heavier cost for being honest, then the burden of openness becomes asymmetrical. This study investigates how AI disclosure statement affects perceptions of writing quality, and whether these effects vary by the author's race and gender. Through a large-scale controlled experiment, both human raters (n = 1,970) and LLM raters (n = 2,520) evaluated a single human-written news article while disclosure statements and author demographics were systematically varied. This approach reflects how both human and algorithmic decisions now influence access to opportunities (e.g., hiring, promotion) and social recognition (e.g., content recommendation algorithms). We find that both human and LLM raters consistently penalize disclosed AI use. However, only LLM raters exhibit demographic interaction effects: they favor articles attributed to women or Black authors when no disclosure is present. But these advantages disappear when AI assistance is revealed. These findings illuminate the complex relationships between AI disclosure and author identity, highlighting disparities between machine and human evaluation patterns. 

**Abstract (ZH)**: 随着AI融入各种类型的人类写作，对AI辅助的透明度要求越来越高。然而，如果透明度建立在不平等的基础上，某些身份群体因诚实而承担更大的成本，那么开放的负担就会变得不对称。本研究探讨了AI披露声明如何影响写作质量的感知，以及这些影响是否因作者的种族和性别而异。通过大规模控制实验，分别有1,970名人类评分者和2,520名LLM评分者评估了一篇单一的人类撰写的新闻文章，同时系统地变化了披露声明和作者的人口统计信息。这种方法反映了现在人类和算法决策如何影响机会获取（例如，招聘、晋升）和社会认可（例如，内容推荐算法）。我们发现，无论是人类评分者还是LLM评分者，都一致惩罚披露的AI使用。然而，只有LLM评分者显示出人口统计交互效应：当没有披露时，他们更倾向于将文章归因于女性或非洲裔作者。但当AI援助被揭示时，这些优势消失。这些发现揭示了AI披露与作者身份之间复杂的关系，突显了机器和人类评估模式之间的差异。 

---
# Age Sensitive Hippocampal Functional Connectivity: New Insights from 3D CNNs and Saliency Mapping 

**Title (ZH)**: 年龄相关海马功能连接性：来自3D CNNs和注意映射的新见解 

**Authors**: Yifei Sun, Marshall A. Dalton, Robert D. Sanders, Yixuan Yuan, Xiang Li, Sharon L. Naismith, Fernando Calamante, Jinglei Lv  

**Link**: [PDF](https://arxiv.org/pdf/2507.01411)  

**Abstract**: Grey matter loss in the hippocampus is a hallmark of neurobiological aging, yet understanding the corresponding changes in its functional connectivity remains limited. Seed-based functional connectivity (FC) analysis enables voxel-wise mapping of the hippocampus's synchronous activity with cortical regions, offering a window into functional reorganization during aging. In this study, we develop an interpretable deep learning framework to predict brain age from hippocampal FC using a three-dimensional convolutional neural network (3D CNN) combined with LayerCAM saliency mapping. This approach maps key hippocampal-cortical connections, particularly with the precuneus, cuneus, posterior cingulate cortex, parahippocampal cortex, left superior parietal lobule, and right superior temporal sulcus, that are highly sensitive to age. Critically, disaggregating anterior and posterior hippocampal FC reveals distinct mapping aligned with their known functional specializations. These findings provide new insights into the functional mechanisms of hippocampal aging and demonstrate the power of explainable deep learning to uncover biologically meaningful patterns in neuroimaging data. 

**Abstract (ZH)**: 海马区灰质损失是神经生物学老化的一个特征，但对其功能连接相应变化的理解仍然有限。基于种子的功能连接分析可以以体素级方式映射海马与皮层区域的同步活动，为理解老化期间的功能重组提供了窗口。在本研究中，我们开发了一个可解释的深度学习框架，使用三维卷积神经网络（3D CNN）结合LayerCAM显著性映射来预测大脑年龄，并揭示了特别易受年龄影响的关键海马-皮层连接，特别是与楔前叶、楔叶、后扣带回、旁海马回、左侧顶叶上回和右侧颞上沟的连接。更重要的是，将前部和后部海马的功能连接分离后，发现了与它们已知的功能特化相一致的分离映射。这些发现为理解海马老化的过程提供了新的见解，并展示了可解释深度学习在揭示神经影像数据中的生物意义模式方面的强大能力。 

---
# Medical-Knowledge Driven Multiple Instance Learning for Classifying Severe Abdominal Anomalies on Prenatal Ultrasound 

**Title (ZH)**: 基于医学知识的多项实例学习在产前超声分类严重腹部异常中的应用 

**Authors**: Huanwen Liang, Jingxian Xu, Yuanji Zhang, Yuhao Huang, Yuhan Zhang, Xin Yang, Ran Li, Xuedong Deng, Yanjun Liu, Guowei Tao, Yun Wu, Sheng Zhao, Xinru Gao, Dong Ni  

**Link**: [PDF](https://arxiv.org/pdf/2507.01401)  

**Abstract**: Fetal abdominal malformations are serious congenital anomalies that require accurate diagnosis to guide pregnancy management and reduce mortality. Although AI has demonstrated significant potential in medical diagnosis, its application to prenatal abdominal anomalies remains limited. Most existing studies focus on image-level classification and rely on standard plane localization, placing less emphasis on case-level diagnosis. In this paper, we develop a case-level multiple instance learning (MIL)-based method, free of standard plane localization, for classifying fetal abdominal anomalies in prenatal ultrasound. Our contribution is three-fold. First, we adopt a mixture-of-attention-experts module (MoAE) to weight different attention heads for various planes. Secondly, we propose a medical-knowledge-driven feature selection module (MFS) to align image features with medical knowledge, performing self-supervised image token selection at the case-level. Finally, we propose a prompt-based prototype learning (PPL) to enhance the MFS. Extensively validated on a large prenatal abdominal ultrasound dataset containing 2,419 cases, with a total of 24,748 images and 6 categories, our proposed method outperforms the state-of-the-art competitors. Codes are available at:this https URL. 

**Abstract (ZH)**: 胎儿腹部畸形是严重先天性疾病，需要准确诊断以指导妊娠管理并降低死亡率。虽然AI在医学诊断中显示出巨大的潜力，但其在产前腹部畸形的应用仍受到限制。现有大多数研究集中于图像级分类，并依赖于标准切面定位，较少关注案例级诊断。本文提出了一种无需标准切面定位的案例级多实例学习（MIL）方法，用于产前超声分类胎儿腹部畸形。我们的贡献包括三个方面：首先，采用混合注意力专家模块（MoAE）为不同切面加权不同的注意力头；其次，提出一种医学知识驱动的特征选择模块（MFS），在案例级进行自监督图像标记选择，使图像特征与医学知识相一致；最后，提出基于提示的原型学习（PPL）以增强MFS。在包含2,419个案例、总计24,748张图像和6个类别的大型产前腹部超声数据集上进行了广泛验证，我们的方法在性能上优于最新竞争对手。代码可访问：this https URL。 

---
# Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy 

**Title (ZH)**: Skywork-Reward-V2: 通过人机协同扩展偏好数据 curated 规模 

**Authors**: Chris Yuhao Liu, Liang Zeng, Yuzhen Xiao, Jujie He, Jiacai Liu, Chaojie Wang, Rui Yan, Wei Shen, Fuxiang Zhang, Jiacheng Xu, Yang Liu, Yahui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.01352)  

**Abstract**: Despite the critical role of reward models (RMs) in reinforcement learning from human feedback (RLHF), current state-of-the-art open RMs perform poorly on most existing evaluation benchmarks, failing to capture the spectrum of nuanced and sophisticated human preferences. Even approaches that incorporate advanced training techniques have not yielded meaningful performance improvements. We hypothesize that this brittleness stems primarily from limitations in preference datasets, which are often narrowly scoped, synthetically labeled, or lack rigorous quality control. To address these challenges, we present a large-scale preference dataset comprising 40 million preference pairs, named SynPref-40M. To enable data curation at scale, we design a human-AI synergistic two-stage pipeline that leverages the complementary strengths of human annotation quality and AI scalability. In this pipeline, humans provide verified annotations, while large language models perform automatic curation based on human guidance. Training on this preference mixture, we introduce Skywork-Reward-V2, a suite of eight reward models ranging from 0.6B to 8B parameters, trained on a carefully curated subset of 26 million preference pairs from SynPref-40M. We demonstrate that Skywork-Reward-V2 is versatile across a wide range of capabilities, including alignment with human preferences, objective correctness, safety, resistance to stylistic biases, and best-of-N scaling, achieving state-of-the-art performance across seven major reward model benchmarks. Ablation studies confirm that the effectiveness of our approach stems not only from data scale but also from high-quality curation. The Skywork-Reward-V2 series represents substantial progress in open reward models, highlighting the untapped potential of existing preference datasets and demonstrating how human-AI curation synergy can unlock significantly higher data quality. 

**Abstract (ZH)**: Despite the Critical Role of Reward Models in Reinforcement Learning from Human Feedback, Current State-of-the-Art Open RMs Perform Poorly on Most Existing Evaluation Benchmarks, Failing to Capture the Spectrum of Nuanced and Sophisticated Human Preferences. We Hypothesize That This Brittleness Stems Primarily from Limitations in Preference Datasets, Which Are Often Narrowly Scoped, Synthetically Labeled, or Lack Rigorous Quality Control. To Address These Challenges, We Present a Large-Scale Preference Dataset Comprising 40 Million Preference Pairs, Named SynPref-40M. To Enable Data Curation at Scale, We Design a Human-AI Synergistic Two-Stage Pipeline That Leverages the Complementary Strengths of Human Annotation Quality and AI Scalability. In This Pipeline, Humans Provide Verified Annotations, While Large Language Models Perform Automatic Curation Based on Human Guidance. Training on This Preference Mixture, We Introduce Skywork-Reward-V2, a Suite of Eight Reward Models Ranging from 0.6B to 8B Parameters, Trained on a Carefully Curated Subset of 26 Million Preference Pairs from SynPref-40M. We Demonstrate That Skywork-Reward-V2 Is Versatile across a Wide Range of Capabilities, Including Alignment with Human Preferences, Objective Correctness, Safety, Resistance to Stylistic Biases, and Best-of-N Scaling, Achieving State-of-the-Art Performance across Seven Major Reward Model Benchmarks. Ablation Studies Confirm That the Effectiveness of Our Approach Stems Not Only from Data Scale but Also from High-Quality Curation. The Skywork-Reward-V2 Series Represents Substantial Progress in Open Reward Models, Highlighting the Untapped Potential of Existing Preference Datasets and Demonstrating How Human-AI Curation Synergy Can Unlock Significantly Higher Data Quality. 

---
# User-guided Generative Source Separation 

**Title (ZH)**: 用户引导的生成式源分离 

**Authors**: Yutong Wen, Minje Kim, Paris Smaragdis  

**Link**: [PDF](https://arxiv.org/pdf/2507.01339)  

**Abstract**: Music source separation (MSS) aims to extract individual instrument sources from their mixture. While most existing methods focus on the widely adopted four-stem separation setup (vocals, bass, drums, and other instruments), this approach lacks the flexibility needed for real-world applications. To address this, we propose GuideSep, a diffusion-based MSS model capable of instrument-agnostic separation beyond the four-stem setup. GuideSep is conditioned on multiple inputs: a waveform mimicry condition, which can be easily provided by humming or playing the target melody, and mel-spectrogram domain masks, which offer additional guidance for separation. Unlike prior approaches that relied on fixed class labels or sound queries, our conditioning scheme, coupled with the generative approach, provides greater flexibility and applicability. Additionally, we design a mask-prediction baseline using the same model architecture to systematically compare predictive and generative approaches. Our objective and subjective evaluations demonstrate that GuideSep achieves high-quality separation while enabling more versatile instrument extraction, highlighting the potential of user participation in the diffusion-based generative process for MSS. Our code and demo page are available at this https URL 

**Abstract (ZH)**: 音乐源分离（MSS）旨在从混合音中提取独立的乐器源。尽管现有方法大多关注于广泛采用的四分体分离设置（人声、贝斯、鼓和其他乐器），但这种方法缺乏现实应用所需的灵活性。为解决这一问题，我们提出GuideSep，这是一种基于扩散的MSS模型，能够超越四分体设置进行乐器无关的分离。GuideSep基于多个输入条件：声音模仿条件，可以通过哼唱或演奏目标旋律轻松提供；以及梅尔谱图域掩码，为分离提供额外指导。与依赖固定类别标签或声音查询的先前方法不同，我们的条件方案结合生成方法提供了更大的灵活性和适用性。此外，我们使用相同模型架构设计了一个掩码预测基准，系统地比较预测和生成方法。我们的客观和主观评估表明，GuideSep实现了高质量的分离，同时使乐器提取更具灵活性，突显了用户参与基于扩散的生成过程在MSS中的潜力。我们的代码和演示页可在以下网址获得。 

---
# Neural Hamiltonian Operator 

**Title (ZH)**: 神经哈密尔顿算子 

**Authors**: Qian Qi  

**Link**: [PDF](https://arxiv.org/pdf/2507.01313)  

**Abstract**: Stochastic control problems in high dimensions are notoriously difficult to solve due to the curse of dimensionality. An alternative to traditional dynamic programming is Pontryagin's Maximum Principle (PMP), which recasts the problem as a system of Forward-Backward Stochastic Differential Equations (FBSDEs). In this paper, we introduce a formal framework for solving such problems with deep learning by defining a \textbf{Neural Hamiltonian Operator (NHO)}. This operator parameterizes the coupled FBSDE dynamics via neural networks that represent the feedback control and an ansatz for the value function's spatial gradient. We show how the optimal NHO can be found by training the underlying networks to enforce the consistency conditions dictated by the PMP. By adopting this operator-theoretic view, we situate the deep FBSDE method within the rigorous language of statistical inference, framing it as a problem of learning an unknown operator from simulated data. This perspective allows us to prove the universal approximation capabilities of NHOs under general martingale drivers and provides a clear lens for analyzing the significant optimization challenges inherent to this class of models. 

**Abstract (ZH)**: 高维随机控制问题由于维数灾难以传统动态规划方法解决著名，一种替代方法是泊里亚克尔最大原理（PMP），它将问题重新表述为前向-后向随机微分方程（FBSDE）系统。本文通过定义一个**神经哈密尔顿算子（NHO）**，介绍了一种利用深度学习求解此类问题的形式化框架。该算子通过神经网络参数化耦合的FBSDE动力学，神经网络表示反馈控制并提供价值函数空间梯度的假设形式。我们展示了如何通过训练底层网络来找到满足PMP所规定的相容条件的最优NHO。采用这种算子观点，将深度FBSDE方法置于统计推断的严格语言框架中，将其视为从模拟数据中学习未知算子的问题。这种视角允许我们在一般鞅驱动条件下证明NHO的普遍逼近能力，并为分析此类模型固有的显著优化挑战提供清晰的视角。 

---
# Capacity Planning and Scheduling for Jobs with Uncertainty in Resource Usage and Duration 

**Title (ZH)**: 具有资源使用量和持续时间不确定性的作业的容量规划与调度 

**Authors**: Sunandita Patra, Mehtab Pathan, Mahmoud Mahfouz, Parisa Zehtabi, Wided Ouaja, Daniele Magazzeni, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2507.01225)  

**Abstract**: Organizations around the world schedule jobs (programs) regularly to perform various tasks dictated by their end users. With the major movement towards using a cloud computing infrastructure, our organization follows a hybrid approach with both cloud and on-prem servers. The objective of this work is to perform capacity planning, i.e., estimate resource requirements, and job scheduling for on-prem grid computing environments. A key contribution of our approach is handling uncertainty in both resource usage and duration of the jobs, a critical aspect in the finance industry where stochastic market conditions significantly influence job characteristics. For capacity planning and scheduling, we simultaneously balance two conflicting objectives: (a) minimize resource usage, and (b) provide high quality-of-service to the end users by completing jobs by their requested deadlines. We propose approximate approaches using deterministic estimators and pair sampling-based constraint programming. Our best approach (pair sampling-based) achieves much lower peak resource usage compared to manual scheduling without compromising on the quality-of-service. 

**Abstract (ZH)**: 全球范围内，组织定期安排作业（程序）以执行各种由最终用户指定的任务。随着云计算基础设施的广泛应用，我们的组织采用混合方式，结合使用云服务器和本地服务器。本研究的目的是进行容量规划，即估算本地网格计算环境所需的资源需求和作业调度。我们方法的关键贡献在于处理资源使用量和作业持续时间的不确定性，这在金融市场中尤为重要，因为在金融市场中，随机市场条件显著影响作业特性。在进行容量规划和调度时，我们同时平衡两个相互冲突的目标：（a）最小化资源使用；（b）通过提供高质量的服务来满足最终用户提出的截止时间要求。我们提出了基于确定性估计和配对抽样约束编程的近似方法。我们最佳的方法（基于配对抽样）在不牺牲服务质量的情况下，实现了显著较低的峰值资源使用量。 

---
# Are Large Brainwave Foundation Models Capable Yet? Insights from Fine-tuning 

**Title (ZH)**: 大型脑波基础模型目前是否足够？从微调获得的见解。 

**Authors**: Na Lee, Konstantinos Barmpas, Yannis Panagakis, Dimitrios Adamos, Nikolaos Laskaris, Stefanos Zafeiriou  

**Link**: [PDF](https://arxiv.org/pdf/2507.01196)  

**Abstract**: Foundation Models have demonstrated significant success across various domains in Artificial Intelligence (AI), yet their capabilities for brainwave modeling remain unclear. In this paper, we comprehensively evaluate current Large Brainwave Foundation Models (LBMs) through systematic fine-tuning experiments across multiple Brain-Computer Interface (BCI) benchmark tasks, including memory tasks and sleep stage classification. Our extensive analysis shows that state-of-the-art LBMs achieve only marginal improvements (0.9%-1.2%) over traditional deep architectures while requiring significantly more parameters (millions vs thousands), raising important questions about their efficiency and applicability in BCI contexts. Moreover, through detailed ablation studies and Low-Rank Adaptation (LoRA), we significantly reduce trainable parameters without performance degradation, while demonstrating that architectural and training inefficiencies limit LBMs' current capabilities. Our experiments span both full model fine-tuning and parameter-efficient adaptation techniques, providing insights into optimal training strategies for BCI applications. We pioneer the application of LoRA to LBMs, revealing that performance benefits generally emerge when adapting multiple neural network components simultaneously. These findings highlight the critical need for domain-specific development strategies to advance LBMs, suggesting that current architectures may require redesign to fully leverage the potential of foundation models in brainwave analysis. 

**Abstract (ZH)**: 大型脑电基础模型在脑-机接口任务中的表现及其改进策略：从全模型微调到低秩适应的全面评估 

---
# AI-guided digital intervention with physiological monitoring reduces intrusive memories after experimental trauma 

**Title (ZH)**: AI引导的数字干预结合生理监测减少实验性创伤后的侵入性记忆 

**Authors**: Megan T. deBettencourt, Sruthi Sakthivel, Emily A. Holmes, Mark Chevillet  

**Link**: [PDF](https://arxiv.org/pdf/2507.01081)  

**Abstract**: Trauma prevalence is vast globally. Evidence-based digital treatments can help, but most require human guidance. Human guides provide tailored instructions and responsiveness to internal cognitive states, but limit scalability. Can generative AI and neurotechnology provide a scalable alternative? Here we test ANTIDOTE, combining AI guidance and pupillometry to automatically deliver and monitor an evidence-based digital treatment, specifically the Imagery Competing Task Intervention (ICTI), to reduce intrusive memories after psychological trauma. One hundred healthy volunteers were exposed to videos of traumatic events and randomly assigned to an intervention or active control condition. As predicted, intervention participants reported significantly fewer intrusive memories over the following week. Post-hoc assessment against clinical rubrics confirmed the AI guide delivered the intervention successfully. Additionally, pupil size tracked intervention engagement and predicted symptom reduction, providing a candidate biomarker of intervention effectiveness. These findings open a path toward rigorous AI-guided digital interventions that can scale to trauma prevalence. 

**Abstract (ZH)**: 全球创伤发病率巨大。基于证据的数字治疗可以有所帮助，但大多需要人工指导。人工指导提供了个性化指令和对内在认知状态的响应，但限制了可扩展性。生成式AI和神经技术能否提供一种可扩展的替代方案？我们测试了ANTIDOTE，结合AI指导和瞳孔ometry，自动提供并监测基于证据的数字治疗，即图像对抗任务干预（ICTI），以减少心理创伤后的侵入性记忆。一百名健康志愿者观看了创伤事件视频，并随机分配到干预组或活性对照组。正如预测的那样，干预组参与者在随后的一周内报告的侵入性记忆显著较少。事后评估显示，AI指导成功地提供了干预。此外，瞳孔大小跟踪干预参与度并预测症状减少，提供了干预有效性的候选生物标志物。这些发现为走向严谨的AI指导数字干预并实现对创伤发病率的扩展打开了路径。 

---
# Empirical Analysis Of Heuristic and Approximation Algorithms for the The Mutual-Visibility Problem 

**Title (ZH)**: 启发式和近似算法对互视问题的实证分析 

**Authors**: Vanja Stojanović, Bor Pangeršič  

**Link**: [PDF](https://arxiv.org/pdf/2507.01076)  

**Abstract**: The NP-complete mutual-visibility (MV) problem currently lacks empirical analysis on its practical behaviour despite theoretical studies. This paper addresses this gap by implementing and evaluating three distinct algorithms - a direct greedy heuristic, a hypergraph-based approximation, and a genetic algorithm - on diverse synthetic graph datasets, including those with analytically known $\mu(G)$ values and general graph models. Our results demonstrate that for smaller graphs, the algorithms consistently achieve MV set sizes aligning with theoretical bounds. However, for larger instances, achieved solution sizes notably diverge from theoretical limits; this, combined with the absence of tight bounds, complicates absolute quality assessment. Nevertheless, validation on known optimal graphs showed the Genetic Algorithm and other heuristics empirically performing best among tested methods. 

**Abstract (ZH)**: NP完全互视（MV）问题当前缺乏对其实际行为的实证分析，尽管已有理论研究。本文通过在包括已知 $\mu(G)$ 值的合成图数据集和一般图模型上实现和评估三种不同的算法——直接贪婪启发式算法、基于超图的近似算法以及遗传算法——来弥补这一空白。我们的结果表明，对于较小的图，算法一致地实现了与理论界限相符的MV集大小。然而，对于较大的实例，达到的解决方案大小明显偏离理论限制；这与缺乏紧的边界一起，使得绝对质量评估变得复杂。不过，对已知最优图的验证显示，遗传算法和其他启发式算法在测试方法中表现最佳。 

---
# Evaluation of a Foundational Model and Stochastic Models for Forecasting Sporadic or Spiky Production Outages of High-Performance Machine Learning Services 

**Title (ZH)**: 对基础模型和随机模型在预测高性能机器学习服务间歇性或突发性生产中断方面的评价 

**Authors**: Keun Soo Yim  

**Link**: [PDF](https://arxiv.org/pdf/2507.01067)  

**Abstract**: Time series forecasting models have diverse real world applications (e.g., from electricity metrics to software workload). Latest foundational models trained for time series forecasting show strengths (e.g., for long sequences and in zero-shot settings). However, foundational model was not yet used for forecasting rare, spiky events, i.e., a challenging target because those are a corner case of extreme events. In this paper, we optimize a state-of-the-art foundational model to forecast sporadic or spiky production outages of high-performance machine learning services powering billions of client devices. We evaluate the forecasting errors of the foundational model compared with classical stochastic forecasting models (e.g., moving average and autoregressive). The analysis helps us understand how each of the evaluated models performs for the sporadic or spiky events. For example, it identifies the key patterns in the target data that are well tracked by the foundational model vs. each of the stochastic models. We use the models with optimal parameters to estimate a year-long outage statistics of a particular root cause with less than 6% value errors. 

**Abstract (ZH)**: 时间序列预测模型在各类实际应用中具有多样性（例如，从电力指标到软件工作负载）。最新的基础模型用于时间序列预测展现了优势（例如，在长序列和零样本设置中的表现）。然而，这些基础模型尚未被用于预测罕见、突发事件，这是一项具有挑战性的目标，因为这些事件是极端事件的一种特例。在本文中，我们优化了一种最新基础模型，以预测高性能机器学习服务驱动的数十亿客户端设备中的间歇性或突发性生产中断。我们将基础模型的预测误差与经典的随机预测模型（如移动平均和自回归模型）进行了比较评估，这有助于理解每种评估模型在间歇性或突发性事件中的表现。例如，它确定了目标数据中的关键模式，这些模式被基础模型良好追踪，而未被各随机模型很好地捕捉。我们使用具有最优参数的模型来估算特定根本原因一年的中断统计数据，误差低于6%。 

---
# FAIR-MATCH: A Multi-Objective Framework for Bias Mitigation in Reciprocal Dating Recommendations 

**Title (ZH)**: FAIR-MATCH：一种用于互惠 dating 推荐中偏见缓解的多目标框架 

**Authors**: Madhav Kotecha  

**Link**: [PDF](https://arxiv.org/pdf/2507.01063)  

**Abstract**: Online dating platforms have fundamentally transformed the formation of romantic relationships, with millions of users worldwide relying on algorithmic matching systems to find compatible partners. However, current recommendation systems in dating applications suffer from significant algorithmic deficiencies, including but not limited to popularity bias, filter bubble effects, and inadequate reciprocity modeling that limit effectiveness and introduce harmful biases. This research integrates foundational work with recent empirical findings to deliver a detailed analysis of dating app recommendation systems, highlighting key issues and suggesting research-backed solutions. Through analysis of reciprocal recommendation frameworks, fairness evaluation metrics, and industry implementations, we demonstrate that current systems achieve modest performance with collaborative filtering reaching 25.1\% while reciprocal methods achieve 28.7\%. Our proposed mathematical framework addresses these limitations through enhanced similarity measures, multi-objective optimization, and fairness-aware algorithms that maintain competitive accuracy while improving demographic representation to reduce algorithmic bias. 

**Abstract (ZH)**: 在线 dating 平台从根本上改变了浪漫关系的形成方式，全球数百万用户依赖算法匹配系统寻找兼容的伴侣。然而，当前 dating 应用中的推荐系统存在显著的算法缺陷，包括但不限于流行度偏差、过滤气泡效应和不充分的互惠建模，这些都限制了系统的有效性并引入了有害的偏差。本研究结合基础工作与最新的实证研究成果，对 dating 平台推荐系统进行了详细分析，指出现存的关键问题，并提出基于研究的解决方案。通过对互惠推荐框架、公平性评估指标及行业实施的分析，我们表明当前系统取得适中的性能，协作过滤达到 25.1%，而互惠方法达到 28.7%。我们提出的数学框架通过增强相似度度量、多目标优化及兼具公平性的算法，在保持竞争力的同时改善了人口统计学代表性，从而减少算法偏差。 

---
# Quantifying Student Success with Generative AI: A Monte Carlo Simulation Informed by Systematic Review 

**Title (ZH)**: 基于系统回顾指导的蒙特卡洛仿真定量衡量学生成功与生成式AI的关系 

**Authors**: Seyma Yaman Kayadibi  

**Link**: [PDF](https://arxiv.org/pdf/2507.01062)  

**Abstract**: The exponential development of generative artificial intelligence (GenAI) technologies like ChatGPT has raised increasing curiosity about their use in higher education, specifically with respect to how students view them, make use of them, and the implications for learning outcomes. This paper employs a hybrid methodological approach involving a systematic literature review and simulation-based modeling to explore student perceptions of GenAI use in the context of higher education. A total of nineteen empirical articles from 2023 through 2025 were selected from the PRISMA-based search targeting the Scopus database. Synthesis of emerging patterns from the literature was achieved by thematic categorization. Six of these had enough quantitative information, i.e., item-level means and standard deviations, to permit probabilistic modeling. One dataset, from the resulting subset, was itself selected as a representative case with which to illustrate inverse-variance weighting by Monte Carlo simulation, by virtue of its well-designed Likert scale format and thematic alignment with the use of computing systems by the researcher.
The simulation provided a composite "Success Score" forecasting the strength of the relationship between student perceptions and learning achievements. Findings reveal that attitude factors concerned with usability and real-world usefulness are significantly better predictors of positive learning achievement than affective or trust-based factors. Such an interdisciplinary perspective provides a unique means of linking thematic results with predictive modelling, resonating with longstanding controversies about the proper use of GenAI tools within the university. 

**Abstract (ZH)**: 生成人工智能（GenAI）技术如ChatGPT的指数级发展引发了对它们在高等教育中应用的日益浓厚兴趣，特别是学生如何看待这些技术、如何使用它们以及对学习成果的影响。本文采用混合方法论，结合系统文献综述和基于仿真的建模，探究GenAI在高等教育中的使用对学生感知的影响。从2023年至2025年使用PRISMA方法在Scopus数据库中筛选出19篇实证文章。通过主题分类整合文献中 emerging 的模式。六篇文章含有足够的定量信息，即项目级别的平均值和标准差，允许进行概率建模。其中一个数据集被选为具有代表性的案例，通过蒙特卡洛仿真的逆方差加权进行示例，因为它拥有精心设计的李克特量表格式，并且主题上与研究人员使用计算系统的模式相吻合。

仿真实验提供了一个综合的“成功分数”，预测学生感知与学习成就之间的关系强度。研究发现，与情感或基于信任的因素相比，与易用性和实际有用性相关的态度因素是积极学习成就的更显著的预测因子。这种跨学科视角为将主题结果与预测建模联系起来提供了一种独特的方式，与关于大学中正确使用GenAI工具的长期争议相呼应。 

---
# XxaCT-NN: Structure Agnostic Multimodal Learning for Materials Science 

**Title (ZH)**: XxaCT-NN: 结构无关的多模态学习方法在材料科学中的应用 

**Authors**: Jithendaraa Subramanian, Linda Hung, Daniel Schweigert, Santosh Suram, Weike Ye  

**Link**: [PDF](https://arxiv.org/pdf/2507.01054)  

**Abstract**: Recent advances in materials discovery have been driven by structure-based models, particularly those using crystal graphs. While effective for computational datasets, these models are impractical for real-world applications where atomic structures are often unknown or difficult to obtain. We propose a scalable multimodal framework that learns directly from elemental composition and X-ray diffraction (XRD) -- two of the more available modalities in experimental workflows without requiring crystal structure input. Our architecture integrates modality-specific encoders with a cross-attention fusion module and is trained on the 5-million-sample Alexandria dataset. We present masked XRD modeling (MXM), and apply MXM and contrastive alignment as self-supervised pretraining strategies. Pretraining yields faster convergence (up to 4.2x speedup) and improves both accuracy and representation quality. We further demonstrate that multimodal performance scales more favorably with dataset size than unimodal baselines, with gains compounding at larger data regimes. Our results establish a path toward structure-free, experimentally grounded foundation models for materials science. 

**Abstract (ZH)**: 近期材料发现领域的进展得益于基于结构的模型，特别是使用晶体图模型。虽然这些模型适用于计算数据集，但在现实应用中，原子结构通常未知或难以获取，这使得这些模型不可行。我们提出了一种可扩展的多模态框架，可以直接从元素组成和X射线衍射（XRD）两种实验流程中更为可用的模态进行学习，无需输入晶体结构。该架构将模态特定编码器与跨注意力融合模块集成，并在包含500万样本的Alexandria数据集上进行训练。我们提出了掩蔽XRD建模（MXM），并使用MXM和对比对齐作为自监督预训练策略。预训练可以更快速地收敛（最高加速4.2倍），并提高准确性和表示质量。我们进一步表明，多模态性能在数据集规模增大时具有更好的扩展性，这种优势在大规模数据集上更为明显。我们的结果为材料科学奠定了无需结构、实验为基础的范式模型的道路。 

---
# Long-Sequence Memory with Temporal Kernels and Dense Hopfield Functionals 

**Title (ZH)**: 长序列记忆的时序核与密集霍普菲尔德函数]." 

**Authors**: Ahmed Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2507.01052)  

**Abstract**: In this study we introduce a novel energy functional for long-sequence memory, building upon the framework of dense Hopfield networks which achieves exponential storage capacity through higher-order interactions. Building upon earlier work on long-sequence Hopfield memory models, we propose a temporal kernal $K(m, k)$ to incorporate temporal dependencies, enabling efficient sequential retrieval of patterns over extended sequences. We demonstrate the successful application of this technique for the storage and sequential retrieval of movies frames which are well suited for this because of the high dimensional vectors that make up each frame creating enough variation between even sequential frames in the high dimensional space. The technique has applications in modern transformer architectures, including efficient long-sequence modeling, memory augmentation, improved attention with temporal bias, and enhanced handling of long-term dependencies in time-series data. Our model offers a promising approach to address the limitations of transformers in long-context tasks, with potential implications for natural language processing, forecasting, and beyond. 

**Abstract (ZH)**: 本研究我们介绍了一种针对长序列记忆的新型能量函数，基于密集霍普菲尔德网络框架，通过更高阶的相互作用实现指数级存储容量。基于早期关于长序列霍普菲尔德记忆模型的工作，我们提出了一种时间内核 $K(m, k)$，以纳入时间依赖性，从而实现对扩展序列中模式的高效逐序列检索。我们展示了该技术在电影帧存储和检索中的成功应用，因为每一帧由高维向量构成，在高维空间中即使顺序帧之间也存在足够的变化。该技术在现代变换器架构中具有应用价值，包括高效长序列建模、记忆增强、带有时间偏置的注意力改进以及时间序列数据中长期依赖关系的增强处理。我们的模型为解决变换器在长上下文任务中的局限性提供了一种有前景的方法，并可能对自然语言处理、预测等领域产生影响。 

---
# Can AI be Consentful? 

**Title (ZH)**: AI能征得同意吗？ 

**Authors**: Giada Pistilli, Bruna Trevelin  

**Link**: [PDF](https://arxiv.org/pdf/2507.01051)  

**Abstract**: The evolution of generative AI systems exposes the challenges of traditional legal and ethical frameworks built around consent. This chapter examines how the conventional notion of consent, while fundamental to data protection and privacy rights, proves insufficient in addressing the implications of AI-generated content derived from personal data. Through legal and ethical analysis, we show that while individuals can consent to the initial use of their data for AI training, they cannot meaningfully consent to the numerous potential outputs their data might enable or the extent to which the output is used or distributed. We identify three fundamental challenges: the scope problem, the temporality problem, and the autonomy trap, which collectively create what we term a ''consent gap'' in AI systems and their surrounding ecosystem. We argue that current legal frameworks inadequately address these emerging challenges, particularly regarding individual autonomy, identity rights, and social responsibility, especially in cases where AI-generated content creates new forms of personal representation beyond the scope of the original consent. By examining how these consent limitations intersect with broader principles of responsible AI (including fairness, transparency, accountability, and autonomy) we demonstrate the need to evolve ethical and legal approaches to consent. 

**Abstract (ZH)**: 生成型AI系统的演进揭示了传统基于同意的法律和伦理框架的挑战。本章探讨了尽管传统的同意概念是数据保护和隐私权的基础，但在处理AI生成内容对个人数据的影响时仍显得不足。通过法律和伦理分析，我们表明，尽管个人可以对数据初始用于AI训练给予同意，但他们无法对数据可能产生的众多潜在输出以及这些输出的使用和分发程度给予有意义的同意。我们识别出三个基本挑战：范围问题、时效性问题和自主性陷阱，这些共同形成了我们所称的“同意缺口”。我们认为现有的法律框架在应对这些新兴挑战方面存在不足，特别是在AI生成内容超出原始同意范围创建新的个人代表性形式时，更缺乏针对个人自主性、身份权利和社会责任的考虑。通过分析这些同意限制与负责任AI的更广泛原则（包括公平性、透明度、可问责性和自主性）的交集，我们证明了需要发展新的伦理和法律上的同意方法。 

---
# Text Detoxification: Data Efficiency, Semantic Preservation and Model Generalization 

**Title (ZH)**: 文本净化：数据效率、语义保留与模型泛化 

**Authors**: Jing Yu, Yibo Zhao, Jiapeng Zhu, Wenming Shao, Bo Pang, Zhao Zhang, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.01050)  

**Abstract**: The widespread dissemination of toxic content on social media poses a serious threat to both online environments and public discourse, highlighting the urgent need for detoxification methods that effectively remove toxicity while preserving the original semantics. However, existing approaches often struggle to simultaneously achieve strong detoxification performance, semantic preservation, and robustness to out-of-distribution data. Moreover, they typically rely on costly, manually annotated parallel corpora while showing poor data efficiency. To address these challenges, we propose a two-stage training framework that jointly optimizes for data efficiency, semantic preservation, and model generalization. We first perform supervised fine-tuning on a small set of high-quality, filtered parallel data to establish a strong initialization. Then, we leverage unlabeled toxic inputs and a custom-designed reward model to train the LLM using Group Relative Policy Optimization. Experimental results demonstrate that our method effectively mitigates the trade-offs faced by previous work, achieving state-of-the-art performance with improved generalization and significantly reduced dependence on annotated data. Our code is available at: this https URL 

**Abstract (ZH)**: 社交媒体上广泛传播的有毒内容对在线环境和公众 discourse 威胁严重，突显出迫切需要既能有效去除有毒内容又能保留原始语义的净化方法。然而，现有方法往往难以同时实现强大的去毒性能、语义保留和对分布外数据的鲁棒性。此外，它们通常依赖于昂贵的手工标注平行语料库，而数据效率较低。为应对这些挑战，我们提出了一种两阶段训练框架，以联合优化数据效率、语义保留和模型泛化能力。首先，在高质量过滤后的平行数据小数据集上进行监督微调，以建立强大的初始化。然后，利用未标记的有毒输入和自定义设计的奖励模型，通过组相对策略优化训练大规模语言模型（LLM）。实验结果表明，我们的方法有效缓解了以往工作的权衡，实现了最先进的性能，并显著减少了对标注数据的依赖。源代码可从如下链接获取：this https URL。 

---
# Sensing Cardiac Health Across Scenarios and Devices: A Multi-Modal Foundation Model Pretrained on Heterogeneous Data from 1.7 Million Individuals 

**Title (ZH)**: 跨场景和设备的 cardiac 健康感知识别：基于 170 万个体异质数据的多模态预训练基础模型 

**Authors**: Xiao Gu, Wei Tang, Jinpei Han, Veer Sangha, Fenglin Liu, Shreyank N Gowda, Antonio H. Ribeiro, Patrick Schwab, Kim Branson, Lei Clifton, Antonio Luiz P. Ribeiro, Zhangdaihong Liu, David A. Clifton  

**Link**: [PDF](https://arxiv.org/pdf/2507.01045)  

**Abstract**: Cardiac biosignals, such as electrocardiograms (ECG) and photoplethysmograms (PPG), are of paramount importance for the diagnosis, prevention, and management of cardiovascular diseases, and have been extensively used in a variety of clinical tasks. Conventional deep learning approaches for analyzing these signals typically rely on homogeneous datasets and static bespoke models, limiting their robustness and generalizability across diverse clinical settings and acquisition protocols. In this study, we present a cardiac sensing foundation model (CSFM) that leverages advanced transformer architectures and a generative, masked pretraining strategy to learn unified representations from vast, heterogeneous health records. Our model is pretrained on an innovative multi-modal integration of data from multiple large-scale datasets (including MIMIC-III-WDB, MIMIC-IV-ECG, and CODE), comprising cardiac signals and the corresponding clinical or machine-generated text reports from approximately 1.7 million individuals. We demonstrate that the embeddings derived from our CSFM not only serve as effective feature extractors across diverse cardiac sensing scenarios, but also enable seamless transfer learning across varying input configurations and sensor modalities. Extensive evaluations across diagnostic tasks, demographic information recognition, vital sign measurement, clinical outcome prediction, and ECG question answering reveal that CSFM consistently outperforms traditional one-modal-one-task approaches. Notably, CSFM exhibits robust performance across multiple ECG lead configurations from standard 12-lead systems to single-lead setups, and in scenarios where only ECG, only PPG, or a combination thereof is available. These findings highlight the potential of CSFM as a versatile and scalable solution, for comprehensive cardiac monitoring. 

**Abstract (ZH)**: 心脏生物信号（如心电图ECG和光体积描记图PPG）对于心血管疾病的诊断、预防和管理至关重要，并在多种临床任务中广泛使用。传统的深度学习方法在分析这些信号时通常依赖于同质数据集和定制模型，这限制了它们在不同临床环境和采集协议下的鲁棒性和泛化能力。在本研究中，我们提出了一种心脏感知基础模型（CSFM），该模型利用先进的变压器架构和生成性掩码预训练策略，从大量异质健康记录中学习统一表示。该模型在包括MIMIC-III-WDB、MIMIC-IV-ECG和CODE在内的多个大规模数据集的多模态集成数据上进行预训练，这些数据集包含了约170万名个体的心电图信号及其相应的临床或机器生成的文本报告。研究结果显示，CSFM提取的嵌入不仅在多种心脏感知场景中作为有效的特征提取器，还能在不同输入配置和传感器模态下实现无缝迁移学习。通过在诊断任务、人口统计信息识别、生理参数测量、临床结果预测和心电图问答等广泛评估中，CSFM表现出优于传统单模态单任务方法的性能。值得注意的是，CSFM在从标准12导联系统到单导联设置等多种心电图导联配置下表现稳健，并在只有心电图、只有光体积描记图或两者结合可用的场景下也表现出色。这些发现突显了CSFM作为多功能和可拓展解决方案的潜力，适用于全面的心脏监测。 

---
# Data Classification with Dynamically Growing and Shrinking Neural Networks 

**Title (ZH)**: 动态扩展与收缩的神经网络的数据分类 

**Authors**: Szymon Świderski, Agnieszka Jastrzębska  

**Link**: [PDF](https://arxiv.org/pdf/2507.01043)  

**Abstract**: The issue of data-driven neural network model construction is one of the core problems in the domain of Artificial Intelligence. A standard approach assumes a fixed architecture with trainable weights. A conceptually more advanced assumption is that we not only train the weights, but also find out the optimal model architecture. We present a new method that realizes just that. This article is an extended version of our conference paper titled "Dynamic Growing and Shrinking of Neural Networks with Monte Carlo Tree Search [26]". In the paper, we show in detail how to create a neural network with a procedure that allows dynamic shrinking and growing of the model while it is being trained. The decision-making mechanism for the architectural design is governed by a Monte Carlo tree search procedure which simulates network behavior and allows to compare several candidate architecture changes to choose the best one. The proposed method was validated using both visual and time series datasets, demonstrating its particular effectiveness in multivariate time series classification. This is attributed to the architecture's ability to adapt dynamically, allowing independent modifications for each time series. The approach is supplemented by Python source code for reproducibility. Experimental evaluations in visual pattern and multivariate time series classification tasks revealed highly promising performance, underscoring the method's robustness and adaptability. 

**Abstract (ZH)**: 数据驱动神经网络模型构建是人工智能领域的核心问题之一。一种标准的做法是假设固定架构并通过可训练的权重进行训练。一个更为先进的概念是不仅训练权重，还寻找最优模型架构。我们提出了一种新方法实现这一点。本文是我们在会议上发表的论文“使用蒙特卡洛树搜索的神经网络的动态生长和收缩[26]”的扩展版本。在论文中，我们详细介绍了如何通过允许模型在训练过程中动态收缩和生长的过程来创建神经网络。架构设计的决策机制由蒙特卡洛树搜索过程管理，该过程模拟网络行为并允许比较多个候选架构变化以选择最佳方案。提出的方 法通过视觉数据集和时间序列数据集验证，展示了其在多变量时间序列分类方面的特别有效性。这归因于架构能够动态适应，允许每个时间序列独立修改。此外，该方法附有Python源代码以提高可再现性。实验评估在视觉模式分类和多变量时间序列分类任务中表明了高度有希望的性能，突显了该方法的稳健性和适应性。 

---
# Can Argus Judge Them All? Comparing VLMs Across Domains 

**Title (ZH)**: Argus能够审判他们吗？跨领域比较VLMs 

**Authors**: Harsh Joshi, Gautam Siddharth Kashyap, Rafiq Ali, Ebad Shabbir, Niharika Jain, Sarthak Jain, Jiechao Gao, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2507.01042)  

**Abstract**: Vision-Language Models (VLMs) are advancing multimodal AI, yet their performance consistency across tasks is underexamined. We benchmark CLIP, BLIP, and LXMERT across diverse datasets spanning retrieval, captioning, and reasoning. Our evaluation includes task accuracy, generation quality, efficiency, and a novel Cross-Dataset Consistency (CDC) metric. CLIP shows strongest generalization (CDC: 0.92), BLIP excels on curated data, and LXMERT leads in structured reasoning. These results expose trade-offs between generalization and specialization, informing industrial deployment of VLMs and guiding development toward robust, task-flexible architectures. 

**Abstract (ZH)**: Vision-Language模型（VLMs）在 multimodal AI 领域取得了进展，但其在不同任务中的性能一致性尚待深入研究。我们通过对CLIP、BLIP和LXMERT在检索、描述和推理等多样数据集上的基准测试，评估了任务准确性、生成质量、效率以及一个新的跨数据集一致性（CDC）指标。CLIP在泛化能力上表现最佳（CDC: 0.92），BLIP在精心策划的数据上表现优异，而LXMERT在结构化推理上处于领先地位。这些结果揭示了泛化能力和专业化的权衡，为VLMs的工业应用提供了指导，并推动了更 robust、更具任务灵活性的架构的发展。 

---
# Fast AI Model Splitting over Edge Networks 

**Title (ZH)**: 边缘网络中快速AI模型拆分 

**Authors**: Zuguang Li, Wen Wu, Shaohua Wu, Songge Zhang, Ye Wang, Xuemin, Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.01041)  

**Abstract**: Split learning (SL) has emerged as a computationally efficient approach for artificial intelligence (AI) model training, which can alleviate device-side computational workloads. However, complex AI model architectures pose high computational complexity to obtain the optimal model splitting. In this paper, we represent an arbitrary AI model as a directed acyclic graph (DAG), and then reformulate the optimal model splitting problem as a minimum s-t cut search problem. To solve the problem, we propose a fast DAG-based model splitting algorithm, which restructures the DAG to enable the optimal model splitting identification via a maximum flow method. Theoretical analysis indicates that the proposed algorithm is optimal. Furthermore, considering AI models with block structures, we propose a block-wise model splitting algorithm to reduce computational complexity. The algorithm abstracts each block, i.e., a component consisting of multiple layers, into a single vertex, thereby obtaining the optimal model splitting via a simplified DAG. Extensive experimental results demonstrate that the proposed algorithms can determine the optimal model splitting within milliseconds, as well as reduce training delay by 24.62%-38.95% in dynamic edge networks as compared to the state-of-the-art benchmarks. 

**Abstract (ZH)**: Split学习（SL）已成为一种高效的人工智能（AI）模型训练方法，可以缓解设备端的计算负担。然而，复杂的AI模型架构导致了获得最优模型分割的高度计算复杂性。在本文中，我们将任意AI模型表示为有向无环图（DAG），并将最优模型分割问题重新表述为最小s-t割搜索问题。为了解决该问题，我们提出了一种基于DAG的快速模型分割算法，该算法重构DAG以通过最大流方法识别最优模型分割。理论分析表明，所提出的算法是最佳的。此外，考虑到具有块结构的AI模型，我们提出了一种块级模型分割算法来降低计算复杂性。该算法将每个块（即由多层组成的组件）抽象为一个顶点，从而通过简化DAG获得最优模型分割。广泛的实验结果表明，所提出的算法可以在毫秒内确定最优模型分割，并且与最先进的基准相比，在动态边缘网络中可以将训练延迟降低24.62%-38.95%。 

---
# Fast Clifford Neural Layers 

**Title (ZH)**: 快速克利福德神经网络层 

**Authors**: Tianxiang Xia, Max Neuwinger, Lin Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.01040)  

**Abstract**: Clifford Neural Layers improve PDE modeling by introducing Clifford Algebra into neural networks. In this project we focus on optimizing the inference of 2/3D Clifford convolutional layers and multivector activation layers for one core CPU performance.
Overall, by testing on a real network block involving Clifford convolutional layers and multivector activation layers, we observe that our implementation is 30% faster than standard PyTorch implementation in relatively large data + network size (>L2 cache).
We open source our code base at this https URL 

**Abstract (ZH)**: Clifford神经层通过将Clifford代数引入神经网络改进PDE建模 

---
# On-Policy Optimization of ANFIS Policies Using Proximal Policy Optimization 

**Title (ZH)**: 使用 aproximimal 策略优化的 ANFIS 策略在线优化 

**Authors**: Kaaustaaub Shankar, Wilhelm Louw, Kelly Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2507.01039)  

**Abstract**: We propose a reinforcement learning (RL) approach for training neuro-fuzzy controllers using Proximal Policy Optimization (PPO). Building on prior work that applied Deep Q-Learning to Adaptive Neuro-Fuzzy Inference Systems (ANFIS), our method replaces the off-policy value-based framework with a stable on-policy actor-critic loop. We evaluate this approach in the CartPole-v1 environment using multiple random seeds and compare its learning performance against ANFIS-Deep Q-Network (DQN) baselines. It was found that PPO-trained fuzzy agents achieved a mean return of 500 +/- 0 on CartPole-v1 after 20000 updates, showcasing less variance than prior DQN-based methods during training and overall faster convergence. These findings suggest that PPO offers a promising pathway for training explainable neuro-fuzzy controllers in reinforcement learning tasks. 

**Abstract (ZH)**: 我们提出了一种使用 proximal policy optimization (PPO) 训练神经模糊控制器的 reinforcement learning (RL) 方法。该方法在先前将深度 Q 学习应用于自适应模糊神经推理系统 (ANFIS) 的工作基础上，用稳定的 on-policy actor-critic 循环替代了离策价值基础框架。我们在 CartPole-v1 环境中使用多个随机种子对该方法进行评估，并将其学习性能与基于 DQN 的 ANFIS 基线方法进行比较。研究发现，经过 PPO 训练的模糊代理在 20000 次更新后于 CartPole-v1 环境中实现了平均回报 500 ± 0，在训练过程中方差较小，并且整体收敛速度更快。这些发现表明，PPO 为在强化学习任务中训练可解释的神经模糊控制器提供了有前景的途径。 

---
# Systemic Constraints of Undecidability 

**Title (ZH)**: 系统不可判定性约束 

**Authors**: Seth Bulin  

**Link**: [PDF](https://arxiv.org/pdf/2507.01036)  

**Abstract**: This paper presents a theory of systemic undecidability, reframing incomputability as a structural property of systems rather than a localized feature of specific functions or problems. We define a notion of causal embedding and prove a closure principle: any subsystem that participates functionally in the computation of an undecidable system inherits its undecidability. This result positions undecidability as a pervasive constraint on prediction, modeling, and epistemic access in both natural and artificial systems. Our framework disarms oracle mimicry and challenges the view that computational limits can be circumvented through architectural innovation. By generalizing classical results into a dynamic systems context, this work augments the logical trajectory of Gödel, Turing, and Chaitin, offering a new perspective of the topology of computability and its interrelation to the boundaries of scientific knowledge. 

**Abstract (ZH)**: 本文提出了系统不可判定性理论，将不可计算性重新定义为系统的一种结构属性，而非特定函数或问题的局部特征。我们定义了一种因果嵌入的概念，并证明了一个封闭原则：任何参与不可判定系统功能计算的子系统都会继承其不可判定性。这一结果将不可判定性定位为对自然和人工系统中预测、建模和知识获取的一种普遍约束。我们的框架消解了Oracle模拟，并挑战了通过体系结构创新绕过计算限制的看法。通过将经典结果推广到动力系统上下文，本文扩展了哥德尔、图灵和柴廷的逻辑轨迹，提供了计算拓扑及其与科学知识边界之间关系的新视角。 

---
# Data-driven Insights for Informed Decision-Making: Applying LSTM Networks for Robust Electricity Forecasting in Libya 

**Title (ZH)**: 基于数据驱动的洞察助力明智决策：在利比亚应用LSTM网络进行稳健的电力预测 

**Authors**: Asma Agaal, Mansour Essgaer, Hend M. Farkash, Zulaiha Ali Othman  

**Link**: [PDF](https://arxiv.org/pdf/2507.01034)  

**Abstract**: Accurate electricity forecasting is crucial for grid stability and energy planning, especially in Benghazi, Libya, where frequent load shedding, generation deficits, and infrastructure limitations persist. This study proposes a data-driven approach to forecast electricity load, generation, and deficits for 2025 using historical data from 2019 (a year marked by instability) and 2023 (a more stable year). Multiple time series models were applied, including ARIMA, seasonal ARIMA, dynamic regression ARIMA, exponential smoothing, extreme gradient boosting, and Long Short-Term Memory (LSTM) neural networks. The dataset was enhanced through missing value imputation, outlier smoothing, and log transformation. Performance was assessed using mean squared error, root mean squared error, mean absolute error, and mean absolute percentage error. LSTM outperformed all other models, showing strong capabilities in modeling non-stationary and seasonal patterns. A key contribution of this work is an optimized LSTM framework that integrates exogenous factors such as temperature and humidity, offering robust performance in forecasting multiple electricity indicators. These results provide practical insights for policymakers and grid operators to enable proactive load management and resource planning in data-scarce, volatile regions. 

**Abstract (ZH)**: 准确的电力 forecasting 对比利电信号稳定和能源规划至关重要，尤其是在频繁的负荷削减、发电不足和基础设施限制持续存在的的班加西地区。本研究提出了一种基于数据的方法，使用2019年（不稳定的年份）和2023年（较为稳定的年份）的历史数据，预测2025年的电力负荷、发电量和赤字。应用了多种时间序列模型，包括ARIMA、季节性ARIMA、动态回归ARIMA、指数平滑、极端梯度提升和长短期记忆（LSTM）神经网络。通过插补缺失值、平滑异常值和对数转换提升了数据集的质量。性能评估使用均方误差、均方根误差、平均绝对误差和平均绝对百分比误差进行。LSTM 在所有模型中表现最佳，展示了在建模非平稳性和季节性模式方面的强大能力。本研究的一个重要贡献是集成了如温度和湿度等外生因素的优化LSTM框架，为多电量指标预测提供了稳健的表现。这些结果为政策制定者和电网运营商提供了在数据稀缺、波动性高的地区进行积极负荷管理和资源规划的实际见解。 

---
# An Uncertainty-Aware Dynamic Decision Framework for Progressive Multi-Omics Integration in Classification Tasks 

**Title (ZH)**: 面向分类任务的渐进多组学集成动态决策框架，考虑不确定性 

**Authors**: Nan Mu, Hongbo Yang, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.01032)  

**Abstract**: Background and Objective: High-throughput multi-omics technologies have proven invaluable for elucidating disease mechanisms and enabling early diagnosis. However, the high cost of multi-omics profiling imposes a significant economic burden, with over reliance on full omics data potentially leading to unnecessary resource consumption. To address these issues, we propose an uncertainty-aware, multi-view dynamic decision framework for omics data classification that aims to achieve high diagnostic accuracy while minimizing testing costs. Methodology: At the single-omics level, we refine the activation functions of neural networks to generate Dirichlet distribution parameters, utilizing subjective logic to quantify both the belief masses and uncertainty mass of classification results. Belief mass reflects the support of a specific omics modality for a disease class, while the uncertainty parameter captures limitations in data quality and model discriminability, providing a more trustworthy basis for decision-making. At the multi omics level, we employ a fusion strategy based on Dempster-Shafer theory to integrate heterogeneous modalities, leveraging their complementarity to boost diagnostic accuracy and robustness. A dynamic decision mechanism is then applied that omics data are incrementally introduced for each patient until either all data sources are utilized or the model confidence exceeds a predefined threshold, potentially before all data sources are utilized. Results and Conclusion: We evaluate our approach on four benchmark multi-omics datasets, ROSMAP, LGG, BRCA, and KIPAN. In three datasets, over 50% of cases achieved accurate classification using a single omics modality, effectively reducing redundant testing. Meanwhile, our method maintains diagnostic performance comparable to full-omics models and preserves essential biological insights. 

**Abstract (ZH)**: 背景与目的：高通量多组学技术已在阐明疾病机制和实现早期诊断方面证明了其 invaluable的价值。然而，多组学表型的高成本对经济造成了显著负担，并可能导致过度依赖全组学数据进而造成不必要的资源消耗。为解决这些问题，我们提出了一种不确定性意识下的多视图动态决策框架，旨在通过最小化测试成本来实现高诊断准确性。方法：在单组学层面，我们改进了神经网络的激活函数以生成狄利克雷分布参数，并利用主观逻辑量化解分类结果的信念质量和不确定性质量。信念质量反映了特定组学模态对疾病类别的支持程度，而不确定性参数则捕捉数据质量和模型可分辨性的限制，为决策提供更可靠的基础。在多组学层面，我们基于Dempster-Shafer理论采用了融合策略以整合异质模态，并利用其互补性以提升诊断准确性和稳健性。然后应用动态决策机制，逐步为每位患者引入组学数据，直到所有数据源都被利用或模型置信度超过预定义阈值，从而可能在利用所有数据源之前停止。结果与结论：我们使用ROSMAP、LGG、BRCA和KIPAN四个基准多组学数据集评估了我们的方法。在三个数据集中，超过50%的病例仅通过单一组学模态即可实现准确分类，有效减少了冗余测试。同时，我们的方法保持了与全组学模型相当的诊断性能，并保留了关键的生物学洞察。 

---
# HPC-AI Coupling Methodology for Scientific Applications 

**Title (ZH)**: HPC-AI联合方法学及其在科学应用中的应用 

**Authors**: Yutong Lu, Dan Huang, Pin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.01025)  

**Abstract**: Artificial intelligence (AI) technologies have fundamentally transformed numerical-based high-performance computing (HPC) applications with data-driven approaches and endeavored to address existing challenges, e.g. high computational intensity, in various scientific domains. In this study, we explore the scenarios of coupling HPC and AI (HPC-AI) in the context of emerging scientific applications, presenting a novel methodology that incorporates three patterns of coupling: surrogate, directive, and coordinate. Each pattern exemplifies a distinct coupling strategy, AI-driven prerequisite, and typical HPC-AI ensembles. Through case studies in materials science, we demonstrate the application and effectiveness of these patterns. The study highlights technical challenges, performance improvements, and implementation details, providing insight into promising perspectives of HPC-AI coupling. The proposed coupling patterns are applicable not only to materials science but also to other scientific domains, offering valuable guidance for future HPC-AI ensembles in scientific discovery. 

**Abstract (ZH)**: 人工智能（AI）技术以数据驱动的方法从根本上改变了基于数值的高性能计算（HPC）应用，并致力于解决现有挑战，例如在各个科学领域中的高计算强度。在本研究中，我们在新兴科学应用的背景下探索高性能计算和人工智能（HPC-AI）的结合场景，提出了一种包含三种耦合模式的新方法：替代模式、指导模式和协调模式。每种模式代表了一种独特的耦合策略、基于AI的前提条件以及典型的HPC-AI结合体。通过材料科学案例研究，我们展示了这些模式的应用和效果。本研究突出了技术挑战、性能改进和实施细节，为HPC-AI耦合提供了有益的视角。提出的耦合模式不仅适用于材料科学，也适用于其他科学领域，为未来的HPC-AI结合体在科学研究中的应用提供了有价值的教学建议。 

---
# Hello Afrika: Speech Commands in Kinyarwanda 

**Title (ZH)**: Hello Afrika: 埔隆沃纳达语语音命令 

**Authors**: George Igwegbe, Martins Awojide, Mboh Bless, Nirel Kadzo  

**Link**: [PDF](https://arxiv.org/pdf/2507.01024)  

**Abstract**: Voice or Speech Commands are a subset of the broader Spoken Word Corpus of a language which are essential for non-contact control of and activation of larger AI systems in devices used in everyday life especially for persons with disabilities. Currently, there is a dearth of speech command models for African languages. The Hello Afrika project aims to address this issue and its first iteration is focused on the Kinyarwanda language since the country has shown interest in developing speech recognition technologies culminating in one of the largest datasets on Mozilla Common Voice. The model was built off a custom speech command corpus made up of general directives, numbers, and a wake word. The final model was deployed on multiple devices (PC, Mobile Phone and Edge Devices) and the performance was assessed using suitable metrics. 

**Abstract (ZH)**: 语音命令是语言语音语料库的一个子集，对于日常生活中设备的非接触控制和激活大型AI系统至关重要，尤其是对于残疾人士。目前，非洲语言的语音命令模型缺乏。Hello Afrika项目旨在解决这一问题，其首个迭代专注于基杭语语言，因为该国对开发语音识别技术表现出兴趣，并产生了Mozilla Common Voice最大的数据集之一。模型基于一个包含通用指令、数字和唤醒词的定制语音命令语料库进行构建。最终模型部署在多个设备（台式机、移动电话和边缘设备）上，并使用合适的指标进行性能评估。 

---
# A Systematic Review of Security Vulnerabilities in Smart Home Devices and Mitigation Techniques 

**Title (ZH)**: 智能家庭设备中的安全漏洞及其缓解技术系统综述 

**Authors**: Mohammed K. Alzaylaee  

**Link**: [PDF](https://arxiv.org/pdf/2507.01018)  

**Abstract**: Smart homes that integrate Internet of Things (IoT) devices face increasing cybersecurity risks, posing significant challenges to these environments. The study explores security threats in smart homes ecosystems, categorizing them into vulnerabilities at the network layer, device level, and those from cloud-based and AI-driven systems. Research findings indicate that post-quantum encryption, coupled with AI-driven anomaly detection, is highly effective in enhancing security; however, computational resource demands present significant challenges. Blockchain authentication together with zero-trust structures builds security resilience, although they need changes to existing infrastructure. The specific security strategies show their effectiveness through ANOVA, Chi-square tests, and Monte Carlo simulations yet lack sufficient scalability according to the results. The research demonstrates the requirement for improvement in cryptographic techniques, alongside AI-enhanced threat detection and adaptive security models which must achieve a balance between performance and efficiency and real-time applicability within smart home ecosystems. 

**Abstract (ZH)**: 智能家庭集成物联网设备面临不断增加的网络安全风险，给这些环境带来重大挑战。该研究探讨了智能家庭生态系统中的安全威胁，将其分类为网络层漏洞、设备级别漏洞以及基于云和AI驱动系统的漏洞。研究发现，结合后量子加密与AI驱动的异常检测在增强安全方面非常有效；然而，计算资源需求也提出了重大挑战。区块链认证与零信任结构构建了安全韧性，但需要对现有基础设施进行更改。具体的安全策略通过ANOVA、卡方检验和蒙特卡洛模拟显示了其有效性，但在可扩展性方面结果表明尚有不足。研究展示了在智能家庭生态系统中改进加密技术、增强AI威胁检测和适应性安全模型的需求，这些模型必须在性能、效率和实时适用性之间找到平衡。 

---
