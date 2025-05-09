# D-CODA: Diffusion for Coordinated Dual-Arm Data Augmentation 

**Title (ZH)**: D-CODA: 扩散驱动的协调双臂数据增强 

**Authors**: I-Chun Arthur Liu, Jason Chen, Gaurav Sukhatme, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2505.04860)  

**Abstract**: Learning bimanual manipulation is challenging due to its high dimensionality and tight coordination required between two arms. Eye-in-hand imitation learning, which uses wrist-mounted cameras, simplifies perception by focusing on task-relevant views. However, collecting diverse demonstrations remains costly, motivating the need for scalable data augmentation. While prior work has explored visual augmentation in single-arm settings, extending these approaches to bimanual manipulation requires generating viewpoint-consistent observations across both arms and producing corresponding action labels that are both valid and feasible. In this work, we propose Diffusion for COordinated Dual-arm Data Augmentation (D-CODA), a method for offline data augmentation tailored to eye-in-hand bimanual imitation learning that trains a diffusion model to synthesize novel, viewpoint-consistent wrist-camera images for both arms while simultaneously generating joint-space action labels. It employs constrained optimization to ensure that augmented states involving gripper-to-object contacts adhere to constraints suitable for bimanual coordination. We evaluate D-CODA on 5 simulated and 3 real-world tasks. Our results across 2250 simulation trials and 300 real-world trials demonstrate that it outperforms baselines and ablations, showing its potential for scalable data augmentation in eye-in-hand bimanual manipulation. Our project website is at: this https URL. 

**Abstract (ZH)**: 基于手腕相机的双臂协调数据增强方法（D-CODA）：面向 Eye-in-Hand 双臂模仿学习的离线数据增强 

---
# Data-Dependent Hidden Markov Model with Off-Road State Determination and Real-Time Viterbi Algorithm for Lane Determination in Autonomous Vehicles 

**Title (ZH)**: 基于数据依赖隐马尔可夫模型的离路状态确定及自主车辆车道识别实时维特比算法 

**Authors**: Mike Stas, Wang Hu, Jay A. Farrell  

**Link**: [PDF](https://arxiv.org/pdf/2505.04763)  

**Abstract**: Lane determination and lane sequence determination are important components for many Connected and Automated Vehicle (CAV) applications. Lane determination has been solved using Hidden Markov Model (HMM) among other methods. The existing HMM literature for lane sequence determination uses empirical definitions with user-modified parameters to calculate HMM probabilities. The probability definitions in the literature can cause breaks in the HMM due to the inability to directly calculate probabilities of off-road positions, requiring post-processing of data. This paper develops a time-varying HMM using the physical properties of the roadway and vehicle, and the stochastic properties of the sensors. This approach yields emission and transition probability models conditioned on the sensor data without parameter tuning. It also accounts for the probability that the vehicle is not in any roadway lane (e.g., on the shoulder or making a U-turn), which eliminates the need for post-processing to deal with breaks in the HMM processing. This approach requires adapting the Viterbi algorithm and the HMM to be conditioned on the sensor data, which are then used to generate the most-likely sequence of lanes the vehicle has traveled. The proposed approach achieves an average accuracy of 95.9%. Compared to the existing literature, this provides an average increase of 2.25% by implementing the proposed transition probability and an average increase of 5.1% by implementing both the proposed transition and emission probabilities. 

**Abstract (ZH)**: 车道确定和车道序列确定是许多Connected and Automated Vehicle (CAV)应用的重要组成部分。本文提出了一种基于路面和车辆的物理特性以及传感器的统计特性的时间变化隐马尔可夫模型，不需要参数调整就能得到基于传感器数据的发射和转换概率模型，同时考虑了车辆不在任何车道（例如在硬肩上或进行U-turn）的可能性，消除了HMM处理过程中需要后处理的问题。该方法通过调整维伯算法和HMM使其基于传感器数据，进而生成车辆最有可能行驶的车道序列。提出的这种方法平均准确率为95.9%。与现有文献相比，通过实施提出的转换概率平均提高了2.25%，通过同时实施提出的转换和发射概率平均提高了5.1%。 

---
# Fitts' List Revisited: An Empirical Study on Function Allocation in a Two-Agent Physical Human-Robot Collaborative Position/Force Task 

**Title (ZH)**: Fitts’定律再探：双人物理人机协作位置/力任务中功能分配的实证研究 

**Authors**: Nicky Mol, J. Micah Prendergast, David A. Abbink, Luka Peternel  

**Link**: [PDF](https://arxiv.org/pdf/2505.04722)  

**Abstract**: In this letter, we investigate whether the classical function allocation holds for physical Human-Robot Collaboration, which is important for providing insights for Industry 5.0 to guide how to best augment rather than replace workers. This study empirically tests the applicability of Fitts' List within physical Human-Robot Collaboration, by conducting a user study (N=26, within-subject design) to evaluate four distinct allocations of position/force control between human and robot in an abstract blending task. We hypothesize that the function in which humans control the position achieves better performance and receives higher user ratings. When allocating position control to the human and force control to the robot, compared to the opposite case, we observed a significant improvement in preventing overblending. This was also perceived better in terms of physical demand and overall system acceptance, while participants experienced greater autonomy, more engagement and less frustration. An interesting insight was that the supervisory role (when the robot controls both position and force control) was rated second best in terms of subjective acceptance. Another surprising insight was that if position control was delegated to the robot, the participants perceived much lower autonomy than when the force control was delegated to the robot. These findings empirically support applying Fitts' principles to static function allocation for physical collaboration, while also revealing important nuanced user experience trade-offs, particularly regarding perceived autonomy when delegating position control. 

**Abstract (ZH)**: 本研究探讨经典的函数分配原则在物理人机协作中的适用性，这对于提供有关未来工业如何最佳辅助而非替代工人的重要见解至关重要。通过一项用户研究（N=26，被试内设计），研究在一项抽象混合任务中测试了四种不同的人工智能与机器人位置/力控制分配方式，以实证检验Fitts列表在物理人机协作中的适用性。研究假设人类控制位置的函数能获得更好的性能并获得更高的用户评分。当人类控制位置控制而机器人控制力控制时，与相反情况相比，我们观察到显著减少了过度混合的情况，并在物理需求和系统整体接受度方面也表现更好，同时参与者感受到更大的自主性、更高的参与度和更低的挫败感。一个有趣的研究结果是：当机器人同时控制位置和力控制时，其在主观接受度方面排名第二。另一个令人惊讶的发现是：如果位置控制委托给机器人，参与者感知到的自主性显著低于力控制委托给机器人的情况。这些发现实证支持将Fitts原则应用于静态函数分配以促进物理协作，同时揭示出有关感知自主性的重要细微用户体验权衡，特别是在分配位置控制时。 

---
# Mapping User Trust in Vision Language Models: Research Landscape, Challenges, and Prospects 

**Title (ZH)**: 视觉语言模型中用户信任映射的研究概述、挑战与前景 

**Authors**: Agnese Chiatti, Sara Bernardini, Lara Shibelski Godoy Piccolo, Viola Schiaffonati, Matteo Matteucci  

**Link**: [PDF](https://arxiv.org/pdf/2505.05318)  

**Abstract**: The rapid adoption of Vision Language Models (VLMs), pre-trained on large image-text and video-text datasets, calls for protecting and informing users about when to trust these systems. This survey reviews studies on trust dynamics in user-VLM interactions, through a multi-disciplinary taxonomy encompassing different cognitive science capabilities, collaboration modes, and agent behaviours. Literature insights and findings from a workshop with prospective VLM users inform preliminary requirements for future VLM trust studies. 

**Abstract (ZH)**: Vision-Language Models的快速采用要求保护和告知用户何时信赖这些系统。本综述通过涵盖不同认知科学能力、合作模式和代理行为的多学科分类，回顾了用户-VLM交互中的信任动态研究。与潜在VLM用户的工作坊文献洞见和发现为未来VLM信任研究提供了初步要求。 

---
# Geometric Fault-Tolerant Neural Network Tracking Control of Unknown Systems on Matrix Lie Groups 

**Title (ZH)**: 未知系统的矩阵李群上几何容错神经网络跟踪控制 

**Authors**: Robin Chhabra, Farzaneh Abdollahi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04725)  

**Abstract**: We present a geometric neural network-based tracking controller for systems evolving on matrix Lie groups under unknown dynamics, actuator faults, and bounded disturbances. Leveraging the left-invariance of the tangent bundle of matrix Lie groups, viewed as an embedded submanifold of the vector space $\R^{N\times N}$, we propose a set of learning rules for neural network weights that are intrinsically compatible with the Lie group structure and do not require explicit parameterization. Exploiting the geometric properties of Lie groups, this approach circumvents parameterization singularities and enables a global search for optimal weights. The ultimate boundedness of all error signals -- including the neural network weights, the coordinate-free configuration error function, and the tracking velocity error -- is established using Lyapunov's direct method. To validate the effectiveness of the proposed method, we provide illustrative simulation results for decentralized formation control of multi-agent systems on the Special Euclidean group. 

**Abstract (ZH)**: 基于几何神经网络的矩阵李群上具有未知动力学、执行器故障和有界干扰系统的跟踪控制器研究 

---
# EcoAgent: An Efficient Edge-Cloud Collaborative Multi-Agent Framework for Mobile Automation 

**Title (ZH)**: EcoAgent：一种高效的边缘-云协作多Agent框架 Mobile Automation 

**Authors**: Biao Yi, Xavier Hu, Yurun Chen, Shengyu Zhang, Hongxia Yang, Fan Wu, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05440)  

**Abstract**: Cloud-based mobile agents powered by (multimodal) large language models ((M)LLMs) offer strong reasoning abilities but suffer from high latency and cost. While fine-tuned (M)SLMs enable edge deployment, they often lose general capabilities and struggle with complex tasks. To address this, we propose EcoAgent, an Edge-Cloud cOllaborative multi-agent framework for mobile automation. EcoAgent features a closed-loop collaboration among a cloud-based Planning Agent and two edge-based agents: the Execution Agent for action execution and the Observation Agent for verifying outcomes. The Observation Agent uses a Pre-Understanding Module to compress screen images into concise text, reducing token usage. In case of failure, the Planning Agent retrieves screen history and replans via a Reflection Module. Experiments on AndroidWorld show that EcoAgent maintains high task success rates while significantly reducing MLLM token consumption, enabling efficient and practical mobile automation. 

**Abstract (ZH)**: 基于云的由（多模态）大型语言模型驱动的移动代理（(M)LLMs）提供了强大的推理能力，但存在高延迟和成本问题。虽然微调的（M）SLMs能够实现边缘部署，但它们往往失去了一般能力，并且难以处理复杂任务。为了解决这个问题，我们提出了一种Edge-Cloud协作多代理框架EcoAgent，用于移动自动化。EcoAgent特征是云基规划代理和两个边缘基代理之间形成闭环合作：执行代理用于执行动作，观察代理用于验证结果。观察代理使用前理解模块将屏幕图像压缩成简洁的文本，减少令牌使用量。在失败情况下，规划代理通过反思模块检索屏幕历史并重新规划。在AndroidWorld的实验显示，EcoAgent能够在大幅减少MLLM令牌消耗的同时保持高任务成功率，实现高效和实用的移动自动化。 

---
# Advancing Neural Network Verification through Hierarchical Safety Abstract Interpretation 

**Title (ZH)**: 通过分层安全性抽象解释推进神经网络验证 

**Authors**: Luca Marzari, Isabella Mastroeni, Alessandro Farinelli  

**Link**: [PDF](https://arxiv.org/pdf/2505.05235)  

**Abstract**: Traditional methods for formal verification (FV) of deep neural networks (DNNs) are constrained by a binary encoding of safety properties, where a model is classified as either safe or unsafe (robust or not robust). This binary encoding fails to capture the nuanced safety levels within a model, often resulting in either overly restrictive or too permissive requirements. In this paper, we introduce a novel problem formulation called Abstract DNN-Verification, which verifies a hierarchical structure of unsafe outputs, providing a more granular analysis of the safety aspect for a given DNN. Crucially, by leveraging abstract interpretation and reasoning about output reachable sets, our approach enables assessing multiple safety levels during the FV process, requiring the same (in the worst case) or even potentially less computational effort than the traditional binary verification approach. Specifically, we demonstrate how this formulation allows rank adversarial inputs according to their abstract safety level violation, offering a more detailed evaluation of the model's safety and robustness. Our contributions include a theoretical exploration of the relationship between our novel abstract safety formulation and existing approaches that employ abstract interpretation for robustness verification, complexity analysis of the novel problem introduced, and an empirical evaluation considering both a complex deep reinforcement learning task (based on Habitat 3.0) and standard DNN-Verification benchmarks. 

**Abstract (ZH)**: 传统方法用于深度神经网络形式验证的局限性及其克服：一种新的抽象DNN验证方法 

---
# ChemRxivQuest: A Curated Chemistry Question-Answer Database Extracted from ChemRxiv Preprints 

**Title (ZH)**: ChemRxivQuest: 一个精选的化学问答数据库，提取自ChemRxiv预印本 

**Authors**: Mahmoud Amiri, Thomas Bocklitz  

**Link**: [PDF](https://arxiv.org/pdf/2505.05232)  

**Abstract**: The rapid expansion of chemistry literature poses significant challenges for researchers seeking to efficiently access domain-specific knowledge. To support advancements in chemistry-focused natural language processing (NLP), we present ChemRxivQuest, a curated dataset of 970 high-quality question-answer (QA) pairs derived from 155 ChemRxiv preprints across 17 subfields of chemistry. Each QA pair is explicitly linked to its source text segment to ensure traceability and contextual accuracy. ChemRxivQuest was constructed using an automated pipeline that combines optical character recognition (OCR), GPT-4o-based QA generation, and a fuzzy matching technique for answer verification. The dataset emphasizes conceptual, mechanistic, applied, and experimental questions, enabling applications in retrieval-based QA systems, search engine development, and fine-tuning of domain-adapted large language models. We analyze the dataset's structure, coverage, and limitations, and outline future directions for expansion and expert validation. ChemRxivQuest provides a foundational resource for chemistry NLP research, education, and tool development. 

**Abstract (ZH)**: 化学文献的迅速扩张给研究人员高效获取领域特定知识带来了重大挑战。为了支持化学方向的自然语言处理(NLP)进展，我们呈现了ChemRxivQuest数据集，该数据集包含了来自155篇ChemRxiv预印本的970个高质量问答(QA)对，覆盖了17个化学子领域。每个QA对都与原始文本段落明确链接，以确保可追溯性和语境准确性。ChemRxivQuest是通过结合光学字符识别(OCR)、基于GPT-4o的问答生成以及模糊匹配技术进行答案验证的自动化管道构建而成的。该数据集强调概念性、机制性、应用性和实验性问题，能够应用于检索型问答系统、搜索引擎开发和领域适配大语言模型的微调。我们分析了数据集的结构、覆盖面和限制，并概述了扩展和专家验证的未来方向。ChemRxivQuest为化学NLP研究、教育和工具开发提供了基础资源。 

---
# Societal and technological progress as sewing an ever-growing, ever-changing, patchy, and polychrome quilt 

**Title (ZH)**: 社会和技术的进步犹如缝制一幅不断扩展、变化莫测、图案繁多且色彩斑斓的Patchworktapestry。 

**Authors**: Joel Z. Leibo, Alexander Sasha Vezhnevets, William A. Cunningham, Sébastien Krier, Manfred Diaz, Simon Osindero  

**Link**: [PDF](https://arxiv.org/pdf/2505.05197)  

**Abstract**: Artificial Intelligence (AI) systems are increasingly placed in positions where their decisions have real consequences, e.g., moderating online spaces, conducting research, and advising on policy. Ensuring they operate in a safe and ethically acceptable fashion is thus critical. However, most solutions have been a form of one-size-fits-all "alignment". We are worried that such systems, which overlook enduring moral diversity, will spark resistance, erode trust, and destabilize our institutions. This paper traces the underlying problem to an often-unstated Axiom of Rational Convergence: the idea that under ideal conditions, rational agents will converge in the limit of conversation on a single ethics. Treating that premise as both optional and doubtful, we propose what we call the appropriateness framework: an alternative approach grounded in conflict theory, cultural evolution, multi-agent systems, and institutional economics. The appropriateness framework treats persistent disagreement as the normal case and designs for it by applying four principles: (1) contextual grounding, (2) community customization, (3) continual adaptation, and (4) polycentric governance. We argue here that adopting these design principles is a good way to shift the main alignment metaphor from moral unification to a more productive metaphor of conflict management, and that taking this step is both desirable and urgent. 

**Abstract (ZH)**: 人工智能系统在决策具有实际后果的情况下，确保其安全和伦理接受性至关重要：超越理性能合的框定 

---
# Is there a half-life for the success rates of AI agents? 

**Title (ZH)**: AI代理的成功率是否有半衰期？ 

**Authors**: Toby Ord  

**Link**: [PDF](https://arxiv.org/pdf/2505.05115)  

**Abstract**: Building on the recent empirical work of Kwa et al. (2025), I show that within their suite of research-engineering tasks the performance of AI agents on longer-duration tasks can be explained by an extremely simple mathematical model -- a constant rate of failing during each minute a human would take to do the task. This implies an exponentially declining success rate with the length of the task and that each agent could be characterised by its own half-life. This empirical regularity allows us to estimate the success rate for an agent at different task lengths. And the fact that this model is a good fit for the data is suggestive of the underlying causes of failure on longer tasks -- that they involve increasingly large sets of subtasks where failing any one fails the task. Whether this model applies more generally on other suites of tasks is unknown and an important subject for further work. 

**Abstract (ZH)**: 基于Kwa等人（2025）的近期实证研究，本文展示了在其研究-工程任务套件中，AI代理在长时间任务上的性能可以用一个极其简单的数学模型来解释——即在人类完成任务所需的时间内的每分钟都会以恒定的失败率失败。这暗示着随着任务长度的增加，成功率呈指数下降，并且每个代理都可以通过其半衰期来表征。这一经验规律允许我们估算不同任务长度下代理的成功率。并且，这一模型能够很好地拟合数据这一事实，表明长时间任务失败的根本原因可能在于涉及越来越多的子任务，只要任何一个子任务失败，整个任务就失败。目前尚不清楚该模型是否适用于其他任务套件，这是一项重要且有待进一步研究的主题。 

---
# A Neuro-Symbolic Framework for Sequence Classification with Relational and Temporal Knowledge 

**Title (ZH)**: 一种基于神经符号框架的序列分类方法，融合关系性和时间性知识 

**Authors**: Luca Salvatore Lorello, Marco Lippi, Stefano Melacci  

**Link**: [PDF](https://arxiv.org/pdf/2505.05106)  

**Abstract**: One of the goals of neuro-symbolic artificial intelligence is to exploit background knowledge to improve the performance of learning tasks. However, most of the existing frameworks focus on the simplified scenario where knowledge does not change over time and does not cover the temporal dimension. In this work we consider the much more challenging problem of knowledge-driven sequence classification where different portions of knowledge must be employed at different timesteps, and temporal relations are available. Our experimental evaluation compares multi-stage neuro-symbolic and neural-only architectures, and it is conducted on a newly-introduced benchmarking framework. Results demonstrate the challenging nature of this novel setting, and also highlight under-explored shortcomings of neuro-symbolic methods, representing a precious reference for future research. 

**Abstract (ZH)**: 神经符号人工智能的一个目标是利用背景知识来改进学习任务的性能。然而，现有的大多数框架集中在知识不随时间变化的简化场景上，并未涵盖时间维度。在这项工作中，我们考虑了更为挑战的知识驱动序列分类问题，在该问题中，必须在不同的时间步骤使用不同部分的知识，并且时间关系可用。我们的实验评估比较了多阶段神经符号和纯粹神经架构，并在新引入的基准框架上进行。结果显示了这一新颖设置的挑战性质，并强调了神经符号方法未充分利用的不足之处，为未来研究提供了宝贵的参考。 

---
# A Reputation System for Large Language Model-based Multi-agent Systems to Avoid the Tragedy of the Commons 

**Title (ZH)**: 基于大型语言模型的多智能体系统中的声誉系统以避免公地悲剧 

**Authors**: Siyue Ren, Wanli Fu, Xinkun Zou, Chen Shen, Yi Cai, Chen Chu, Zhen Wang, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05029)  

**Abstract**: The tragedy of the commons, where individual self-interest leads to collectively disastrous outcomes, is a pervasive challenge in human society. Recent studies have demonstrated that similar phenomena can arise in generative multi-agent systems (MASs). To address this challenge, this paper explores the use of reputation systems as a remedy. We propose RepuNet, a dynamic, dual-level reputation framework that models both agent-level reputation dynamics and system-level network evolution. Specifically, driven by direct interactions and indirect gossip, agents form reputations for both themselves and their peers, and decide whether to connect or disconnect other agents for future interactions. Through two distinct scenarios, we show that RepuNet effectively mitigates the 'tragedy of the commons', promoting and sustaining cooperation in generative MASs. Moreover, we find that reputation systems can give rise to rich emergent behaviors in generative MASs, such as the formation of cooperative clusters, the social isolation of exploitative agents, and the preference for sharing positive gossip rather than negative ones. 

**Abstract (ZH)**: 公地悲剧：个体自我利益导致集体灾难性后果的现象在人类社会中普遍存在，近期研究表明此类现象也可能出现在生成性多智能体系统中。为应对这一挑战，本文探讨了声誉系统作为一种解决方案的应用。我们提出了RepuNet，一种动态的双层声誉框架，用于建模智能体层面的声誉动态和系统层面的网络演化。通过两种不同的场景，我们展示了RepuNet如何有效地缓解“公地悲剧”，促进和维持生成性多智能体系统中的合作。此外，我们发现声誉系统可以在生成性多智能体系统中引发丰富的新涌现行为，如合作集群的形成、剥削性智能体的社会孤立以及更倾向于传播正面消息而非负面消息。 

---
# Foam-Agent: Towards Automated Intelligent CFD Workflows 

**Title (ZH)**: 泡沫代理：迈向自动化智能CFD工作流 

**Authors**: Ling Yue, Nithin Somasekharan, Yadi Cao, Shaowu Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.04997)  

**Abstract**: Computational Fluid Dynamics (CFD) is an essential simulation tool in various engineering disciplines, but it often requires substantial domain expertise and manual configuration, creating barriers to entry. We present Foam-Agent, a multi-agent framework that automates complex OpenFOAM-based CFD simulation workflows from natural language inputs. Our innovation includes (1) a hierarchical multi-index retrieval system with specialized indices for different simulation aspects, (2) a dependency-aware file generation system that provides consistency management across configuration files, and (3) an iterative error correction mechanism that diagnoses and resolves simulation failures without human intervention. Through comprehensive evaluation on the dataset of 110 simulation tasks, Foam-Agent achieves an 83.6% success rate with Claude 3.5 Sonnet, significantly outperforming existing frameworks (55.5% for MetaOpenFOAM and 37.3% for OpenFOAM-GPT). Ablation studies demonstrate the critical contribution of each system component, with the specialized error correction mechanism providing a 36.4% performance improvement. Foam-Agent substantially lowers the CFD expertise threshold while maintaining modeling accuracy, demonstrating the potential of specialized multi-agent systems to democratize access to complex scientific simulation tools. The code is public at this https URL 

**Abstract (ZH)**: 基于多agents的自然语言驱动的OpenFOAM复杂CFD仿真工作流自动化框架 

---
# Position: The AI Conference Peer Review Crisis Demands Author Feedback and Reviewer Rewards 

**Title (ZH)**: 位置：人工智能会议同行评审危机亟需作者反馈和评审员奖励 

**Authors**: Jaeho Kim, Yunseok Lee, Seulki Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.04966)  

**Abstract**: The peer review process in major artificial intelligence (AI) conferences faces unprecedented challenges with the surge of paper submissions (exceeding 10,000 submissions per venue), accompanied by growing concerns over review quality and reviewer responsibility. This position paper argues for the need to transform the traditional one-way review system into a bi-directional feedback loop where authors evaluate review quality and reviewers earn formal accreditation, creating an accountability framework that promotes a sustainable, high-quality peer review system. The current review system can be viewed as an interaction between three parties: the authors, reviewers, and system (i.e., conference), where we posit that all three parties share responsibility for the current problems. However, issues with authors can only be addressed through policy enforcement and detection tools, and ethical concerns can only be corrected through self-reflection. As such, this paper focuses on reforming reviewer accountability with systematic rewards through two key mechanisms: (1) a two-stage bi-directional review system that allows authors to evaluate reviews while minimizing retaliatory behavior, (2)a systematic reviewer reward system that incentivizes quality reviewing. We ask for the community's strong interest in these problems and the reforms that are needed to enhance the peer review process. 

**Abstract (ZH)**: 重大人工智能会议的peer review过程面临着前所未有的挑战：随着投稿量激增（每个会场超过10,000篇投稿）和对评审质量及评审员责任的日益关注，需要将传统的单向评审系统转变为双向反馈循环，其中作者评估评审质量，评审员获得正式认证，从而建立一个促进可持续高质量评审系统的问责框架。当前的评审系统可视为作者、评审员和系统（即会议）之间的交互，我们提出所有三方均需对当前问题负责。然而，作者方面的问题只能通过政策执行和检测工具来解决，伦理问题只能通过自我反省来纠正。因此，本文着重从系统性奖励机制改革评审员问责制，主要包括：（1）两阶段双向评审系统，允许作者评估评审而不助长报复行为；（2）系统性评审员奖励系统，激励高质量评审。我们呼吁社区对这些问题及其所需的改革给予强烈关注，以提升同行评审过程的质量。 

---
# CRAFT: Cultural Russian-Oriented Dataset Adaptation for Focused Text-to-Image Generation 

**Title (ZH)**: CRAFT：面向聚焦文本到图像生成的文化俄语导向数据集适配 

**Authors**: Viacheslav Vasilev, Vladimir Arkhipkin, Julia Agafonova, Tatiana Nikulina, Evelina Mironova, Alisa Shichanina, Nikolai Gerasimenko, Mikhail Shoytov, Denis Dimitrov  

**Link**: [PDF](https://arxiv.org/pdf/2505.04851)  

**Abstract**: Despite the fact that popular text-to-image generation models cope well with international and general cultural queries, they have a significant knowledge gap regarding individual cultures. This is due to the content of existing large training datasets collected on the Internet, which are predominantly based on Western European or American popular culture. Meanwhile, the lack of cultural adaptation of the model can lead to incorrect results, a decrease in the generation quality, and the spread of stereotypes and offensive content. In an effort to address this issue, we examine the concept of cultural code and recognize the critical importance of its understanding by modern image generation models, an issue that has not been sufficiently addressed in the research community to date. We propose the methodology for collecting and processing the data necessary to form a dataset based on the cultural code, in particular the Russian one. We explore how the collected data affects the quality of generations in the national domain and analyze the effectiveness of our approach using the Kandinsky 3.1 text-to-image model. Human evaluation results demonstrate an increase in the level of awareness of Russian culture in the model. 

**Abstract (ZH)**: 尽管现有的文本到图像生成模型在处理国际和通用文化查询方面表现良好，但在处理个体文化方面存在明显的知识缺口。这主要是因为现有大型训练数据集的内容主要基于西方欧洲或美国的流行文化，收集于互联网。同时，模型的文化适应不足可能导致错误结果、生成质量下降以及刻板印象和有害内容的传播。为解决这一问题，我们探讨了文化代码的概念，并认识到现代图像生成模型对其理解的迫切重要性，这一问题至今尚未在研究界得到充分关注。我们提出了一种收集和处理数据的方法，以形成基于文化代码、特别是俄罗斯文化代码的数据库。我们探究了收集的数据如何影响国内领域的生成质量，并使用Kandinsky 3.1文本到图像模型分析我们方法的有效性。人类评估结果显示，模型对俄罗斯文化的意识水平有所提高。 

---
# Is there Value in Reinforcement Learning? 

**Title (ZH)**: 强化学习是否有价值？ 

**Authors**: Lior Fox, Yonatan Loewenstein  

**Link**: [PDF](https://arxiv.org/pdf/2505.04822)  

**Abstract**: Action-values play a central role in popular Reinforcement Learing (RL) models of behavior. Yet, the idea that action-values are explicitly represented has been extensively debated. Critics had therefore repeatedly suggested that policy-gradient (PG) models should be favored over value-based (VB) ones, as a potential solution for this dilemma. Here we argue that this solution is unsatisfying. This is because PG methods are not, in fact, "Value-free" -- while they do not rely on an explicit representation of Value for acting (stimulus-response mapping), they do require it for learning. Hence, switching to PG models is, per se, insufficient for eliminating Value from models of behavior. More broadly, the requirement for a representation of Value stems from the underlying assumptions regarding the optimization objective posed by the standard RL framework, not from the particular algorithm chosen to solve it. Previous studies mostly took these standard RL assumptions for granted, as part of their conceptualization or problem modeling, while debating the different methods used to optimize it (i.e., PG or VB). We propose that, instead, the focus of the debate should shift to critically evaluating the underlying modeling assumptions. Such evaluation is particularly important from an experimental perspective. Indeed, the very notion of Value must be reconsidered when standard assumptions (e.g., risk neutrality, full-observability, Markovian environment, exponential discounting) are relaxed, as is likely in natural settings. Finally, we use the Value debate as a case study to argue in favor of a more nuanced, algorithmic rather than statistical, view of what constitutes "a model" in cognitive sciences. Our analysis suggests that besides "parametric" statistical complexity, additional aspects such as computational complexity must also be taken into account when evaluating model complexity. 

**Abstract (ZH)**: 行动值在行为的强化学习模型中扮演核心角色，然而行动值显式表示的想法一直受到广泛讨论。我们argue指出，改用策略梯度模型并不能从根本上消除行动值，因为策略梯度方法尽管不依赖于行动的显式价值表示（刺激-响应映射），但在学习过程中仍需依赖价值表示。因此，切换到策略梯度模型本身不足以消除行为模型中的价值因素。更广泛而言，价值表示的需求源自标准强化学习框架所提出的优化目标的潜在假设，而非所选择的具体算法。以往研究大多假定这些标准假设，并将其作为概念化或问题建模的一部分，而争论的是不同的优化方法（即策略梯度或基于价值的方法）。我们提出，争论的焦点应该转移到对潜在建模假设的批判性评估上。这种评估从实验角度来看尤为重要。事实上，当放松标准假设（如无风险偏好、完全可观测性、马尔可夫环境、指数折扣）时，必须重新考虑价值这一概念，这在自然环境中很可能是事实。最后，我们使用价值争议作为案例研究，主张认知科学中“模型”的构成应采取更为细腻的方法论视角，而不是单纯的统计视角。我们的分析表明，在评估模型复杂度时，除统计复杂性外，还需要考虑计算复杂性等因素。 

---
# Dynamic Location Search for Identifying Maximum Weighted Independent Sets in Complex Networks 

**Title (ZH)**: 复杂网络中最大加权独立集的动态位置搜索 

**Authors**: Enqiang Zhu, Chenkai Hao, Chanjuan Liu, Yongsheng Rao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04674)  

**Abstract**: While Artificial intelligence (AI), including Generative AI, are effective at generating high-quality traffic data and optimization solutions in intelligent transportation systems (ITSs), these techniques often demand significant training time and computational resources, especially in large-scale and complex scenarios. To address this, we introduce a novel and efficient algorithm for solving the maximum weighted independent set (MWIS) problem, which can be used to model many ITSs applications, such as traffic signal control and vehicle routing. Given the NP-hard nature of the MWIS problem, our proposed algorithm, DynLS, incorporates three key innovations to solve it effectively. First, it uses a scores-based adaptive vertex perturbation (SAVP) technique to accelerate convergence, particularly in sparse graphs. Second, it includes a region location mechanism (RLM) to help escape local optima by dynamically adjusting the search space. Finally, it employs a novel variable neighborhood descent strategy, ComLS, which combines vertex exchange strategies with a reward mechanism to guide the search toward high-quality solutions. Our experimental results demonstrate DynLS's superior performance, consistently delivering high-quality solutions within 1000 seconds. DynLS outperformed five leading algorithms across 360 test instances, achieving the best solution for 350 instances and surpassing the second-best algorithm, Cyclic-Fast, by 177 instances. Moreover, DynLS matched Cyclic-Fast's convergence speed, highlighting its efficiency and practicality. This research represents a significant advancement in heuristic algorithms for the MWIS problem, offering a promising approach to aid AI techniques in optimizing intelligent transportation systems. 

**Abstract (ZH)**: 一种用于解决最大加权独立集问题的高效算法DynLS及其在智能交通系统中的应用 

---
# Flow-GRPO: Training Flow Matching Models via Online RL 

**Title (ZH)**: 基于在线强化学习的流匹配模型训练方法 

**Authors**: Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05470)  

**Abstract**: We propose Flow-GRPO, the first method integrating online reinforcement learning (RL) into flow matching models. Our approach uses two key strategies: (1) an ODE-to-SDE conversion that transforms a deterministic Ordinary Differential Equation (ODE) into an equivalent Stochastic Differential Equation (SDE) that matches the original model's marginal distribution at all timesteps, enabling statistical sampling for RL exploration; and (2) a Denoising Reduction strategy that reduces training denoising steps while retaining the original inference timestep number, significantly improving sampling efficiency without performance degradation. Empirically, Flow-GRPO is effective across multiple text-to-image tasks. For complex compositions, RL-tuned SD3.5 generates nearly perfect object counts, spatial relations, and fine-grained attributes, boosting GenEval accuracy from $63\%$ to $95\%$. In visual text rendering, its accuracy improves from $59\%$ to $92\%$, significantly enhancing text generation. Flow-GRPO also achieves substantial gains in human preference alignment. Notably, little to no reward hacking occurred, meaning rewards did not increase at the cost of image quality or diversity, and both remained stable in our experiments. 

**Abstract (ZH)**: Flow-GRPO：将在线强化学习集成到流匹配模型中的首次尝试 

---
# Reasoning Models Don't Always Say What They Think 

**Title (ZH)**: Reasoning Models 不总是说它们想说的 

**Authors**: Yanda Chen, Joe Benton, Ansh Radhakrishnan, Jonathan Uesato, Carson Denison, John Schulman, Arushi Somani, Peter Hase, Misha Wagner, Fabien Roger, Vlad Mikulik, Samuel R. Bowman, Jan Leike, Jared Kaplan, Ethan Perez  

**Link**: [PDF](https://arxiv.org/pdf/2505.05410)  

**Abstract**: Chain-of-thought (CoT) offers a potential boon for AI safety as it allows monitoring a model's CoT to try to understand its intentions and reasoning processes. However, the effectiveness of such monitoring hinges on CoTs faithfully representing models' actual reasoning processes. We evaluate CoT faithfulness of state-of-the-art reasoning models across 6 reasoning hints presented in the prompts and find: (1) for most settings and models tested, CoTs reveal their usage of hints in at least 1% of examples where they use the hint, but the reveal rate is often below 20%, (2) outcome-based reinforcement learning initially improves faithfulness but plateaus without saturating, and (3) when reinforcement learning increases how frequently hints are used (reward hacking), the propensity to verbalize them does not increase, even without training against a CoT monitor. These results suggest that CoT monitoring is a promising way of noticing undesired behaviors during training and evaluations, but that it is not sufficient to rule them out. They also suggest that in settings like ours where CoT reasoning is not necessary, test-time monitoring of CoTs is unlikely to reliably catch rare and catastrophic unexpected behaviors. 

**Abstract (ZH)**: Chain-of-Thought监控对于AI安全的潜在益处及其局限性：基于最新推理模型的评估 

---
# CART-ELC: Oblique Decision Tree Induction via Exhaustive Search 

**Title (ZH)**: CART-ELC：基于穷尽搜索的斜向决策树诱导 

**Authors**: Andrew D. Laack  

**Link**: [PDF](https://arxiv.org/pdf/2505.05402)  

**Abstract**: Oblique decision trees have attracted attention due to their potential for improved classification performance over traditional axis-aligned decision trees. However, methods that rely on exhaustive search to find oblique splits face computational challenges. As a result, they have not been widely explored. We introduce a novel algorithm, Classification and Regression Tree - Exhaustive Linear Combinations (CART-ELC), for inducing oblique decision trees that performs an exhaustive search on a restricted set of hyperplanes. We then investigate the algorithm's computational complexity and its predictive capabilities. Our results demonstrate that CART-ELC consistently achieves competitive performance on small datasets, often yielding statistically significant improvements in classification accuracy relative to existing decision tree induction algorithms, while frequently producing shallower, simpler, and thus more interpretable trees. 

**Abstract (ZH)**: 斜决策树由于潜在的分类性能改进而受到了关注，但依赖于详尽搜索以找到斜分割的方法面临计算挑战，因此它们并未得到广泛研究。我们提出了一种新的算法——分类和回归树-详尽线性组合（CART-ELC），该算法在受限的超平面集上进行详尽搜索以诱导斜决策树。然后，我们探讨了该算法的计算复杂性和预测能力。我们的结果表明，CART-ELC在小型数据集上能够一致地实现竞争性能，通常在分类准确性方面相对于现有决策树诱导算法表现出统计意义上的显著改进，同时经常生成更浅、更简单、因此更具解释性的树结构。 

---
# Threshold Modulation for Online Test-Time Adaptation of Spiking Neural Networks 

**Title (ZH)**: 基于阈值调制的在线测试时SNN的自适应调整 

**Authors**: Kejie Zhao, Wenjia Hua, Aiersi Tuerhong, Luziwei Leng, Yuxin Ma, Qinghua Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.05375)  

**Abstract**: Recently, spiking neural networks (SNNs), deployed on neuromorphic chips, provide highly efficient solutions on edge devices in different scenarios. However, their ability to adapt to distribution shifts after deployment has become a crucial challenge. Online test-time adaptation (OTTA) offers a promising solution by enabling models to dynamically adjust to new data distributions without requiring source data or labeled target samples. Nevertheless, existing OTTA methods are largely designed for traditional artificial neural networks and are not well-suited for SNNs. To address this gap, we propose a low-power, neuromorphic chip-friendly online test-time adaptation framework, aiming to enhance model generalization under distribution shifts. The proposed approach is called Threshold Modulation (TM), which dynamically adjusts the firing threshold through neuronal dynamics-inspired normalization, being more compatible with neuromorphic hardware. Experimental results on benchmark datasets demonstrate the effectiveness of this method in improving the robustness of SNNs against distribution shifts while maintaining low computational cost. The proposed method offers a practical solution for online test-time adaptation of SNNs, providing inspiration for the design of future neuromorphic chips. The demo code is available at this http URL. 

**Abstract (ZH)**: 近来，部署在神经形态芯片上的脉冲神经网络（SNNs）在不同场景下的边缘设备上提供了高效的解决方案。然而，在部署后适应分布偏移的能力已成为一个关键挑战。在线测试时适应（OTTA）通过使模型能够动态调整以适应新的数据分布，而无需源数据或标记的目标样本，提供了一种有前景的解决方案。然而，现有的OTTA方法主要针对传统的人工神经网络设计，不适用于SNNs。为了解决这一问题，我们提出了一种低功耗、适用于神经形态芯片的在线测试时适应框架，旨在在分布偏移下增强模型的泛化能力。该提出的方案称为阈值调制（TM），通过神经元动力学启发的归一化动态调整放电阈值，更适配于神经形态硬件。基准数据集上的实验结果证明了该方法在提高SNNs对分布偏移的鲁棒性的同时保持了低计算成本的有效性。该方法提供了一种实际的解决方案，用于SNNs的在线测试时适应，并为未来神经形态芯片的设计提供了启示。完整的演示代码可在以下网址获取。 

---
# High-fidelity Grain Growth Modeling: Leveraging Deep Learning for Fast Computations 

**Title (ZH)**: 高保真晶粒生长建模：利用深度学习实现快速计算 

**Authors**: Pungponhavoan Tep, Marc Bernacki  

**Link**: [PDF](https://arxiv.org/pdf/2505.05354)  

**Abstract**: Grain growth simulation is crucial for predicting metallic material microstructure evolution during annealing and resulting final mechanical properties, but traditional partial differential equation-based methods are computationally expensive, creating bottlenecks in materials design and manufacturing. In this work, we introduce a machine learning framework that combines a Convolutional Long Short-Term Memory networks with an Autoencoder to efficiently predict grain growth evolution. Our approach captures both spatial and temporal aspects of grain evolution while encoding high-dimensional grain structure data into a compact latent space for pattern learning, enhanced by a novel composite loss function combining Mean Squared Error, Structural Similarity Index Measurement, and Boundary Preservation to maintain structural integrity of grain boundary topology of the prediction. Results demonstrated that our machine learning approach accelerates grain growth prediction by up to \SI{89}{\times} faster, reducing computation time from \SI{10}{\minute} to approximately \SI{10}{\second} while maintaining high-fidelity predictions. The best model (S-30-30) achieving a structural similarity score of \SI{86.71}{\percent} and mean grain size error of just \SI{0.07}{\percent}. All models accurately captured grain boundary topology, morphology, and size distributions. This approach enables rapid microstructural prediction for applications where conventional simulations are prohibitively time-consuming, potentially accelerating innovation in materials science and manufacturing. 

**Abstract (ZH)**: 基于卷积长短期记忆网络与自动编码器的机器学习框架加速晶粒生长演化预测 

---
# Scalable Chain of Thoughts via Elastic Reasoning 

**Title (ZH)**: 弹性推理驱动的可扩展思维链 

**Authors**: Yuhui Xu, Hanze Dong, Lei Wang, Doyen Sahoo, Junnan Li, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.05315)  

**Abstract**: Large reasoning models (LRMs) have achieved remarkable progress on complex tasks by generating extended chains of thought (CoT). However, their uncontrolled output lengths pose significant challenges for real-world deployment, where inference-time budgets on tokens, latency, or compute are strictly constrained. We propose Elastic Reasoning, a novel framework for scalable chain of thoughts that explicitly separates reasoning into two phases--thinking and solution--with independently allocated budgets. At test time, Elastic Reasoning prioritize that completeness of solution segments, significantly improving reliability under tight resource constraints. To train models that are robust to truncated thinking, we introduce a lightweight budget-constrained rollout strategy, integrated into GRPO, which teaches the model to reason adaptively when the thinking process is cut short and generalizes effectively to unseen budget constraints without additional training. Empirical results on mathematical (AIME, MATH500) and programming (LiveCodeBench, Codeforces) benchmarks demonstrate that Elastic Reasoning performs robustly under strict budget constraints, while incurring significantly lower training cost than baseline methods. Remarkably, our approach also produces more concise and efficient reasoning even in unconstrained settings. Elastic Reasoning offers a principled and practical solution to the pressing challenge of controllable reasoning at scale. 

**Abstract (ZH)**: Elastic Reasoning：一种可扩展的可控推理框架 

---
# T-T: Table Transformer for Tagging-based Aspect Sentiment Triplet Extraction 

**Title (ZH)**: T-T: 基于标签的aspect情感三元组提取的表格变换器 

**Authors**: Kun Peng, Chaodong Tong, Cong Cao, Hao Peng, Qian Li, Guanlin Wu, Lei Jiang, Yanbing Liu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05271)  

**Abstract**: Aspect sentiment triplet extraction (ASTE) aims to extract triplets composed of aspect terms, opinion terms, and sentiment polarities from given sentences. The table tagging method is a popular approach to addressing this task, which encodes a sentence into a 2-dimensional table, allowing for the tagging of relations between any two words. Previous efforts have focused on designing various downstream relation learning modules to better capture interactions between tokens in the table, revealing that a stronger capability to capture relations can lead to greater improvements in the model. Motivated by this, we attempt to directly utilize transformer layers as downstream relation learning modules. Due to the powerful semantic modeling capability of transformers, it is foreseeable that this will lead to excellent improvement. However, owing to the quadratic relation between the length of the table and the length of the input sentence sequence, using transformers directly faces two challenges: overly long table sequences and unfair local attention interaction. To address these challenges, we propose a novel Table-Transformer (T-T) for the tagging-based ASTE method. Specifically, we introduce a stripe attention mechanism with a loop-shift strategy to tackle these challenges. The former modifies the global attention mechanism to only attend to a 2-dimensional local attention window, while the latter facilitates interaction between different attention windows. Extensive and comprehensive experiments demonstrate that the T-T, as a downstream relation learning module, achieves state-of-the-art performance with lower computational costs. 

**Abstract (ZH)**: 面向表格的aspect情感三元组提取（Table-based Aspect Sentiment Triplet Extraction, ASTE）旨在从给定句子中提取由aspect术语、意见术语和情感极性组成的三元组。表标记方法是解决这一任务的一种流行方法，该方法将句子编码为二维表，允许对任何两个词之间的关系进行标注。先前的努力集中在设计各种下游关系学习模块，以更好地捕捉表中token之间的交互，揭示出捕捉关系的能力越强，模型的改进程度越大。受此启发，我们尝试直接利用transformer层作为下游关系学习模块。由于transformer强大的语义建模能力，这预计将导致显著的改进。然而，由于表的长度与输入句子序列长度之间存在二次关系，直接使用transformer面临两个挑战：过长的表序列和不公的局部注意力交互。为解决这些挑战，我们提出了一个新颖的Table-Transformer（T-T）用于基于表标记的ASTE方法。具体地，我们引入了一种带循环移位策略的条带注意力机制以应对这些挑战。前者将全局注意力机制修改为仅关注二维局部注意力窗口，而后者促进了不同注意力窗口之间的交互。详尽且综合的实验表明，作为下游关系学习模块，T-T实现了最先进的性能，并具有更低的计算成本。 

---
# Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration 

**Title (ZH)**: 基于状态建模和对抗性探索的协作多智能体 reinforcement learning 改进方法 

**Authors**: Andreas Kontogiannis, Konstantinos Papathanasiou, Yi Shen, Giorgos Stamou, Michael M. Zavlanos, George Vouros  

**Link**: [PDF](https://arxiv.org/pdf/2505.05262)  

**Abstract**: Learning to cooperate in distributed partially observable environments with no communication abilities poses significant challenges for multi-agent deep reinforcement learning (MARL). This paper addresses key concerns in this domain, focusing on inferring state representations from individual agent observations and leveraging these representations to enhance agents' exploration and collaborative task execution policies. To this end, we propose a novel state modelling framework for cooperative MARL, where agents infer meaningful belief representations of the non-observable state, with respect to optimizing their own policies, while filtering redundant and less informative joint state information. Building upon this framework, we propose the MARL SMPE algorithm. In SMPE, agents enhance their own policy's discriminative abilities under partial observability, explicitly by incorporating their beliefs into the policy network, and implicitly by adopting an adversarial type of exploration policies which encourages agents to discover novel, high-value states while improving the discriminative abilities of others. Experimentally, we show that SMPE outperforms state-of-the-art MARL algorithms in complex fully cooperative tasks from the MPE, LBF, and RWARE benchmarks. 

**Abstract (ZH)**: 在无通信能力的分布式部分可观测环境中学习合作对多智能体深度强化学习（MARL）提出了重大挑战。本文针对该领域的关键问题，集中于从个体智能体观察中推断状态表示，并利用这些表示增强智能体的探索和协作任务执行策略。为此，我们提出了一种新的合作MARL的状态建模框架，其中智能体推断与优化自身策略相关的有意义的信念表示，同时过滤冗余和低信息量的联合状态信息。在此基础上，我们提出了MARL SMPE算法。在SMPE中，智能体通过将自身的信念融入策略网络中，明确地增强自身策略的鉴别能力，并通过采用一种对抗型的探索策略，隐式地促进自身和他人鉴别能力的提升，同时发现新的、高价值状态。实验结果显示，SMPE在MPE、LBF和RWARE基准中的复杂完全协同任务中优于最先进的MARL算法。 

---
# Put CASH on Bandits: A Max K-Armed Problem for Automated Machine Learning 

**Title (ZH)**: 把CASH onto bandits：一个自动机器学习的极大K臂问题 

**Authors**: Amir Rezaei Balef, Claire Vernade, Katharina Eggensperger  

**Link**: [PDF](https://arxiv.org/pdf/2505.05226)  

**Abstract**: The Combined Algorithm Selection and Hyperparameter optimization (CASH) is a challenging resource allocation problem in the field of AutoML. We propose MaxUCB, a max $k$-armed bandit method to trade off exploring different model classes and conducting hyperparameter optimization. MaxUCB is specifically designed for the light-tailed and bounded reward distributions arising in this setting and, thus, provides an efficient alternative compared to classic max $k$-armed bandit methods assuming heavy-tailed reward distributions. We theoretically and empirically evaluate our method on four standard AutoML benchmarks, demonstrating superior performance over prior approaches. 

**Abstract (ZH)**: 结合算法选择与超参数优化的MaxUCB方法：一种适用于AutoML领域的高效资源分配策略 

---
# Incentive-Aware Machine Learning; Robustness, Fairness, Improvement & Causality 

**Title (ZH)**: 激励aware机器学习；稳健性、公平性、改进与因果性 

**Authors**: Chara Podimata  

**Link**: [PDF](https://arxiv.org/pdf/2505.05211)  

**Abstract**: The article explores the emerging domain of incentive-aware machine learning (ML), which focuses on algorithmic decision-making in contexts where individuals can strategically modify their inputs to influence outcomes. It categorizes the research into three perspectives: robustness, aiming to design models resilient to "gaming"; fairness, analyzing the societal impacts of such systems; and improvement/causality, recognizing situations where strategic actions lead to genuine personal or societal improvement. The paper introduces a unified framework encapsulating models for these perspectives, including offline, online, and causal settings, and highlights key challenges such as differentiating between gaming and improvement and addressing heterogeneity among agents. By synthesizing findings from diverse works, we outline theoretical advancements and practical solutions for robust, fair, and causally-informed incentive-aware ML systems. 

**Abstract (ZH)**: 本文探讨了激励感知机器学习这一新兴领域，重点关注个体可以通过战略性修改输入来影响结果的环境中的算法决策。文章从三个方面分类研究：鲁棒性，旨在设计能够在“游戏”面前坚固的模型；公平性，分析此类系统的社会影响；以及改进/因果性，认识到战略性行为可能导致真实的个人或社会改进。文章介绍了一个统一框架，涵盖这些视角下的模型，包括离线、在线和因果设置，并强调了识别“游戏”与改进之间的差异和应对代理异质性等关键挑战。通过综合多元研究的发现，本文概述了稳健、公平和因果驱动的激励感知机器学习系统的理论进展和实用解决方案。 

---
# LAPSO: A Unified Optimization View for Learning-Augmented Power System Operations 

**Title (ZH)**: LAPSO：学习增强电力系统运营的统一优化视角 

**Authors**: Wangkun Xu, Zhongda Chu, Fei Teng  

**Link**: [PDF](https://arxiv.org/pdf/2505.05203)  

**Abstract**: With the high penetration of renewables, traditional model-based power system operation is challenged to deliver economic, stable, and robust decisions. Machine learning has emerged as a powerful modeling tool for capturing complex dynamics to address these challenges. However, its separate design often lacks systematic integration with existing methods. To fill the gap, this paper proposes a holistic framework of Learning-Augmented Power System Operations (LAPSO, pronounced as Lap-So). Adopting a native optimization perspective, LAPSO is centered on the operation stage and aims to break the boundary between temporally siloed power system tasks, such as forecast, operation and control, while unifying the objectives of machine learning and model-based optimizations at both training and inference stages. Systematic analysis and simulations demonstrate the effectiveness of applying LAPSO in designing new integrated algorithms, such as stability-constrained optimization (SCO) and objective-based forecasting (OBF), while enabling end-to-end tracing of different sources of uncertainties. In addition, a dedicated Python package-lapso is introduced to automatically augment existing power system optimization models with learnable components. All code and data are available at this https URL. 

**Abstract (ZH)**: 随着可再生能源的高渗透率，传统的基于模型的电力系统运行模式面临着提供经济、稳定和 robust 决策的挑战。机器学习作为一种强大的建模工具，被用于捕捉复杂动力学以应对这些挑战。然而，其单独设计往往缺乏与现有方法的系统集成。为了弥补这一差距，本文提出了一种整体框架，称为学习增强电力系统运行（LAPSO，发音为 Lap-So）。从原生优化视角出发，LAPSO以运行阶段为中心，旨在打破时间上分离的电力系统任务（如预测、运行和控制）之间的边界，同时在训练和推理阶段统一机器学习和基于模型优化的目标。系统分析和仿真表明，LAPSO在设计新的集成算法（如稳定性约束优化（SCO）和目标导向预测（OBF））方面有效，同时允许对不同来源的不确定性进行端到端跟踪。此外，还介绍了一个专门的 Python 包 lapso，用于自动增强现有的电力系统优化模型中的可学习组件。所有代码和数据可在以下网址获得。 

---
# Concept-Based Unsupervised Domain Adaptation 

**Title (ZH)**: 基于概念的无监督领域适应 

**Authors**: Xinyue Xu, Yueying Hu, Hui Tang, Yi Qin, Lu Mi, Hao Wang, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.05195)  

**Abstract**: Concept Bottleneck Models (CBMs) enhance interpretability by explaining predictions through human-understandable concepts but typically assume that training and test data share the same distribution. This assumption often fails under domain shifts, leading to degraded performance and poor generalization. To address these limitations and improve the robustness of CBMs, we propose the Concept-based Unsupervised Domain Adaptation (CUDA) framework. CUDA is designed to: (1) align concept representations across domains using adversarial training, (2) introduce a relaxation threshold to allow minor domain-specific differences in concept distributions, thereby preventing performance drop due to over-constraints of these distributions, (3) infer concepts directly in the target domain without requiring labeled concept data, enabling CBMs to adapt to diverse domains, and (4) integrate concept learning into conventional domain adaptation (DA) with theoretical guarantees, improving interpretability and establishing new benchmarks for DA. Experiments demonstrate that our approach significantly outperforms the state-of-the-art CBM and DA methods on real-world datasets. 

**Abstract (ZH)**: 基于概念的无监督领域适应 (CUDA) 框架 

---
# Stochastic Variational Propagation: Local, Scalable and Efficient Alternative to Backpropagation 

**Title (ZH)**: 随机变分传播：反向传播的局部、可扩展且efficient的替代方法 

**Authors**: Bojian Yin, Federico Corradi  

**Link**: [PDF](https://arxiv.org/pdf/2505.05181)  

**Abstract**: Backpropagation (BP) is the cornerstone of deep learning, but its reliance on global gradient synchronization limits scalability and imposes significant memory overhead. We propose Stochastic Variational Propagation (SVP), a scalable alternative that reframes training as hierarchical variational inference. SVP treats layer activations as latent variables and optimizes local Evidence Lower Bounds (ELBOs), enabling independent, local updates while preserving global coherence. However, directly applying KL divergence in layer-wise ELBOs risks inter-layer's representation collapse due to excessive compression. To prevent this, SVP projects activations into low-dimensional spaces via fixed random matrices, ensuring information preservation and representational diversity. Combined with a feature alignment loss for inter-layer consistency, SVP achieves competitive accuracy with BP across diverse architectures (MLPs, CNNs, Transformers) and datasets (MNIST to ImageNet), reduces memory usage by up to 4x, and significantly improves scalability. More broadly, SVP introduces a probabilistic perspective to deep representation learning, opening pathways toward more modular and interpretable neural network design. 

**Abstract (ZH)**: Stochastic Variational Propagation: A Scalable Alternative for Deep Learning 

---
# Guiding Evolutionary AutoEncoder Training with Activation-Based Pruning Operators 

**Title (ZH)**: 基于激活值的剪枝操作引导自动编码器进化训练 

**Authors**: Steven Jorgensen, Erik Hemberg, Jamal Toutouh, Una-May O'Reilly  

**Link**: [PDF](https://arxiv.org/pdf/2505.05138)  

**Abstract**: This study explores a novel approach to neural network pruning using evolutionary computation, focusing on simultaneously pruning the encoder and decoder of an autoencoder. We introduce two new mutation operators that use layer activations to guide weight pruning. Our findings reveal that one of these activation-informed operators outperforms random pruning, resulting in more efficient autoencoders with comparable performance to canonically trained models. Prior work has established that autoencoder training is effective and scalable with a spatial coevolutionary algorithm that cooperatively coevolves a population of encoders with a population of decoders, rather than one autoencoder. We evaluate how the same activity-guided mutation operators transfer to this context. We find that random pruning is better than guided pruning, in the coevolutionary setting. This suggests activation-based guidance proves more effective in low-dimensional pruning environments, where constrained sample spaces can lead to deviations from true uniformity in randomization. Conversely, population-driven strategies enhance robustness by expanding the total pruning dimensionality, achieving statistically uniform randomness that better preserves system dynamics. We experiment with pruning according to different schedules and present best combinations of operator and schedule for the canonical and coevolving populations cases. 

**Abstract (ZH)**: 该研究探讨了一种使用进化计算进行神经网络剪枝的新型方法，重点关注同时剪枝自编码器的编码器和解码器。我们引入了两种新的变异算子，利用层激活来指导权重剪枝。我们的研究发现，其中一种基于激活的算子优于随机剪枝，从而实现了更具效率的自编码器，并且在性能上与传统训练模型相当。之前的研究表明，使用时空协作演化算法可以有效且可扩展地训练自编码器，该算法协同演化编码器和解码器种群，而非单个自编码器。我们评估了相同活动引导的变异算子在这种上下文中的应用。我们发现，在协同演化设置中，随机剪枝优于引导剪枝，这表明基于激活的引导在低维度剪枝环境中更为有效，受限的样本空间可能导致随机化偏离真正的均匀性。相反，种群驱动的策略通过扩展总剪枝维度来增强鲁棒性，实现统计上均匀的随机性，从而更好地保持系统动力学。我们根据不同的剪枝时间表进行实验，并展示了在标准和协同演化种群情况下最优的算子和时间表组合。 

---
# Beyond Low-rank Decomposition: A Shortcut Approach for Efficient On-Device Learning 

**Title (ZH)**: 超越低秩分解：一种高效的本地设备学习快捷方法 

**Authors**: Le-Trung Nguyen, Ael Quelennec, Van-Tam Nguyen, Enzo Tartaglione  

**Link**: [PDF](https://arxiv.org/pdf/2505.05086)  

**Abstract**: On-device learning has emerged as a promising direction for AI development, particularly because of its potential to reduce latency issues and mitigate privacy risks associated with device-server communication, while improving energy efficiency. Despite these advantages, significant memory and computational constraints still represent major challenges for its deployment. Drawing on previous studies on low-rank decomposition methods that address activation memory bottlenecks in backpropagation, we propose a novel shortcut approach as an alternative. Our analysis and experiments demonstrate that our method can reduce activation memory usage, even up to $120.09\times$ compared to vanilla training, while also reducing overall training FLOPs up to $1.86\times$ when evaluated on traditional benchmarks. 

**Abstract (ZH)**: 设备端学习已成为AI发展的一个有前途的方向，特别是在减少设备-服务器通信带来的延迟问题和隐私风险、提升能源效率方面展现出潜力。尽管存在这些优势，内存和计算能力的限制仍然是其广泛应用的主要挑战。基于以往针对反向传播中激活记忆瓶颈的低秩分解方法研究，我们提出了一种新颖的捷径方法作为替代方案。我们的分析和实验表明，该方法可以减少激活记忆使用量，最多可达120.09倍，与标准训练相比，同时在传统基准上将整体训练FLOPs降低至最多1.86倍。 

---
# Teochew-Wild: The First In-the-wild Teochew Dataset with Orthographic Annotations 

**Title (ZH)**: 潮州野生：首个带有音节注释的潮州方言野外数据集 

**Authors**: Linrong Pan, Chenglong Jiang, Gaoze Hou, Ying Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.05056)  

**Abstract**: This paper reports the construction of the Teochew-Wild, a speech corpus of the Teochew dialect. The corpus includes 18.9 hours of in-the-wild Teochew speech data from multiple speakers, covering both formal and colloquial expressions, with precise orthographic and pinyin annotations. Additionally, we provide supplementary text processing tools and resources to propel research and applications in speech tasks for this low-resource language, such as automatic speech recognition (ASR) and text-to-speech (TTS). To the best of our knowledge, this is the first publicly available Teochew dataset with accurate orthographic annotations. We conduct experiments on the corpus, and the results validate its effectiveness in ASR and TTS tasks. 

**Abstract (ZH)**: 本论文报告了潮州野生语料库的构建，该语料库包含来自多位讲者的18.9小时的潮州方言野生语音数据，涵盖正式和非正式表达，并附有精确的拼写和拼音注释。此外，我们还提供了辅助文本处理工具和资源，以促进对这种低资源语言的语音任务研究和应用，如自动语音识别（ASR）和文本到语音（TTS）。据我们所知，这是首个包含准确拼写注释的公开潮州方言数据集。我们在语料库上进行了实验，结果验证了其在ASR和TTS任务中的有效性。 

---
# Generating Reliable Synthetic Clinical Trial Data: The Role of Hyperparameter Optimization and Domain Constraints 

**Title (ZH)**: 生成可靠的合成临床试验数据：超参数优化和领域约束的作用 

**Authors**: Waldemar Hahn, Jan-Niklas Eckardt, Christoph Röllig, Martin Sedlmayr, Jan Moritz Middeke, Markus Wolfien  

**Link**: [PDF](https://arxiv.org/pdf/2505.05019)  

**Abstract**: The generation of synthetic clinical trial data offers a promising approach to mitigating privacy concerns and data accessibility limitations in medical research. However, ensuring that synthetic datasets maintain high fidelity, utility, and adherence to domain-specific constraints remains a key challenge. While hyperparameter optimization (HPO) has been shown to improve generative model performance, the effectiveness of different optimization strategies for synthetic clinical data remains unclear. This study systematically evaluates four HPO strategies across eight generative models, comparing single-metric optimization against compound metric optimization approaches. Our results demonstrate that HPO consistently improves synthetic data quality, with TVAE, CTGAN, and CTAB-GAN+ achieving improvements of up to 60%, 39%, and 38%, respectively. Compound metric optimization outperformed single-metric strategies, producing more balanced and generalizable synthetic datasets. Interestingly, HPO alone is insufficient to ensure clinically valid synthetic data, as all models exhibited violations of fundamental survival constraints. Preprocessing and postprocessing played a crucial role in reducing these violations, as models lacking robust processing steps produced invalid data in up to 61% of cases. These findings underscore the necessity of integrating explicit domain knowledge alongside HPO to create high quality synthetic datasets. Our study provides actionable recommendations for improving synthetic data generation, with future research needed to refine metric selection and validate these findings on larger datasets to enhance clinical applicability. 

**Abstract (ZH)**: 合成临床试验数据的生成为缓解医疗研究中的隐私担忧和数据可访问性限制提供了有前景的方法。然而，确保合成数据集保持高保真度、实用性和遵循领域特定约束仍然是一个关键挑战。虽然超参数优化（HPO）已被证明可以提高生成模型的性能，但不同优化策略对合成临床数据的有效性尚不清楚。本研究系统地评估了四种HPO策略在八种生成模型中的应用，比较了单指标优化与复合指标优化方法。研究结果表明，HPO一致地提高了合成数据的质量，其中TVAE、CTGAN和CTAB-GAN+分别实现了多达60%、39%和38%的改进。复合指标优化优于单指标策略，产生了更平衡和更具通用性的合成数据集。有趣的是，仅HPO不足以确保临床有效的合成数据，所有模型均表现出违反基本生存约束的现象。预处理和后处理在减少这些违反现象方面发挥了关键作用，缺乏稳健处理步骤的模型在多达61%的情况下产生了无效数据。这些发现强调了在HPO过程中结合显式领域知识的必要性，以生成高质量的合成数据集。本研究提供了改进合成数据生成的可操作建议，未来研究需要进一步细化指标选择并在更大规模的数据集上验证这些发现，以增强临床应用性。 

---
# An Agent-Based Modeling Approach to Free-Text Keyboard Dynamics for Continuous Authentication 

**Title (ZH)**: 基于代理模型的自由文本键盘动态建模连续认证方法 

**Authors**: Roberto Dillon, Arushi  

**Link**: [PDF](https://arxiv.org/pdf/2505.05015)  

**Abstract**: Continuous authentication systems leveraging free-text keyboard dynamics offer a promising additional layer of security in a multifactor authentication setup that can be used in a transparent way with no impact on user experience. This study investigates the efficacy of behavioral biometrics by employing an Agent-Based Model (ABM) to simulate diverse typing profiles across mechanical and membrane keyboards. Specifically, we generated synthetic keystroke data from five unique agents, capturing features related to dwell time, flight time, and error rates within sliding 5-second windows updated every second. Two machine learning approaches, One-Class Support Vector Machine (OC-SVM) and Random Forest (RF), were evaluated for user verification. Results revealed a stark contrast in performance: while One-Class SVM failed to differentiate individual users within each group, Random Forest achieved robust intra-keyboard user recognition (Accuracy > 0.7) but struggled to generalize across keyboards for the same user, highlighting the significant impact of keyboard hardware on typing behavior. These findings suggest that: (1) keyboard-specific user profiles may be necessary for reliable authentication, and (2) ensemble methods like RF outperform One-Class SVM in capturing fine-grained user-specific patterns. 

**Abstract (ZH)**: 基于自由文本键盘动态的持续认证系统在多重因素认证设置中提供了一种增强的安全层，可以在不影响用户体验的情况下以透明方式使用。本研究通过运用基于代理的模型（ABM）模拟机械键盘和薄膜键盘的多样化打字模式，探讨行为生物特征的有效性。具体地，我们从五个独特的代理生成合成按键数据，在每秒更新一次的滑动5秒窗口内捕捉与停留时间、飞行时间和错误率相关的特点。评估了Two机器学习方法，即One-Class支持向量机（OC-SVM）和随机森林（RF），用于用户验证。结果表明，One-Class SVM在区分同一组内的个体用户方面表现不佳，而随机森林实现了稳健的键盘内用户识别（准确率＞0.7），但在同一用户跨键盘上的泛化方面遇到困难，突显了键盘硬件对打字行为的显著影响。这些发现表明：（1）特定键盘的用户配置文件可能是可靠认证所必需的，（2）集成方法如随机森林在捕捉细粒度的用户特定模式方面优于One-Class SVM。 

---
# Decomposition of Probabilities of Causation with Two Mediators 

**Title (ZH)**: 两中介变量下因果概率的分解 

**Authors**: Yuta Kawakami, Jin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.04983)  

**Abstract**: Mediation analysis for probabilities of causation (PoC) provides a fundamental framework for evaluating the necessity and sufficiency of treatment in provoking an event through different causal pathways. One of the primary objectives of causal mediation analysis is to decompose the total effect into path-specific components. In this study, we investigate the path-specific probability of necessity and sufficiency (PNS) to decompose the total PNS into path-specific components along distinct causal pathways between treatment and outcome, incorporating two mediators. We define the path-specific PNS for decomposition and provide an identification theorem. Furthermore, we conduct numerical experiments to assess the properties of the proposed estimators from finite samples and demonstrate their practical application using a real-world educational dataset. 

**Abstract (ZH)**: 概率介导分析中的概率成因（PoC）的中介分析提供了一种基本框架，用于评估不同因果路径中治疗诱发事件的必要性和充分性。因果中介分析的主要目标之一是将总效应分解为特定路径的组件。在本研究中，我们探讨特定路径的概率必要性和充分性（PNS），将其总PNS分解为治疗与结果之间不同因果路径的特定路径组件，同时考虑两个中介变量。我们定义了用于分解的特定路径PNS，并提供了识别定理。此外，我们通过数值实验评估了所提估计量在有限样本中的性质，并使用实际教育数据集展示了其实际应用。 

---
# ChainMarks: Securing DNN Watermark with Cryptographic Chain 

**Title (ZH)**: ChainMarks: 用加密链保护DNN水印 

**Authors**: Brian Choi, Shu Wang, Isabelle Choi, Kun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.04977)  

**Abstract**: With the widespread deployment of deep neural network (DNN) models, dynamic watermarking techniques are being used to protect the intellectual property of model owners. However, recent studies have shown that existing watermarking schemes are vulnerable to watermark removal and ambiguity attacks. Besides, the vague criteria for determining watermark presence further increase the likelihood of such attacks. In this paper, we propose a secure DNN watermarking scheme named ChainMarks, which generates secure and robust watermarks by introducing a cryptographic chain into the trigger inputs and utilizes a two-phase Monte Carlo method for determining watermark presence. First, ChainMarks generates trigger inputs as a watermark dataset by repeatedly applying a hash function over a secret key, where the target labels associated with trigger inputs are generated from the digital signature of model owner. Then, the watermarked model is produced by training a DNN over both the original and watermark datasets. To verify watermarks, we compare the predicted labels of trigger inputs with the target labels and determine ownership with a more accurate decision threshold that considers the classification probability of specific models. Experimental results show that ChainMarks exhibits higher levels of robustness and security compared to state-of-the-art watermarking schemes. With a better marginal utility, ChainMarks provides a higher probability guarantee of watermark presence in DNN models with the same level of watermark accuracy. 

**Abstract (ZH)**: ChainMarks：一种基于加密链的深度神经网络水印方案 

---
# Moments of Causal Effects 

**Title (ZH)**: 因果效应的矩 

**Authors**: Yuta Kawakami, Jin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.04971)  

**Abstract**: The moments of random variables are fundamental statistical measures for characterizing the shape of a probability distribution, encompassing metrics such as mean, variance, skewness, and kurtosis. Additionally, the product moments, including covariance and correlation, reveal the relationships between multiple random variables. On the other hand, the primary focus of causal inference is the evaluation of causal effects, which are defined as the difference between two potential outcomes. While traditional causal effect assessment focuses on the average causal effect, this work provides definitions, identification theorems, and bounds for moments and product moments of causal effects to analyze their distribution and relationships. We conduct experiments to illustrate the estimation of the moments of causal effects from finite samples and demonstrate their practical application using a real-world medical dataset. 

**Abstract (ZH)**: 随机变量的矩是描述概率分布形状的基本统计量，包括均值、方差、偏度和峰度等指标。此外，包含协方差和相关系数的乘积矩揭示了多个随机变量之间的关系。另一方面，因果推断的主要关注点是评估因果效应，这些效应定义为两个潜在结果之间的差异。虽然传统的因果效应评估主要关注平均因果效应，但本工作提供了因果效应矩和乘积矩的定义、识别定理及其界，以便分析其分布和关系。我们通过实验说明如何从有限样本中估计因果效应的矩，并通过实际医疗数据集演示其实际应用。 

---
# Graffe: Graph Representation Learning via Diffusion Probabilistic Models 

**Title (ZH)**: Graffe: 图表示学习 via 扩散概率模型 

**Authors**: Dingshuo Chen, Shuchen Xue, Liuji Chen, Yingheng Wang, Qiang Liu, Shu Wu, Zhi-Ming Ma, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04956)  

**Abstract**: Diffusion probabilistic models (DPMs), widely recognized for their potential to generate high-quality samples, tend to go unnoticed in representation learning. While recent progress has highlighted their potential for capturing visual semantics, adapting DPMs to graph representation learning remains in its infancy. In this paper, we introduce Graffe, a self-supervised diffusion model proposed for graph representation learning. It features a graph encoder that distills a source graph into a compact representation, which, in turn, serves as the condition to guide the denoising process of the diffusion decoder. To evaluate the effectiveness of our model, we first explore the theoretical foundations of applying diffusion models to representation learning, proving that the denoising objective implicitly maximizes the conditional mutual information between data and its representation. Specifically, we prove that the negative logarithm of the denoising score matching loss is a tractable lower bound for the conditional mutual information. Empirically, we conduct a series of case studies to validate our theoretical insights. In addition, Graffe delivers competitive results under the linear probing setting on node and graph classification tasks, achieving state-of-the-art performance on 9 of the 11 real-world datasets. These findings indicate that powerful generative models, especially diffusion models, serve as an effective tool for graph representation learning. 

**Abstract (ZH)**: 自监督扩散模型Grafite及其在图表示学习中的应用 

---
# Structural Alignment in Link Prediction 

**Title (ZH)**: 链接预测中的结构对齐 

**Authors**: Jeffrey Seathrún Sardina  

**Link**: [PDF](https://arxiv.org/pdf/2505.04939)  

**Abstract**: While Knowledge Graphs (KGs) have become increasingly popular across various scientific disciplines for their ability to model and interlink huge quantities of data, essentially all real-world KGs are known to be incomplete. As such, with the growth of KG use has been a concurrent development of machine learning tools designed to predict missing information in KGs, which is referred to as the Link Prediction Task. The majority of state-of-the-art link predictors to date have followed an embedding-based paradigm. In this paradigm, it is assumed that the information content of a KG is best represented by the (individual) vector representations of its nodes and edges, and that therefore node and edge embeddings are particularly well-suited to performing link prediction.
This thesis proposes an alternative perspective on the field's approach to link prediction and KG data modelling. Specifically, this work re-analyses KGs and state-of-the-art link predictors from a graph-structure-first perspective that models the information content of a KG in terms of whole triples, rather than individual nodes and edges.
Following a literature review and two core sets of experiments, this thesis concludes that a structure-first perspective on KGs and link prediction is both viable and useful for understanding KG learning and for enabling cross-KG transfer learning for the link prediction task. This observation is used to create and propose the Structural Alignment Hypothesis, which postulates that link prediction can be understood and modelled as a structural task.
All code and data used for this thesis are open-sourced. This thesis was written bilingually, with the main document in English and an informal extended summary in Irish. An Irish-language translation dictionary of machine learning terms (the Foclóir Tráchtais) created for this work is open-sourced as well. 

**Abstract (ZH)**: 虽然知识图谱（KGs）在各个科学领域因其建模大量数据并实现数据链接的能力而变得日益流行，但实际上所有现实世界的KGs都存在着不完整性。随着KGs应用的增长，开发用于预测KG中缺失信息的机器学习工具也得到了同步发展，这一任务被称为链接预测任务。迄今为止，大多数先进的链接预测器都遵循基于嵌入的方法。在这种方法中，假设知识图谱的信息内容可以用其节点和边的向量表示最好，并且因此节点和边的嵌入特别适合进行链接预测。

本文对该领域在链接预测和KG数据建模方面的做法提出了不同的观点。具体而言，本文从图结构优先的角度重新分析了KGs和最先进的链接预测器，以整三元组来建模KG的信息内容，而不是单独的节点和边。

在文献综述和两组核心实验之后，本文得出结论，从图结构优先的角度理解KGs和链接预测既是可行的，也有助于理解KG学习，并使链接预测任务跨KG迁移学习成为可能。本文基于这一观察提出了结构对齐假说，认为链接预测可以被理解为一个结构任务并进行建模。

本文使用的所有代码和数据均已开源。本文使用英语为主文档，并附有爱尔兰语的非正式扩展摘要。本文还开发了一个机器学习术语的爱尔兰语翻译词典（Foclóir Tráchtais），该词典也已开源。 

---
# Fair Uncertainty Quantification for Depression Prediction 

**Title (ZH)**: 抑郁症预测中的公平不确定性量化 

**Authors**: Yonghong Li, Xiuzhuang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.04931)  

**Abstract**: Trustworthy depression prediction based on deep learning, incorporating both predictive reliability and algorithmic fairness across diverse demographic groups, is crucial for clinical application. Recently, achieving reliable depression predictions through uncertainty quantification has attracted increasing attention. However, few studies have focused on the fairness of uncertainty quantification (UQ) in depression prediction. In this work, we investigate the algorithmic fairness of UQ, namely Equal Opportunity Coverage (EOC) fairness, and propose Fair Uncertainty Quantification (FUQ) for depression prediction. FUQ pursues reliable and fair depression predictions through group-based analysis. Specifically, we first group all the participants by different sensitive attributes and leverage conformal prediction to quantify uncertainty within each demographic group, which provides a theoretically guaranteed and valid way to quantify uncertainty for depression prediction and facilitates the investigation of fairness across different demographic groups. Furthermore, we propose a fairness-aware optimization strategy that formulates fairness as a constrained optimization problem under EOC constraints. This enables the model to preserve predictive reliability while adapting to the heterogeneous uncertainty levels across demographic groups, thereby achieving optimal fairness. Through extensive evaluations on several visual and audio depression datasets, our approach demonstrates its effectiveness. 

**Abstract (ZH)**: 基于深度学习的可信赖抑郁预测：结合预测可靠性与跨不同人群组的算法公平性对于临床应用至关重要。最近，通过不确定性量化实现可靠的抑郁预测吸引了越来越多的关注。然而，很少有研究关注抑郁预测中不确定性量化（UQ）的公平性。在本文中，我们研究了UQ的算法公平性，即平等机会覆盖（EOC）公平性，并提出了一种新的方法，即公平不确定性量化（FUQ）以用于抑郁预测。FUQ 通过基于群体的分析追求可靠的和公平的抑郁预测。具体来说，我们首先根据不同的敏感属性对所有参与者进行分组，并利用校准预测来量化每个群体内的不确定性，这种方法为抑郁预测提供了一种理论保证且有效的不确定性量化方式，并有助于在不同人群组之间调查公平性问题。此外，我们提出了一个公平性感知的优化策略，将公平性作为在EOC约束下的约束优化问题进行建模。这种方法使得模型在保持预测可靠性的同时，能够适应不同人群组之间的异质不确定性水平，从而实现最优公平性。通过在多个视觉和音频抑郁数据集上的广泛评估，我们的方法证实了其有效性。 

---
# Physics-Assisted and Topology-Informed Deep Learning for Weather Prediction 

**Title (ZH)**: 物理学辅助和拓扑驱动的深度学习天气预报 

**Authors**: Jiaqi Zheng, Qing Ling, Yerong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04918)  

**Abstract**: Although deep learning models have demonstrated remarkable potential in weather prediction, most of them overlook either the \textbf{physics} of the underlying weather evolution or the \textbf{topology} of the Earth's surface. In light of these disadvantages, we develop PASSAT, a novel Physics-ASSisted And Topology-informed deep learning model for weather prediction. PASSAT attributes the weather evolution to two key factors: (i) the advection process that can be characterized by the advection equation and the Navier-Stokes equation; (ii) the Earth-atmosphere interaction that is difficult to both model and calculate. PASSAT also takes the topology of the Earth's surface into consideration, other than simply treating it as a plane. With these considerations, PASSAT numerically solves the advection equation and the Navier-Stokes equation on the spherical manifold, utilizes a spherical graph neural network to capture the Earth-atmosphere interaction, and generates the initial velocity fields that are critical to solving the advection equation from the same spherical graph neural network. In the $5.625^\circ$-resolution ERA5 data set, PASSAT outperforms both the state-of-the-art deep learning-based weather prediction models and the operational numerical weather prediction model IFS T42. Code and checkpoint are available at this https URL. 

**Abstract (ZH)**: 虽然深度学习模型在天气预测方面展示了 remarkable 的潜力，但大多数模型要么忽略了基础天气演变过程的 **物理规律**，要么忽略了地球表面的 **拓扑结构**。鉴于这些缺点，我们提出了 PASSAT，这是一种新的结合物理辅助和拓扑信息的深度学习模型，用于天气预测。PASSAT 将天气演变归因于两个关键因素：（i）可以由对流方程和纳维-斯托克斯方程描述的对流过程；（ii）地球与大气的相互作用，这一作用既难以建模也难以计算。PASSAT 还考虑了地球表面的拓扑结构，而不仅仅是将其视为平面。通过这些考虑，PASSAT 在球面流形上数值求解对流方程和纳维-斯托克斯方程，利用球面图神经网络捕获地球与大气之间的相互作用，并从同一个球面图神经网络生成用于求解对流方程的初始速度场。在 $5.625^\circ$ 分辨率的 ERA5 数据集中，PASSAT 在天气预测性能上超过了最先进的基于深度学习的天气预测模型和现有的操作数值天气预报模型 IFS T42。代码和检查点可在以下链接获取。 

---
# An Open-Source Dual-Loss Embedding Model for Semantic Retrieval in Higher Education 

**Title (ZH)**: 开源双损失嵌入模型在高等教育中的语义检索 

**Authors**: Ramteja Sajja, Yusuf Sermet, Ibrahim Demir  

**Link**: [PDF](https://arxiv.org/pdf/2505.04916)  

**Abstract**: Recent advances in AI have catalyzed the adoption of intelligent educational tools, yet many semantic retrieval systems remain ill-suited to the unique linguistic and structural characteristics of academic content. This study presents two open-source embedding models fine-tuned for educational question answering, particularly in the context of course syllabi. A synthetic dataset of 3,197 sentence pairs, spanning synonymous terminology, paraphrased questions, and implicit-explicit mappings, was constructed through a combination of manual curation and large language model (LLM)-assisted generation. Two training strategies were evaluated: (1) a baseline model fine-tuned using MultipleNegativesRankingLoss (MNRL), and (2) a dual-loss model that combines MNRL with CosineSimilarityLoss to improve both semantic ranking and similarity calibration. Evaluations were conducted on 28 university course syllabi using a fixed set of natural language questions categorized into course, faculty, and teaching assistant information. Results demonstrate that both fine-tuned models outperform strong open-source baselines, including all-MiniLM-L6-v2 and multi-qa-MiniLM-L6-cos-v1, and that the dual-loss model narrows the performance gap with high-performing proprietary embeddings such as OpenAI's text-embedding-3 series. This work contributes reusable, domain-aligned embedding models and provides a replicable framework for educational semantic retrieval, supporting downstream applications such as academic chatbots, retrieval-augmented generation (RAG) systems, and learning management system (LMS) integrations. 

**Abstract (ZH)**: Recent Advances in AIhave Catalyzed the Adoption of Intelligent Educational Tools: Fine-Tuned Embedding Models for Academic Question Answering in Course Syllabi 

---
# SpatialPrompting: Keyframe-driven Zero-Shot Spatial Reasoning with Off-the-Shelf Multimodal Large Language Models 

**Title (ZH)**: 空间提示：以关键帧驱动的零样本空间推理与即用型多模态大型语言模型 

**Authors**: Shun Taguchi, Hideki Deguchi, Takumi Hamazaki, Hiroyuki Sakai  

**Link**: [PDF](https://arxiv.org/pdf/2505.04911)  

**Abstract**: This study introduces SpatialPrompting, a novel framework that harnesses the emergent reasoning capabilities of off-the-shelf multimodal large language models to achieve zero-shot spatial reasoning in three-dimensional (3D) environments. Unlike existing methods that rely on expensive 3D-specific fine-tuning with specialized 3D inputs such as point clouds or voxel-based features, SpatialPrompting employs a keyframe-driven prompt generation strategy. This framework uses metrics such as vision-language similarity, Mahalanobis distance, field of view, and image sharpness to select a diverse and informative set of keyframes from image sequences and then integrates them with corresponding camera pose data to effectively abstract spatial relationships and infer complex 3D structures. The proposed framework not only establishes a new paradigm for flexible spatial reasoning that utilizes intuitive visual and positional cues but also achieves state-of-the-art zero-shot performance on benchmark datasets, such as ScanQA and SQA3D, across several metrics. The proposed method effectively eliminates the need for specialized 3D inputs and fine-tuning, offering a simpler and more scalable alternative to conventional approaches. 

**Abstract (ZH)**: SpatialPrompting: 一种利用现成多模态大语言模型新兴推理能力实现三维环境零样本空间推理的新型框架 

---
# Precise gradient descent training dynamics for finite-width multi-layer neural networks 

**Title (ZH)**: 有限宽度多层神经网络的精确梯度下降训练动力学 

**Authors**: Qiyang Han, Masaaki Imaizumi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04898)  

**Abstract**: In this paper, we provide the first precise distributional characterization of gradient descent iterates for general multi-layer neural networks under the canonical single-index regression model, in the `finite-width proportional regime' where the sample size and feature dimension grow proportionally while the network width and depth remain bounded. Our non-asymptotic state evolution theory captures Gaussian fluctuations in first-layer weights and concentration in deeper-layer weights, and remains valid for non-Gaussian features.
Our theory differs from existing neural tangent kernel (NTK), mean-field (MF) theories and tensor program (TP) in several key aspects. First, our theory operates in the finite-width regime whereas these existing theories are fundamentally infinite-width. Second, our theory allows weights to evolve from individual initializations beyond the lazy training regime, whereas NTK and MF are either frozen at or only weakly sensitive to initialization, and TP relies on special initialization schemes. Third, our theory characterizes both training and generalization errors for general multi-layer neural networks beyond the uniform convergence regime, whereas existing theories study generalization almost exclusively in two-layer settings.
As a statistical application, we show that vanilla gradient descent can be augmented to yield consistent estimates of the generalization error at each iteration, which can be used to guide early stopping and hyperparameter tuning. As a further theoretical implication, we show that despite model misspecification, the model learned by gradient descent retains the structure of a single-index function with an effective signal determined by a linear combination of the true signal and the initialization. 

**Abstract (ZH)**: 在有限宽度比例 regime 下广义多层神经网络在单指数量化回归模型下梯度下降迭代的首个精确分布描述：不同于现有神经 tangent 核（NTK）、均场（MF）理论和张量程序（TP）的若干关键差异 

---
# Clustering with Communication: A Variational Framework for Single Cell Representation Learning 

**Title (ZH)**: 通信驱动聚类：单细胞表示学习的变分框架 

**Authors**: Cong Qi, Yeqing Chen, Jie Zhang, Wei Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04891)  

**Abstract**: Single-cell RNA sequencing (scRNA-seq) has revealed complex cellular heterogeneity, but recent studies emphasize that understanding biological function also requires modeling cell-cell communication (CCC), the signaling interactions mediated by ligand-receptor pairs that coordinate cellular behavior. Tools like CellChat have demonstrated that CCC plays a critical role in processes such as cell differentiation, tissue regeneration, and immune response, and that transcriptomic data inherently encodes rich information about intercellular signaling. We propose CCCVAE, a novel variational autoencoder framework that incorporates CCC signals into single-cell representation learning. By leveraging a communication-aware kernel derived from ligand-receptor interactions and a sparse Gaussian process, CCCVAE encodes biologically informed priors into the latent space. Unlike conventional VAEs that treat each cell independently, CCCVAE encourages latent embeddings to reflect both transcriptional similarity and intercellular signaling context. Empirical results across four scRNA-seq datasets show that CCCVAE improves clustering performance, achieving higher evaluation scores than standard VAE baselines. This work demonstrates the value of embedding biological priors into deep generative models for unsupervised single-cell analysis. 

**Abstract (ZH)**: 单细胞RNA测序(scRNA-seq)揭示了复杂的细胞异质性，但最近的研究强调，理解生物功能还需要建模细胞间通讯(CCC)，即由配体-受体配对介导的信号相互作用，协调细胞行为。像CellChat这样的工具已经证明，CCC在细胞分化、组织再生和免疫反应等过程中扮演了关键角色，并且转录组数据本身就包含了丰富的细胞间信号传导信息。我们提出了一种新颖的变分自编码器框架CCCVAE，该框架将CCC信号整合到单细胞表示学习中。CCCVAE通过利用源自配体-受体相互作用的通信感知核和稀疏高斯过程，将生物学先验知识编码到潜在空间中。与传统的独立处理每个细胞的VAE不同，CCCVAE鼓励潜在嵌入反映转录相似性和细胞间信号传导上下文。在四个scRNA-seq数据集上的实验证明，CCCVAE提高了聚类性能，评估得分高于标准VAE基线。这项工作展示了将生物学先验嵌入到深度生成模型中进行无监督单细胞分析的价值。 

---
# QBR: A Question-Bank-Based Approach to Fine-Grained Legal Knowledge Retrieval for the General Public 

**Title (ZH)**: 基于问题集的方法：面向普通公众的细粒度法律知识检索 

**Authors**: Mingruo Yuan, Ben Kao, Tien-Hsuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04883)  

**Abstract**: Retrieval of legal knowledge by the general public is a challenging problem due to the technicality of the professional knowledge and the lack of fundamental understanding by laypersons on the subject. Traditional information retrieval techniques assume that users are capable of formulating succinct and precise queries for effective document retrieval. In practice, however, the wide gap between the highly technical contents and untrained users makes legal knowledge retrieval very difficult. We propose a methodology, called QBR, which employs a Questions Bank (QB) as an effective medium for bridging the knowledge gap. We show how the QB is used to derive training samples to enhance the embedding of knowledge units within documents, which leads to effective fine-grained knowledge retrieval. We discuss and evaluate through experiments various advantages of QBR over traditional methods. These include more accurate, efficient, and explainable document retrieval, better comprehension of retrieval results, and highly effective fine-grained knowledge retrieval. We also present some case studies and show that QBR achieves social impact by assisting citizens to resolve everyday legal concerns. 

**Abstract (ZH)**: 普通公众获取法律知识是一个具有挑战性的问题，由于专业知识的技术性以及非专业人士对该主题的基本理解不足。传统的信息检索技术假设用户能够制定简洁和精确的查询以实现有效的文档检索。然而，在实践中，高度技术性的内容与未受训练的用户之间的巨大差距使得法律知识检索非常困难。我们提出了一种名为QBR的方法，该方法利用知识问答库（QB）作为弥合知识差距的有效媒介。我们展示了如何利用QB来生成训练样本以增强文档中的知识单元嵌入，从而实现有效的细粒度知识检索。我们通过实验讨论并评估了QBR相对于传统方法的各种优势，包括更准确、更高效的文档检索、更好的检索结果理解以及非常有效的细粒度知识检索。我们还呈现了一些案例研究，并展示了QBR通过帮助公民解决日常生活中的法律问题而产生的社会影响。 

---
# ConCISE: Confidence-guided Compression in Step-by-step Efficient Reasoning 

**Title (ZH)**: ConCISE: 基于信心引导的逐步高效推理压缩 

**Authors**: Ziqing Qiao, Yongheng Deng, Jiali Zeng, Dong Wang, Lai Wei, Fandong Meng, Jie Zhou, Ju Ren, Yaoxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04881)  

**Abstract**: Large Reasoning Models (LRMs) perform strongly in complex reasoning tasks via Chain-of-Thought (CoT) prompting, but often suffer from verbose outputs caused by redundant content, increasing computational overhead, and degrading user experience. Existing compression methods either operate post-hoc pruning, risking disruption to reasoning coherence, or rely on sampling-based selection, which fails to intervene effectively during generation. In this work, we introduce a confidence-guided perspective to explain the emergence of redundant reflection in LRMs, identifying two key patterns: Confidence Deficit, where the model reconsiders correct steps due to low internal confidence, and Termination Delay, where reasoning continues even after reaching a confident answer. Based on this analysis, we propose ConCISE (Confidence-guided Compression In Step-by-step Efficient Reasoning), a framework that simplifies reasoning chains by reinforcing the model's confidence during inference, thus preventing the generation of redundant reflection steps. It integrates Confidence Injection to stabilize intermediate steps and Early Stopping to terminate reasoning when confidence is sufficient. Extensive experiments demonstrate that fine-tuning LRMs on ConCISE-generated data yields significantly shorter outputs, reducing length by up to approximately 50% under SimPO, while maintaining high task accuracy. ConCISE consistently outperforms existing baselines across multiple reasoning benchmarks. 

**Abstract (ZH)**: Large Reasoning Models (LRMs)通过Chain-of-Thought (CoT)提示在复杂推理任务中表现出色，但常常因冗余内容导致冗长输出，增加计算开销并降低用户体验。现有压缩方法要么在事后剪枝操作，有可能破坏推理连贯性，要么依赖于基于采样的选择，这在生成过程中无法有效干预。在本项工作中，我们引入了一种基于信心的视角来解释LRMs中冗余反思的产生，识别出两种关键模式：信心赤字，模型因内部信心不足而重新考虑正确的步骤；以及终止延迟，即使在得到自信的答案后，推理仍然继续。基于此分析，我们提出了ConCISE（基于信心的压缩在逐步高效推理中），一种框架，在推理过程中增强模型的信心，从而防止生成冗余反思步骤。该框架集成了信心注入以稳定中间步骤，并采用早期停止以在信心充足时终止推理。广泛实验证明，通过对ConCISE生成的数据进行微调，可以显著缩短输出长度，在SimPO下减少约50%的长度，同时保持高任务准确性。ConCISE在多种推理基准测试中均优于现有基线。 

---
# Learning from Loss Landscape: Generalizable Mixed-Precision Quantization via Adaptive Sharpness-Aware Gradient Aligning 

**Title (ZH)**: 基于损失景观学习：自适应锐度感知梯度对齐的可迁移混合精度量化 

**Authors**: Lianbo Ma, Jianlun Ma, Yuee Zhou, Guoyang Xie, Qiang He, Zhichao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04877)  

**Abstract**: Mixed Precision Quantization (MPQ) has become an essential technique for optimizing neural network by determining the optimal bitwidth per layer. Existing MPQ methods, however, face a major hurdle: they require a computationally expensive search for quantization policies on large-scale datasets. To resolve this issue, we introduce a novel approach that first searches for quantization policies on small datasets and then generalizes them to large-scale datasets. This approach simplifies the process, eliminating the need for large-scale quantization fine-tuning and only necessitating model weight adjustment. Our method is characterized by three key techniques: sharpness-aware minimization for enhanced quantization generalization, implicit gradient direction alignment to handle gradient conflicts among different optimization objectives, and an adaptive perturbation radius to accelerate optimization. Both theoretical analysis and experimental results validate our approach. Using the CIFAR10 dataset (just 0.5\% the size of ImageNet training data) for MPQ policy search, we achieved equivalent accuracy on ImageNet with a significantly lower computational cost, while improving efficiency by up to 150% over the baselines. 

**Abstract (ZH)**: 基于小型数据集的迁移量化策略：Mixed Precision Quantization (MPQ) 已成为通过确定每层的最优位宽来优化神经网络的关键技术。现有 MPQ 方法面临的主要挑战是，在大规模数据集上寻找量化策略计算成本高昂。为解决这一问题，我们提出了一种新颖的方法，该方法首先在小型数据集上搜索量化策略，然后将这些策略推广到大规模数据集。该方法简化了过程，消除了对大规模量化微调的需求，只需调整模型权重。我们的方法具有三个关键技术：增强量化泛化的尖锐性感知最小化、处理不同优化目标之间梯度冲突的隐式梯度方向对齐，以及加速优化的自适应扰动半径。理论分析和实验结果均验证了该方法的有效性。使用仅相当于 ImageNet 训练数据0.5％大小的CIFAR10数据集进行 MPQ 策略搜索，我们在 ImageNet 上实现了相当于的精度，计算成本显著降低，同时相对于 baseline 方法效率提升高达150％。 

---
# Federated Learning for Cyber Physical Systems: A Comprehensive Survey 

**Title (ZH)**: 联邦学习在 cyber-physical 系统中的应用：一个综合综述 

**Authors**: Minh K. Quan, Pubudu N. Pathirana, Mayuri Wijayasundara, Sujeeva Setunge, Dinh C. Nguyen, Christopher G. Brinton, David J. Love, H. Vincent Poor  

**Link**: [PDF](https://arxiv.org/pdf/2505.04873)  

**Abstract**: The integration of machine learning (ML) in cyber physical systems (CPS) is a complex task due to the challenges that arise in terms of real-time decision making, safety, reliability, device heterogeneity, and data privacy. There are also open research questions that must be addressed in order to fully realize the potential of ML in CPS. Federated learning (FL), a distributed approach to ML, has become increasingly popular in recent years. It allows models to be trained using data from decentralized sources. This approach has been gaining popularity in the CPS field, as it integrates computer, communication, and physical processes. Therefore, the purpose of this work is to provide a comprehensive analysis of the most recent developments of FL-CPS, including the numerous application areas, system topologies, and algorithms developed in recent years. The paper starts by discussing recent advances in both FL and CPS, followed by their integration. Then, the paper compares the application of FL in CPS with its applications in the internet of things (IoT) in further depth to show their connections and distinctions. Furthermore, the article scrutinizes how FL is utilized in critical CPS applications, e.g., intelligent transportation systems, cybersecurity services, smart cities, and smart healthcare solutions. The study also includes critical insights and lessons learned from various FL-CPS implementations. The paper's concluding section delves into significant concerns and suggests avenues for further research in this fast-paced and dynamic era. 

**Abstract (ZH)**: 机器学习在 cyber 物理系统中的集成：联邦学习的进展与挑战 

---
# Quantum-Inspired Optimization Process for Data Imputation 

**Title (ZH)**: 量子启发的优化过程用于数据插补 

**Authors**: Nishikanta Mohanty, Bikash K. Behera, Badsah Mukherjee, Christopher Ferrie  

**Link**: [PDF](https://arxiv.org/pdf/2505.04841)  

**Abstract**: Data imputation is a critical step in data pre-processing, particularly for datasets with missing or unreliable values. This study introduces a novel quantum-inspired imputation framework evaluated on the UCI Diabetes dataset, which contains biologically implausible missing values across several clinical features. The method integrates Principal Component Analysis (PCA) with quantum-assisted rotations, optimized through gradient-free classical optimizers -COBYLA, Simulated Annealing, and Differential Evolution to reconstruct missing values while preserving statistical fidelity. Reconstructed values are constrained within +/-2 standard deviations of original feature distributions, avoiding unrealistic clustering around central tendencies. This approach achieves a substantial and statistically significant improvement, including an average reduction of over 85% in Wasserstein distance and Kolmogorov-Smirnov test p-values between 0.18 and 0.22, compared to p-values > 0.99 in classical methods such as Mean, KNN, and MICE. The method also eliminates zero-value artifacts and enhances the realism and variability of imputed data. By combining quantum-inspired transformations with a scalable classical framework, this methodology provides a robust solution for imputation tasks in domains such as healthcare and AI pipelines, where data quality and integrity are crucial. 

**Abstract (ZH)**: 基于量子启发的新型插补框架在UCI糖尿病数据集上的评估 

---
# Piecewise Constant Spectral Graph Neural Network 

**Title (ZH)**: 分段常数谱图神经网络 

**Authors**: Vahan Martirosyan, Jhony H. Giraldo, Fragkiskos D. Malliaros  

**Link**: [PDF](https://arxiv.org/pdf/2505.04808)  

**Abstract**: Graph Neural Networks (GNNs) have achieved significant success across various domains by leveraging graph structures in data. Existing spectral GNNs, which use low-degree polynomial filters to capture graph spectral properties, may not fully identify the graph's spectral characteristics because of the polynomial's small degree. However, increasing the polynomial degree is computationally expensive and beyond certain thresholds leads to performance plateaus or degradation. In this paper, we introduce the Piecewise Constant Spectral Graph Neural Network(PieCoN) to address these challenges. PieCoN combines constant spectral filters with polynomial filters to provide a more flexible way to leverage the graph structure. By adaptively partitioning the spectrum into intervals, our approach increases the range of spectral properties that can be effectively learned. Experiments on nine benchmark datasets, including both homophilic and heterophilic graphs, demonstrate that PieCoN is particularly effective on heterophilic datasets, highlighting its potential for a wide range of applications. 

**Abstract (ZH)**: Piecewise Constant Spectral Graph Neural Network (PieCoN) 

---
# Confabulation dynamics in a reservoir computer: Filling in the gaps with untrained attractors 

**Title (ZH)**: reservoir计算机中的虚构动态：用未训练的吸引子填补空白 

**Authors**: Jack O'Hagan, Andrew Keane, Andrew Flynn  

**Link**: [PDF](https://arxiv.org/pdf/2505.04792)  

**Abstract**: Artificial Intelligence has advanced significantly in recent years thanks to innovations in the design and training of artificial neural networks (ANNs). Despite these advancements, we still understand relatively little about how elementary forms of ANNs learn, fail to learn, and generate false information without the intent to deceive, a phenomenon known as `confabulation'. To provide some foundational insight, in this paper we analyse how confabulation occurs in reservoir computers (RCs): a dynamical system in the form of an ANN. RCs are particularly useful to study as they are known to confabulate in a well-defined way: when RCs are trained to reconstruct the dynamics of a given attractor, they sometimes construct an attractor that they were not trained to construct, a so-called `untrained attractor' (UA). This paper sheds light on the role played by UAs when reconstruction fails and their influence when modelling transitions between reconstructed attractors. Based on our results, we conclude that UAs are an intrinsic feature of learning systems whose state spaces are bounded, and that this means of confabulation may be present in systems beyond RCs. 

**Abstract (ZH)**: 近年来，由于人工神经网络（ANNs）在设计和训练方面的创新，人工智能取得了显著进展。尽管取得了这些进步，我们仍对基本形式的ANNs如何学习、无法学习以及生成虚假信息（无意欺骗的现象，称之为“自欺”）知之甚少。为了提供一些基础性的见解，本文分析了“水库计算机”（RCs）中自欺现象是如何发生的：作为一种动态系统，ANN的形式，RCs以其在特定情况下自欺的方式而特别有用：当RCs被训练以重构给定吸引子的动力学时，有时会构建一个它们没有被训练的吸引子，称为“未训练吸引子”（UA）。本文探讨了当重构失败时未训练吸引子的作用及其在建模重构吸引子之间过渡时的影响。根据我们的研究结果，我们得出结论认为，未训练吸引子是学习系统中固有的特征，这意味着这种自欺手段可能存在于RCs之外的系统中。 

---
# Replay to Remember (R2R): An Efficient Uncertainty-driven Unsupervised Continual Learning Framework Using Generative Replay 

**Title (ZH)**: 回忆以记住 (R2R): 一种基于不确定性驱动的生成回放高效无监督连续学习框架 

**Authors**: Sriram Mandalika, Harsha Vardhan, Athira Nambiar  

**Link**: [PDF](https://arxiv.org/pdf/2505.04787)  

**Abstract**: Continual Learning entails progressively acquiring knowledge from new data while retaining previously acquired knowledge, thereby mitigating ``Catastrophic Forgetting'' in neural networks. Our work presents a novel uncertainty-driven Unsupervised Continual Learning framework using Generative Replay, namely ``Replay to Remember (R2R)''. The proposed R2R architecture efficiently uses unlabelled and synthetic labelled data in a balanced proportion using a cluster-level uncertainty-driven feedback mechanism and a VLM-powered generative replay module. Unlike traditional memory-buffer methods that depend on pretrained models and pseudo-labels, our R2R framework operates without any prior training. It leverages visual features from unlabeled data and adapts continuously using clustering-based uncertainty estimation coupled with dynamic thresholding. Concurrently, a generative replay mechanism along with DeepSeek-R1 powered CLIP VLM produces labelled synthetic data representative of past experiences, resembling biological visual thinking that replays memory to remember and act in new, unseen tasks. Extensive experimental analyses are carried out in CIFAR-10, CIFAR-100, CINIC-10, SVHN and TinyImageNet datasets. Our proposed R2R approach improves knowledge retention, achieving a state-of-the-art performance of 98.13%, 73.06%, 93.41%, 95.18%, 59.74%, respectively, surpassing state-of-the-art performance by over 4.36%. 

**Abstract (ZH)**: 持续学习涉及从新数据中逐步获取知识同时保留之前获取的知识，从而减轻神经网络中的“灾难性遗忘”。我们的工作提出了一种基于不确定性驱动的无监督持续学习框架，名为“回忆以记起（R2R）”，该框架利用生成回放机制。提出的R2R架构通过簇级不确定性驱动的反馈机制和基于VLM的生成回放模块，高效地在平衡比例下使用未标记和合成标记数据。与依赖预训练模型和伪标签的传统记忆缓冲方法不同，我们的R2R框架无需任何先验训练。它利用未标记数据的视觉特征，并结合基于聚类的不确定性估计和动态阈值进行持续适应。同时，生成回放机制与DeepSeek-R1增强的CLIP VLM合作生成代表过往经历的标记合成数据，类似于生物视觉思考中的回忆记忆以记住并在新、未见任务中采取行动。在CIFAR-10、CIFAR-100、CINIC-10、SVHN和TinyImageNet数据集上进行了广泛实验分析。我们提出的R2R方法提高了知识保留，分别达到了98.13%、73.06%、93.41%、95.18%、59.74%的性能，超过现有最佳性能4.36%以上。 

---
# Flower Across Time and Media: Sentiment Analysis of Tang Song Poetry and Visual Correspondence 

**Title (ZH)**: .time与媒介中的花朵：唐宋诗词的情感分析与视觉对应研究 

**Authors**: Shuai Gong, Tiange Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.04785)  

**Abstract**: The Tang (618 to 907) and Song (960 to 1279) dynasties witnessed an extraordinary flourishing of Chinese cultural expression, where floral motifs served as a dynamic medium for both poetic sentiment and artistic design. While previous scholarship has examined these domains independently, the systematic correlation between evolving literary emotions and visual culture remains underexplored. This study addresses that gap by employing BERT-based sentiment analysis to quantify emotional patterns in floral imagery across Tang Song poetry, then validating these patterns against contemporaneous developments in decorative this http URL approach builds upon recent advances in computational humanities while remaining grounded in traditional sinological methods. By applying a fine tuned BERT model to analyze peony and plum blossom imagery in classical poetry, we detect measurable shifts in emotional connotations between the Tang and Song periods. These textual patterns are then cross berenced with visual evidence from textiles, ceramics, and other material culture, revealing previously unrecognized synergies between literary expression and artistic representation. 

**Abstract (ZH)**: 唐代（618-907）和宋代（960-1279）见证了 Chinese 文化表达的 extraordinary 繁荣，其中花饰图案成为抒情和艺术设计的动态媒介。尽管以往的研究已经分别探讨了这些领域，但文学情感演变与视觉文化的系统性关联仍待深入探索。本研究通过使用基于 BERT 的情感分析来量化唐宋诗歌中花饰意象的情感模式，然后将这些模式与同期装饰艺术的发展进行比对，从而填补了这一空白。该研究建立在人文计算的最新进展之上，同时仍然根植于传统的汉学方法。通过应用微调后的 BERT 模型分析古典诗歌中的牡丹和梅花意象，我们发现唐代和宋代之间情感内涵存在可量化的转变。然后将这些文本模式与纺织品、陶瓷和其他物质文化中的视觉证据进行交叉验证，揭示了文学表达与艺术表现之间的先前未被认识的协同关系。 

---
# Exploring Zero-Shot App Review Classification with ChatGPT: Challenges and Potential 

**Title (ZH)**: 基于ChatGPT的零样本应用评论分类：挑战与潜力 

**Authors**: Mohit Chaudhary, Chirag Jain, Preethu Rose Anish  

**Link**: [PDF](https://arxiv.org/pdf/2505.04759)  

**Abstract**: App reviews are a critical source of user feedback, offering valuable insights into an app's performance, features, usability, and overall user experience. Effectively analyzing these reviews is essential for guiding app development, prioritizing feature updates, and enhancing user satisfaction. Classifying reviews into functional and non-functional requirements play a pivotal role in distinguishing feedback related to specific app features (functional requirements) from feedback concerning broader quality attributes, such as performance, usability, and reliability (non-functional requirements). Both categories are integral to informed development decisions. Traditional approaches to classifying app reviews are hindered by the need for large, domain-specific datasets, which are often costly and time-consuming to curate. This study explores the potential of zero-shot learning with ChatGPT for classifying app reviews into four categories: functional requirement, non-functional requirement, both, or neither. We evaluate ChatGPT's performance on a benchmark dataset of 1,880 manually annotated reviews from ten diverse apps spanning multiple domains. Our findings demonstrate that ChatGPT achieves a robust F1 score of 0.842 in review classification, despite certain challenges and limitations. Additionally, we examine how factors such as review readability and length impact classification accuracy and conduct a manual analysis to identify review categories more prone to misclassification. 

**Abstract (ZH)**: 应用程序评论是用户反馈的关键来源，提供了关于应用程序性能、功能、可用性和整体用户体验的重要见解。有效地分析这些评论对于指导应用程序开发、优先考虑功能更新以及提升用户满意度至关重要。将评论分类为功能性要求和非功能性要求在区分与具体应用程序功能相关的反馈（功能性要求）和与更广泛的Quality属性相关的反馈（非功能性要求）方面起到关键作用。两类反馈对于明智的开发决策都是不可或缺的。传统的应用程序评论分类方法受限于需要大量的特定领域数据集，这些数据集往往成本高昂且耗时。本研究探讨了使用ChatGPT进行零-shot学习在将应用程序评论分类为四种类别（功能性要求、非功能性要求、两者皆是或两者都不是）方面的潜力。我们评估了ChatGPT在1,880条人工标注的评论基准数据集中（来自十个不同领域的应用程序）的表现。研究结果表明，尽管存在某些挑战和限制，ChatGPT在评论分类中的F1分数达到了0.842。此外，我们还研究了评论可读性和长度等因素如何影响分类准确性，并进行了人工分析以识别更易出现分类错误的评论类别。 

---
# Advanced Deep Learning Approaches for Automated Recognition of Cuneiform Symbols 

**Title (ZH)**: 先进深度学习方法在楔形符号自动识别中的应用 

**Authors**: Shahad Elshehaby, Alavikunhu Panthakkan, Hussain Al-Ahmad, Mina Al-Saad  

**Link**: [PDF](https://arxiv.org/pdf/2505.04678)  

**Abstract**: This paper presents a thoroughly automated method for identifying and interpreting cuneiform characters via advanced deep-learning algorithms. Five distinct deep-learning models were trained on a comprehensive dataset of cuneiform characters and evaluated according to critical performance metrics, including accuracy and precision. Two models demonstrated outstanding performance and were subsequently assessed using cuneiform symbols from the Hammurabi law acquisition, notably Hammurabi Law 1. Each model effectively recognized the relevant Akkadian meanings of the symbols and delivered precise English translations. Future work will investigate ensemble and stacking approaches to optimize performance, utilizing hybrid architectures to improve detection accuracy and reliability. This research explores the linguistic relationships between Akkadian, an ancient Mesopotamian language, and Arabic, emphasizing their historical and cultural linkages. This study demonstrates the capability of deep learning to decipher ancient scripts by merging computational linguistics with archaeology, therefore providing significant insights for the comprehension and conservation of human history. 

**Abstract (ZH)**: 本文提出了一种完全自动化的方法，通过先进深度学习算法识别和解释楔形文字字符。五种不同的深度学习模型在楔形文字字符的全面数据集上进行训练，并根据关键性能指标（包括准确率和精密度）进行评估。其中两种模型表现突出，并随后使用汉谟拉比法典中的楔形文字符号（尤其是汉谟拉比法典第1条）进行了评估，每种模型都能有效地识别相关的阿卡德语意义，并提供精确的英语翻译。未来工作将研究集成和堆叠方法以优化性能，并利用混合架构提高检测准确性和可靠性。本文探讨了阿卡德语与阿拉伯语之间的语言关系，强调了它们的历史和文化联系。该研究展示了深度学习在合并计算语言学与考古学以解译古代文字方面的能力，从而为理解和保护人类历史提供了重要的见解。 

---
# Proceedings The 13th International Workshop on Theorem proving components for Educational software 

**Title (ZH)**: 第13届定理证明组件在教育软件研讨会论文集 

**Authors**: Julien Narboux, Walther Neuper, Pedro Quaresma  

**Link**: [PDF](https://arxiv.org/pdf/2505.04677)  

**Abstract**: The ThEdu series pursues the smooth transition from an intuitive way of doing mathematics at secondary school to a more formal approach to the subject in STEM education while favoring software support for this transition by exploiting the power of theorem-proving technologies.  What follows is a brief description of how the present volume contributes to this enterprise.  The 13th International Workshop on Theorem Proving Components for Educational Software (ThEdu'24), was a satellite event of the CADE29, part of IJCAR 2024, Nancy, France. ThEdu'24 was a vibrant workshop, with one invited talk by Jeremy Avigad (Carnegie Mellon University) and 14 submitted talks. An open call for papers was then issued and attracted 9 submissions. Eight of those submissions have been accepted by our reviewers. The resulting revised papers are collected in the present volume. The contributions in this volume are a faithful representation of the wide spectrum of ThEdu, ranging from those more focused on the automated deduction research, not losing track of the possible applications in an educational setting, to those focused on the applications, in educational settings, of automated deduction tools and methods. We, the volume editors, hope that this collection of papers will further promote the development of theorem-proving-based software and that it will allow to improve the mutual understanding between computer scientists, mathematicians, and stakeholders in education. While this volume goes to press, the next edition of the ThEdu workshop is being prepared: ThEdu'25 will be a satellite event of the 30th international Conference on Automated DEduction (CADE-30), July 28th - August 2nd, 2025, Stuttgart, Germany. 

**Abstract (ZH)**: The ThEdu 系列追求从中学直观的数学方法平滑过渡到 STEM 教育中更为形式化的处理方法，并通过利用定理证明技术的力量来支持这一过渡。以下是本卷对此项工作的贡献的简要描述。第十三届定理证明组件教育软件国际研讨会（ThEdu'24）是 CADE29（IJCAR 2024 的卫星会议）的一部分，在法国南锡举行。ThEdu'24 是一个充满活力的工作组，包括一场由 Jeremy Avigad（卡内基梅隆大学）举办的特邀演讲，以及 14 场提交的演讲。随后发出了一篇论文征稿，并收到了 9 篇投稿。其中 8 篇投稿被我们的审稿人接受。这些经过修订的论文收录在本卷中。本卷中的贡献忠实反映了 ThEdu 的广泛范围，从更多关注自动推理研究，但不忽视教育应用，到更多关注自动化推理工具和方法在教育环境中的应用。我们，作为该卷的编辑，希望这些论文的集合将进一步促进基于定理证明的软件的发展，并有助于提高计算机科学家、数学家和教育利益相关者之间的相互理解。与此同时，ThEdu 工作坊的下一届会议正在筹备中：ThEdu'25 将作为第 30 届自动推理国际会议（CADE-30）的卫星会议，在 2025 年 7 月 28 日至 8 月 2 日在德国斯图加特举行。 

---
# Language translation, and change of accent for speech-to-speech task using diffusion model 

**Title (ZH)**: 使用扩散模型进行语言翻译及语音换音的语音到语音任务 

**Authors**: Abhishek Mishra, Ritesh Sur Chowdhury, Vartul Bahuguna, Isha Pandey, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2505.04639)  

**Abstract**: Speech-to-speech translation (S2ST) aims to convert spoken input in one language to spoken output in another, typically focusing on either language translation or accent adaptation. However, effective cross-cultural communication requires handling both aspects simultaneously - translating content while adapting the speaker's accent to match the target language context. In this work, we propose a unified approach for simultaneous speech translation and change of accent, a task that remains underexplored in current literature. Our method reformulates the problem as a conditional generation task, where target speech is generated based on phonemes and guided by target speech features. Leveraging the power of diffusion models, known for high-fidelity generative capabilities, we adapt text-to-image diffusion strategies by conditioning on source speech transcriptions and generating Mel spectrograms representing the target speech with desired linguistic and accentual attributes. This integrated framework enables joint optimization of translation and accent adaptation, offering a more parameter-efficient and effective model compared to traditional pipelines. 

**Abstract (ZH)**: 跨文化口语翻译与口音转换（S2ST）旨在将一种语言的口语输入转换为另一种语言的口语输出，通常专注于语言翻译或口音适应。然而，有效的跨文化沟通要求同时处理这两个方面：内容翻译与说话人口音的适配以匹配目标语言背景。在本文中，我们提出了一种统一的方法，用于同时进行口语翻译和口音转换，这是一个当前文献中研究不足的任务。我们的方法将问题重新表述为条件生成任务，其中目标口语基于音素生成并由目标口语特征引导。利用扩散模型的强大生成能力，我们通过在源口语转录的基础上进行条件设置，并生成表示具有所需语言和口音属性的目标口语的梅尔频谱图，来适应文本到图像的扩散策略。通过这种集成框架，可以实现翻译和口音转换的联合优化，相比传统流水线方法，提供了一个更具参数效率和有效性的模型。 

---
# From Dialect Gaps to Identity Maps: Tackling Variability in Speaker Verification 

**Title (ZH)**: 从方言差距到身份映射：应对说话人验证中的变异性 

**Authors**: Abdulhady Abas Abdullah, Soran Badawi, Dana A. Abdullah, Dana Rasul Hamad, Hanan Abdulrahman Taher, Sabat Salih Muhamad, Aram Mahmood Ahmed, Bryar A. Hassan, Sirwan Abdolwahed Aula, Tarik A. Rashid  

**Link**: [PDF](https://arxiv.org/pdf/2505.04629)  

**Abstract**: The complexity and difficulties of Kurdish speaker detection among its several dialects are investigated in this work. Because of its great phonetic and lexical differences, Kurdish with several dialects including Kurmanji, Sorani, and Hawrami offers special challenges for speaker recognition systems. The main difficulties in building a strong speaker identification system capable of precisely identifying speakers across several dialects are investigated in this work. To raise the accuracy and dependability of these systems, it also suggests solutions like sophisticated machine learning approaches, data augmentation tactics, and the building of thorough dialect-specific corpus. The results show that customized strategies for every dialect together with cross-dialect training greatly enhance recognition performance. 

**Abstract (ZH)**: Kurdish方言 Speaker识别的复杂性与困难及其解决方案研究 

---
# Toward Holistic Evaluation of Recommender Systems Powered by Generative Models 

**Title (ZH)**: 面向生成模型驱动的推荐系统综合性评估 

**Authors**: Yashar Deldjoo, Nikhil Mehta, Maheswaran Sathiamoorthy, Shuai Zhang, Pablo Castells, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2504.06667)  

**Abstract**: Recommender systems powered by generative models (Gen-RecSys) extend beyond classical item ranking by producing open-ended content, which simultaneously unlocks richer user experiences and introduces new risks. On one hand, these systems can enhance personalization and appeal through dynamic explanations and multi-turn dialogues. On the other hand, they might venture into unknown territory-hallucinating nonexistent items, amplifying bias, or leaking private information. Traditional accuracy metrics cannot fully capture these challenges, as they fail to measure factual correctness, content safety, or alignment with user intent.
This paper makes two main contributions. First, we categorize the evaluation challenges of Gen-RecSys into two groups: (i) existing concerns that are exacerbated by generative outputs (e.g., bias, privacy) and (ii) entirely new risks (e.g., item hallucinations, contradictory explanations). Second, we propose a holistic evaluation approach that includes scenario-based assessments and multi-metric checks-incorporating relevance, factual grounding, bias detection, and policy compliance. Our goal is to provide a guiding framework so researchers and practitioners can thoroughly assess Gen-RecSys, ensuring effective personalization and responsible deployment. 

**Abstract (ZH)**: 基于生成模型的推荐系统（Gen-RecSys）超越了经典项排序，通过生成开放式内容同时提升了用户体验并引入了新的风险。一方面，这些系统可以通过动态解释和多轮对话增强个性化和吸引力。另一方面，它们可能闯入未知领域，如虚构不存在的项目、放大偏见或泄露私人信息。传统准确性指标无法充分捕捉这些挑战，因为它们未能衡量事实正确性、内容安全性或与用户意图的一致性。

本文主要贡献两个方面：首先，我们将Gen-RecSys的评估挑战分为两类：（i）生成输出加剧的现有问题（如偏见、隐私），和（ii）全新的风险（如项目幻觉、矛盾的解释）。其次，我们提出了一种综合的评估方法，包括基于场景的评估和多指标检查，涵盖了相关性、事实基础、偏见检测和政策合规性。我们的目标是提供一个指导框架，使研究者和实践者能够全面评估Gen-RecSys，确保有效的个性化和负责任的部署。 

---
