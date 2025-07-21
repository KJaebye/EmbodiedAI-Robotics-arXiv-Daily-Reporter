# MorphIt: Flexible Spherical Approximation of Robot Morphology for Representation-driven Adaptation 

**Title (ZH)**: MorphIt: 适用于表示驱动适应的机器人形态的灵活球形近似 

**Authors**: Nataliya Nechyporenko, Yutong Zhang, Sean Campbell, Alessandro Roncone  

**Link**: [PDF](https://arxiv.org/pdf/2507.14061)  

**Abstract**: What if a robot could rethink its own morphological representation to better meet the demands of diverse tasks? Most robotic systems today treat their physical form as a fixed constraint rather than an adaptive resource, forcing the same rigid geometric representation to serve applications with vastly different computational and precision requirements. We introduce MorphIt, a novel algorithm for approximating robot morphology using spherical primitives that balances geometric accuracy with computational efficiency. Unlike existing approaches that rely on either labor-intensive manual specification or inflexible computational methods, MorphIt implements an automatic gradient-based optimization framework with tunable parameters that provides explicit control over the physical fidelity versus computational cost tradeoff. Quantitative evaluations demonstrate that MorphIt outperforms baseline approaches (Variational Sphere Set Approximation and Adaptive Medial-Axis Approximation) across multiple metrics, achieving better mesh approximation with fewer spheres and reduced computational overhead. Our experiments show enhanced robot capabilities in collision detection accuracy, contact-rich interaction simulation, and navigation through confined spaces. By dynamically adapting geometric representations to task requirements, robots can now exploit their physical embodiment as an active resource rather than an inflexible parameter, opening new frontiers for manipulation in environments where physical form must continuously balance precision with computational tractability. 

**Abstract (ZH)**: 如果机器人能够重新构想自身的形态表示以更好地满足多样化任务的需求会怎样？当前大多数机器人系统将其实体形态视为固定的约束而非可适应的资源，被迫使用僵化的几何表示来服务于具有截然不同计算和精度要求的应用。我们引入了MorphIt，一种用于使用球形原语近似机器人形态的新算法，该算法平衡了几何精度与计算效率。与依赖于劳动密集型手工指定或不灵活计算方法的现有方法不同，MorphIt 实现了一种自动基于梯度的优化框架，具有可调参数，可提供对物理精度与计算成本权衡的显式控制。定量评估表明，MorphIt 在多个指标上优于基线方法（可变球集近似和自适应中轴逼近），使用更少的球体实现了更好的网格近似并减少了计算开销。我们的实验展示了机器人在碰撞检测准确性、富含接触的交互模拟以及在受限空间中的导航方面的增强能力。通过动态适应几何表示以满足任务需求，机器人现在可以将其物理形态作为活性资源而非僵化的参数加以利用，从而在那些物理形态必须不断平衡精度与计算可处理性的环境中开拓新的操控前沿。 

---
# EdgeVLA: Efficient Vision-Language-Action Models 

**Title (ZH)**: EdgeVLA: 高效的视觉-语言-动作模型 

**Authors**: Paweł Budzianowski, Wesley Maa, Matthew Freed, Jingxiang Mo, Winston Hsiao, Aaron Xie, Tomasz Młoduchowski, Viraj Tipnis, Benjamin Bolte  

**Link**: [PDF](https://arxiv.org/pdf/2507.14049)  

**Abstract**: Vision-Language Models (VLMs) have emerged as a promising approach to address the data scarcity challenge in robotics, enabling the development of generalizable visuomotor control policies. While models like OpenVLA showcase the potential of this paradigm, deploying large-scale VLMs on resource-constrained mobile manipulation systems remains a significant hurdle. This paper introduces Edge VLA (EVLA), a novel approach designed to significantly enhance the inference speed of Vision-Language-Action (VLA) models. EVLA maintains the representational power of these models while enabling real-time performance on edge devices. We achieve this through two key innovations: 1) Eliminating the autoregressive requirement for end-effector position prediction, leading to a 7x speedup in inference, and 2) Leveraging the efficiency of Small Language Models (SLMs), demonstrating comparable training performance to larger models with significantly reduced computational demands. Our early results demonstrate that EVLA achieves comparable training characteristics to OpenVLA while offering substantial gains in inference speed and memory efficiency. We release our model checkpoints and training \href{this https URL }{codebase} to foster further research. 

**Abstract (ZH)**: Vision-Language-Action模型(EVLA)：边端实时视觉-语言-行动模型 

---
# A segmented robot grasping perception neural network for edge AI 

**Title (ZH)**: 边缘AI中的分段机器人抓取感知神经网络 

**Authors**: Casper Bröcheler, Thomas Vroom, Derrick Timmermans, Alan van den Akker, Guangzhi Tang, Charalampos S. Kouzinopoulos, Rico Möckel  

**Link**: [PDF](https://arxiv.org/pdf/2507.13970)  

**Abstract**: Robotic grasping, the ability of robots to reliably secure and manipulate objects of varying shapes, sizes and orientations, is a complex task that requires precise perception and control. Deep neural networks have shown remarkable success in grasp synthesis by learning rich and abstract representations of objects. When deployed at the edge, these models can enable low-latency, low-power inference, making real-time grasping feasible in resource-constrained environments. This work implements Heatmap-Guided Grasp Detection, an end-to-end framework for the detection of 6-Dof grasp poses, on the GAP9 RISC-V System-on-Chip. The model is optimised using hardware-aware techniques, including input dimensionality reduction, model partitioning, and quantisation. Experimental evaluation on the GraspNet-1Billion benchmark validates the feasibility of fully on-chip inference, highlighting the potential of low-power MCUs for real-time, autonomous manipulation. 

**Abstract (ZH)**: 基于Heatmap引导的6-自由度抓取检测在GAP9 RISC-V系统芯片上的实现 

---
# NeHMO: Neural Hamilton-Jacobi Reachability Learning for Decentralized Safe Multi-Agent Motion Planning 

**Title (ZH)**: NeHMO: 基于神经哈密尔顿-雅可比可达性学习的分布式安全多Agent运动规划 

**Authors**: Qingyi Chen, Ahmed H. Qureshi  

**Link**: [PDF](https://arxiv.org/pdf/2507.13940)  

**Abstract**: Safe Multi-Agent Motion Planning (MAMP) is a significant challenge in robotics. Despite substantial advancements, existing methods often face a dilemma. Decentralized algorithms typically rely on predicting the behavior of other agents, sharing contracts, or maintaining communication for safety, while centralized approaches struggle with scalability and real-time decision-making. To address these challenges, we introduce Neural Hamilton-Jacobi Reachability Learning (HJR) for Decentralized Multi-Agent Motion Planning. Our method provides scalable neural HJR modeling to tackle high-dimensional configuration spaces and capture worst-case collision and safety constraints between agents. We further propose a decentralized trajectory optimization framework that incorporates the learned HJR solutions to solve MAMP tasks in real-time. We demonstrate that our method is both scalable and data-efficient, enabling the solution of MAMP problems in higher-dimensional scenarios with complex collision constraints. Our approach generalizes across various dynamical systems, including a 12-dimensional dual-arm setup, and outperforms a range of state-of-the-art techniques in successfully addressing challenging MAMP tasks. Video demonstrations are available at this https URL. 

**Abstract (ZH)**: 安全多Agent运动规划（Safe Multi-Agent Motion Planning, MAMP）是机器人技术中的一个重要挑战。尽管取得了显著进展，现有方法通常面临着困境。去中心化算法通常依赖于预测其他Agent的行为、共享合约或保持通信以确保安全性，而中心化方法则难以实现可扩展性和实时决策。为了解决这些问题，我们提出了用于去中心化多Agent运动规划的神经哈密尔顿-雅可比可达性学习（Neural Hamilton-Jacobi Reachability Learning, NJRL）方法。该方法提供了可扩展的神经NJVL建模，以应对高维配置空间并捕捉代理间的最坏情况碰撞和安全性约束。我们进一步提出了一种去中心化轨迹优化框架，该框架结合了学习到的NJVL解决方案，以实时解决MAMP任务。我们证明了该方法既可扩展又高效，能够在更高维度且有复杂碰撞约束的情景下解决MAMP问题。该方法适用于多种动力学系统，包括12维双臂设置，并在成功应对具有挑战性的MAMP任务方面优于多种现有先进技术。视频演示可在该链接获取。 

---
# Safety Certification in the Latent space using Control Barrier Functions and World Models 

**Title (ZH)**: 在潜在空间中使用控制约束函数和世界模型进行安全性认证 

**Authors**: Mehul Anand, Shishir Kolathaya  

**Link**: [PDF](https://arxiv.org/pdf/2507.13871)  

**Abstract**: Synthesising safe controllers from visual data typically requires extensive supervised labelling of safety-critical data, which is often impractical in real-world settings. Recent advances in world models enable reliable prediction in latent spaces, opening new avenues for scalable and data-efficient safe control. In this work, we introduce a semi-supervised framework that leverages control barrier certificates (CBCs) learned in the latent space of a world model to synthesise safe visuomotor policies. Our approach jointly learns a neural barrier function and a safe controller using limited labelled data, while exploiting the predictive power of modern vision transformers for latent dynamics modelling. 

**Abstract (ZH)**: 基于视觉数据合成安全控制器通常需要对安全关键数据进行广泛的监督标注，这在实际应用场景中往往不可行。近期世界模型的发展使得在潜在空间中进行可靠预测成为可能，从而为可扩展且数据效率高的安全控制开辟了新的途径。在本工作中，我们引入了一种半监督框架，该框架利用世界模型潜在空间中学习到的控制屏障证书（CBCs）来合成安全的视觉-运动策略。我们的方法在有限的标注数据下联合学习神经屏障函数和安全控制器，并利用现代视觉变压器的预测能力进行潜在动态建模。 

---
# Iteratively Learning Muscle Memory for Legged Robots to Master Adaptive and High Precision Locomotion 

**Title (ZH)**: 迭代学习肌肉记忆以实现腿部机器人适应性和高精度运动掌握 

**Authors**: Jing Cheng, Yasser G. Alqaham, Zhenyu Gan, Amit K. Sanyal  

**Link**: [PDF](https://arxiv.org/pdf/2507.13662)  

**Abstract**: This paper presents a scalable and adaptive control framework for legged robots that integrates Iterative Learning Control (ILC) with a biologically inspired torque library (TL), analogous to muscle memory. The proposed method addresses key challenges in robotic locomotion, including accurate trajectory tracking under unmodeled dynamics and external disturbances. By leveraging the repetitive nature of periodic gaits and extending ILC to nonperiodic tasks, the framework enhances accuracy and generalization across diverse locomotion scenarios. The control architecture is data-enabled, combining a physics-based model derived from hybrid-system trajectory optimization with real-time learning to compensate for model uncertainties and external disturbances. A central contribution is the development of a generalized TL that stores learned control profiles and enables rapid adaptation to changes in speed, terrain, and gravitational conditions-eliminating the need for repeated learning and significantly reducing online computation. The approach is validated on the bipedal robot Cassie and the quadrupedal robot A1 through extensive simulations and hardware experiments. Results demonstrate that the proposed framework reduces joint tracking errors by up to 85% within a few seconds and enables reliable execution of both periodic and nonperiodic gaits, including slope traversal and terrain adaptation. Compared to state-of-the-art whole-body controllers, the learned skills eliminate the need for online computation during execution and achieve control update rates exceeding 30x those of existing methods. These findings highlight the effectiveness of integrating ILC with torque memory as a highly data-efficient and practical solution for legged locomotion in unstructured and dynamic environments. 

**Abstract (ZH)**: 一种将迭代学习控制与生物启发扭矩库结合的可扩展自适应腿足机器人控制框架 

---
# A Study of Teleoperation Methods in a Simulated Virtual Eye Surgery Environment 

**Title (ZH)**: 虚拟眼科手术环境中遥控操作方法的研究 

**Authors**: Haoran Wang, Yasamin Foroutani, Matthew Nepo, Mercedes Rodriguez, Ji Ma, Jean-Pierre Hubschman, Tsu-Chin Tsao, Jacob Rosen  

**Link**: [PDF](https://arxiv.org/pdf/2507.13654)  

**Abstract**: This paper examines the performance of Inside and Outside Control modes at various scaling factors in a simulated vitreoretinal surgical setting. The IRISS teleoperated surgical system's console (cockpit) was adapted to project a simulated microscope view of an intraocular setup to a virtual reality (VR) headset. Five experienced vitreoretinal surgeons and five engineers with no surgical experience used the system to perform tasks common to vitreoretinal surgery. Experimental results indicate that Inside Control methods at higher scaling factors (20 or 30) achieved the best performance overall, though the optimal scaling factor may vary by task and complexity. Optimizing control methods and scaling factors could lead to improvements in surgical efficiency and accuracy, as well as minimize risks in future robotic-assisted intraocular procedures. 

**Abstract (ZH)**: 本文在模拟玻璃体视网膜手术环境中，研究了不同缩放因子下Inside和Outside控制模式的性能。IRISS远程操作手术系统的控制台（仪表板）被适配，以将模拟的手术室内视图投影到虚拟现实（VR）头显上。五名经验丰富的玻璃体视网膜外科医生和五名无手术经验的工程师使用该系统执行常见的玻璃体视网膜手术任务。实验结果表明，在较高的缩放因子（20或30）下，Inside控制方法的整体性能最佳，尽管最优的缩放因子可能因任务和复杂度而异。优化控制方法和缩放因子可能有助于提高手术效率和准确性，并在未来的机器人辅助眼内手术中降低风险。 

---
# SCOPE for Hexapod Gait Generation 

**Title (ZH)**: 六足机器人步态生成的SCOPE方法 

**Authors**: Jim O'Connor, Jay B. Nash, Derin Gezgin, Gary B. Parker  

**Link**: [PDF](https://arxiv.org/pdf/2507.13539)  

**Abstract**: Evolutionary methods have previously been shown to be an effective learning method for walking gaits on hexapod robots. However, the ability of these algorithms to evolve an effective policy rapidly degrades as the input space becomes more complex. This degradation is due to the exponential growth of the solution space, resulting from an increasing parameter count to handle a more complex input. In order to address this challenge, we introduce Sparse Cosine Optimized Policy Evolution (SCOPE). SCOPE utilizes the Discrete Cosine Transform (DCT) to learn directly from the feature coefficients of an input matrix. By truncating the coefficient matrix returned by the DCT, we can reduce the dimensionality of an input while retaining the highest energy features of the original input. We demonstrate the effectiveness of this method by using SCOPE to learn the gait of a hexapod robot. The hexapod controller is given a matrix input containing time-series information of previous poses, which are then transformed to gait parameters by an evolved policy. In this task, the addition of SCOPE to a reference algorithm achieves a 20% increase in efficacy. SCOPE achieves this result by reducing the total input size of the time-series pose data from 2700 to 54, a 98% decrease. Additionally, SCOPE is capable of compressing an input to any output shape, provided that each output dimension is no greater than the corresponding input dimension. This paper demonstrates that SCOPE is capable of significantly compressing the size of an input to an evolved controller, resulting in a statistically significant gain in efficacy. 

**Abstract (ZH)**: Sparse Cosine Optimized Policy Evolution for Effective Learning of Hexapod Walking Gaits 

---
# Conceptual and Design Principles for a Self-Referential Algorithm Mimicking Neuronal Assembly Functions 

**Title (ZH)**: 自我参照算法的设计原理：模仿神经元组装功能 

**Authors**: Paolo Totaro, Alberto Mangiante  

**Link**: [PDF](https://arxiv.org/pdf/2507.14011)  

**Abstract**: This article proposes a method to formalise models of cognitive processes grounded in experience, considering experience from the perspective of a living system and not from that of an observer of the living system. The perspective of a living system is defined by the need of the system to preserve the vital equilibria. The method is based on an algorithmic schema that we call Environment Generative Operator (EGO) and uses a self-referential language developed for this purpose which we call E-language. EGO simulates cognitive processes as operations on neuron assemblies as understood by Hebb. In this article we present an EGO prototype (EGO-P) which has already been implemented and tested. 

**Abstract (ZH)**: 本文提出了一种方法，用于形式化基于经验的认知过程模型，从一个生活系统的视角而非观察者视角来考虑经验。生活系统的视角定义为系统保持生命平衡的需要。该方法基于一种我们称之为环境生成算子（EGO）的算法框架，并使用专门为这一目的开发的一种自参照语言，称为E语言。EGO将认知过程模拟为对海伯所理解的神经元集合的操作。本文介绍了已经实现并测试的一种EGO原型（EGO-P）。 

---
# Generative AI-Driven High-Fidelity Human Motion Simulation 

**Title (ZH)**: 由生成式AI驱动的高保真人体运动模拟 

**Authors**: Hari Iyer, Neel Macwan, Atharva Jitendra Hude, Heejin Jeong, Shenghan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.14097)  

**Abstract**: Human motion simulation (HMS) supports cost-effective evaluation of worker behavior, safety, and productivity in industrial tasks. However, existing methods often suffer from low motion fidelity. This study introduces Generative-AI-Enabled HMS (G-AI-HMS), which integrates text-to-text and text-to-motion models to enhance simulation quality for physical tasks. G-AI-HMS tackles two key challenges: (1) translating task descriptions into motion-aware language using Large Language Models aligned with MotionGPT's training vocabulary, and (2) validating AI-enhanced motions against real human movements using computer vision. Posture estimation algorithms are applied to real-time videos to extract joint landmarks, and motion similarity metrics are used to compare them with AI-enhanced sequences. In a case study involving eight tasks, the AI-enhanced motions showed lower error than human created descriptions in most scenarios, performing better in six tasks based on spatial accuracy, four tasks based on alignment after pose normalization, and seven tasks based on overall temporal similarity. Statistical analysis showed that AI-enhanced prompts significantly (p $<$ 0.0001) reduced joint error and temporal misalignment while retaining comparable posture accuracy. 

**Abstract (ZH)**: 基于生成AI的Humanmotion仿真：提高工业任务中动作保真度的方法 

---
# From Extraction to Synthesis: Entangled Heuristics for Agent-Augmented Strategic Reasoning 

**Title (ZH)**: 从提取到综合：代理增强战略推理中的纠缠启发式方法 

**Authors**: Renato Ghisellini, Remo Pareschi, Marco Pedroni, Giovanni Battista Raggi  

**Link**: [PDF](https://arxiv.org/pdf/2507.13768)  

**Abstract**: We present a hybrid architecture for agent-augmented strategic reasoning, combining heuristic extraction, semantic activation, and compositional synthesis. Drawing on sources ranging from classical military theory to contemporary corporate strategy, our model activates and composes multiple heuristics through a process of semantic interdependence inspired by research in quantum cognition. Unlike traditional decision engines that select the best rule, our system fuses conflicting heuristics into coherent and context-sensitive narratives, guided by semantic interaction modeling and rhetorical framing. We demonstrate the framework via a Meta vs. FTC case study, with preliminary validation through semantic metrics. Limitations and extensions (e.g., dynamic interference tuning) are discussed. 

**Abstract (ZH)**: 基于启发式提取、语义激活和组合合成的代理增强战略推理混合架构 

---
# Binarizing Physics-Inspired GNNs for Combinatorial Optimization 

**Title (ZH)**: 基于物理启发的二值化GNNs在组合优化中的应用 

**Authors**: Martin Krutský, Gustav Šír, Vyacheslav Kungurtsev, Georgios Korpas  

**Link**: [PDF](https://arxiv.org/pdf/2507.13703)  

**Abstract**: Physics-inspired graph neural networks (PI-GNNs) have been utilized as an efficient unsupervised framework for relaxing combinatorial optimization problems encoded through a specific graph structure and loss, reflecting dependencies between the problem's variables. While the framework has yielded promising results in various combinatorial problems, we show that the performance of PI-GNNs systematically plummets with an increasing density of the combinatorial problem graphs. Our analysis reveals an interesting phase transition in the PI-GNNs' training dynamics, associated with degenerate solutions for the denser problems, highlighting a discrepancy between the relaxed, real-valued model outputs and the binary-valued problem solutions. To address the discrepancy, we propose principled alternatives to the naive strategy used in PI-GNNs by building on insights from fuzzy logic and binarized neural networks. Our experiments demonstrate that the portfolio of proposed methods significantly improves the performance of PI-GNNs in increasingly dense settings. 

**Abstract (ZH)**: 物理启发的图神经网络（PI-GNNs）已被用作一个高效的无监督框架，用于通过特定的图结构和损失函数来放松编码组合优化问题，反映变量之间的依赖关系。尽管该框架在各种组合问题中取得了令人鼓舞的结果，我们发现PI-GNNs在组合问题图密度增加时其性能系统性地下降。我们的分析揭示了PI-GNNs训练动力学中有趣的相转变，与更密集的问题相关的退化解，强调了放松的实值模型输出与二进制值问题解之间的差距。为了弥合这一差距，我们基于模糊逻辑和二值神经网络的见解，提出了一系列原则性的替代方案，以改进PI-GNNs的原始策略。实验结果表明，所提出方法的组合在日益密集的设置中显著提高了PI-GNNs的性能。 

---
# Humans learn to prefer trustworthy AI over human partners 

**Title (ZH)**: 人类更偏好可信赖的AI伙伴 

**Authors**: Yaomin Jiang, Levin Brinkmann, Anne-Marie Nussberger, Ivan Soraperra, Jean-François Bonnefon, Iyad Rahwan  

**Link**: [PDF](https://arxiv.org/pdf/2507.13524)  

**Abstract**: Partner selection is crucial for cooperation and hinges on communication. As artificial agents, especially those powered by large language models (LLMs), become more autonomous, intelligent, and persuasive, they compete with humans for partnerships. Yet little is known about how humans select between human and AI partners and adapt under AI-induced competition pressure. We constructed a communication-based partner selection game and examined the dynamics in hybrid mini-societies of humans and bots powered by a state-of-the-art LLM. Through three experiments (N = 975), we found that bots, though more prosocial than humans and linguistically distinguishable, were not selected preferentially when their identity was hidden. Instead, humans misattributed bots' behaviour to humans and vice versa. Disclosing bots' identity induced a dual effect: it reduced bots' initial chances of being selected but allowed them to gradually outcompete humans by facilitating human learning about the behaviour of each partner type. These findings show how AI can reshape social interaction in mixed societies and inform the design of more effective and cooperative hybrid systems. 

**Abstract (ZH)**: 人工伙伴选择对于合作至关重要且依赖于沟通。随着尤其是由大规模语言模型（LLMs）驱动的智能代理变得更为自主、智能和有说服力，它们在合作伙伴中与人类竞争。然而，关于人类在人工智能引起的竞争压力下选择人类和AI合作伙伴及其适应机制，我们知之甚少。我们构建了一个基于沟通的人工伙伴选择博弈，并考察了人类和由先进LLM驱动的智能代理组成的混合小社会中的动态变化。通过三项实验（N=975），我们发现，尽管智能代理比人类更有利他性且在语言上有明显区分，但在其身份未被揭示时，并未被优先选择。相反，人类错误地将智能代理的行为归因给人类，反之亦然。揭示智能代理的身份产生了双重效果：它减少了智能代理最初被选择的机会，但让他们能够通过促进人类对每种合作伙伴类型行为的理解而逐步胜过人类。这些发现展示了AI如何重新塑造混合社会中的社会互动，并为设计更有效的混合系统提供指导。 

---
# ERR@HRI 2.0 Challenge: Multimodal Detection of Errors and Failures in Human-Robot Conversations 

**Title (ZH)**: ERR@HRI 2.0 挑战赛：人类-机器人对话中多模态错误和故障检测 

**Authors**: Shiye Cao, Maia Stiber, Amama Mahmood, Maria Teresa Parreira, Wendy Ju, Micol Spitale, Hatice Gunes, Chien-Ming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.13468)  

**Abstract**: The integration of large language models (LLMs) into conversational robots has made human-robot conversations more dynamic. Yet, LLM-powered conversational robots remain prone to errors, e.g., misunderstanding user intent, prematurely interrupting users, or failing to respond altogether. Detecting and addressing these failures is critical for preventing conversational breakdowns, avoiding task disruptions, and sustaining user trust. To tackle this problem, the ERR@HRI 2.0 Challenge provides a multimodal dataset of LLM-powered conversational robot failures during human-robot conversations and encourages researchers to benchmark machine learning models designed to detect robot failures. The dataset includes 16 hours of dyadic human-robot interactions, incorporating facial, speech, and head movement features. Each interaction is annotated with the presence or absence of robot errors from the system perspective, and perceived user intention to correct for a mismatch between robot behavior and user expectation. Participants are invited to form teams and develop machine learning models that detect these failures using multimodal data. Submissions will be evaluated using various performance metrics, including detection accuracy and false positive rate. This challenge represents another key step toward improving failure detection in human-robot interaction through social signal analysis. 

**Abstract (ZH)**: 大型语言模型（LLMs）在对话机器人中的集成使得人机对话更加动态，但由LLM驱动的对话机器人仍然容易出现错误，例如误解用户意图、过早中断用户或完全无法响应。检测和解决这些故障对于防止对话中断、避免任务中断并维持用户信任至关重要。为应对这一挑战，ERR@HRI 2.0竞赛提供了一种多模态数据集，其中包括LLM驱动的对话机器人在人机对话过程中出现的故障，并鼓励研究人员使用机器学习模型来检测机器人故障。该数据集包括16小时的双向人机互动，整合了面部、语音和头部动作特征。每项互动都从系统视角标注了机器人故障的有无，并考虑到用户意图的感知，以纠正机器人行为与用户期望之间的不一致。参赛者被邀请组队并开发能够使用多模态数据检测这些故障的机器学习模型。提交将根据检测准确率和假阳性率等性能指标进行评估。该挑战代表了通过社会信号分析提高人机交互中故障检测的又一个重要步骤。 

---
# Enhancing Spatial Reasoning in Vision-Language Models via Chain-of-Thought Prompting and Reinforcement Learning 

**Title (ZH)**: 通过链式思考提示和强化学习提升视觉语言模型的空间推理能力 

**Authors**: Binbin Ji, Siddharth Agrawal, Qiance Tang, Yvonne Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.13362)  

**Abstract**: This study investigates the spatial reasoning capabilities of vision-language models (VLMs) through Chain-of-Thought (CoT) prompting and reinforcement learning. We begin by evaluating the impact of different prompting strategies and find that simple CoT formats, where the model generates a reasoning step before the answer, not only fail to help, but can even harm the model's original performance. In contrast, structured multi-stage prompting based on scene graphs (SceneGraph CoT) significantly improves spatial reasoning accuracy. Furthermore, to improve spatial reasoning ability, we fine-tune models using Group Relative Policy Optimization (GRPO) on the SAT dataset and evaluate their performance on CVBench. Compared to supervised fine-tuning (SFT), GRPO achieves higher accuracy on Pass@1 evaluations and demonstrates superior robustness under out-of-distribution (OOD) conditions. In particular, we find that SFT overfits to surface-level linguistic patterns and may degrade performance when test-time phrasing changes (e.g., from "closer to" to "farther from"). GRPO, on the other hand, generalizes more reliably and maintains stable performance under such shifts. Our findings provide insights into how reinforcement learning and structured prompting improve the spatial reasoning capabilities and generalization behavior of modern VLMs. All code is open source at: this https URL 

**Abstract (ZH)**: 本研究通过链式思考（CoT）提示和强化学习探讨了视觉语言模型（VLMs）的空间推理能力。我们首先评估了不同提示策略的影响，并发现简单形式的CoT格式（模型在给出答案前生成一个推理步骤），不仅没有帮助，反而可能损害模型的原始性能。相比之下，基于场景图的结构化多阶段提示（SceneGraph CoT）显著提高了空间推理的准确性。为进一步提高空间推理能力，我们使用组相对策略优化（GRPO）对SAT数据集进行微调，并在CVBench上评估其性能。与监督微调（SFT）相比，GRPO在Pass@1评估中实现了更高的准确率，并在异常分布（OOD）条件下表现出更好的鲁棒性。特别是，我们发现SFT过度拟合于表面语言模式，在测试时措辞改变（如从“靠近”变为“远离”）时可能会损害性能。GRPO则更可靠地泛化，并在这些变化下保持了稳定的性能。我们的发现提供了关于强化学习和结构化提示如何改善现代VLMs的空间推理能力和泛化行为的见解。所有代码均开源于：this https URL。 

---
# Generalist Bimanual Manipulation via Foundation Video Diffusion Models 

**Title (ZH)**: 双手通用操作 via 基础视频扩散模型 

**Authors**: Yao Feng, Hengkai Tan, Xinyi Mao, Guodong Liu, Shuhe Huang, Chendong Xiang, Hang Su, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.12898)  

**Abstract**: Bimanual robotic manipulation, which involves the coordinated control of two robotic arms, is foundational for solving challenging tasks. Despite recent progress in general-purpose manipulation, data scarcity and embodiment heterogeneity remain serious obstacles to further scaling up in bimanual settings. In this paper, we introduce VIdeo Diffusion for Action Reasoning (VIDAR), a two-stage framework that leverages large-scale, diffusion-based video pre-training and a novel masked inverse dynamics model for action prediction. We pre-train the video diffusion model on 750K multi-view videos from three real-world bimanual robot platforms, utilizing a unified observation space that encodes robot, camera, task, and scene contexts. Our masked inverse dynamics model learns masks to extract action-relevant information from generated trajectories without requiring pixel-level labels, and the masks can effectively generalize to unseen backgrounds. Our experiments demonstrate that with only 20 minutes of human demonstrations on an unseen robot platform (only 1% of typical data requirements), VIDAR generalizes to unseen tasks and backgrounds with strong semantic understanding, surpassing state-of-the-art methods. Our findings highlight the potential of video foundation models, coupled with masked action prediction, to enable scalable and generalizable robotic manipulation in diverse real-world settings. 

**Abstract (ZH)**: 双臂机器人操控，即通过协调控制两台机器人手臂来解决复杂任务的基础方法，尽管在通用操控方面取得了进展，但在双臂设置中，数据稀疏性和实体异质性仍然是扩大规模的重大障碍。在本文中，我们介绍了基于视频扩散的动作推理框架（VIDAR），该框架利用大规模的基于扩散的视频预训练和一种新的掩码反向动力学模型进行动作预测。我们使用统一的观测空间对来自三个真实世界双臂机器人平台的750K多视角视频进行视频扩散模型的预训练，该观测空间编码了机器人、相机、任务和场景上下文。我们的掩码反向动力学模型学习掩码以在生成的轨迹中提取与动作相关的信息，而不需要像素级标签，该掩码可以有效泛化到未见的背景中。我们的实验表明，在一个未见过的机器人平台上仅通过20分钟的人类示范（即普通数据需求的1%），VIDAR能够泛化到未见过的任务和背景中，展现出强大的语义理解，超越了当前最先进的方法。我们的研究结果强调了视频基础模型与掩码动作预测相结合的潜力，以在多种真实世界环境中实现可扩展和可泛化的机器人操控。 

---
