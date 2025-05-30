# Accessible and Pedagogically-Grounded Explainability for Human-Robot Interaction: A Framework Based on UDL and Symbolic Interfaces 

**Title (ZH)**: 基于UDL和符号接口的人机交互可访问性和以教学为导向的解释性框架 

**Authors**: Francisco J. Rodríguez Lera, Raquel Fernández Hernández, Sonia Lopez González, Miguel Angel González-Santamarta, Francisco Jesús Rodríguez Sedano, Camino Fernandez Llamas  

**Link**: [PDF](https://arxiv.org/pdf/2504.06189)  

**Abstract**: This paper presents a novel framework for accessible and pedagogically-grounded robot explainability, designed to support human-robot interaction (HRI) with users who have diverse cognitive, communicative, or learning needs. We combine principles from Universal Design for Learning (UDL) and Universal Design (UD) with symbolic communication strategies to facilitate the alignment of mental models between humans and robots. Our approach employs Asterics Grid and ARASAAC pictograms as a multimodal, interpretable front-end, integrated with a lightweight HTTP-to-ROS 2 bridge that enables real-time interaction and explanation triggering. We emphasize that explainability is not a one-way function but a bidirectional process, where human understanding and robot transparency must co-evolve. We further argue that in educational or assistive contexts, the role of a human mediator (e.g., a teacher) may be essential to support shared understanding. We validate our framework with examples of multimodal explanation boards and discuss how it can be extended to different scenarios in education, assistive robotics, and inclusive AI. 

**Abstract (ZH)**: 一种面向多元认知、沟通或学习需求的可访问性与教学导向的机器人可解释性新框架：支持共融设计的人机交互 

---
# ViTaMIn: Learning Contact-Rich Tasks Through Robot-Free Visuo-Tactile Manipulation Interface 

**Title (ZH)**: ViTaMIn: 通过机器人无介入的视触觉 manipulation 接口学习接触密集型任务 

**Authors**: Fangchen Liu, Chuanyu Li, Yihua Qin, Ankit Shaw, Jing Xu, Pieter Abbeel, Rui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06156)  

**Abstract**: Tactile information plays a crucial role for humans and robots to interact effectively with their environment, particularly for tasks requiring the understanding of contact properties. Solving such dexterous manipulation tasks often relies on imitation learning from demonstration datasets, which are typically collected via teleoperation systems and often demand substantial time and effort. To address these challenges, we present ViTaMIn, an embodiment-free manipulation interface that seamlessly integrates visual and tactile sensing into a hand-held gripper, enabling data collection without the need for teleoperation. Our design employs a compliant Fin Ray gripper with tactile sensing, allowing operators to perceive force feedback during manipulation for more intuitive operation. Additionally, we propose a multimodal representation learning strategy to obtain pre-trained tactile representations, improving data efficiency and policy robustness. Experiments on seven contact-rich manipulation tasks demonstrate that ViTaMIn significantly outperforms baseline methods, demonstrating its effectiveness for complex manipulation tasks. 

**Abstract (ZH)**: 触觉信息在人类和机器人与其环境有效交互中扮演重要角色，尤其对于需要理解接触性质的任务。解决这类灵巧 manipulation 任务通常依赖于从演示数据集中进行模仿学习，这些数据集通常通过遥操作系统收集，往往需要大量时间和精力。为应对这些挑战，我们提出了一种无需体感的 manipulation 接口 ViTaMIn，该接口无缝集成视觉和触觉传感器于手持式夹爪中，使得在无需遥操作的情况下收集数据成为可能。我们的设计采用顺应性 Fin Ray 夹爪并配备触觉传感器，使操作者在 manipulation 过程中能够感知力反馈，从而实现更直观的操作。此外，我们提出了一种多模态表示学习策略以获得预训练的触觉表示，提高数据效率和策略鲁棒性。在七个富含接触的任务上的实验表明，ViTaMIn 显著优于基线方法，证明了其在复杂 manipulation 任务中的有效性。 

---
# MAPLE: Encoding Dexterous Robotic Manipulation Priors Learned From Egocentric Videos 

**Title (ZH)**: MAPLE: 编码来自第一人称视频学习到的灵巧机器人操作先验知识 

**Authors**: Alexey Gavryushin, Xi Wang, Robert J. S. Malate, Chenyu Yang, Xiangyi Jia, Shubh Goel, Davide Liconti, René Zurbrügg, Robert K. Katzschmann, Marc Pollefeys  

**Link**: [PDF](https://arxiv.org/pdf/2504.06084)  

**Abstract**: Large-scale egocentric video datasets capture diverse human activities across a wide range of scenarios, offering rich and detailed insights into how humans interact with objects, especially those that require fine-grained dexterous control. Such complex, dexterous skills with precise controls are crucial for many robotic manipulation tasks, yet are often insufficiently addressed by traditional data-driven approaches to robotic manipulation. To address this gap, we leverage manipulation priors learned from large-scale egocentric video datasets to improve policy learning for dexterous robotic manipulation tasks. We present MAPLE, a novel method for dexterous robotic manipulation that exploits rich manipulation priors to enable efficient policy learning and better performance on diverse, complex manipulation tasks. Specifically, we predict hand-object contact points and detailed hand poses at the moment of hand-object contact and use the learned features to train policies for downstream manipulation tasks. Experimental results demonstrate the effectiveness of MAPLE across existing simulation benchmarks, as well as a newly designed set of challenging simulation tasks, which require fine-grained object control and complex dexterous skills. The benefits of MAPLE are further highlighted in real-world experiments using a dexterous robotic hand, whereas simultaneous evaluation across both simulation and real-world experiments has remained underexplored in prior work. 

**Abstract (ZH)**: 大规模第一人称视频数据集捕获了广泛场景下的多样化人类活动，提供了关于人类如何精细操控物体的丰富而详细的见解，尤其是那些需要精细灵巧控制的技能。这些复杂的灵巧技能对于许多机器人操作任务至关重要，但传统的数据驱动方法经常未能充分解决这些问题。为了解决这一问题，我们利用从大规模第一人称视频数据集中学习到的操作先验来改进灵巧机器人操作任务的策略学习。我们提出了MAPLE，一种新颖的灵巧机器人操作方法，利用丰富的操作先验以实现高效策略学习并在多样而复杂的操作任务中取得更好性能。具体来说，我们预测手物接触点和手物接触瞬间的详细手部姿态，并使用学习到的特征来训练下游操作任务的策略。实验结果表明，MAPLE在现有的仿真基准测试以及新设计的一系列具有精细物体控制需求和复杂灵巧技能的挑战性仿真任务中都表现出有效性。在真实世界实验中使用灵巧机器人手进一步突显了MAPLE的优势，而同时在仿真和真实世界实验中进行评估的研究在此之前尚不多见。 

---
# Deep RL-based Autonomous Navigation of Micro Aerial Vehicles (MAVs) in a complex GPS-denied Indoor Environment 

**Title (ZH)**: 基于深度强化学习的微型 aerial 车辆在复杂GPS受限室内环境中的自主导航 

**Authors**: Amit Kumar Singh, Prasanth Kumar Duba, P. Rajalakshmi  

**Link**: [PDF](https://arxiv.org/pdf/2504.05918)  

**Abstract**: The Autonomy of Unmanned Aerial Vehicles (UAVs) in indoor environments poses significant challenges due to the lack of reliable GPS signals in enclosed spaces such as warehouses, factories, and indoor facilities. Micro Aerial Vehicles (MAVs) are preferred for navigating in these complex, GPS-denied scenarios because of their agility, low power consumption, and limited computational capabilities. In this paper, we propose a Reinforcement Learning based Deep-Proximal Policy Optimization (D-PPO) algorithm to enhance realtime navigation through improving the computation efficiency. The end-to-end network is trained in 3D realistic meta-environments created using the Unreal Engine. With these trained meta-weights, the MAV system underwent extensive experimental trials in real-world indoor environments. The results indicate that the proposed method reduces computational latency by 91\% during training period without significant degradation in performance. The algorithm was tested on a DJI Tello drone, yielding similar results. 

**Abstract (ZH)**: 无人驾驶航空车辆（UAVs）在室内环境中的自主导航由于仓库、工厂等封闭空间缺乏可靠的GPS信号而面临重大挑战。微型空中车辆（MAVs）因其灵活、低功耗和有限的计算能力，成为在这些复杂且GPS受限场景中导航的优选方案。本文提出了一种基于强化学习的深度近端策略优化（D-PPO）算法，通过提高计算效率来增强实时导航性能。端到端的网络在使用Unreal Engine创建的三维实时元环境中进行训练。利用这些训练好的元权重，MAV系统在实际室内环境中进行了广泛的实验测试。结果表明，在训练期间，所提出的方法减少了91%的计算延迟，同时性能未出现显著下降。该算法在大疆Tello无人机上进行了测试，结果相似。 

---
# PTRL: Prior Transfer Deep Reinforcement Learning for Legged Robots Locomotion 

**Title (ZH)**: PTRL：先验转移深度强化学习在腿足机器人运动中的应用 

**Authors**: Haodong Huang, Shilong Sun, Zida Zhao, Hailin Huang, Changqing Shen, Wenfu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05629)  

**Abstract**: In the field of legged robot motion control, reinforcement learning (RL) holds great promise but faces two major challenges: high computational cost for training individual robots and poor generalization of trained models. To address these problems, this paper proposes a novel framework called Prior Transfer Reinforcement Learning (PTRL), which improves both training efficiency and model transferability across different robots. Drawing inspiration from model transfer techniques in deep learning, PTRL introduces a fine-tuning mechanism that selectively freezes layers of the policy network during transfer, making it the first to apply such a method in RL. The framework consists of three stages: pre-training on a source robot using the Proximal Policy Optimization (PPO) algorithm, transferring the learned policy to a target robot, and fine-tuning with partial network freezing. Extensive experiments on various robot platforms confirm that this approach significantly reduces training time while maintaining or even improving performance. Moreover, the study quantitatively analyzes how the ratio of frozen layers affects transfer results, providing valuable insights into optimizing the process. The experimental outcomes show that PTRL achieves better walking control performance and demonstrates strong generalization and adaptability, offering a promising solution for efficient and scalable RL-based control of legged robots. 

**Abstract (ZH)**: 基于腿部机器人运动控制领域的强化学习：Prior Transfer Reinforcement Learning (PTRL)框架的研究 

---
# Trust Through Transparency: Explainable Social Navigation for Autonomous Mobile Robots via Vision-Language Models 

**Title (ZH)**: 信任通过透明性：基于视觉语言模型的自主移动机器人可解释的社会导航 

**Authors**: Oluwadamilola Sotomi, Devika Kodi, Aliasghar Arab  

**Link**: [PDF](https://arxiv.org/pdf/2504.05477)  

**Abstract**: Service and assistive robots are increasingly being deployed in dynamic social environments; however, ensuring transparent and explainable interactions remains a significant challenge. This paper presents a multimodal explainability module that integrates vision language models and heat maps to improve transparency during navigation. The proposed system enables robots to perceive, analyze, and articulate their observations through natural language summaries. User studies (n=30) showed a preference of majority for real-time explanations, indicating improved trust and understanding. Our experiments were validated through confusion matrix analysis to assess the level of agreement with human expectations. Our experimental and simulation results emphasize the effectiveness of explainability in autonomous navigation, enhancing trust and interpretability. 

**Abstract (ZH)**: 服务和辅助机器人在动态社会环境中越来越多地被部署；然而，确保透明和可解释的交互仍是一项重大挑战。本文提出了一种多模态可解释模块，将视觉语言模型和热图结合，以提高导航过程中的透明度。所提出的系统使机器人能够通过自然语言摘要感知、分析和表述其观察结果。用户研究（n=30）表明，大多数用户更偏好实时解释，这表明改善了信任和理解。通过混淆矩阵分析验证了我们的实验结果，以评估与人类预期的一致性水平。我们的实验和仿真结果强调了可解释性在自主导航中的有效性，增强了信任和可解释性。 

---
# Sim4EndoR: A Reinforcement Learning Centered Simulation Platform for Task Automation of Endovascular Robotics 

**Title (ZH)**: Sim4EndoR：一种以强化学习为中心的内vascular机器人任务自动化仿真平台 

**Authors**: Tianliang Yao, Madaoji Ban, Bo Lu, Zhiqiang Pei, Peng Qi  

**Link**: [PDF](https://arxiv.org/pdf/2504.05330)  

**Abstract**: Robotic-assisted percutaneous coronary intervention (PCI) holds considerable promise for elevating precision and safety in cardiovascular procedures. Nevertheless, current systems heavily depend on human operators, resulting in variability and the potential for human error. To tackle these challenges, Sim4EndoR, an innovative reinforcement learning (RL) based simulation environment, is first introduced to bolster task-level autonomy in PCI. This platform offers a comprehensive and risk-free environment for the development, evaluation, and refinement of potential autonomous systems, enhancing data collection efficiency and minimizing the need for costly hardware trials. A notable aspect of the groundbreaking Sim4EndoR is its reward function, which takes into account the anatomical constraints of the vascular environment, utilizing the geometric characteristics of vessels to steer the learning process. By seamlessly integrating advanced physical simulations with neural network-driven policy learning, Sim4EndoR fosters efficient sim-to-real translation, paving the way for safer, more consistent robotic interventions in clinical practice, ultimately improving patient outcomes. 

**Abstract (ZH)**: 机器人辅助经皮冠状动脉介入治疗（PCI）在提升心血管手术精确性和安全性方面展现出显著潜力。然而，现有系统仍高度依赖人工操作，导致操作变异性和潜在的人为错误。为应对这些挑战，首次引入了基于强化学习（RL）的创新模拟环境Sim4EndoR，以增强PCI任务级别的自主性。该平台提供了一个全面且无风险的环境，用于开发、评估和完善潜在的自主系统，提高数据收集效率并减少昂贵硬件试验的需求。Sim4EndoR 的一大突破在于其奖励函数，该函数考虑了血管环境的解剖约束，利用血管的几何特征引导学习过程。通过无缝集成高级物理模拟与神经网络驱动的策略学习，Sim4EndoR 促进了从模拟到现实的有效转换，为临床实践中更安全、更一致的机器人干预铺平道路，最终改善患者预后。 

---
# Real-Time Model Predictive Control for the Swing-Up Problem of an Underactuated Double Pendulum 

**Title (ZH)**: 实时模型预测控制在欠驱动双摆的直立问题中的应用 

**Authors**: Blanka Burchard, Franek Stark  

**Link**: [PDF](https://arxiv.org/pdf/2504.05363)  

**Abstract**: The 3rd AI Olympics with RealAIGym competition poses the challenge of developing a global policy that can swing up and stabilize an underactuated 2-link system Acrobot and/or Pendubot from any configuration in the state space. This paper presents an optimal control-based approach using a real-time Nonlinear Model Predictive Control (MPC). The results show that the controller achieves good performance and robustness and can reliably handle disturbances. 

**Abstract (ZH)**: 第三届AI Olympics配合RealAIGym竞赛提出了在全球任意状态配置下摆动并稳定欠驱动双连杆系统Acrobot和/or Pendubot的挑战。本文提出了一种基于最优控制的实时非线性模型预测控制（MPC）方法。结果表明，控制器性能优良、鲁棒性强，能够可靠地处理干扰。 

---
# Continual Learning of Multiple Cognitive Functions with Brain-inspired Temporal Development Mechanism 

**Title (ZH)**: 基于脑启发的时间发展机制的多种认知功能连续学习 

**Authors**: Bing Han, Feifei Zhao, Yinqian Sun, Wenxuan Pan, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05621)  

**Abstract**: Cognitive functions in current artificial intelligence networks are tied to the exponential increase in network scale, whereas the human brain can continuously learn hundreds of cognitive functions with remarkably low energy consumption. This advantage is in part due to the brain cross-regional temporal development mechanisms, where the progressive formation, reorganization, and pruning of connections from basic to advanced regions, facilitate knowledge transfer and prevent network redundancy. Inspired by these, we propose the Continual Learning of Multiple Cognitive Functions with Brain-inspired Temporal Development Mechanism(TD-MCL), enabling cognitive enhancement from simple to complex in Perception-Motor-Interaction(PMI) multiple cognitive task scenarios. The TD-MCL model proposes the sequential evolution of long-range connections between different cognitive modules to promote positive knowledge transfer, while using feedback-guided local connection inhibition and pruning to effectively eliminate redundancies in previous tasks, reducing energy consumption while preserving acquired knowledge. Experiments show that the proposed method can achieve continual learning capabilities while reducing network scale, without introducing regularization, replay, or freezing strategies, and achieving superior accuracy on new tasks compared to direct learning. The proposed method shows that the brain's developmental mechanisms offer a valuable reference for exploring biologically plausible, low-energy enhancements of general cognitive abilities. 

**Abstract (ZH)**: 基于大脑启发的时间发展机制实现多认知功能的持续学习（TD-MCL）：从简单到复杂感知-运动-交互（PMI）多认知任务场景中的认知增强 

---
# Interactive Explanations for Reinforcement-Learning Agents 

**Title (ZH)**: 强化学习代理的交互式解释 

**Authors**: Yotam Amitai, Ofra Amir, Guy Avni  

**Link**: [PDF](https://arxiv.org/pdf/2504.05393)  

**Abstract**: As reinforcement learning methods increasingly amass accomplishments, the need for comprehending their solutions becomes more crucial. Most explainable reinforcement learning (XRL) methods generate a static explanation depicting their developers' intuition of what should be explained and how. In contrast, literature from the social sciences proposes that meaningful explanations are structured as a dialog between the explainer and the explainee, suggesting a more active role for the user and her communication with the agent. In this paper, we present ASQ-IT -- an interactive explanation system that presents video clips of the agent acting in its environment based on queries given by the user that describe temporal properties of behaviors of interest. Our approach is based on formal methods: queries in ASQ-IT's user interface map to a fragment of Linear Temporal Logic over finite traces (LTLf), which we developed, and our algorithm for query processing is based on automata theory. User studies show that end-users can understand and formulate queries in ASQ-IT and that using ASQ-IT assists users in identifying faulty agent behaviors. 

**Abstract (ZH)**: 基于查询的交互解释系统ASQ-IT：一种基于形式方法的时间逻辑片段查询处理方法 

---
# Multi-fidelity Reinforcement Learning Control for Complex Dynamical Systems 

**Title (ZH)**: 多保真强化学习控制复杂动力学系统 

**Authors**: Luning Sun, Xin-Yang Liu, Siyan Zhao, Aditya Grover, Jian-Xun Wang, Jayaraman J. Thiagarajan  

**Link**: [PDF](https://arxiv.org/pdf/2504.05588)  

**Abstract**: Controlling instabilities in complex dynamical systems is challenging in scientific and engineering applications. Deep reinforcement learning (DRL) has seen promising results for applications in different scientific applications. The many-query nature of control tasks requires multiple interactions with real environments of the underlying physics. However, it is usually sparse to collect from the experiments or expensive to simulate for complex dynamics. Alternatively, controlling surrogate modeling could mitigate the computational cost issue. However, a fast and accurate learning-based model by offline training makes it very hard to get accurate pointwise dynamics when the dynamics are chaotic. To bridge this gap, the current work proposes a multi-fidelity reinforcement learning (MFRL) framework that leverages differentiable hybrid models for control tasks, where a physics-based hybrid model is corrected by limited high-fidelity data. We also proposed a spectrum-based reward function for RL learning. The effect of the proposed framework is demonstrated on two complex dynamics in physics. The statistics of the MFRL control result match that computed from many-query evaluations of the high-fidelity environments and outperform other SOTA baselines. 

**Abstract (ZH)**: 复杂动力学系统中控制稳定性的问题在科学和工程应用中具有挑战性。深度强化学习（DRL）在不同科学应用中的前景令人鼓舞。由于控制任务的多查询性质，需要与基础物理的实际情况进行多次互动。然而，从实验中收集数据通常是稀疏的，对于复杂动力学而言，模拟则通常非常昂贵。作为替代方案，代理模型控制可以缓解计算成本问题。然而，通过离线训练获得快速且准确的学习模型使得在动力学混沌时难以获得准确的点wise动力学。为解决这一问题，当前工作提出了一种多保真度强化学习（MFRL）框架，利用可微混合模型进行控制任务，其中基于物理的混合模型通过有限的高保真数据进行修正。我们还提出了基于频谱的奖励函数以供RL学习。所提出的框架在物理学中的两个复杂动力学问题上得到了验证，其统计结果与高保真环境的多查询评估结果相符，并优于其他最先进的基准方法。 

---
# Deep Reinforcement Learning Algorithms for Option Hedging 

**Title (ZH)**: 深度强化学习算法在期权对冲中的应用 

**Authors**: Andrei Neagu, Frédéric Godin, Leila Kosseim  

**Link**: [PDF](https://arxiv.org/pdf/2504.05521)  

**Abstract**: Dynamic hedging is a financial strategy that consists in periodically transacting one or multiple financial assets to offset the risk associated with a correlated liability. Deep Reinforcement Learning (DRL) algorithms have been used to find optimal solutions to dynamic hedging problems by framing them as sequential decision-making problems. However, most previous work assesses the performance of only one or two DRL algorithms, making an objective comparison across algorithms difficult. In this paper, we compare the performance of eight DRL algorithms in the context of dynamic hedging; Monte Carlo Policy Gradient (MCPG), Proximal Policy Optimization (PPO), along with four variants of Deep Q-Learning (DQL) and two variants of Deep Deterministic Policy Gradient (DDPG). Two of these variants represent a novel application to the task of dynamic hedging. In our experiments, we use the Black-Scholes delta hedge as a baseline and simulate the dataset using a GJR-GARCH(1,1) model. Results show that MCPG, followed by PPO, obtain the best performance in terms of the root semi-quadratic penalty. Moreover, MCPG is the only algorithm to outperform the Black-Scholes delta hedge baseline with the allotted computational budget, possibly due to the sparsity of rewards in our environment. 

**Abstract (ZH)**: 动态对冲是一种金融策略，涉及周期性交易一个或多个金融资产以抵消与相关负债相关的风险。深度强化学习（DRL）算法已被用于通过将其表述为顺序决策问题来寻找动态对冲问题的最优解。然而，大部分先前的工作仅评估了一两种DRL算法的表现，这使得算法之间的客观比较变得困难。在这篇论文中，我们比较了八种DRL算法在动态对冲中的表现；蒙特卡洛策略梯度（MCPG）、 proportional策略优化（PPO），以及四种深度Q学习（DQL）和两种深度确定性策略梯度（DDPG）的变体。在这两种变体中，其中一种是首次应用于动态对冲任务。在我们的实验中，我们使用布莱克-斯科尔斯Delta对冲作为基线，并使用GJR-GARCH(1,1)模型模拟数据集。结果表明，MCPG之后是PPO，在根半四次惩罚方面获得最佳表现。此外，MCPG是唯一一种在给定计算预算内表现优于布莱克-斯科尔斯Delta对冲基线的算法，这可能是由于我们环境中奖励的稀疏性。 

---
# The Role of Environment Access in Agnostic Reinforcement Learning 

**Title (ZH)**: 环境访问在agnostic强化学习中的作用 

**Authors**: Akshay Krishnamurthy, Gene Li, Ayush Sekhari  

**Link**: [PDF](https://arxiv.org/pdf/2504.05405)  

**Abstract**: We study Reinforcement Learning (RL) in environments with large state spaces, where function approximation is required for sample-efficient learning. Departing from a long history of prior work, we consider the weakest possible form of function approximation, called agnostic policy learning, where the learner seeks to find the best policy in a given class $\Pi$, with no guarantee that $\Pi$ contains an optimal policy for the underlying task. Although it is known that sample-efficient agnostic policy learning is not possible in the standard online RL setting without further assumptions, we investigate the extent to which this can be overcome with stronger forms of access to the environment. Specifically, we show that: 1. Agnostic policy learning remains statistically intractable when given access to a local simulator, from which one can reset to any previously seen state. This result holds even when the policy class is realizable, and stands in contrast to a positive result of [MFR24] showing that value-based learning under realizability is tractable with local simulator access. 2. Agnostic policy learning remains statistically intractable when given online access to a reset distribution with good coverage properties over the state space (the so-called $\mu$-reset setting). We also study stronger forms of function approximation for policy learning, showing that PSDP [BKSN03] and CPI [KL02] provably fail in the absence of policy completeness. 3. On a positive note, agnostic policy learning is statistically tractable for Block MDPs with access to both of the above reset models. We establish this via a new algorithm that carefully constructs a policy emulator: a tabular MDP with a small state space that approximates the value functions of all policies $\pi \in \Pi$. These values are approximated without any explicit value function class. 

**Abstract (ZH)**: 我们研究具有大规模状态空间环境中的强化学习（RL），在这种情况下需要使用功能近似以实现样本高效学习。不同于以往工作的长期历史，我们考虑功能近似中最弱的形式，即无放大型策略学习，其中学习者寻求在给定策略类$\Pi$中找到最优策略，但没有保证$\Pi$中包含底层任务的最优策略。虽然在标准在线RL设置中，无需进一步假设无法在无放大型策略学习中实现样本高效学习，但我们研究了更强环境访问形式在这种限制下的克服程度。具体而言，我们展示了：1. 即使策略类是可以实现的，当具有访问局部模拟器的能力时（可以重置到已见过的任何状态），无放大型策略学习仍然统计上不可行。这一结果与[MFR24]中关于在局部模拟器访问下基于值的学习可以通过对实现性假设进行处理而变得可行的积极结果形成了对比。2. 即使具有对具有良好状态空间覆盖性质的重置分布的在线访问（所谓的$\mu$-重置设置），无放大型策略学习仍然统计上不可行。我们还研究了策略学习中的更强形式的功能近似，展示了PSDP [BKSN03]和CPI [KL02]在缺少策略完备性时会失效。3. 在一个积极的方面，当具有上述两种重置模型的访问时，无放大型策略学习对于Block MDP来说是统计上可处理的。我们通过一个新的算法建立这一结论，该算法仔细构建了一个策略模拟器：一个具有小状态空间的表格MDP，它可以近似所有策略$\pi \in \Pi$的价值函数。这些价值是在没有任何显式价值函数类的情况下近似的。 

---
