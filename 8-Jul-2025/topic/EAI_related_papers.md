# Action Space Reduction Strategies for Reinforcement Learning in Autonomous Driving 

**Title (ZH)**: 自主驾驶中强化学习的动作空间缩减策略 

**Authors**: Elahe Delavari, Feeza Khan Khanzada, Jaerock Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2507.05251)  

**Abstract**: Reinforcement Learning (RL) offers a promising framework for autonomous driving by enabling agents to learn control policies through interaction with environments. However, large and high-dimensional action spaces often used to support fine-grained control can impede training efficiency and increase exploration costs. In this study, we introduce and evaluate two novel structured action space modification strategies for RL in autonomous driving: dynamic masking and relative action space reduction. These approaches are systematically compared against fixed reduction schemes and full action space baselines to assess their impact on policy learning and performance. Our framework leverages a multimodal Proximal Policy Optimization agent that processes both semantic image sequences and scalar vehicle states. The proposed dynamic and relative strategies incorporate real-time action masking based on context and state transitions, preserving action consistency while eliminating invalid or suboptimal choices. Through comprehensive experiments across diverse driving routes, we show that action space reduction significantly improves training stability and policy performance. The dynamic and relative schemes, in particular, achieve a favorable balance between learning speed, control precision, and generalization. These findings highlight the importance of context-aware action space design for scalable and reliable RL in autonomous driving tasks. 

**Abstract (ZH)**: 强化学习（RL）为自主驾驶提供了有前途的框架，通过使代理通过与环境的交互来学习控制策略。然而，用于支持精细控制的大型和高维动作空间往往会阻碍训练效率并增加探索成本。在本研究中，我们引入并评估了两种新的强化学习在自主驾驶中的结构化动作空间修改策略：动态遮蔽和相对动作空间缩减。这些方法系统地与固定缩减方案和全动作空间基线进行比较，以评估其对策略学习和性能的影响。我们的框架采用了多模态的密切策略优化代理，处理语义图像序列和标量车辆状态。所提出的动态和相对策略基于上下文和状态转换实时遮蔽动作，保持动作一致性同时消除无效或次优选择。通过在多种驾驶路线上的全面实验，我们展示了动作空间缩减显著提高了训练稳定性和策略性能。特别是，动态和相对方案在学习速度、控制精度和泛化能力之间取得了有利的平衡。这些发现强调了用于可扩展和可靠的自主驾驶任务中强化学习的上下文感知动作空间设计的重要性。 

---
# StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling 

**Title (ZH)**: StreamVLN: 基于慢快速上下文建模的流式视觉-语言导航 

**Authors**: Meng Wei, Chenyang Wan, Xiqian Yu, Tai Wang, Yuqiang Yang, Xiaohan Mao, Chenming Zhu, Wenzhe Cai, Hanqing Wang, Yilun Chen, Xihui Liu, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05240)  

**Abstract**: Vision-and-Language Navigation (VLN) in real-world settings requires agents to process continuous visual streams and generate actions with low latency grounded in language instructions. While Video-based Large Language Models (Video-LLMs) have driven recent progress, current VLN methods based on Video-LLM often face trade-offs among fine-grained visual understanding, long-term context modeling and computational efficiency. We introduce StreamVLN, a streaming VLN framework that employs a hybrid slow-fast context modeling strategy to support multi-modal reasoning over interleaved vision, language and action inputs. The fast-streaming dialogue context facilitates responsive action generation through a sliding-window of active dialogues, while the slow-updating memory context compresses historical visual states using a 3D-aware token pruning strategy. With this slow-fast design, StreamVLN achieves coherent multi-turn dialogue through efficient KV cache reuse, supporting long video streams with bounded context size and inference cost. Experiments on VLN-CE benchmarks demonstrate state-of-the-art performance with stable low latency, ensuring robustness and efficiency in real-world deployment. The project page is: \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 实时光学-语言导航（VLN）在现实场景中要求代理处理连续的视觉流并基于语言指令以低延迟生成动作。虽然基于视频的大语言模型（Video-LLM）推动了近期的发展，但当前基于Video-LLM的VLN方法经常在细粒度的视觉理解、长期上下文建模和计算效率之间存在权衡。我们引入了StreamVLN，这是一种流式VLN框架，采用混合慢速-快速上下文建模策略，支持交错的视觉、语言和动作输入的多模态推理。快速流式对话上下文通过滑动窗口中的活跃对话促进响应式动作生成，而缓慢更新的记忆上下文则通过3D意识的token剪枝策略压缩历史视觉状态。利用这种慢速-快速设计，StreamVLN通过高效的关键值缓存重用实现一致的多轮对话，支持具有限定上下文大小和推理成本的长视频流。在VLN-CE基准测试中的实验展示了最先进的性能，并保证了在实际部署中的稳健性和效率。项目页面：\[this https URL\]。 

---
# NavigScene: Bridging Local Perception and Global Navigation for Beyond-Visual-Range Autonomous Driving 

**Title (ZH)**: NavigScene: 连接局部感知与全局导航以实现超视距自动驾驶 

**Authors**: Qucheng Peng, Chen Bai, Guoxiang Zhang, Bo Xu, Xiaotong Liu, Xiaoyin Zheng, Chen Chen, Cheng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.05227)  

**Abstract**: Autonomous driving systems have made significant advances in Q&A, perception, prediction, and planning based on local visual information, yet they struggle to incorporate broader navigational context that human drivers routinely utilize. We address this critical gap between local sensor data and global navigation information by proposing NavigScene, an auxiliary navigation-guided natural language dataset that simulates a human-like driving environment within autonomous driving systems. Moreover, we develop three complementary paradigms to leverage NavigScene: (1) Navigation-guided Reasoning, which enhances vision-language models by incorporating navigation context into the prompting approach; (2) Navigation-guided Preference Optimization, a reinforcement learning method that extends Direct Preference Optimization to improve vision-language model responses by establishing preferences for navigation-relevant summarized information; and (3) Navigation-guided Vision-Language-Action model, which integrates navigation guidance and vision-language models with conventional driving models through feature fusion. Extensive experiments demonstrate that our approaches significantly improve performance across perception, prediction, planning, and question-answering tasks by enabling reasoning capabilities beyond visual range and improving generalization to diverse driving scenarios. This work represents a significant step toward more comprehensive autonomous driving systems capable of navigating complex, unfamiliar environments with greater reliability and safety. 

**Abstract (ZH)**: 自主驾驶系统基于局部视觉信息在问答、感知、预测和规划方面取得了显著进展，但难以整合人类驾驶员常用的更广泛的导航上下文。我们通过提出NavigScene辅助导航引导自然语言数据集来弥补局部传感器数据与全局导航信息之间的关键差距，该数据集在自主驾驶系统中模拟了类似人类驾驶的环境。此外，我们开发了三种互补的范式来利用NavigScene：（1）导航引导的推理，通过将导航上下文纳入提示方法来增强视觉语言模型；（2）导航引导的偏好优化，一种强化学习方法，扩展了直接偏好优化，通过建立与导航相关的信息偏好来改善视觉语言模型的响应；（3）导航引导的视觉语言行动模型，通过特征融合将导航引导和视觉语言模型与传统的驾驶模型集成。大量实验表明，我们的方法通过增强推理能力并提高对多样化驾驶场景的泛化能力，在感知、预测、规划和问答任务中显著提高了性能。这项工作代表了朝着更加全面的自主驾驶系统迈出了重要一步，这些系统能够在更可靠和安全的情况下导航复杂且不熟悉的环境。 

---
# EmbodieDreamer: Advancing Real2Sim2Real Transfer for Policy Training via Embodied World Modeling 

**Title (ZH)**: EmbodieDreamer: 通过具身世界建模促进从真实到模拟再到真实的策略训练转移 

**Authors**: Boyuan Wang, Xinpan Meng, Xiaofeng Wang, Zheng Zhu, Angen Ye, Yang Wang, Zhiqin Yang, Chaojun Ni, Guan Huang, Xingang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05198)  

**Abstract**: The rapid advancement of Embodied AI has led to an increasing demand for large-scale, high-quality real-world data. However, collecting such embodied data remains costly and inefficient. As a result, simulation environments have become a crucial surrogate for training robot policies. Yet, the significant Real2Sim2Real gap remains a critical bottleneck, particularly in terms of physical dynamics and visual appearance. To address this challenge, we propose EmbodieDreamer, a novel framework that reduces the Real2Sim2Real gap from both the physics and appearance perspectives. Specifically, we propose PhysAligner, a differentiable physics module designed to reduce the Real2Sim physical gap. It jointly optimizes robot-specific parameters such as control gains and friction coefficients to better align simulated dynamics with real-world observations. In addition, we introduce VisAligner, which incorporates a conditional video diffusion model to bridge the Sim2Real appearance gap by translating low-fidelity simulated renderings into photorealistic videos conditioned on simulation states, enabling high-fidelity visual transfer. Extensive experiments validate the effectiveness of EmbodieDreamer. The proposed PhysAligner reduces physical parameter estimation error by 3.74% compared to simulated annealing methods while improving optimization speed by 89.91\%. Moreover, training robot policies in the generated photorealistic environment leads to a 29.17% improvement in the average task success rate across real-world tasks after reinforcement learning. Code, model and data will be publicly available. 

**Abstract (ZH)**: 基于物理和视觉的Embodied AI从现实到模拟再到现实的差距缩小方法 

---
# LERa: Replanning with Visual Feedback in Instruction Following 

**Title (ZH)**: LERa：基于视觉反馈的指令跟随重规划 

**Authors**: Svyatoslav Pchelintsev, Maxim Patratskiy, Anatoly Onishchenko, Alexandr Korchemnyi, Aleksandr Medvedev, Uliana Vinogradova, Ilya Galuzinsky, Aleksey Postnikov, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2507.05135)  

**Abstract**: Large Language Models are increasingly used in robotics for task planning, but their reliance on textual inputs limits their adaptability to real-world changes and failures. To address these challenges, we propose LERa - Look, Explain, Replan - a Visual Language Model-based replanning approach that utilizes visual feedback. Unlike existing methods, LERa requires only a raw RGB image, a natural language instruction, an initial task plan, and failure detection - without additional information such as object detection or predefined conditions that may be unavailable in a given scenario. The replanning process consists of three steps: (i) Look, where LERa generates a scene description and identifies errors; (ii) Explain, where it provides corrective guidance; and (iii) Replan, where it modifies the plan accordingly. LERa is adaptable to various agent architectures and can handle errors from both dynamic scene changes and task execution failures. We evaluate LERa on the newly introduced ALFRED-ChaOS and VirtualHome-ChaOS datasets, achieving a 40% improvement over baselines in dynamic environments. In tabletop manipulation tasks with a predefined probability of task failure within the PyBullet simulator, LERa improves success rates by up to 67%. Further experiments, including real-world trials with a tabletop manipulator robot, confirm LERa's effectiveness in replanning. We demonstrate that LERa is a robust and adaptable solution for error-aware task execution in robotics. The code is available at this https URL. 

**Abstract (ZH)**: 视觉语言模型基于的重规划方法：看、解释、重规划 

---
# VerifyLLM: LLM-Based Pre-Execution Task Plan Verification for Robots 

**Title (ZH)**: VerifyLLM：基于LLM的预执行任务计划验证方法 

**Authors**: Danil S. Grigorev, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2507.05118)  

**Abstract**: In the field of robotics, researchers face a critical challenge in ensuring reliable and efficient task planning. Verifying high-level task plans before execution significantly reduces errors and enhance the overall performance of these systems. In this paper, we propose an architecture for automatically verifying high-level task plans before their execution in simulator or real-world environments. Leveraging Large Language Models (LLMs), our approach consists of two key steps: first, the conversion of natural language instructions into Linear Temporal Logic (LTL), followed by a comprehensive analysis of action sequences. The module uses the reasoning capabilities of the LLM to evaluate logical coherence and identify potential gaps in the plan. Rigorous testing on datasets of varying complexity demonstrates the broad applicability of the module to household tasks. We contribute to improving the reliability and efficiency of task planning and addresses the critical need for robust pre-execution verification in autonomous systems. The code is available at this https URL. 

**Abstract (ZH)**: 在机器人领域，研究人员面临确保可靠和高效任务规划的关键挑战。在执行前验证高级任务计划可以显著减少错误并提升这些系统的整体性能。本文提出了一种架构，在模拟器或真实环境中的任务执行前自动验证高级任务计划。该方法利用大型语言模型（LLMs），主要包括两步：首先将自然语言指令转换为线性时序逻辑（LTL），然后对该行动计划序列进行全面分析。模块利用LLM的推理能力评估逻辑连贯性并识别潜在的计划缺口。复杂度各异的数据集上的严格测试表明，该模块适用于家庭任务。本文有助于提高任务规划的可靠性和效率，并满足自主系统在执行前进行 robust 验证的迫切需求。代码见 <https://github.com/XXXXX>。 

---
# Piggyback Camera: Easy-to-Deploy Visual Surveillance by Mobile Sensing on Commercial Robot Vacuums 

**Title (ZH)**: 搭车摄像头：基于商用扫地机器人的移动传感视觉监控易部署方案 

**Authors**: Ryo Yonetani  

**Link**: [PDF](https://arxiv.org/pdf/2507.04910)  

**Abstract**: This paper presents Piggyback Camera, an easy-to-deploy system for visual surveillance using commercial robot vacuums. Rather than requiring access to internal robot systems, our approach mounts a smartphone equipped with a camera and Inertial Measurement Unit (IMU) on the robot, making it applicable to any commercial robot without hardware modifications. The system estimates robot poses through neural inertial navigation and efficiently captures images at regular spatial intervals throughout the cleaning task. We develop a novel test-time data augmentation method called Rotation-Augmented Ensemble (RAE) to mitigate domain gaps in neural inertial navigation. A loop closure method that exploits robot cleaning patterns further refines these estimated poses. We demonstrate the system with an object mapping application that analyzes captured images to geo-localize objects in the environment. Experimental evaluation in retail environments shows that our approach achieves 0.83 m relative pose error for robot localization and 0.97 m positional error for object mapping of over 100 items. 

**Abstract (ZH)**: 基于商用吸尘器的搭车摄像头视觉监控系统 

---
# Training-free Generation of Temporally Consistent Rewards from VLMs 

**Title (ZH)**: 无需训练的时空一致奖励生成从VLMs 

**Authors**: Yinuo Zhao, Jiale Yuan, Zhiyuan Xu, Xiaoshuai Hao, Xinyi Zhang, Kun Wu, Zhengping Che, Chi Harold Liu, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04789)  

**Abstract**: Recent advances in vision-language models (VLMs) have significantly improved performance in embodied tasks such as goal decomposition and visual comprehension. However, providing accurate rewards for robotic manipulation without fine-tuning VLMs remains challenging due to the absence of domain-specific robotic knowledge in pre-trained datasets and high computational costs that hinder real-time applicability. To address this, we propose $\mathrm{T}^2$-VLM, a novel training-free, temporally consistent framework that generates accurate rewards through tracking the status changes in VLM-derived subgoals. Specifically, our method first queries the VLM to establish spatially aware subgoals and an initial completion estimate before each round of interaction. We then employ a Bayesian tracking algorithm to update the goal completion status dynamically, using subgoal hidden states to generate structured rewards for reinforcement learning (RL) agents. This approach enhances long-horizon decision-making and improves failure recovery capabilities with RL. Extensive experiments indicate that $\mathrm{T}^2$-VLM achieves state-of-the-art performance in two robot manipulation benchmarks, demonstrating superior reward accuracy with reduced computation consumption. We believe our approach not only advances reward generation techniques but also contributes to the broader field of embodied AI. Project website: this https URL. 

**Abstract (ZH)**: Recent advances in 视觉-语言模型(VLMs)显著提升了目标分解和视觉理解等沉浸式任务的表现。然而，要在不微调VLMs的情况下为机器人操作提供准确的奖励仍然具有挑战性，这是因为预训练数据集中缺乏特定领域的机器人知识，以及高昂的计算成本限制了实时适用性。为了解决这一问题，我们提出了一种新的无需训练、时序一致的框架$\mathrm{T}^2$-VLM，该框架通过跟踪由VLM生成的子目标状态变化来生成准确的奖励。具体而言，我们的方法首先在每次交互轮次前查询VLM以建立空间意识子目标和初步完成估计。然后，我们采用贝叶斯跟踪算法动态更新目标完成状态，并利用子目标隐藏状态为强化学习(RL)代理生成结构化奖励。该方法增强了长期决策制定能力，并通过RL提高了故障恢复能力。大量实验表明，$\mathrm{T}^2$-VLM在两个机器人操作基准测试中达到了最先进的性能，展示了在减少计算消耗的同时更高的奖励准确性。我们相信，我们的方法不仅推动了奖励生成技术的发展，还对更广泛的沉浸式AI领域做出了贡献。 

---
# MOSU: Autonomous Long-range Robot Navigation with Multi-modal Scene Understanding 

**Title (ZH)**: MOSU: 多模态场景理解驱动的自主长距离机器人导航 

**Authors**: Jing Liang, Kasun Weerakoon, Daeun Song, Senthurbavan Kirubaharan, Xuesu Xiao, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2507.04686)  

**Abstract**: We present MOSU, a novel autonomous long-range navigation system that enhances global navigation for mobile robots through multimodal perception and on-road scene understanding. MOSU addresses the outdoor robot navigation challenge by integrating geometric, semantic, and contextual information to ensure comprehensive scene understanding. The system combines GPS and QGIS map-based routing for high-level global path planning and multi-modal trajectory generation for local navigation refinement. For trajectory generation, MOSU leverages multi-modalities: LiDAR-based geometric data for precise obstacle avoidance, image-based semantic segmentation for traversability assessment, and Vision-Language Models (VLMs) to capture social context and enable the robot to adhere to social norms in complex environments. This multi-modal integration improves scene understanding and enhances traversability, allowing the robot to adapt to diverse outdoor conditions. We evaluate our system in real-world on-road environments and benchmark it on the GND dataset, achieving a 10% improvement in traversability on navigable terrains while maintaining a comparable navigation distance to existing global navigation methods. 

**Abstract (ZH)**: MOSU：一种通过多模态感知和道路场景理解增强移动机器人远程导航的新型自主导航系统 

---
# DRAE: Dynamic Retrieval-Augmented Expert Networks for Lifelong Learning and Task Adaptation in Robotics 

**Title (ZH)**: 动态检索增强专家网络：面向机器人终身学习和任务适应的研究 

**Authors**: Yayu Long, Kewei Chen, Long Jin, Mingsheng Shang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04661)  

**Abstract**: We introduce Dynamic Retrieval-Augmented Expert Networks (DRAE), a groundbreaking architecture that addresses the challenges of lifelong learning, catastrophic forgetting, and task adaptation by combining the dynamic routing capabilities of Mixture-of-Experts (MoE); leveraging the knowledge-enhancement power of Retrieval-Augmented Generation (RAG); incorporating a novel hierarchical reinforcement learning (RL) framework; and coordinating through ReflexNet-SchemaPlanner-HyperOptima (RSHO).DRAE dynamically routes expert models via a sparse MoE gating mechanism, enabling efficient resource allocation while leveraging external knowledge through parametric retrieval (P-RAG) to augment the learning process. We propose a new RL framework with ReflexNet for low-level task execution, SchemaPlanner for symbolic reasoning, and HyperOptima for long-term context modeling, ensuring continuous adaptation and memory retention. Experimental results show that DRAE significantly outperforms baseline approaches in long-term task retention and knowledge reuse, achieving an average task success rate of 82.5% across a set of dynamic robotic manipulation tasks, compared to 74.2% for traditional MoE models. Furthermore, DRAE maintains an extremely low forgetting rate, outperforming state-of-the-art methods in catastrophic forgetting mitigation. These results demonstrate the effectiveness of our approach in enabling flexible, scalable, and efficient lifelong learning for robotics. 

**Abstract (ZH)**: 动态检索增强专家网络（DRAE）：一种应对终身学习挑战的创新架构 

---
# Bio-Inspired Hybrid Map: Spatial Implicit Local Frames and Topological Map for Mobile Cobot Navigation 

**Title (ZH)**: 生物启发的混合地图：空间隐式局部帧与拓扑地图用于移动协作机器人导航 

**Authors**: Tuan Dang, Manfred Huber  

**Link**: [PDF](https://arxiv.org/pdf/2507.04649)  

**Abstract**: Navigation is a fundamental capacity for mobile robots, enabling them to operate autonomously in complex and dynamic environments. Conventional approaches use probabilistic models to localize robots and build maps simultaneously using sensor observations. Recent approaches employ human-inspired learning, such as imitation and reinforcement learning, to navigate robots more effectively. However, these methods suffer from high computational costs, global map inconsistency, and poor generalization to unseen environments. This paper presents a novel method inspired by how humans perceive and navigate themselves effectively in novel environments. Specifically, we first build local frames that mimic how humans represent essential spatial information in the short term. Points in local frames are hybrid representations, including spatial information and learned features, so-called spatial-implicit local frames. Then, we integrate spatial-implicit local frames into the global topological map represented as a factor graph. Lastly, we developed a novel navigation algorithm based on Rapid-Exploring Random Tree Star (RRT*) that leverages spatial-implicit local frames and the topological map to navigate effectively in environments. To validate our approach, we conduct extensive experiments in real-world datasets and in-lab environments. We open our source code at this https URL}{this https URL. 

**Abstract (ZH)**: 移动机器人在复杂和动态环境中的导航是一项基本能力，使其能够自主操作。传统方法使用概率模型同时利用传感器观测值进行机器人定位和建图。近年来，通过模仿和强化学习等受人类启发的学习方法，提高了机器人导航的效率。然而，这些方法存在计算成本高、全局地图不一致和对未见环境泛化能力差的问题。本论文提出了一种受人类如何在新环境中有效感知和导航启发的新方法。具体而言，我们首先构建局部框架，模拟人类在短期中表示关键空间信息的方式。局部框架中的点是混合表示，包括空间信息和学习特征，称为空间隐式局部框架。然后，我们将空间隐式局部框架整合到由因子图表示的全局拓扑地图中。最后，我们开发了一种基于快速扩展随机树星（RRT*）的新导航算法，该算法利用空间隐式局部框架和拓扑地图有效地在环境中导航。为了验证我们的方法，我们在真实世界数据集和实验室环境中进行了广泛的实验。我们已开源代码，详见此链接：this https URL此链接：this https URL。 

---
# IDAGC: Adaptive Generalized Human-Robot Collaboration via Human Intent Estimation and Multimodal Policy Learning 

**Title (ZH)**: IDAGC：基于人类意图估计和多模态策略学习的自适应通用人类-机器人协作 

**Authors**: Haotian Liu, Yuchuang Tong, Guanchen Liu, Zhaojie Ju, Zhengtao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04620)  

**Abstract**: In Human-Robot Collaboration (HRC), which encompasses physical interaction and remote cooperation, accurate estimation of human intentions and seamless switching of collaboration modes to adjust robot behavior remain paramount challenges. To address these issues, we propose an Intent-Driven Adaptive Generalized Collaboration (IDAGC) framework that leverages multimodal data and human intent estimation to facilitate adaptive policy learning across multi-tasks in diverse scenarios, thereby facilitating autonomous inference of collaboration modes and dynamic adjustment of robotic actions. This framework overcomes the limitations of existing HRC methods, which are typically restricted to a single collaboration mode and lack the capacity to identify and transition between diverse states. Central to our framework is a predictive model that captures the interdependencies among vision, language, force, and robot state data to accurately recognize human intentions with a Conditional Variational Autoencoder (CVAE) and automatically switch collaboration modes. By employing dedicated encoders for each modality and integrating extracted features through a Transformer decoder, the framework efficiently learns multi-task policies, while force data optimizes compliance control and intent estimation accuracy during physical interactions. Experiments highlights our framework's practical potential to advance the comprehensive development of HRC. 

**Abstract (ZH)**: 在人类-机器人协作（HRC）中，涵盖物理交互和远程合作，准确估计人类意图和无缝切换协作模式以调整机器人行为仍然是主要挑战。为解决这些问题，我们提出了一种基于意图的自适应通用协作（IDAGC）框架，该框架利用多模态数据和人类意图估计来促进多任务在多种场景下的自适应策略学习，从而实现协作模式的自主推理和机器人类动的动态调整。该框架克服了现有HRC方法的限制，这些方法通常局限于单一协作模式，并且缺乏在不同状态下识别和转换的能力。本框架的核心是一个预测模型，该模型捕捉视觉、语言、力和机器人状态数据之间的相互依赖关系，通过条件变分自编码器（CVAE）准确识别人类意图，并自动切换协作模式。通过为每种模态使用专门的编码器并利用Transformer解码器集成提取特征，框架高效地学习多任务策略，同时力数据优化物理交互中的顺应控制和意图估计准确性。实验突出了我们框架在全面推动HRC发展的实际潜力。 

---
# VLM-TDP: VLM-guided Trajectory-conditioned Diffusion Policy for Robust Long-Horizon Manipulation 

**Title (ZH)**: VLM-TDP: 由VLM引导的轨迹条件化扩散策略以实现稳健的长时_horizon �操作 

**Authors**: Kefeng Huang, Tingguang Li, Yuzhen Liu, Zhe Zhang, Jiankun Wang, Lei Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.04524)  

**Abstract**: Diffusion policy has demonstrated promising performance in the field of robotic manipulation. However, its effectiveness has been primarily limited in short-horizon tasks, and its performance significantly degrades in the presence of image noise. To address these limitations, we propose a VLM-guided trajectory-conditioned diffusion policy (VLM-TDP) for robust and long-horizon manipulation. Specifically, the proposed method leverages state-of-the-art vision-language models (VLMs) to decompose long-horizon tasks into concise, manageable sub-tasks, while also innovatively generating voxel-based trajectories for each sub-task. The generated trajectories serve as a crucial conditioning factor, effectively steering the diffusion policy and substantially enhancing its performance. The proposed Trajectory-conditioned Diffusion Policy (TDP) is trained on trajectories derived from demonstration data and validated using the trajectories generated by the VLM. Simulation experimental results indicate that our method significantly outperforms classical diffusion policies, achieving an average 44% increase in success rate, over 100% improvement in long-horizon tasks, and a 20% reduction in performance degradation in challenging conditions, such as noisy images or altered environments. These findings are further reinforced by our real-world experiments, where the performance gap becomes even more pronounced in long-horizon tasks. Videos are available on this https URL 

**Abstract (ZH)**: 基于VLM引导的轨迹条件扩散策略（VLM-TDP）在鲁棒长 horizon 操作中的应用 

---
# SimLauncher: Launching Sample-Efficient Real-world Robotic Reinforcement Learning via Simulation Pre-training 

**Title (ZH)**: SimLauncher：通过模拟预训练实现高效现实机器人强化学习的样本启动器 

**Authors**: Mingdong Wu, Lehong Wu, Yizhuo Wu, Weiyao Huang, Hongwei Fan, Zheyuan Hu, Haoran Geng, Jinzhou Li, Jiahe Ying, Long Yang, Yuanpei Chen, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.04452)  

**Abstract**: Autonomous learning of dexterous, long-horizon robotic skills has been a longstanding pursuit of embodied AI. Recent advances in robotic reinforcement learning (RL) have demonstrated remarkable performance and robustness in real-world visuomotor control tasks. However, applying RL in the real world faces challenges such as low sample efficiency, slow exploration, and significant reliance on human intervention. In contrast, simulators offer a safe and efficient environment for extensive exploration and data collection, while the visual sim-to-real gap, often a limiting factor, can be mitigated using real-to-sim techniques. Building on these, we propose SimLauncher, a novel framework that combines the strengths of real-world RL and real-to-sim-to-real approaches to overcome these challenges. Specifically, we first pre-train a visuomotor policy in the digital twin simulation environment, which then benefits real-world RL in two ways: (1) bootstrapping target values using extensive simulated demonstrations and real-world demonstrations derived from pre-trained policy rollouts, and (2) Incorporating action proposals from the pre-trained policy for better exploration. We conduct comprehensive experiments across multi-stage, contact-rich, and dexterous hand manipulation tasks. Compared to prior real-world RL approaches, SimLauncher significantly improves sample efficiency and achieves near-perfect success rates. We hope this work serves as a proof of concept and inspires further research on leveraging large-scale simulation pre-training to benefit real-world robotic RL. 

**Abstract (ZH)**: 自主学习灵巧、长期 horizon 机器人技能是嵌入式人工智能长期追求的目标。近期在机器人强化学习（RL）方面的进展已在实际世界的视觉-运动控制任务中展示了卓越的性能和鲁棒性。然而，在现实世界中应用RL面临着样本效率低、探索缓慢以及对人类干预依赖性强等挑战。相比之下，模拟器提供了安全且高效的环境，用于广泛的探索和数据收集，而现实到模拟的视觉差距常常是一个限制因素，可以通过现实到模拟技术来缓解。在此基础上，我们提出了SimLauncher这一新颖框架，该框架结合了现实世界RL和现实到模拟再到现实方法的优势，以克服这些挑战。具体来说，我们首先在数字孪生模拟环境中预训练一个视觉-运动策略，该策略在两个方面有助于现实世界的RL：（1）利用广泛的模拟演示和由预训练策略执行卷出的真实世界演示来启动目标值；（2）结合预训练策略的动作提案以提高探索效果。我们在多阶段、接触丰富和灵巧手操作任务中进行了全面的实验。与先前的现实世界RL方法相比，SimLauncher显著提高了样本效率，并实现了近乎完美的成功率。我们希望这项工作能够作为概念验证，并激励进一步研究，利用大规模模拟预训练来提升现实世界机器人RL的效果。 

---
# Free-Space Optical Communication-Driven NMPC Framework for Multi-Rotor Aerial Vehicles in Structured Inspection Scenarios 

**Title (ZH)**: 自由空间光学通信驱动的NMPC框架在结构化检查场景中的多旋翼无人机应用 

**Authors**: Giuseppe Silano, Daniel Bonilla Licea, Hajar El Hammouti, Martin Saska  

**Link**: [PDF](https://arxiv.org/pdf/2507.04443)  

**Abstract**: This paper introduces a Nonlinear Model Predictive Control (NMPC) framework for communication-aware motion planning of Multi-Rotor Aerial Vehicles (MRAVs) using Free-Space Optical (FSO) links. The scenario involves MRAVs equipped with body-fixed optical transmitters and Unmanned Ground Vehicles (UGVs) acting as mobile relays, each outfitted with fixed conical Field-of-View (FoV) receivers. The controller integrates optical connectivity constraints into the NMPC formulation to ensure beam alignment and minimum link quality, while also enabling UGV tracking and obstacle avoidance. The method supports both coplanar and tilted MRAV configurations. MATLAB simulations demonstrate its feasibility and effectiveness. 

**Abstract (ZH)**: 基于自由空间光学链接的多旋翼无人机通信感知运动规划的非线性模型预测控制框架 

---
# "Hi AirStar, Guide Me to the Badminton Court." 

**Title (ZH)**: Hi AirStar, 引领我到羽毛球场地。 

**Authors**: Ziqin Wang, Jinyu Chen, Xiangyi Zheng, Qinan Liao, Linjiang Huang, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04430)  

**Abstract**: Unmanned Aerial Vehicles, operating in environments with relatively few obstacles, offer high maneuverability and full three-dimensional mobility. This allows them to rapidly approach objects and perform a wide range of tasks often challenging for ground robots, making them ideal for exploration, inspection, aerial imaging, and everyday assistance. In this paper, we introduce AirStar, a UAV-centric embodied platform that turns a UAV into an intelligent aerial assistant: a large language model acts as the cognitive core for environmental understanding, contextual reasoning, and task planning. AirStar accepts natural interaction through voice commands and gestures, removing the need for a remote controller and significantly broadening its user base. It combines geospatial knowledge-driven long-distance navigation with contextual reasoning for fine-grained short-range control, resulting in an efficient and accurate vision-and-language navigation (VLN) this http URL, the system also offers built-in capabilities such as cross-modal question answering, intelligent filming, and target tracking. With a highly extensible framework, it supports seamless integration of new functionalities, paving the way toward a general-purpose, instruction-driven intelligent UAV agent. The supplementary PPT is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 无人驾驶航空器在较少障碍的环境中操作，具有高机动性和全三维移动性，能够快速接近物体并执行多种任务，这些任务往往对地面机器人具有挑战性，使其成为探索、检查、空中成像和日常生活助手的理想选择。本文介绍了一种以无人驾驶航空器为中心的自主平台AirStar，该平台将无人驾驶航空器转变为智能空中助手：一个大型语言模型作为认知核心，用于环境理解、上下文推理和任务规划。AirStar 通过语音命令和手势接受自然交互，去掉了远程控制器的需要，显著扩展了其用户基础。它结合了基于地理空间知识的远程导航与基于上下文的精细近距离控制，实现了高效的视觉-语言导航（VLN）。此外，该系统还内置了跨模态问答、智能拍摄和目标跟踪等功能。凭借高度可扩展的框架，它支持无缝集成新功能，朝着通用目的、指令驱动的智能无人驾驶航空器代理迈出了一步。更多资料请参见附录PPT，链接：https://supplementary.example.com/ppt。 

---
# Implicit Dual-Control for Visibility-Aware Navigation in Unstructured Environments 

**Title (ZH)**: 隐式双控制在无结构环境中的可见性感知导航 

**Authors**: Benjamin Johnson, Qilun Zhu, Robert Prucka, Morgan Barron, Miriam Figueroa-Santos, Matthew Castanier  

**Link**: [PDF](https://arxiv.org/pdf/2507.04371)  

**Abstract**: Navigating complex, cluttered, and unstructured environments that are a priori unknown presents significant challenges for autonomous ground vehicles, particularly when operating with a limited field of view(FOV) resulting in frequent occlusion and unobserved space. This paper introduces a novel visibility-aware model predictive path integral framework(VA-MPPI). Formulated as a dual control problem where perceptual uncertainties and control decisions are intertwined, it reasons over perception uncertainty evolution within a unified planning and control pipeline. Unlike traditional methods that rely on explicit uncertainty objectives, the VA-MPPI controller implicitly balances exploration and exploitation, reducing uncertainty only when system performance would be increased. The VA-MPPI framework is evaluated in simulation against deterministic and prescient controllers across multiple scenarios, including a cluttered urban alleyway and an occluded off-road environment. The results demonstrate that VA-MPPI significantly improves safety by reducing collision with unseen obstacles while maintaining competitive performance. For example, in the off-road scenario with 400 control samples, the VA-MPPI controller achieved a success rate of 84%, compared to only 8% for the deterministic controller, with all VA-MPPI failures arising from unmet stopping criteria rather than collisions. Furthermore, the controller implicitly avoids unobserved space, improving safety without explicit directives. The proposed framework highlights the potential for robust, visibility-aware navigation in unstructured and occluded environments, paving the way for future advancements in autonomous ground vehicle systems. 

**Abstract (ZH)**: 面向未知复杂、拥挤和无结构环境的自主地面车辆导航：一种新的基于视线的模型预测路径积分框架（VA-MPPI） 

---
# MLLM-Fabric: Multimodal Large Language Model-Driven Robotic Framework for Fabric Sorting and Selection 

**Title (ZH)**: MLLM-Fabric：多模态大规模语言模型驱动的纺织品分拣与选择机器人框架 

**Authors**: Liman Wang, Hanyang Zhong, Tianyuan Wang, Shan Luo, Jihong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04351)  

**Abstract**: Choosing the right fabric is crucial to meet functional and quality requirements in robotic applications for textile manufacturing, apparel production, and smart retail. We present MLLM-Fabric, a robotic framework powered by multimodal large language models (MLLMs) for fabric sorting and selection. The system includes a robotic arm, a camera, a visuotactile sensor, and a pressure sensor. It employs supervised fine-tuning and multimodal explanation-guided knowledge distillation to accurately classify and rank fabric properties. To facilitate further research, we release a dataset of 220 unique fabric samples, including RGB images and synchronized visuotactile and pressure data. Experimental results show that our Fabric-Llama-90B model consistently outperforms pretrained vision-language baselines in both property ranking accuracy and selection reliability. 

**Abstract (ZH)**: 选择合适的织物对于纺织制造、服装生产及智能零售中的机器人应用至关重要。我们提出了一种基于多模态大型语言模型（MLLMs）的机器人框架MLLM-Fabric，用于织物分类和选择。该系统包括机械臂、摄像头、视触觉传感器和压力传感器。它采用监督微调和多模态解释引导的知识 distillation 技术，准确分类和排序织物属性。为促进进一步研究，我们发布了包含220种独特织物样本的数据集，其中包括RGB图像和同步的视触觉及压力数据。实验结果表明，我们的Fabric-Llama-90B模型在属性排序准确性和选择可靠性方面均优于预训练的视觉-语言基线模型。 

---
# Wavelet Policy: Lifting Scheme for Policy Learning in Long-Horizon Tasks 

**Title (ZH)**: 小波策略：长时_horizon任务中的策略学习提升方案 

**Authors**: Hao Huang, Shuaihang Yuan, Geeta Chandra Raju Bethala, Congcong Wen, Anthony Tzes, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04331)  

**Abstract**: Policy learning focuses on devising strategies for agents in embodied artificial intelligence systems to perform optimal actions based on their perceived states. One of the key challenges in policy learning involves handling complex, long-horizon tasks that require managing extensive sequences of actions and observations with multiple modes. Wavelet analysis offers significant advantages in signal processing, notably in decomposing signals at multiple scales to capture both global trends and fine-grained details. In this work, we introduce a novel wavelet policy learning framework that utilizes wavelet transformations to enhance policy learning. Our approach leverages learnable multi-scale wavelet decomposition to facilitate detailed observation analysis and robust action planning over extended sequences. We detail the design and implementation of our wavelet policy, which incorporates lifting schemes for effective multi-resolution analysis and action generation. This framework is evaluated across multiple complex scenarios, including robotic manipulation, self-driving, and multi-robot collaboration, demonstrating the effectiveness of our method in improving the precision and reliability of the learned policy. 

**Abstract (ZH)**: 基于小波变换的策略学习框架：多尺度分析在复杂任务中的应用 

---
# AutoLayout: Closed-Loop Layout Synthesis via Slow-Fast Collaborative Reasoning 

**Title (ZH)**: AutoLayout: 通过慢速-快速协作推理实现闭环布局合成 

**Authors**: Weixing Chen, Dafeng Chi, Yang Liu, Yuxi Yang, Yexin Zhang, Yuzheng Zhuang, Xingyue Quan, Jianye Hao, Guanbin Li, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.04293)  

**Abstract**: The automated generation of layouts is vital for embodied intelligence and autonomous systems, supporting applications from virtual environment construction to home robot deployment. Current approaches, however, suffer from spatial hallucination and struggle with balancing semantic fidelity and physical plausibility, often producing layouts with deficits such as floating or overlapping objects and misaligned stacking relation. In this paper, we propose AutoLayout, a fully automated method that integrates a closed-loop self-validation process within a dual-system framework. Specifically, a slow system harnesses detailed reasoning with a Reasoning-Reflection-Generation (RRG) pipeline to extract object attributes and spatial constraints. Then, a fast system generates discrete coordinate sets and a topological relation set that are jointly validated. To mitigate the limitations of handcrafted rules, we further introduce an LLM-based Adaptive Relation Library (ARL) for generating and evaluating layouts. Through the implementation of Slow-Fast Collaborative Reasoning, the AutoLayout efficiently generates layouts after thorough deliberation, effectively mitigating spatial hallucination. Its self-validation mechanism establishes a closed-loop process that iteratively corrects potential errors, achieving a balance between physical stability and semantic consistency. The effectiveness of AutoLayout was validated across 8 distinct scenarios, where it demonstrated a significant 10.1% improvement over SOTA methods in terms of physical plausibility, semantic consistency, and functional completeness. 

**Abstract (ZH)**: 自动布局生成对于具身智能和自主系统至关重要，支持从虚拟环境构建到家庭机器人部署的各种应用。然而，当前的方法面临着空间幻觉的问题，并且难以在语义准确性和物理可行性之间取得平衡，经常产生诸如浮空对象、重叠对象和对齐不当堆叠关系的布局缺陷。在本文中，我们提出了一种名为AutoLayout的全自动方法，该方法在双系统框架内集成了一个闭环自验证过程。具体而言，慢系统利用推理-反思-生成（RRG）管道进行详细的推理提取对象属性和空间约束。然后，快系统生成离散坐标集和拓扑关系集，并联合验证。为了减轻手工艺规则的局限性，我们进一步引入了基于LLM的自适应关系库（ARL）来生成和评估布局。通过实施Slow-Fast协作推理，AutoLayout在深入讨论后高效地生成布局，有效减轻了空间幻觉。其自验证机制建立了一个闭环过程，迭代纠正潜在错误，实现了物理稳定性和语义一致性之间的平衡。AutoLayout在8种不同场景中的有效性得到了验证，与SOTA方法相比，在物理可行性、语义一致性和功能完整性方面分别实现了10.1%的显著改进。 

---
# Efficient Learning of A Unified Policy For Whole-body Manipulation and Locomotion Skills 

**Title (ZH)**: 整体 manipulation 和运动技能统一策略的高效学习 

**Authors**: Dianyong Hou, Chengrui Zhu, Zhen Zhang, Zhibin Li, Chuang Guo, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04229)  

**Abstract**: Equipping quadruped robots with manipulators provides unique loco-manipulation capabilities, enabling diverse practical applications. This integration creates a more complex system that has increased difficulties in modeling and control. Reinforcement learning (RL) offers a promising solution to address these challenges by learning optimal control policies through interaction. Nevertheless, RL methods often struggle with local optima when exploring large solution spaces for motion and manipulation tasks. To overcome these limitations, we propose a novel approach that integrates an explicit kinematic model of the manipulator into the RL framework. This integration provides feedback on the mapping of the body postures to the manipulator's workspace, guiding the RL exploration process and effectively mitigating the local optima issue. Our algorithm has been successfully deployed on a DeepRobotics X20 quadruped robot equipped with a Unitree Z1 manipulator, and extensive experimental results demonstrate the superior performance of this approach. 

**Abstract (ZH)**: 装备 manipulator 的四足机器人提供了独特的 locomo-manipulation 能力，使其能够在多种实际应用中发挥作用。这种集成增加了系统的复杂性，给建模和控制带来了更大的难度。强化学习 (RL) 通过交互学习最优控制策略，为应对这些挑战提供了有希望的解决方案。然而，当在动作和操作任务的大解决方案空间中探索时，RL 方法往往难以摆脱局部最优。为克服这些限制，我们提出了一种新的方法，将 manipulator 的显式运动学模型集成到 RL 框架中。这种集成提供了有关机器人姿态与 manipulator 工作空间映射的反馈，指导 RL 探索过程，并有效地缓解了局部最优的问题。我们的算法已在装备 Unitree Z1 manipulator 的 DeepRobotics X20 四足机器人上成功实现，并且广泛的实验结果表明了该方法的优越性能。 

---
# Learning Humanoid Arm Motion via Centroidal Momentum Regularized Multi-Agent Reinforcement Learning 

**Title (ZH)**: 基于质心动量正则化的多智能体强化学习的人形臂运动学习 

**Authors**: Ho Jae Lee, Se Hwan Jeon, Sangbae Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.04140)  

**Abstract**: Humans naturally swing their arms during locomotion to regulate whole-body dynamics, reduce angular momentum, and help maintain balance. Inspired by this principle, we present a limb-level multi-agent reinforcement learning (RL) framework that enables coordinated whole-body control of humanoid robots through emergent arm motion. Our approach employs separate actor-critic structures for the arms and legs, trained with centralized critics but decentralized actors that share only base states and centroidal angular momentum (CAM) observations, allowing each agent to specialize in task-relevant behaviors through modular reward design. The arm agent guided by CAM tracking and damping rewards promotes arm motions that reduce overall angular momentum and vertical ground reaction moments, contributing to improved balance during locomotion or under external perturbations. Comparative studies with single-agent and alternative multi-agent baselines further validate the effectiveness of our approach. Finally, we deploy the learned policy on a humanoid platform, achieving robust performance across diverse locomotion tasks, including flat-ground walking, rough terrain traversal, and stair climbing. 

**Abstract (ZH)**: 人类在运动过程中自然摆动双臂以调节全身动力学、减少角动量并帮助维持平衡。受此启发，我们提出了一种肢体级多智能体强化学习（RL）框架，通过涌现的臂部运动实现类人机器人全身协调控制。该方法为臂和腿分别采用独立的	actor-critic结构，使用集中式批评家但分散式演员，后者仅共享基础状态和质心角动量（CAM）观察结果，通过模块化的奖励设计使每个代理专注于与任务相关的行为。由CAM跟踪和阻尼奖励引导的臂部代理促进了减少总体角动量和地面反作用力矩的臂部运动，从而在步行或外部扰动下改善了平衡。与单智能体和替代多智能体基线的比较研究进一步验证了该方法的有效性。最终，我们在类人平台上部署所学习的策略，实现了在多种运动任务中稳健的性能，包括平坦地面步行、崎岖地形穿越和楼梯攀爬。 

---
# Are Learning-Based Approaches Ready for Real-World Indoor Navigation? A Case for Imitation Learning 

**Title (ZH)**: 基于学习的方法ready for实际室内导航了吗？模仿学习的案例分析 

**Authors**: Nigitha Selvaraj, Alex Mitrevski, Sebastian Houben  

**Link**: [PDF](https://arxiv.org/pdf/2507.04086)  

**Abstract**: Traditional indoor robot navigation methods provide a reliable solution when adapted to constrained scenarios, but lack flexibility or require manual re-tuning when deployed in more complex settings. In contrast, learning-based approaches learn directly from sensor data and environmental interactions, enabling easier adaptability. While significant work has been presented in the context of learning navigation policies, learning-based methods are rarely compared to traditional navigation methods directly, which is a problem for their ultimate acceptance in general navigation contexts. In this work, we explore the viability of imitation learning (IL) for indoor navigation, using expert (joystick) demonstrations to train various navigation policy networks based on RGB images, LiDAR, and a combination of both, and we compare our IL approach to a traditional potential field-based navigation method. We evaluate the approach on a physical mobile robot platform equipped with a 2D LiDAR and a camera in an indoor university environment. Our multimodal model demonstrates superior navigation capabilities in most scenarios, but faces challenges in dynamic environments, likely due to limited diversity in the demonstrations. Nevertheless, the ability to learn directly from data and generalise across layouts suggests that IL can be a practical navigation approach, and potentially a useful initialisation strategy for subsequent lifelong learning. 

**Abstract (ZH)**: 基于模仿学习的室内导航研究：将专家演示用于RGB图像、LiDAR及两者结合的导航策略训练并与传统势场导航方法对比 

---
# Generalized Locomotion in Out-of-distribution Conditions with Robust Transformer 

**Title (ZH)**: 在分布外条件下具有稳健性的通用运动学习 

**Authors**: Lingxiao Guo, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.04039)  

**Abstract**: To succeed in the real world, robots must deal with situations that differ from those seen during training. Those out-of-distribution situations for legged robot mainly include challenging dynamic gaps and perceptual gaps. Here we study the problem of robust locomotion in such novel situations. While previous methods usually rely on designing elaborate training and adaptation techniques, we approach the problem from a network model perspective. Our approach, RObust Locomotion Transformer(ROLT),a variation of transformer,could achieve robustness in a variety of unseen conditions. ROLT introduces two key designs: body tokenization and consistent dropout. Body tokenization supports knowledge share across different limbs, which boosts generalization ability of the network. Meanwhile, a novel dropout strategy enhances the policy's robustness to unseen perceptual noise. We conduct extensive experiments both on quadruped and hexapod robots. Results demonstrate that ROLT is more robust than existing methods. Although trained in only a few dynamic settings, the learned policy generalizes well to multiple unseen dynamic conditions. Additionally, despite training with clean observations, the model handles challenging corruption noise during testing. 

**Abstract (ZH)**: 面对未知情况的稳健腿部机器人运动研究：RObust Locomotion Transformer（ROLT）方法探究 

---
# RwoR: Generating Robot Demonstrations from Human Hand Collection for Policy Learning without Robot 

**Title (ZH)**: RwoR: 从人类手部采集生成机器人演示以在无机器人情况下进行策略学习 

**Authors**: Liang Heng, Xiaoqi Li, Shangqing Mao, Jiaming Liu, Ruolin Liu, Jingli Wei, Yu-Kai Wang, Yueru Jia, Chenyang Gu, Rui Zhao, Shanghang Zhang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.03930)  

**Abstract**: Recent advancements in imitation learning have shown promising results in robotic manipulation, driven by the availability of high-quality training data. To improve data collection efficiency, some approaches focus on developing specialized teleoperation devices for robot control, while others directly use human hand demonstrations to obtain training this http URL, the former requires both a robotic system and a skilled operator, limiting scalability, while the latter faces challenges in aligning the visual gap between human hand demonstrations and the deployed robot this http URL address this, we propose a human hand data collection system combined with our hand-to-gripper generative model, which translates human hand demonstrations into robot gripper demonstrations, effectively bridging the observation this http URL, a GoPro fisheye camera is mounted on the human wrist to capture human hand this http URL then train a generative model on a self-collected dataset of paired human hand and UMI gripper demonstrations, which have been processed using a tailored data pre-processing strategy to ensure alignment in both timestamps and this http URL, given only human hand demonstrations, we are able to automatically extract the corresponding SE(3) actions and integrate them with high-quality generated robot demonstrations through our generation pipeline for training robotic policy this http URL experiments, the robust manipulation performance demonstrates not only the quality of the generated robot demonstrations but also the efficiency and practicality of our data collection this http URL demonstrations can be found at: this https URL 

**Abstract (ZH)**: 最近在模仿学习方面的进展展示了其在机器人操作中的有希望的结果，这得益于高质量训练数据的可用性。为了提高数据采集效率，一些方法专注于开发专门的远程操作设备以控制机器人，而另一些方法直接利用人类手部演示来获得训练数据。前者需要一个机器人系统和一个熟练的操作员，限制了其可扩展性，而后者则面临人类手部演示与部署机器人之间视觉差异的对齐问题。为了解决这个问题，我们提出了一种结合手到 gripper 生成模型的人类手部数据采集系统，该系统将人类手部演示转化为机器人 gripper 演示，有效地弥合了观察之间的差距。我们使用鱼眼相机安装在人类手腕上，捕捉人类手部演示。然后在一个自收集的人类手部和UMI gripper 演示配对数据集上训练生成模型，该数据集经过定制的数据预处理策略处理以确保在时间戳和空间上的对齐。仅给定人类手部演示，我们能够自动生成对应的SE(3)动作，并通过我们的生成管道与高质量的机器人演示集结合，用于训练机器人策略。实验结果不仅证明了生成的机器人演示质量，还展示了我们的数据采集方法的高效性和实用性。生成的演示可以在以下网址找到：[相关网址]。 

---
# DK-RRT: Deep Koopman RRT for Collision-Aware Motion Planning of Space Manipulators in Dynamic Debris Environments 

**Title (ZH)**: 基于深度科莫邦曼RRT的空间 manipulator 动态碎片环境中的避碰运动规划 

**Authors**: Qi Chen, Rui Liu, Kangtong Mo, Boli Zhang, Dezhi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.03878)  

**Abstract**: Trajectory planning for robotic manipulators operating in dynamic orbital debris environments poses significant challenges due to complex obstacle movements and uncertainties. This paper presents Deep Koopman RRT (DK-RRT), an advanced collision-aware motion planning framework integrating deep learning with Koopman operator theory and Rapidly-exploring Random Trees (RRT). DK-RRT leverages deep neural networks to identify efficient nonlinear embeddings of debris dynamics, enhancing Koopman-based predictions and enabling accurate, proactive planning in real-time. By continuously refining predictive models through online sensor feedback, DK-RRT effectively navigates the manipulator through evolving obstacle fields. Simulation studies demonstrate DK-RRT's superior performance in terms of adaptability, robustness, and computational efficiency compared to traditional RRT and conventional Koopman-based planning, highlighting its potential for autonomous space manipulation tasks. 

**Abstract (ZH)**: 基于动态轨道 debris 环境下的机械臂轨迹规划面临着复杂障碍物运动和不确定性带来的显著挑战。本文提出了一种将深度学习、Koopman 操作符理论和随机树算法（RRT）相结合的先进防碰撞轨迹规划框架 Deep Koopman RRT（DK-RRT）。DK-RRT 利用深度神经网络识别高效的非线性 debris 动力学嵌入，增强基于 Koopman 的预测能力，实现实时的精确、主动规划。通过不断利用在线传感器反馈优化预测模型，DK-RRT 有效地引导机械臂穿越不断变化的障碍物场。仿真研究结果表明，相比于传统 RRT 和常规的基于 Koopman 的规划方法，DK-RRT 在适应性、鲁棒性和计算效率方面具有明显优势，展示了其在自主太空操作任务中的潜力。 

---
# Image-driven Robot Drawing with Rapid Lognormal Movements 

**Title (ZH)**: 基于图像的机器人快速对数正态运动绘图 

**Authors**: Daniel Berio, Guillaume Clivaz, Michael Stroh, Oliver Deussen, Réjean Plamondon, Sylvain Calinon, Frederic Fol Leymarie  

**Link**: [PDF](https://arxiv.org/pdf/2507.03166)  

**Abstract**: Large image generation and vision models, combined with differentiable rendering technologies, have become powerful tools for generating paths that can be drawn or painted by a robot. However, these tools often overlook the intrinsic physicality of the human drawing/writing act, which is usually executed with skillful hand/arm gestures. Taking this into account is important for the visual aesthetics of the results and for the development of closer and more intuitive artist-robot collaboration scenarios. We present a method that bridges this gap by enabling gradient-based optimization of natural human-like motions guided by cost functions defined in image space. To this end, we use the sigma-lognormal model of human hand/arm movements, with an adaptation that enables its use in conjunction with a differentiable vector graphics (DiffVG) renderer. We demonstrate how this pipeline can be used to generate feasible trajectories for a robot by combining image-driven objectives with a minimum-time smoothing criterion. We demonstrate applications with generation and robotic reproduction of synthetic graffiti as well as image abstraction. 

**Abstract (ZH)**: 大规模图像生成与视觉模型结合差分渲染技术，可以通过梯度优化自然人类运动以指导生成可由机器人绘制的路径，同时考虑人类绘画/书写行为的内在物理特性，这对于视觉美学和促进更紧密更直观的人机艺术家协作场景的发展至关重要。我们提出了一种方法，通过使用sigma-lognormal手/臂运动模型及其适应性，使其能够与可微分向量图形渲染器（DiffVG）结合使用，从而实现基于图像空间定义的成本函数的梯度优化。我们展示如何通过结合图像驱动的目标和最短时间平滑准则来生成可行的机器人轨迹。我们展示了生成和机器人复制合成涂鸦以及图像抽象的应用。 

---
# Personalised Explanations in Long-term Human-Robot Interactions 

**Title (ZH)**: 长期内个性化解释的人机互动 

**Authors**: Ferran Gebellí, Anaís Garrell, Jan-Gerrit Habekost, Séverin Lemaignan, Stefan Wermter, Raquel Ros  

**Link**: [PDF](https://arxiv.org/pdf/2507.03049)  

**Abstract**: In the field of Human-Robot Interaction (HRI), a fundamental challenge is to facilitate human understanding of robots. The emerging domain of eXplainable HRI (XHRI) investigates methods to generate explanations and evaluate their impact on human-robot interactions. Previous works have highlighted the need to personalise the level of detail of these explanations to enhance usability and comprehension. Our paper presents a framework designed to update and retrieve user knowledge-memory models, allowing for adapting the explanations' level of detail while referencing previously acquired concepts. Three architectures based on our proposed framework that use Large Language Models (LLMs) are evaluated in two distinct scenarios: a hospital patrolling robot and a kitchen assistant robot. Experimental results demonstrate that a two-stage architecture, which first generates an explanation and then personalises it, is the framework architecture that effectively reduces the level of detail only when there is related user knowledge. 

**Abstract (ZH)**: 在人机交互（HRI）领域，一个基本挑战是促进人类对机器人的理解。可解释人机交互（XHRI）这一新兴领域研究生成解释的方法及其对人机交互影响的评估。先前的研究强调需要个性化这些解释的详细程度以提高可用性和理解度。本文提出了一种框架，用于更新和检索用户知识-记忆模型，使解释的详细程度能够根据不同用户的先前概念进行调整。基于本文提出框架的三种使用大型语言模型（LLMs）的架构在两种不同场景中进行了评估：巡逻机器人和厨房助手机器人。实验结果表明，两阶段架构，首先生成解释然后个性化解释，是在用户具有相关知识时有效降低解释详细程度的框架架构。 

---
# Closed-Form Robustness Bounds for Second-Order Pruning of Neural Controller Policies 

**Title (ZH)**: 闭式稳健性界估计：神经控制策略的第二阶剪枝 

**Authors**: Maksym Shamrai  

**Link**: [PDF](https://arxiv.org/pdf/2507.02953)  

**Abstract**: Deep neural policies have unlocked agile flight for quadcopters, adaptive grasping for manipulators, and reliable navigation for ground robots, yet their millions of weights conflict with the tight memory and real-time constraints of embedded microcontrollers. Second-order pruning methods, such as Optimal Brain Damage (OBD) and its variants, including Optimal Brain Surgeon (OBS) and the recent SparseGPT, compress networks in a single pass by leveraging the local Hessian, achieving far higher sparsity than magnitude thresholding. Despite their success in vision and language, the consequences of such weight removal on closed-loop stability, tracking accuracy, and safety have remained unclear. We present the first mathematically rigorous robustness analysis of second-order pruning in nonlinear discrete-time control. The system evolves under a continuous transition map, while the controller is an $L$-layer multilayer perceptron with ReLU-type activations that are globally 1-Lipschitz. Pruning the weight matrix of layer $k$ replaces $W_k$ with $W_k+\delta W_k$, producing the perturbed parameter vector $\widehat{\Theta}=\Theta+\delta\Theta$ and the pruned policy $\pi(\cdot;\widehat{\Theta})$. For every input state $s\in X$ we derive the closed-form inequality $
\|\pi(s;\Theta)-\pi(s;\widehat{\Theta})\|_2 \le C_k(s)\,\|\delta W_k\|_2, $
where the constant $C_k(s)$ depends only on unpruned spectral norms and biases, and can be evaluated in closed form from a single forward pass. The derived bounds specify, prior to field deployment, the maximal admissible pruning magnitude compatible with a prescribed control-error threshold. By linking second-order network compression with closed-loop performance guarantees, our work narrows a crucial gap between modern deep-learning tooling and the robustness demands of safety-critical autonomous systems. 

**Abstract (ZH)**: 第二阶剪枝方法在非线性离散时间控制中的稳健性分析：连接现代深度学习工具与安全关键自主系统的需求 

---
# From Marginal to Joint Predictions: Evaluating Scene-Consistent Trajectory Prediction Approaches for Automated Driving 

**Title (ZH)**: 从边缘到联合预测：评估场景一致的轨迹预测方法在自动驾驶中的性能 

**Authors**: Fabian Konstantinidis, Ariel Dallari Guerreiro, Raphael Trumpp, Moritz Sackmann, Ulrich Hofmann, Marco Caccamo, Christoph Stiller  

**Link**: [PDF](https://arxiv.org/pdf/2507.05254)  

**Abstract**: Accurate motion prediction of surrounding traffic participants is crucial for the safe and efficient operation of automated vehicles in dynamic environments. Marginal prediction models commonly forecast each agent's future trajectories independently, often leading to sub-optimal planning decisions for an automated vehicle. In contrast, joint prediction models explicitly account for the interactions between agents, yielding socially and physically consistent predictions on a scene level. However, existing approaches differ not only in their problem formulation but also in the model architectures and implementation details used, making it difficult to compare them. In this work, we systematically investigate different approaches to joint motion prediction, including post-processing of the marginal predictions, explicitly training the model for joint predictions, and framing the problem as a generative task. We evaluate each approach in terms of prediction accuracy, multi-modality, and inference efficiency, offering a comprehensive analysis of the strengths and limitations of each approach. Several prediction examples are available at this https URL. 

**Abstract (ZH)**: 周围的交通参与者准确运动预测对于动态环境下自动驾驶车辆的安全高效运行至关重要。单个预测模型通常独立预测每个代理的未来轨迹，这往往会导致自动驾驶车辆规划决策的次优结果。相比之下，联合预测模型明确考虑代理之间的交互，从而在场景级别上提供社会上和物理上一致的预测。然而，现有的方法不仅在问题表述上不同，还在所使用的大模型架构和实现细节上有所不同，这使得它们难以进行比较。在本文中，我们系统地研究了不同的联合运动预测方法，包括对单个预测的后处理、明确训练模型进行联合预测以及将问题定义为生成任务。我们从预测准确性、多模态性和推理效率等方面评估每种方法，全面分析每种方法的优点和局限性。一些预测示例可在以下链接查阅：this https URL。 

---
# Critiques of World Models 

**Title (ZH)**: 世界模型的批判 

**Authors**: Eric Xing, Mingkai Deng, Jinyu Hou, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.05169)  

**Abstract**: World Model, the supposed algorithmic surrogate of the real-world environment which biological agents experience with and act upon, has been an emerging topic in recent years because of the rising needs to develop virtual agents with artificial (general) intelligence. There has been much debate on what a world model really is, how to build it, how to use it, and how to evaluate it. In this essay, starting from the imagination in the famed Sci-Fi classic Dune, and drawing inspiration from the concept of "hypothetical thinking" in psychology literature, we offer critiques of several schools of thoughts on world modeling, and argue the primary goal of a world model to be simulating all actionable possibilities of the real world for purposeful reasoning and acting. Building on the critiques, we propose a new architecture for a general-purpose world model, based on hierarchical, multi-level, and mixed continuous/discrete representations, and a generative and self-supervision learning framework, with an outlook of a Physical, Agentic, and Nested (PAN) AGI system enabled by such a model. 

**Abstract (ZH)**: 世界模型：作为一种生物代理体验和作用于真实世界环境的算法替代品，近年来由于开发具有人工（通用）智能的虚拟代理的需求增加，已经成为一个新兴话题。关于世界模型的本质、构建方法、使用方式以及评估标准，存在着诸多争论。本文从著名的科幻经典《徐古》中的想象出发，借鉴心理学文献中的“假设思维”概念，对几种世界建模学派的观点进行了批判，并提出世界模型的主要目标是在目的性推理和行动中模拟所有可行动的可能性。基于这些批判，我们提出了一种新的通用世界模型架构，基于分层、多级和混合连续/离散表示，并提出了一种生成性和自我监督学习框架，展望在这种模型支持下能够实现具身、物理和嵌套（PAN）的人工通用智能系统。 

---
# VOTE: Vision-Language-Action Optimization with Trajectory Ensemble Voting 

**Title (ZH)**: VOTE：轨迹集成投票的视觉-语言-动作优化 

**Authors**: Juyi Lin, Amir Taherin, Arash Akbari, Arman Akbari, Lei Lu, Guangyu Chen, Taskin Padir, Xiaomeng Yang, Weiwei Chen, Yiqian Li, Xue Lin, David Kaeli, Pu Zhao, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05116)  

**Abstract**: Recent large-scale Vision Language Action (VLA) models have shown superior performance in robotic manipulation tasks guided by natural language. However, their generalization remains limited when applied to novel objects or unfamiliar environments that lie outside the training distribution. To address this, many existing approaches integrate additional components such as depth estimation, segmentation, or even diffusion to improve generalization, at the cost of adding significant computation overhead, resulting in low efficiency. This motivates the exploration of efficient action prediction methods, which are independent of additional high-level visual representations or diffusion techniques. In this work, we propose VOTE, an efficient and general framework for the optimization and acceleration of VLA models. In details, we propose a novel tokenizer-free fine-tuning approach for parallel accurate action prediction, which reduces computational overhead and accelerates inference speed. Additionally, we adopt an ensemble voting strategy for the action sampling, which significantly improves model performance and enhances generalization. Experimental results show that our method achieves state-of-the-art performance with 35$\times$ faster inference and 145 Hz throughput. All the details and codes will be open-sourced. 

**Abstract (ZH)**: 近期的大规模视觉语言动作（VLA）模型在由自然语言指导的机器人 manipulation 任务中表现出色。然而，当应用于训练分布之外的新奇对象或不熟悉的环境时，其泛化能力仍然有限。为了解决这一问题，许多现有方法通过集成深度估计、分割，甚至扩散等额外组件来提高泛化能力，但会增加显著的计算开销，导致效率低下。这促使探索与额外高级视觉表示或扩散技术无关的高效动作预测方法。在本文中，我们提出了VOTE，一种用于提升视觉语言动作模型优化和加速的一般框架。该方法提出了一种新颖的无分词器并行准确动作预测的微调方法，减少了计算开销并加快了推理速度。此外，我们采用了集合投票策略进行动作采样，显著提升了模型性能并增强了泛化能力。实验结果表明，我们的方法在推理速度提高35倍且吞吐量达到145 Hz的情况下达到了最先进的性能，并且所有细节和代码将开源。 

---
# From Autonomy to Agency: Agentic Vehicles for Human-Centered Mobility Systems 

**Title (ZH)**: 从自主到能动：为以人为中心的移动系统设计能动车辆 

**Authors**: Jiangbo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04996)  

**Abstract**: Autonomy, from the Greek autos (self) and nomos (law), refers to the capacity to operate according to internal rules without external control. Accordingly, autonomous vehicles (AuVs) are defined as systems capable of perceiving their environment and executing preprogrammed tasks independently of external input. However, both research and real-world deployments increasingly showcase vehicles that demonstrate behaviors beyond this definition (including the SAE levels 1 to 6), such as interaction with humans and machines, goal adaptation, contextual reasoning, external tool use, and long-term planning, particularly with the integration of large language models (LLMs) and agentic AI systems. These developments reveal a conceptual gap between technical autonomy and the broader cognitive and social capabilities needed for future human-centered mobility systems. To address this, we introduce the concept of agentic vehicles (AgVs), referring to vehicles that integrate agentic AI to reason, adapt, and interact within complex environments. This paper presents a systems-level framework to characterize AgVs, focusing on their cognitive and communicative layers and differentiating them from conventional AuVs. It synthesizes relevant advances in agentic AI, robotics, multi-agent systems, and human-machine interaction, and highlights how agentic AI, through high-level reasoning and tool use, can function not merely as computational tools but as interactive agents embedded in mobility ecosystems. The paper concludes by identifying key challenges in the development and governance of AgVs, including safety, real-time control, public acceptance, ethical alignment, and regulatory frameworks. 

**Abstract (ZH)**: 自主性，源自希腊语的autos（自我）和nomos（法则），指的是根据内部规则自主运行而不受外部控制的能力。相应地，自主车辆（AuVs）被定义为能够独立于外部输入感知其环境并执行预编程任务的系统。然而，研究和实际部署越来越多地展示了表现出超出这一定义的行为的车辆（包括SAE等级1至6的车辆），如与人类和机器的交互、目标适应、情境推理、外部工具使用以及长期规划，特别是在大型语言模型（LLMs）和代理型人工智能系统的集成中。这些发展揭示了技术自主性与未来以人为中心的移动系统所需更广泛的认知和社会能力之间的概念差距。为应对这一差距，我们引入了代理型车辆（AgVs）的概念，指的是整合代理型人工智能以在复杂环境中进行推理、适应和交互的车辆。本文提出了一种系统级框架来界定AgVs，重点关注其认知和通信层，并将其与传统的AuVs区分开来。该文综合了代理型人工智能、机器人技术、多agent系统和人机交互领域的相关进展，并指出代理型人工智能如何通过高层次推理和工具使用，不仅作为计算工具，而且作为嵌入在移动生态系统中的交互代理发挥作用。论文最后指出了代理型车辆开发和治理中的关键挑战，包括安全性、实时控制、公众接受度、伦理对齐和监管框架。 

---
# Accelerated Online Reinforcement Learning using Auxiliary Start State Distributions 

**Title (ZH)**: 使用辅助起始状态分布加速在线强化学习 

**Authors**: Aman Mehra, Alexandre Capone, Jeff Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2507.04606)  

**Abstract**: A long-standing problem in online reinforcement learning (RL) is of ensuring sample efficiency, which stems from an inability to explore environments efficiently. Most attempts at efficient exploration tackle this problem in a setting where learning begins from scratch, without prior information available to bootstrap learning. However, such approaches fail to leverage expert demonstrations and simulators that can reset to arbitrary states. These affordances are valuable resources that offer enormous potential to guide exploration and speed up learning. In this paper, we explore how a small number of expert demonstrations and a simulator allowing arbitrary resets can accelerate learning during online RL. We find that training with a suitable choice of an auxiliary start state distribution that may differ from the true start state distribution of the underlying Markov Decision Process can significantly improve sample efficiency. We find that using a notion of safety to inform the choice of this auxiliary distribution significantly accelerates learning. By using episode length information as a way to operationalize this notion, we demonstrate state-of-the-art sample efficiency on a sparse-reward hard-exploration environment. 

**Abstract (ZH)**: 在线强化学习中确保样本效率的长期问题源于有效探索环境的能力不足。大多数高效探索的尝试是在没有先前信息可供利用以加速学习的情况下，从头开始学习。然而，这些方法未能利用可以重置到任意状态的专家演示和模拟器。这些功能是宝贵的资源，具有极大的潜力来引导探索并加速学习。在本文中，我们探讨了如何通过少量专家演示和允许任意重置的模拟器来加速在线强化学习中的学习。我们发现，使用一个与底层马尔可夫决策过程的真实起始状态分布可能不同的适当辅助起始状态分布进行训练，可以显著提高样本效率。我们发现，使用安全性的概念来指导这种辅助分布的选择可以显著加快学习速度。通过使用 Episode 长度信息来实现这一概念，我们展示了在稀疏奖励和困难探索环境中达到最先进的样本效率。 

---
# Grounded Gesture Generation: Language, Motion, and Space 

**Title (ZH)**: 基于语境的手势生成：语言、动作与空间 

**Authors**: Anna Deichler, Jim O'Regan, Teo Guichoux, David Johansson, Jonas Beskow  

**Link**: [PDF](https://arxiv.org/pdf/2507.04522)  

**Abstract**: Human motion generation has advanced rapidly in recent years, yet the critical problem of creating spatially grounded, context-aware gestures has been largely overlooked. Existing models typically specialize either in descriptive motion generation, such as locomotion and object interaction, or in isolated co-speech gesture synthesis aligned with utterance semantics. However, both lines of work often treat motion and environmental grounding separately, limiting advances toward embodied, communicative agents. To address this gap, our work introduces a multimodal dataset and framework for grounded gesture generation, combining two key resources: (1) a synthetic dataset of spatially grounded referential gestures, and (2) MM-Conv, a VR-based dataset capturing two-party dialogues. Together, they provide over 7.7 hours of synchronized motion, speech, and 3D scene information, standardized in the HumanML3D format. Our framework further connects to a physics-based simulator, enabling synthetic data generation and situated evaluation. By bridging gesture modeling and spatial grounding, our contribution establishes a foundation for advancing research in situated gesture generation and grounded multimodal interaction.
Project page: this https URL 

**Abstract (ZH)**: 人类运动生成近年来取得了快速进展，但创建空间上具grounded性、情境aware性的手势的关键问题迄今已被广泛关注不足。现有模型通常专门处理描述性运动生成，如行进和物体交互，或者孤立的共声手势合成，与语义对齐。然而，这两方面的研究往往将运动和环境grounding分开处理，限制了具身、交际型代理的研究进展。为弥补这一空白，我们的工作引入了一个多模态数据集和框架，用于生成grounded手势，结合了两个关键资源：（1）一个空间上具grounded性的参考手势合成数据集，（2）一个基于VR的对话数据集，捕捉两人的对话。它们共同提供了超过7.7小时的同步运动、语音和三维场景信息，并统一使用了HumanML3D格式。我们的框架进一步与物理基础模拟器连接，使合成数据生成和情境评估成为可能。通过将手势建模与空间grounding相结合，我们的贡献为推进情境手势生成和多模态交互的研究奠定了基础。 

---
# Thousand-Brains Systems: Sensorimotor Intelligence for Rapid, Robust Learning and Inference 

**Title (ZH)**: 千脑系统：感知运动智能以实现快速稳健的学习与推理 

**Authors**: Niels Leadholm, Viviane Clay, Scott Knudstrup, Hojae Lee, Jeff Hawkins  

**Link**: [PDF](https://arxiv.org/pdf/2507.04494)  

**Abstract**: Current AI systems achieve impressive performance on many tasks, yet they lack core attributes of biological intelligence, including rapid, continual learning, representations grounded in sensorimotor interactions, and structured knowledge that enables efficient generalization. Neuroscience theory suggests that mammals evolved flexible intelligence through the replication of a semi-independent, sensorimotor module, a functional unit known as a cortical column. To address the disparity between biological and artificial intelligence, thousand-brains systems were proposed as a means of mirroring the architecture of cortical columns and their interactions.
In the current work, we evaluate the unique properties of Monty, the first implementation of a thousand-brains system. We focus on 3D object perception, and in particular, the combined task of object recognition and pose estimation. Utilizing the YCB dataset of household objects, we first assess Monty's use of sensorimotor learning to build structured representations, finding that these enable robust generalization. These representations include an emphasis on classifying objects by their global shape, as well as a natural ability to detect object symmetries. We then explore Monty's use of model-free and model-based policies to enable rapid inference by supporting principled movements. We find that such policies complement Monty's modular architecture, a design that can accommodate communication between modules to further accelerate inference speed via a novel `voting' algorithm. Finally, we examine Monty's use of associative, Hebbian-like binding to enable rapid, continual, and computationally efficient learning, properties that compare favorably to current deep learning architectures. While Monty is still in a nascent stage of development, these findings support thousand-brains systems as a powerful and promising new approach to AI. 

**Abstract (ZH)**: 当前的AI系统在许多任务上取得了令人印象深刻的性能，但缺乏生物智能的核心属性，包括快速、持续的学习，基于传感器动作交互的表示，以及能够高效泛化的结构化知识。神经科学理论表明，哺乳动物通过复制半独立的、传感器动作模块，即脑皮层柱这一功能单位，进化出了灵活的智能。为了缩小生物智能与人工智能之间的差距， thousand-brains 系统被提出作为模仿脑皮层柱及其交互架构的一种手段。
在当前的研究中，我们评估了Monty的特点，Monty是第一个实现thousand-brains系统的实例。我们集中在三维物体感知，尤其是物体识别和姿态估计的综合任务上。利用家庭用品的YCB数据集，我们首先评估Monty如何利用传感器动作学习构建结构化的表示，发现这些表示能够实现稳健的泛化。这些表示包括强调通过整体形状分类物体，并且自然具备检测物体对称性的能力。然后，我们探讨了Monty如何利用无模型和基于模型的策略来实现快速推理，通过支持原理性的运动来支持。我们发现，这些策略与Monty的模块化架构相补充，这种设计可以通过一种新颖的“投票”算法在模块之间进行通信，进一步加速推理速度。最后，我们研究了Monty利用类似艾宾浩斯绑定来实现快速、持续、计算高效的学习，这些特性与当前的深度学习架构相比具有优势。尽管Monty仍处于发展的初级阶段，但这些发现支持thousand-brains系统作为一种强大而有前途的新AI方法。 

---
# DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge 

**Title (ZH)**: DreamVLA：融入全面世界知识的视觉-语言-行动模型 

**Authors**: Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, XinQiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, He Wang, Zhizheng Zhang, Li Yi, Wenjun Zeng, Xin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2507.04447)  

**Abstract**: Recent advances in vision-language-action (VLA) models have shown promise in integrating image generation with action prediction to improve generalization and reasoning in robot manipulation. However, existing methods are limited to challenging image-based forecasting, which suffers from redundant information and lacks comprehensive and critical world knowledge, including dynamic, spatial and semantic information. To address these limitations, we propose DreamVLA, a novel VLA framework that integrates comprehensive world knowledge forecasting to enable inverse dynamics modeling, thereby establishing a perception-prediction-action loop for manipulation tasks. Specifically, DreamVLA introduces a dynamic-region-guided world knowledge prediction, integrated with the spatial and semantic cues, which provide compact yet comprehensive representations for action planning. This design aligns with how humans interact with the world by first forming abstract multimodal reasoning chains before acting. To mitigate interference among the dynamic, spatial and semantic information during training, we adopt a block-wise structured attention mechanism that masks their mutual attention, preventing information leakage and keeping each representation clean and disentangled. Moreover, to model the conditional distribution over future actions, we employ a diffusion-based transformer that disentangles action representations from shared latent features. Extensive experiments on both real-world and simulation environments demonstrate that DreamVLA achieves 76.7% success rate on real robot tasks and 4.44 average length on the CALVIN ABC-D benchmarks. 

**Abstract (ZH)**: 近期在视觉-语言-动作（VLA）模型方面的进展展示了将图像生成与动作预测相结合以提高机器人操作中泛化能力和推理的潜力。然而，现有方法受限于基于图像的预测挑战，这些预测包含冗余信息且缺乏综合性和关键的世界知识，包括动态、空间和语义信息。为了解决这些限制，我们提出了DreamVLA，一种新颖的VLA框架，能够整合全面的世界知识预测以实现逆动力学建模，从而建立感知-预测-动作循环以完成操作任务。具体来说，DreamVLA 引入了动态区域引导的世界知识预测，结合了空间和语义线索，为动作规划提供了紧凑且综合的表示。此设计符合人类在行动前先形成抽象多模态推理链的方式。为了在训练过程中减轻动态、空间和语义信息之间的相互干扰，我们采用了块结构化的注意力机制，屏蔽它们之间的相互注意，防止信息泄露并保持每个表示的清晰和独立。此外，为了建模未来动作的条件分布，我们采用了基于扩散的变换器来独立动作表示与共享的潜在特征。在现实世界和模拟环境中的广泛实验表明，DreamVLA 在真实机器人任务中的成功率达到了76.7%，并在CALVIN ABC-D基准测试中实现了4.44的平均长度。 

---
# Mission-Aligned Learning-Informed Control of Autonomous Systems: Formulation and Foundations 

**Title (ZH)**: 自主系统的目标导向学习驱动控制：形式化与基础理论 

**Authors**: Vyacheslav Kungurtsev, Gustav Sir, Akhil Anand, Sebastien Gros, Haozhe Tian, Homayoun Hamedmoghadam  

**Link**: [PDF](https://arxiv.org/pdf/2507.04356)  

**Abstract**: Research, innovation and practical capital investment have been increasing rapidly toward the realization of autonomous physical agents. This includes industrial and service robots, unmanned aerial vehicles, embedded control devices, and a number of other realizations of cybernetic/mechatronic implementations of intelligent autonomous devices. In this paper, we consider a stylized version of robotic care, which would normally involve a two-level Reinforcement Learning procedure that trains a policy for both lower level physical movement decisions as well as higher level conceptual tasks and their sub-components. In order to deliver greater safety and reliability in the system, we present the general formulation of this as a two-level optimization scheme which incorporates control at the lower level, and classical planning at the higher level, integrated with a capacity for learning. This synergistic integration of multiple methodologies -- control, classical planning, and RL -- presents an opportunity for greater insight for algorithm development, leading to more efficient and reliable performance. Here, the notion of reliability pertains to physical safety and interpretability into an otherwise black box operation of autonomous agents, concerning users and regulators. This work presents the necessary background and general formulation of the optimization framework, detailing each component and its integration with the others. 

**Abstract (ZH)**: 研究、创新和实际资本投资正迅速增加以实现自主物理代理。这包括工业和服务业机器人、无人驾驶航空车辆、嵌入式控制设备以及许多其他基于信息技术和机电一体化的智能自主设备。本文考虑了一种精简版的机器人护理，通常涉及两层强化学习程序，用于训练低层物理动作决策和高层概念任务及其子组件的策略。为了提高系统的安全性和可靠性，我们提出将这一过程作为包含低层控制和高层经典规划的两层优化方案，结合学习能力。多种方法——控制、经典规划和RL——的这种协同集成为算法开发提供了更多的洞察力，从而实现更高效和可靠的性能。在这里，可靠性的概念涉及物理安全和对自主代理黑盒操作的可解释性，对于用户和监管机构而言。本文提供了优化框架的必要背景和一般性形式，并详细说明了每个组成部分及其与其他组成部分的集成。 

---
# Human-centered AI with focus on Human-robot interaction (Book chapter) 

**Title (ZH)**: 以人为中心的AI——以人机交互为重点（书章节） 

**Authors**: Alireza Mortezapour, Giuliana Vitiello  

**Link**: [PDF](https://arxiv.org/pdf/2507.04095)  

**Abstract**: Modern social robots can be considered the descendants of steam engines from the First Industrial Revolution (IR 1.0) and industrial robotic arms from the Third Industrial Revolution (IR 3.0). As some time has passed since the introduction of these robots during the Fourth Industrial Revolution (IR 4.0), challenges and issues in their interaction with humans have emerged, leading researchers to conclude that, like any other AI-based technology, these robots must also be human-centered to meet the needs of their users. This chapter aims to introduce humans and their needs in interactions with robots, ranging from short-term, one-on-one interactions (micro-level) to long-term, macro-level needs at the societal scale. Building upon the principles of human-centered AI, this chapter presents, for the first time, a new framework of human needs called the Dual Pyramid. This framework encompasses a comprehensive list of human needs in robot interactions, from the most fundamental, robot effectiveness to macro level requirements, such as the collaboration with robots in achieving the United Nations 17 Sustainable Development Goals. 

**Abstract (ZH)**: 现代社交机器人可以被视为第一工业革命（IR 1.0）中的蒸汽发动机和第三工业革命（IR 3.0）中的工业机器人臂的后裔。随着第四工业革命（IR 4.0）中这些机器人的引入时间推移，它们与人类互动中出现了挑战和问题，促使研究人员认识到，就像其他任何基于人工智能的技术一样，这些机器人也必须以人类为中心，以满足用户的需求。本章旨在介绍人类及其在与机器人互动中的需求，从短期一对一互动（微观层面）到长期、宏观层面的需求（社会层面）。基于以人为本的人工智能原则，本章首次提出了一种新的需求框架，称为双棱锥框架。该框架涵盖了机器人互动中全面的人类需求清单，从最基本的有效性需求到宏观层面的要求，如与机器人合作实现联合国17项可持续发展目标。 

---
# Breaking Imitation Bottlenecks: Reinforced Diffusion Powers Diverse Trajectory Generation 

**Title (ZH)**: 打破imitation瓶颈：强化扩散赋能多样化轨迹生成 

**Authors**: Ziying Song, Lin Liu, Hongyu Pan, Bencheng Liao, Mingzhe Guo, Lei Yang, Yongchang Zhang, Shaoqing Xu, Caiyan Jia, Yadan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2507.04049)  

**Abstract**: Most end-to-end autonomous driving methods rely on imitation learning from single expert demonstrations, often leading to conservative and homogeneous behaviors that limit generalization in complex real-world scenarios. In this work, we propose DIVER, an end-to-end driving framework that integrates reinforcement learning with diffusion-based generation to produce diverse and feasible trajectories. At the core of DIVER lies a reinforced diffusion-based generation mechanism. First, the model conditions on map elements and surrounding agents to generate multiple reference trajectories from a single ground-truth trajectory, alleviating the limitations of imitation learning that arise from relying solely on single expert demonstrations. Second, reinforcement learning is employed to guide the diffusion process, where reward-based supervision enforces safety and diversity constraints on the generated trajectories, thereby enhancing their practicality and generalization capability. Furthermore, to address the limitations of L2-based open-loop metrics in capturing trajectory diversity, we propose a novel Diversity metric to evaluate the diversity of multi-mode this http URL experiments on the closed-loop NAVSIM and Bench2Drive benchmarks, as well as the open-loop nuScenes dataset, demonstrate that DIVER significantly improves trajectory diversity, effectively addressing the mode collapse problem inherent in imitation learning. 

**Abstract (ZH)**: 一种结合强化学习与扩散生成的端到端驾驶框架：多样性和可行性并重的轨迹生成 

---
# MedGemma Technical Report 

**Title (ZH)**: MedGemma技术报告 

**Authors**: Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, Atilla Kiraly, Madeleine Traverse, Timo Kohlberger, Shawn Xu, Fayaz Jamil, Cían Hughes, Charles Lau, Justin Chen, Fereshteh Mahvar, Liron Yatziv, Tiffany Chen, Bram Sterling, Stefanie Anna Baby, Susanna Maria Baby, Jeremy Lai, Samuel Schmidgall, Lu Yang, Kejia Chen, Per Bjornsson, Shashir Reddy, Ryan Brush, Kenneth Philbrick, Howard Hu, Howard Yang, Richa Tiwari, Sunny Jansen, Preeti Singh, Yun Liu, Shekoofeh Azizi, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Riviere, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean-bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Elena Buchatskaya, Jean-Baptiste Alayrac, Dmitry, Lepikhin, Vlad Feinberg, Sebastian Borgeaud, Alek Andreev, Cassidy Hardin, Robert Dadashi, Léonard Hussenot, Armand Joulin, Olivier Bachem, Yossi Matias, Katherine Chou, Avinatan Hassidim, Kavi Goel, Clement Farabet, Joelle Barral, Tris Warkentin, Jonathon Shlens, David Fleet, Victor Cotruta, Omar Sanseviero, Gus Martins, Phoebe Kirk, Anand Rao, Shravya Shetty, David F. Steiner, Can Kirmizibayrak, Rory Pilgrim, Daniel Golden, Lin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05201)  

**Abstract**: Artificial intelligence (AI) has significant potential in healthcare applications, but its training and deployment faces challenges due to healthcare's diverse data, complex tasks, and the need to preserve privacy. Foundation models that perform well on medical tasks and require less task-specific tuning data are critical to accelerate the development of healthcare AI applications. We introduce MedGemma, a collection of medical vision-language foundation models based on Gemma 3 4B and 27B. MedGemma demonstrates advanced medical understanding and reasoning on images and text, significantly exceeding the performance of similar-sized generative models and approaching the performance of task-specific models, while maintaining the general capabilities of the Gemma 3 base models. For out-of-distribution tasks, MedGemma achieves 2.6-10% improvement on medical multimodal question answering, 15.5-18.1% improvement on chest X-ray finding classification, and 10.8% improvement on agentic evaluations compared to the base models. Fine-tuning MedGemma further improves performance in subdomains, reducing errors in electronic health record information retrieval by 50% and reaching comparable performance to existing specialized state-of-the-art methods for pneumothorax classification and histopathology patch classification. We additionally introduce MedSigLIP, a medically-tuned vision encoder derived from SigLIP. MedSigLIP powers the visual understanding capabilities of MedGemma and as an encoder achieves comparable or better performance than specialized medical image encoders. Taken together, the MedGemma collection provides a strong foundation of medical image and text capabilities, with potential to significantly accelerate medical research and development of downstream applications. The MedGemma collection, including tutorials and model weights, can be found at this https URL. 

**Abstract (ZH)**: 人工智能（AI）在医疗应用中有巨大的潜力，但由于医疗数据的多样性、任务的复杂性以及需要保护隐私，其训练和部署面临着挑战。能够很好地完成医疗任务且需要较少的特定任务调优数据的基础模型对于加速医疗AI应用的发展至关重要。我们介绍了MedGemma，这是一种基于Gemma 3 4B和27B的医疗视觉-语言基础模型集合。MedGemma在图像和文本上展示了高级的医学理解和推理能力，显著超过了同类生成模型的性能，并接近特定任务模型的性能，同时保持了Gemma 3基础模型的一般能力。对于分布外任务，MedGemma在医疗多模态问答上的表现提高了2.6-10%，在胸部X光检查分类上的表现提高了15.5-18.1%，在代理评估上的表现提高了10.8%，优于基础模型。进一步微调MedGemma在子领域进一步提高了性能，减少了电子健康记录信息检索的错误50%，并在气胸分类和组织病理学斑块分类方面达到了现有专门方法的相当性能。此外，我们还介绍了MedSigLIP，这是一种基于SigLIP的医学调优视觉编码器。MedSigLIP增强了MedGemma的视觉理解能力，作为编码器，其性能与专门的医学图像编码器相当或更好。总体而言，MedGemma集合提供了一种强大的医学图像和文本能力基础，有望显著加速医学研究和下游应用的发展。MedGemma集合，包括教程和模型权重，可在以下链接找到：this https URL。 

---
# When Imitation Learning Outperforms Reinforcement Learning in Surgical Action Planning 

**Title (ZH)**: 当模仿学习在手术动作规划中表现优于强化学习时 

**Authors**: Maxence Boels, Harry Robertshaw, Alejandro Granados, Prokar Dasgupta, Sebastien Ourselin  

**Link**: [PDF](https://arxiv.org/pdf/2507.05011)  

**Abstract**: Surgical action planning requires predicting future instrument-verb-target triplets for real-time assistance. While teleoperated robotic surgery provides natural expert demonstrations for imitation learning (IL), reinforcement learning (RL) could potentially discover superior strategies through exploration. We present the first comprehensive comparison of IL versus RL for surgical action planning on CholecT50. Our Dual-task Autoregressive Imitation Learning (DARIL) baseline achieves 34.6% action triplet recognition mAP and 33.6% next frame prediction mAP with smooth planning degradation to 29.2% at 10-second horizons. We evaluated three RL variants: world model-based RL, direct video RL, and inverse RL enhancement. Surprisingly, all RL approaches underperformed DARIL i.e. world model RL dropped to 3.1% mAP at 10s while direct video RL achieved only 15.9%. Our analysis reveals that distribution matching on expert-annotated test sets systematically favors IL over potentially valid RL policies that differ from training demonstrations. This challenges assumptions about RL superiority in sequential decision making and provides crucial insights for surgical AI development. 

**Abstract (ZH)**: 手术动作规划需要预测未来的器械-动词-目标三元组以实现实时辅助。虽然远程操作的机器人手术提供了自然的 expert 示范用于模仿学习（IL），强化学习（RL）则有可能通过探索发现更优策略。我们首次在 CholecT50 上全面比较了 IL 与 RL 在手术动作规划中的应用。我们的双任务自回归模仿学习（DARIL）基线实现 34.6% 的动作三元组识别 mAP 和 33.6% 的下一帧预测 mAP，并且在 10 秒时间窗口内平滑下降到 29.2%。我们评估了三种 RL 变体：基于世界模型的 RL、直接视频 RL 和逆 RL 增强。令人惊讶的是，所有 RL 方法均劣于 DARIL，即基于世界模型的 RL 在 10 秒时仅达到 3.1% 的 mAP，而直接视频 RL 仅达到 15.9%。我们的分析表明，针对专家标注的测试集进行分布匹配系统性地偏向于 IL 而非差异训练示例的潜在有效 RL 策略。这挑战了在顺序决策中 RL 优越性的假设，并为手术 AI 的发展提供了重要见解。 

---
# Anomalous Decision Discovery using Inverse Reinforcement Learning 

**Title (ZH)**: 使用逆强化学习发现异常决策 

**Authors**: Ashish Bastola, Mert D. Pesé, Long Cheng, Jonathon Smereka, Abolfazl Razi  

**Link**: [PDF](https://arxiv.org/pdf/2507.04464)  

**Abstract**: Anomaly detection plays a critical role in Autonomous Vehicles (AVs) by identifying unusual behaviors through perception systems that could compromise safety and lead to hazardous situations. Current approaches, which often rely on predefined thresholds or supervised learning paradigms, exhibit reduced efficacy when confronted with unseen scenarios, sensor noise, and occlusions, leading to potential safety-critical failures. Moreover, supervised methods require large annotated datasets, limiting their real-world feasibility. To address these gaps, we propose an anomaly detection framework based on Inverse Reinforcement Learning (IRL) to infer latent driving intentions from sequential perception data, thus enabling robust identification. Specifically, we present Trajectory-Reward Guided Adaptive Pre-training (TRAP), a novel IRL framework for anomaly detection, to address two critical limitations of existing methods: noise robustness and generalization to unseen scenarios. Our core innovation is implicitly learning temporal credit assignments via reward and worst-case supervision. We leverage pre-training with variable-horizon sampling to maximize time-to-consequence, resulting in early detection of behavior deviation. Experiments on 14,000+ simulated trajectories demonstrate state-of-the-art performance, achieving 0.90 AUC and 82.2\% F1-score - outperforming similarly trained supervised and unsupervised baselines by 39\% on Recall and 12\% on F1-score, respectively. Similar performance is achieved while exhibiting robustness to various noise types and generalization to unseen anomaly types. Our code will be available at: this https URL 

**Abstract (ZH)**: 自主驾驶中的异常检测通过感知系统识别可能危及安全的异常行为，发挥着关键作用。当前的方法通常依赖预定义的阈值或监督学习范式，在遇到未知场景、传感器噪声和遮挡时效用降低，可能导致潜在的安全关键性故障。此外，监督方法需要大规模标注数据集，限制了其实用性。为解决这些问题，我们提出了一种基于逆强化学习（IRL）的异常检测框架，以从序列感知数据中推断隐含的驾驶意图，从而实现稳健的识别。具体来说，我们呈现了轨迹-奖励引导自适应预训练（TRAP），这是一种新颖的用于异常检测的IRL框架，解决了现有方法的两个关键问题：噪声鲁棒性和对未知场景的泛化能力。我们的核心创新是通过奖励和最坏情况监督隐式学习时间信用分配。我们利用变时窗采样的预训练以最大化后果时间，实现行为偏离的早期检测。在14,000多个模拟轨迹上的实验展示了最先进的性能，AUC达到0.90，F1分数为82.2%，分别比同样训练的监督和无监督基线在召回率上高出39%、F1分数上高出12%。同时，该方法在面对不同类型的噪声和未见过的异常类型时表现出鲁棒性。代码将在此处提供：this https URL 

---
# WebSynthesis: World-Model-Guided MCTS for Efficient WebUI-Trajectory Synthesis 

**Title (ZH)**: WebSynthesis: 世界模型引导的 Monte Carlo 森林搜索高效网页UI轨迹合成 

**Authors**: Yifei Gao, Junhong Ye, Jiaqi Wang, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04370)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly improved the capabilities of web agents. However, effectively navigating complex and dynamic web environments still requires more advanced trajectory-level planning and execution. Prior studies have addressed self-improving agents by collecting extensive GUI trajectories from real-environment interactions. Despite their effectiveness, these approaches encounter two critical challenges: (1) Uncontrollable environment states, where real or sandboxed web environments often yield unstable and non-deterministic feedback, complicating the reproduction and debugging of agent behaviors; and (2) High API costs, as generating even a single interaction trajectory can involve hundreds of queries, leading to considerable API usage and computational expenses. To address these limitations and enable scalable self-improvement for agents, we propose WebSynthesis, a novel framework for trajectory synthesis and training. WebSynthesis leverages a learned world model to simulate virtual web environments, allowing a policy agent to perform efficient and reversible tree-based planning. This approach supports the large-scale generation of diverse and high-quality trajectories, which are subsequently utilized to refine the agent's policy. Experimental results demonstrate that an agent trained using WebSynthesis on a small-scale synthetic dataset achieves performance comparable to or even surpassing that of models trained on large-scale real-world data. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）显著提升了网络代理的能力。然而，有效地在复杂和动态的网络环境中导航仍然需要更高级的轨迹级规划和执行。先前的研究通过收集来自真实环境交互的广泛GUI轨迹来处理自我改进的代理。尽管这些方法有效，但它们遇到了两个关键挑战：（1）不可控的环境状态，其中真实的或沙箱化的网络环境通常会导致不稳定和非确定性的反馈，复杂化了代理行为的重现和调试；（2）高昂的API成本，生成单个交互轨迹可能需要数百次查询，导致大量的API使用和计算开销。为了解决这些限制并使代理的自我改进可扩展，我们提出WebSynthesis，一种新型的轨迹合成与训练框架。WebSynthesis利用学习到的世界模型来模拟虚拟网络环境，使策略代理能够进行高效且可逆的树状规划。这种方法支持大规模生成多样化和高质量的轨迹，随后用于细化代理策略。实验结果表明，使用WebSynthesis在小型合成数据集上训练的代理，在性能上与或甚至超越了在大规模真实世界数据上训练的模型。 

---
# Multi-Agent Reasoning for Cardiovascular Imaging Phenotype Analysis 

**Title (ZH)**: 多agent推理在心血管影像表型分析中的应用 

**Authors**: Weitong Zhang, Mengyun Qiao, Chengqi Zang, Steven Niederer, Paul M Matthews, Wenjia Bai, Bernhard Kainz  

**Link**: [PDF](https://arxiv.org/pdf/2507.03460)  

**Abstract**: Identifying the associations between imaging phenotypes and disease risk factors and outcomes is essential for understanding disease mechanisms and improving diagnosis and prognosis models. However, traditional approaches rely on human-driven hypothesis testing and selection of association factors, often overlooking complex, non-linear dependencies among imaging phenotypes and other multi-modal data. To address this, we introduce a Multi-agent Exploratory Synergy for the Heart (MESHAgents) framework that leverages large language models as agents to dynamically elicit, surface, and decide confounders and phenotypes in association studies, using cardiovascular imaging as a proof of concept. Specifically, we orchestrate a multi-disciplinary team of AI agents -- spanning cardiology, biomechanics, statistics, and clinical research -- which spontaneously generate and converge on insights through iterative, self-organizing reasoning. The framework dynamically synthesizes statistical correlations with multi-expert consensus, providing an automated pipeline for phenome-wide association studies (PheWAS). We demonstrate the system's capabilities through a population-based study of imaging phenotypes of the heart and aorta. MESHAgents autonomously uncovered correlations between imaging phenotypes and a wide range of non-imaging factors, identifying additional confounder variables beyond standard demographic factors. Validation on diagnosis tasks reveals that MESHAgents-discovered phenotypes achieve performance comparable to expert-selected phenotypes, with mean AUC differences as small as -0.004 on disease classification tasks. Notably, the recall score improves for 6 out of 9 disease types. Our framework provides clinically relevant imaging phenotypes with transparent reasoning, offering a scalable alternative to expert-driven methods. 

**Abstract (ZH)**: 利用多智能体探索协同框架（MESHAgents）识别影像表型与疾病风险因素及结局之间的关联对于理解疾病机制和改进诊断及预后模型至关重要。传统方法依赖于人工驱动的假设测试和关联因素的选择，往往会忽略影像表型与其他多模态数据之间复杂且非线性的依赖关系。为解决这一问题，我们引入了一种基于多智能体探索协同框架（MESHAgents），利用大语言模型作为智能体动态地提出、呈现和决定混杂因素和表型，在心血管影像方面作为概念验证。具体而言，我们协调了一个跨心脏科、生物力学、统计学和临床研究的多学科AI智能体团队，通过迭代的自我组织推理自发生成并收敛于见解。该框架动态地综合了多元专家共识的统计相关性，提供了一种自动化的全表型关联研究（PheWAS）管道。我们通过基于人群的心脏和主动脉影像表型研究展示了系统的功能。MESHAgents自主发现了影像表型与非影像因素之间的多种相关性，识别出标准化的demographic因素之外的额外混杂变量。在诊断任务上的验证表明，MESHAgents发现的表型在疾病分类任务上的性能与专家选择的表型相当，平均AUC差异仅为-0.004。值得注意的是，6种疾病类型的召回率有所提高。该框架提供了具有透明推理的临床相关影像表型，提供了一种可扩展的专家驱动方法替代方案。 

---
# LTLCrit: A Temporal Logic-based LLM Critic for Safe and Efficient Embodied Agents 

**Title (ZH)**: 基于时序逻辑的LLM批评家：用于安全高效体现式代理的LTLCrit 

**Authors**: Anand Gokhale, Vaibhav Srivastava, Francesco Bullo  

**Link**: [PDF](https://arxiv.org/pdf/2507.03293)  

**Abstract**: Large language models (LLMs) have demonstrated promise in reasoning tasks and general decision-making in static environments. In long-term planning tasks, however, errors tend to accumulate, often leading to unsafe or inefficient behavior, limiting their use in general-purpose settings. We propose a modular actor-critic architecture in which an LLM actor is guided by LTLCrit, a trajectory-level LLM critic that communicates via linear temporal logic (LTL). Our setup combines the reasoning strengths of language models with the guarantees of formal logic. The actor selects high-level actions from natural language observations, while the critic analyzes full trajectories and proposes new LTL constraints that shield the actor from future unsafe or inefficient behavior. The architecture supports both fixed, hand-specified safety constraints and adaptive, learned soft constraints that promote long-term efficiency. Our architecture is model-agnostic: any LLM-based planner can serve as the actor, and LTLCrit serves as a logic-generating wrapper. We formalize planning as graph traversal under symbolic constraints, allowing LTLCrit to analyze failed or suboptimal trajectories and generate new temporal logic rules that improve future behavior. We evaluate our system on the Minecraft diamond-mining benchmark, achieving 100% completion rates and improving efficiency compared to baseline LLM planners. Our results suggest that enabling LLMs to supervise each other through logic is a powerful and flexible paradigm for safe, generalizable decision making. 

**Abstract (ZH)**: 大型语言模型在长期规划任务中的模块化actor-critic架构：通过线性时序逻辑实现安全与效率 

---
# UrbanMind: Towards Urban General Intelligence via Tool-Enhanced Retrieval-Augmented Generation and Multilevel Optimization 

**Title (ZH)**: UrbanMind：通过工具增强的检索增强生成和多层优化 toward 城市通用智能 

**Authors**: Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, Ye Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04706)  

**Abstract**: Urban general intelligence (UGI) refers to the capacity of AI systems to autonomously perceive, reason, and act within dynamic and complex urban environments. In this paper, we introduce UrbanMind, a tool-enhanced retrieval-augmented generation (RAG) framework designed to facilitate UGI. Central to UrbanMind is a novel architecture based on Continual Retrieval-Augmented MoE-based LLM (C-RAG-LLM), which dynamically incorporates domain-specific knowledge and evolving urban data to support long-term adaptability. The architecture of C-RAG-LLM aligns naturally with a multilevel optimization framework, where different layers are treated as interdependent sub-problems. Each layer has distinct objectives and can be optimized either independently or jointly through a hierarchical learning process. The framework is highly flexible, supporting both end-to-end training and partial layer-wise optimization based on resource or deployment constraints. To remain adaptive under data drift, it is further integrated with an incremental corpus updating mechanism. Evaluations on real-world urban tasks of a variety of complexity verify the effectiveness of the proposed framework. This work presents a promising step toward the realization of general-purpose LLM agents in future urban environments. 

**Abstract (ZH)**: 城市通用智能（UGI）是指AI系统在动态复杂的城市环境中自主感知、推理和行动的能力。本文介绍了UrbanMind，一种工具增强的检索增强生成（RAG）框架，旨在促进UGI的实现。UrbanMind的核心是一种新颖的基于连续检索增强MoE基大语言模型（C-RAG-LLM）的架构，该架构能够动态地纳入领域特定知识和不断变化的城市数据，以支持长期适应性。C-RAG-LLM架构自然地与多级优化框架相契合，其中不同层次被视为相互依赖的子问题。每一层都有其独特的目标，并可以通过分层学习过程独立或联合进行优化。该框架具有高度灵活性，可以根据资源或部署约束支持端到端训练和部分层次优化。为保持在数据漂移下的适用性，进一步集成了增量语料库更新机制。对复杂程度各异的现实城市任务的评估验证了所提出框架的有效性。本文展示了向未来城市环境中实现通用大语言模型代理的重要一步。 

---
# Tempo-R0: A Video-MLLM for Temporal Video Grounding through Efficient Temporal Sensing Reinforcement Learning 

**Title (ZH)**: Tempo-R0：一种通过高效时间感知强化学习进行-temporal视频约束的视频-MLLM 

**Authors**: Feng Yue, Zhaoxing Zhang, Junming Jiao, Zhengyu Liang, Shiwen Cao, Feifei Zhang, Rong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.04702)  

**Abstract**: Temporal Video Grounding (TVG), which requires pinpointing relevant temporal segments from video based on language query, has always been a highly challenging task in the field of video understanding. Videos often have a larger volume of information and redundancy than texts or images. Models should present comprehensive understanding of the whole video to accurately retrieve query-relevant clips. We thus propose Tempo-R0: a Video Multimodal Large Language Model (Video-MLLM) for the temporal video grounding task via multimodal temporal sensing reinforcement. Specifically, during the preprocessing stage of our pipeline, we employ Self-adaptive Attention Allocation (SAA) method based on frame content variation to efficiently use the MLLM's limited attention. The Explicit Timestamp-modal Aligned (ETA) method is also utilized to strengthen our model's capability to perceive the boundaries of events in the video. In the fine-tuning part of our pipeline, we creatively apply Partial Irrelevance Refusing-based Group Relative Policy Optimization (PIR-GRPO) in TVG area to foster model's temporal reasoning from not only accepting relevant video-query pairs but also refusing irrelevant ones. Experiments demonstrate that our method accomplishes a notable advantage over SOTA solutions by around 3.5% on both the original QVHighlights testbench and its corrected version with more reasonable ground truth annotations. 

**Abstract (ZH)**: 基于语言查询的视频时序定位（Temporal Video Grounding, TVG）：多模态时空感知强化的视频大规模语言模型（Video-MLLM） 

---
# Hierarchical Intent-guided Optimization with Pluggable LLM-Driven Semantics for Session-based Recommendation 

**Title (ZH)**: 基于层次化意图引导的可插拔LLM驱动语义会话推荐优化 

**Authors**: Jinpeng Chen, Jianxiang He, Huan Li, Senzhang Wang, Yuan Cao, Kaimin Wei, Zhenye Yang, Ye Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.04623)  

**Abstract**: Session-based Recommendation (SBR) aims to predict the next item a user will likely engage with, using their interaction sequence within an anonymous session. Existing SBR models often focus only on single-session information, ignoring inter-session relationships and valuable cross-session insights. Some methods try to include inter-session data but struggle with noise and irrelevant information, reducing performance. Additionally, most models rely on item ID co-occurrence and overlook rich semantic details, limiting their ability to capture fine-grained item features. To address these challenges, we propose a novel hierarchical intent-guided optimization approach with pluggable LLM-driven semantic learning for session-based recommendations, called HIPHOP. First, we introduce a pluggable embedding module based on large language models (LLMs) to generate high-quality semantic representations, enhancing item embeddings. Second, HIPHOP utilizes graph neural networks (GNNs) to model item transition relationships and incorporates a dynamic multi-intent capturing module to address users' diverse interests within a session. Additionally, we design a hierarchical inter-session similarity learning module, guided by user intent, to capture global and local session relationships, effectively exploring users' long-term and short-term interests. To mitigate noise, an intent-guided denoising strategy is applied during inter-session learning. Finally, we enhance the model's discriminative capability by using contrastive learning to optimize session representations. Experiments on multiple datasets show that HIPHOP significantly outperforms existing methods, demonstrating its effectiveness in improving recommendation quality. Our code is available: this https URL. 

**Abstract (ZH)**: 基于会话的推荐（SBR）旨在使用用户在匿名会话内的交互序列来预测用户可能 engagement 的下一个项目。现有的SBR模型通常仅关注单会话信息，忽视了会话间的关系和跨会话有价值的见解。一些方法尝试包含会话间数据，但难以处理噪音和无关信息，影响了性能。此外，大多数模型依赖于物品ID共现，忽略了丰富的语义细节，限制了它们捕捉细粒度物品特征的能力。为了解决这些挑战，我们提出了一种名为HIPHOP的新颖层次意图引导优化方法，该方法结合可插拔的大语言模型驱动的语义学习，用于基于会话的推荐。首先，我们基于大规模语言模型引入了一个可插拔的嵌入模块，生成高质量的语义表示，增强物品嵌入。其次，HIPHOP利用图神经网络（GNNs）建模项目转换关系，并结合一个动态多意图捕捉模块以应对会话内用户 varied 的兴趣。此外，我们设计了一个层次化的会话间相似性学习模块，受用户意图引导，以捕捉全局和局部会话关系，有效探索用户长期和短期兴趣。为了减轻噪音，在会话间学习期间应用意图引导的去噪策略。最后，通过对比学习增强模型的辨别能力，优化会话表示。在多个数据集上的实验结果显示，HIPHOP显著优于现有方法，证明了其在提高推荐质量方面的有效性。我们的代码可在以下链接获取：this https URL。 

---
# Multimodal LLM Integrated Semantic Communications for 6G Immersive Experiences 

**Title (ZH)**: 多模态LLM集成语义通信以实现6G沉浸式体验 

**Authors**: Yusong Zhang, Yuxuan Sun, Lei Guo, Wei Chen, Bo Ai, Deniz Gunduz  

**Link**: [PDF](https://arxiv.org/pdf/2507.04621)  

**Abstract**: 6G networks promise revolutionary immersive communication experiences including augmented reality (AR), virtual reality (VR), and holographic communications. These applications demand high-dimensional multimodal data transmission and intelligent data processing in real-time, which is extremely challenging over resource-limited wireless communication systems. Moreover, a joint understanding of the environment, context, and user intent is essential to deliver task-relevant content effectively. This article presents a novel multimodal large language model (MLLM) integrated semantic communications framework, termed MLLM-SC, which fully leverages reasoning and generative capabilities of pre-trained foundation models for context-aware and task-oriented wireless communication. The MLLM-SC framework adopts a device-edge collaborative architecture. At the edge, MLLM-empowered semantic guidance module analyzes multimodal inputs, user intents, and channel conditions to generate importance-aware attention maps prioritizing semantically critical information. An importance-aware semantic encoder and a resource-adaptive semantic decoder are jointly designed and optimized, which can utilize the semantic guidance for adaptive bandwidth allocation and high-quality content reconstruction or generation. Extensive case studies on visual question answering for AR/VR applications and diffusion-driven image generation validate the effectiveness of MLLM-SC. 

**Abstract (ZH)**: 6G网络承诺提供革命性的沉浸式通信体验，包括增强现实（AR）、虚拟现实（VR）和全息通信。这些应用要求在资源有限的无线通信系统中进行实时的高维多模态数据传输和智能数据处理，这极具挑战性。此外，理解和联合环境、上下文以及用户意图对于有效提供任务相关的内容至关重要。本文提出了一种新颖的多模态大型语言模型（MLLM）集成语义通信框架，称为MLLM-SC，该框架充分利用预训练基础模型的推理和生成能力，实现具有上下文感知和任务导向的无线通信。MLLM-SC框架采用设备-边缘协作架构。在边缘处，MLLM赋能的语义指导模块分析多模态输入、用户意图和信道条件，生成注重语义关键信息的重要性感知注意力图。重要性感知语义编码器和资源自适应语义解码器被联合设计和优化，能够利用语义指导进行自适应带宽分配和高质量内容的重建或生成。广泛应用于AR/VR应用的视觉问答和扩散驱动图像生成的案例研究验证了MLLM-SC的有效性。 

---
# Pedestrian Intention Prediction via Vision-Language Foundation Models 

**Title (ZH)**: 基于视觉-语言基础模型的行人意图预测 

**Authors**: Mohsen Azarmi, Mahdi Rezaei, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04141)  

**Abstract**: Prediction of pedestrian crossing intention is a critical function in autonomous vehicles. Conventional vision-based methods of crossing intention prediction often struggle with generalizability, context understanding, and causal reasoning. This study explores the potential of vision-language foundation models (VLFMs) for predicting pedestrian crossing intentions by integrating multimodal data through hierarchical prompt templates. The methodology incorporates contextual information, including visual frames, physical cues observations, and ego-vehicle dynamics, into systematically refined prompts to guide VLFMs effectively in intention prediction. Experiments were conducted on three common datasets-JAAD, PIE, and FU-PIP. Results demonstrate that incorporating vehicle speed, its variations over time, and time-conscious prompts significantly enhances the prediction accuracy up to 19.8%. Additionally, optimised prompts generated via an automatic prompt engineering framework yielded 12.5% further accuracy gains. These findings highlight the superior performance of VLFMs compared to conventional vision-based models, offering enhanced generalisation and contextual understanding for autonomous driving applications. 

**Abstract (ZH)**: 基于视觉-语言基础模型的多模态人行横道穿越意图预测 

---
# Accurate and Efficient World Modeling with Masked Latent Transformers 

**Title (ZH)**: 掩码潜在变换器实现高效准确的世界建模 

**Authors**: Maxime Burchi, Radu Timofte  

**Link**: [PDF](https://arxiv.org/pdf/2507.04075)  

**Abstract**: The Dreamer algorithm has recently obtained remarkable performance across diverse environment domains by training powerful agents with simulated trajectories. However, the compressed nature of its world model's latent space can result in the loss of crucial information, negatively affecting the agent's performance. Recent approaches, such as $\Delta$-IRIS and DIAMOND, address this limitation by training more accurate world models. However, these methods require training agents directly from pixels, which reduces training efficiency and prevents the agent from benefiting from the inner representations learned by the world model. In this work, we propose an alternative approach to world modeling that is both accurate and efficient. We introduce EMERALD (Efficient MaskEd latent tRAnsformer worLD model), a world model using a spatial latent state with MaskGIT predictions to generate accurate trajectories in latent space and improve the agent performance. On the Crafter benchmark, EMERALD achieves new state-of-the-art performance, becoming the first method to surpass human experts performance within 10M environment steps. Our method also succeeds to unlock all 22 Crafter achievements at least once during evaluation. 

**Abstract (ZH)**: EMERALD：高效掩码潜在空间转换世界模型 

---
# RLVER: Reinforcement Learning with Verifiable Emotion Rewards for Empathetic Agents 

**Title (ZH)**: RLVER：可验证情绪reward的强化学习方法用于 empathy代理 

**Authors**: Peisong Wang, Ruotian Ma, Bang Zhang, Xingyu Chen, Zhiwei He, Kang Luo, Qingsong Lv, Qingxuan Jiang, Zheng Xie, Shanyi Wang, Yuan Li, Fanghua Ye, Jian Li, Yifan Yang, Zhaopeng Tu, Xiaolong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.03112)  

**Abstract**: Large language models (LLMs) excel at logical and algorithmic reasoning, yet their emotional intelligence (EQ) still lags far behind their cognitive prowess. While reinforcement learning from verifiable rewards (RLVR) has advanced in other domains, its application to dialogue-especially for emotional intelligence-remains underexplored. In this work, we introduce RLVER, the first end-to-end reinforcement learning framework that leverages verifiable emotion rewards from simulated users to cultivate higher-order empathetic abilities in LLMs. Within this framework, self-consistent affective simulated users engage in dialogue rollouts and produce deterministic emotion scores during conversations, serving as reward signals to guide the LLM's learning. Fine-tuning publicly available Qwen2.5-7B-Instruct model with PPO boosts its Sentient-Benchmark score from 13.3 to 79.2 while largely preserving mathematical and coding competence. Extensive experiments reveal that: (i) RLVER consistently improves multiple dialogue capabilities; (ii) Thinking and non-thinking models show distinct trends--thinking models excel in empathy and insight, while non-thinking models favor action; (iii) GRPO often yields stable gains, while PPO can push certain capabilities to a higher ceiling; (iv) More challenging environments are not always better-moderate ones can yield stronger outcomes. Our results show that RLVER is a practical route toward emotionally intelligent and broadly capable language agents. 

**Abstract (ZH)**: 大型语言模型在逻辑和算法推理方面表现卓越，但在情感 intelligence（EQ）方面仍远远落后于其认知能力。尽管可验证奖励强化学习（RLVR）在其他领域取得了进展，但在对话中特别是情感 intelligence 方面的应用仍待探索。在这项工作中，我们引入了 RLVER，这是第一个利用模拟用户的情感验证奖励来培养大型语言模型高阶同理能力的端到端强化学习框架。在此框架中，自我一致的情感模拟用户参与对话展开，并在会话中生成确定性的情感得分，作为奖励信号引导大型语言模型的学习。使用 PPO 算法 fine-tune 公开可用的 Qwen2.5-7B-Instruct 模型，使其 Sentient-Benchmark 得分从 13.3 提升到 79.2，同时主要保留了数学和编程能力。广泛的实验表明：（i）RLVER 一致地提高多种对话能力；（ii）思考模型和非思考模型表现出不同的趋势——思考模型在同理心和洞察力方面表现出色，而非思考模型则更侧重于行动；（iii）GRPO 经常产生稳定的收益，而 PPO 可以推动某些能力达到更高的上限；（iv）更具挑战性的环境并不总是更好的选择——适度的环境可能产生更强的效果。我们的结果表明，RLVER 是实现具备情感 intelligence 和广泛能力的语言代理的现实途径。 

---
