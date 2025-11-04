# GenDexHand: Generative Simulation for Dexterous Hands 

**Title (ZH)**: GenDexHand: 生成模拟灵巧手 

**Authors**: Feng Chen, Zhuxiu Xu, Tianzhe Chu, Xunzhe Zhou, Li Sun, Zewen Wu, Shenghua Gao, Zhongyu Li, Yanchao Yang, Yi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.01791)  

**Abstract**: Data scarcity remains a fundamental bottleneck for embodied intelligence. Existing approaches use large language models (LLMs) to automate gripper-based simulation generation, but they transfer poorly to dexterous manipulation, which demands more specialized environment design. Meanwhile, dexterous manipulation tasks are inherently more difficult due to their higher degrees of freedom. Massively generating feasible and trainable dexterous hand tasks remains an open challenge. To this end, we present GenDexHand, a generative simulation pipeline that autonomously produces diverse robotic tasks and environments for dexterous manipulation. GenDexHand introduces a closed-loop refinement process that adjusts object placements and scales based on vision-language model (VLM) feedback, substantially improving the average quality of generated environments. Each task is further decomposed into sub-tasks to enable sequential reinforcement learning, reducing training time and increasing success rates. Our work provides a viable path toward scalable training of diverse dexterous hand behaviors in embodied intelligence by offering a simulation-based solution to synthetic data generation. Our website: this https URL. 

**Abstract (ZH)**: 数据稀缺仍然是实体智能的基本瓶颈。现有方法使用大型语言模型（LLMs）自动化生成基于机械臂的仿真，但这些方法在对需求更多专业化环境设计的灵巧操作任务上表现不佳。同时，由于具备更高的自由度，灵巧操作任务本身更加困难。大规模生成可行且可训练的灵巧手任务仍然是一个开放挑战。为此，我们提出GenDexHand，一种自动生成用于灵巧操作的多样化机器人任务和环境的生成仿真流水线。GenDexHand引入了一个基于视觉语言模型（VLM）反馈的闭环细化过程，显著提高了生成环境的平均质量。每个任务进一步分解为子任务，以实现序列强化学习，从而减少训练时间和提高成功率。我们的工作通过提供基于仿真的合成数据生成解决方案，为实体智能中多样化灵巧手行为的大规模训练提供了可行路径。我们的网站：[this https URL]。 

---
# MOBIUS: A Multi-Modal Bipedal Robot that can Walk, Crawl, Climb, and Roll 

**Title (ZH)**: MOBIUS：一种可以行走、爬行、攀爬和滚动的多模态双足机器人 

**Authors**: Alexander Schperberg, Yusuke Tanaka, Stefano Di Cairano, Dennis Hong  

**Link**: [PDF](https://arxiv.org/pdf/2511.01774)  

**Abstract**: This article presents a Multi-Modal Bipedal Intelligent Urban Scout robot (MOBIUS) capable of walking, crawling, climbing, and rolling. MOBIUS features four limbs--two 6-DoF arms with two-finger grippers for manipulation and climbing, and two 4-DoF legs for locomotion--enabling smooth transitions across diverse terrains without reconfiguration. A hybrid control architecture combines reinforcement learning-based locomotion with model-based predictive and admittance control enhanced for safety by a Reference Governor toward compliant contact interactions. A high-level MIQCP planner autonomously selects locomotion modes to balance stability and energy efficiency. Hardware experiments demonstrate robust gait transitions, dynamic climbing, and full-body load support via pinch grasp. Overall, MOBIUS demonstrates the importance of tight integration between morphology, high-level planning, and control to enable mobile loco-manipulation and grasping, substantially expanding its interaction capabilities, workspace, and traversability. 

**Abstract (ZH)**: 一种多模态双足智能城市侦察机器人（MOBIUS）：具备行走、爬行、攀登和滚动能力的多模态双足智能城市侦察机器人 

---
# Lightweight Learning from Actuation-Space Demonstrations via Flow Matching for Whole-Body Soft Robotic Grasping 

**Title (ZH)**: 基于流匹配的整体软体机器人抓取动作空间示范轻量学习 

**Authors**: Liudi Yang, Yang Bai, Yuhao Wang, Ibrahim Alsarraj, Gitta Kutyniok, Zhanchi Wang, Ke Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01770)  

**Abstract**: Robotic grasping under uncertainty remains a fundamental challenge due to its uncertain and contact-rich nature. Traditional rigid robotic hands, with limited degrees of freedom and compliance, rely on complex model-based and heavy feedback controllers to manage such interactions. Soft robots, by contrast, exhibit embodied mechanical intelligence: their underactuated structures and passive flexibility of their whole body, naturally accommodate uncertain contacts and enable adaptive behaviors. To harness this capability, we propose a lightweight actuation-space learning framework that infers distributional control representations for whole-body soft robotic grasping, directly from deterministic demonstrations using a flow matching model (Rectified Flow),without requiring dense sensing or heavy control loops. Using only 30 demonstrations (less than 8% of the reachable workspace), the learned policy achieves a 97.5% grasp success rate across the whole workspace, generalizes to grasped-object size variations of +-33%, and maintains stable performance when the robot's dynamic response is directly adjusted by scaling the execution time from 20% to 200%. These results demonstrate that actuation-space learning, by leveraging its passive redundant DOFs and flexibility, converts the body's mechanics into functional control intelligence and substantially reduces the burden on central controllers for this uncertain-rich task. 

**Abstract (ZH)**: 软体机器人在不确定性下的抓取仍是一项基本挑战：基于体内部份机械智能的轻量级驱动空间学习框架 

---
# Unified Diffusion VLA: Vision-Language-Action Model via Joint Discrete Denoising Diffusion Process 

**Title (ZH)**: 统一扩散VLA模型：通过联合离散去噪扩散过程实现的多模态动作模型 

**Authors**: Jiayi Chen, Wenxuan Song, Pengxiang Ding, Ziyang Zhou, Han Zhao, Feilong Tang, Donglin Wang, Haoang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.01718)  

**Abstract**: Vision-language-action (VLA) models aim to understand natural language instructions and visual observations and to execute corresponding actions as an embodied agent. Recent work integrates future images into the understanding-acting loop, yielding unified VLAs that jointly understand, generate, and act -- reading text and images and producing future images and actions. However, these models either rely on external experts for modality unification or treat image generation and action prediction as separate processes, limiting the benefits of direct synergy between these tasks. Our core philosophy is to optimize generation and action jointly through a synchronous denoising process, where the iterative refinement enables actions to evolve from initialization, under constant and sufficient visual guidance. We ground this philosophy in our proposed Unified Diffusion VLA and Joint Discrete Denoising Diffusion Process (JD3P), which is a joint diffusion process that integrates multiple modalities into a single denoising trajectory to serve as the key mechanism enabling understanding, generation, and acting to be intrinsically synergistic. Our model and theory are built on a unified tokenized space of all modalities and a hybrid attention mechanism. We further propose a two-stage training pipeline and several inference-time techniques that optimize performance and efficiency. Our approach achieves state-of-the-art performance on benchmarks such as CALVIN, LIBERO, and SimplerEnv with 4$\times$ faster inference than autoregressive methods, and we demonstrate its effectiveness through in-depth analysis and real-world evaluations. Our project page is available at this https URL. 

**Abstract (ZH)**: vision-language-action (VLA) 模型旨在理解自然语言指令和视觉观察，并作为具身代理执行相应动作。近期的研究将未来图像整合进理解-执行循环中，产生了能够联合理解、生成和执行的统一 VLA——阅读文本和图像并生成未来图像和动作。然而，这些模型要么依赖外部专家进行模态统一，要么将图像生成和动作预测视为独立过程，限制了这些任务直接协同效应的发挥。我们的核心理念是通过同步去噪过程优化生成和动作，迭代精炼使动作在持续且充足的视觉引导下从初始化演变。我们基于提出的 Unified Diffusion VLA 和 Joint Discrete Denoising Diffusion Process (JD3P) 实现了这一理念，JD3P 是一种联合去噪过程，将多种模态整合到单一去噪轨迹中，作为使理解、生成和执行内在协同的关键机制。我们的模型和理论建立在一个统一的多模态标记空间以及一种混合注意力机制之上。我们还提出了两阶段训练管道和几种推理时的技术，以优化性能和效率。我们的方法在 CALVIN、LIBERO 和 SimplerEnv 等基准测试中达到最先进的性能，推理速度比自回归方法快 4 倍，并通过深入分析和实际评测证明了其有效性。我们的项目页面可访问此处。 

---
# MARS: Multi-Agent Robotic System with Multimodal Large Language Models for Assistive Intelligence 

**Title (ZH)**: 多模态大型语言模型驱动的多Agent机器人系统辅助智能 

**Authors**: Renjun Gao, Peiyan Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2511.01594)  

**Abstract**: Multimodal large language models (MLLMs) have shown remarkable capabilities in cross-modal understanding and reasoning, offering new opportunities for intelligent assistive systems, yet existing systems still struggle with risk-aware planning, user personalization, and grounding language plans into executable skills in cluttered homes. We introduce MARS - a Multi-Agent Robotic System powered by MLLMs for assistive intelligence and designed for smart home robots supporting people with disabilities. The system integrates four agents: a visual perception agent for extracting semantic and spatial features from environment images, a risk assessment agent for identifying and prioritizing hazards, a planning agent for generating executable action sequences, and an evaluation agent for iterative optimization. By combining multimodal perception with hierarchical multi-agent decision-making, the framework enables adaptive, risk-aware, and personalized assistance in dynamic indoor environments. Experiments on multiple datasets demonstrate the superior overall performance of the proposed system in risk-aware planning and coordinated multi-agent execution compared with state-of-the-art multimodal models. The proposed approach also highlights the potential of collaborative AI for practical assistive scenarios and provides a generalizable methodology for deploying MLLM-enabled multi-agent systems in real-world environments. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在跨模态理解和推理方面展现了显著的能力，为智能辅助系统提供了新的机遇，但现有系统仍难以应对风险感知规划、用户个性化以及在杂乱居家环境中将语言计划转化为可执行技能的挑战。我们介绍了MARS——一种由MLLMs驱动的多Agent机器人系统，旨在为支持残疾人的智能家用机器人提供辅助智能。该系统整合了四个代理：视觉感知代理，用于从环境图像中提取语义和空间特征；风险评估代理，用于识别和优先处理风险；规划代理，用于生成可执行的动作序列；评估代理，用于迭代优化。通过结合多模态感知与分层多Agent决策，该框架能够在动态室内环境中提供适配性强、风险感知和个性化的辅助。实验在多个数据集上展示了与最先进的多模态模型相比，所提出系统在风险感知规划和协调多Agent执行方面的优越整体性能。所提出的方法还强调了协作AI在实际辅助场景中的潜力，并提供了一种在现实环境中部署基于MLLM的多Agent系统的普遍方法。 

---
# Floor Plan-Guided Visual Navigation Incorporating Depth and Directional Cues 

**Title (ZH)**: 基于楼层平面图的视觉导航：结合深度和方向线索 

**Authors**: Wei Huang, Jiaxin Li, Zang Wan, Huijun Di, Wei Liang, Zhu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01493)  

**Abstract**: Guiding an agent to a specific target in indoor environments based solely on RGB inputs and a floor plan is a promising yet challenging problem. Although existing methods have made significant progress, two challenges remain unresolved. First, the modality gap between egocentric RGB observations and the floor plan hinders the integration of visual and spatial information for both local obstacle avoidance and global planning. Second, accurate localization is critical for navigation performance, but remains challenging at deployment in unseen environments due to the lack of explicit geometric alignment between RGB inputs and floor plans. We propose a novel diffusion-based policy, denoted as GlocDiff, which integrates global path planning from the floor plan with local depth-aware features derived from RGB observations. The floor plan offers explicit global guidance, while the depth features provide implicit geometric cues, collectively enabling precise prediction of optimal navigation directions and robust obstacle avoidance. Moreover, GlocDiff introduces noise perturbation during training to enhance robustness against pose estimation errors, and we find that combining this with a relatively stable VO module during inference results in significantly improved navigation performance. Extensive experiments on the FloNa benchmark demonstrate GlocDiff's efficiency and effectiveness in achieving superior navigation performance, and the success of real-world deployments also highlights its potential for widespread practical applications. 

**Abstract (ZH)**: 基于RGB输入和楼层平面图引导代理在室内环境中导航：一种新的扩散策略的研究 

---
# AERMANI-VLM: Structured Prompting and Reasoning for Aerial Manipulation with Vision Language Models 

**Title (ZH)**: AERMANI-VLM：基于视觉语言模型的结构化提示与推理在空中 manipulation 中的应用 

**Authors**: Sarthak Mishra, Rishabh Dev Yadav, Avirup Das, Saksham Gupta, Wei Pan, Spandan Roy  

**Link**: [PDF](https://arxiv.org/pdf/2511.01472)  

**Abstract**: The rapid progress of vision--language models (VLMs) has sparked growing interest in robotic control, where natural language can express the operation goals while visual feedback links perception to action. However, directly deploying VLM-driven policies on aerial manipulators remains unsafe and unreliable since the generated actions are often inconsistent, hallucination-prone, and dynamically infeasible for flight. In this work, we present AERMANI-VLM, the first framework to adapt pretrained VLMs for aerial manipulation by separating high-level reasoning from low-level control, without any task-specific fine-tuning. Our framework encodes natural language instructions, task context, and safety constraints into a structured prompt that guides the model to generate a step-by-step reasoning trace in natural language. This reasoning output is used to select from a predefined library of discrete, flight-safe skills, ensuring interpretable and temporally consistent execution. By decoupling symbolic reasoning from physical action, AERMANI-VLM mitigates hallucinated commands and prevents unsafe behavior, enabling robust task completion. We validate the framework in both simulation and hardware on diverse multi-step pick-and-place tasks, demonstrating strong generalization to previously unseen commands, objects, and environments. 

**Abstract (ZH)**: 视觉-语言模型(VLMs)的迅速进展激发了机器人控制领域日益增长的兴趣，其中自然语言可以表达操作目标，视觉反馈则将感知与行动联系起来。然而，直接在空中 manipulator 上部署由 VLM 驱动的策略仍然存在安全隐患且不可靠，因为生成的动作往往不一致、幻觉性强且在飞行中难以实现。在本文中，我们提出 AERMANI-VLM，这是首个通过将高层推理与低层控制分离来适应预训练 VLMs 的框架，无需任何任务特定的微调。该框架将自然语言指令、任务上下文和安全约束编码为结构化的提示，引导模型生成自然语言形式的逐步推理痕迹。该推理输出用于从预定义的、飞行安全的离散技能库中选择技能，确保可解释性和时间一致性执行。通过将符号推理与物理动作解耦，AERMANI-VLM 减少了幻觉命令并防止了不安全行为，从而使任务完成变得稳健。我们在仿真和硬件上针对多样化的多步骤拾放任务验证了该框架，展示了其在未见过的命令、物体和环境中的强大泛化能力。 

---
# FoldPath: End-to-End Object-Centric Motion Generation via Modulated Implicit Paths 

**Title (ZH)**: FoldPath: 基于调制隐式路径的端到端物体中心运动生成 

**Authors**: Paolo Rabino, Gabriele Tiboni, Tatiana Tommasi  

**Link**: [PDF](https://arxiv.org/pdf/2511.01407)  

**Abstract**: Object-Centric Motion Generation (OCMG) is instrumental in advancing automated manufacturing processes, particularly in domains requiring high-precision expert robotic motions, such as spray painting and welding. To realize effective automation, robust algorithms are essential for generating extended, object-aware trajectories across intricate 3D geometries. However, contemporary OCMG techniques are either based on ad-hoc heuristics or employ learning-based pipelines that are still reliant on sensitive post-processing steps to generate executable paths. We introduce FoldPath, a novel, end-to-end, neural field based method for OCMG. Unlike prior deep learning approaches that predict discrete sequences of end-effector waypoints, FoldPath learns the robot motion as a continuous function, thus implicitly encoding smooth output paths. This paradigm shift eliminates the need for brittle post-processing steps that concatenate and order the predicted discrete waypoints. Particularly, our approach demonstrates superior predictive performance compared to recently proposed learning-based methods, and attains generalization capabilities even in real industrial settings, where only a limited amount of 70 expert samples are provided. We validate FoldPath through comprehensive experiments in a realistic simulation environment and introduce new, rigorous metrics designed to comprehensively evaluate long-horizon robotic paths, thus advancing the OCMG task towards practical maturity. 

**Abstract (ZH)**: 基于对象的运动生成（OCMG）对于推进自动化制造过程至关重要，特别是在需要高精度专家级机器人动作的领域，如喷漆和焊接。为了实现有效的自动化，生成跨复杂3D几何结构的对象感知长期轨迹的稳健算法是必不可少的。然而，当前的OCMG技术要么基于随意的启发式方法，要么采用的学习管道仍然依赖于敏感的后处理步骤以生成可执行路径。我们提出了一种新颖的端到端神经场方法FoldPath，用于OCMG。与先前的深度学习方法预测离散末端执行器航点的序列不同，FoldPath 学习机器人的运动作为连续函数，从而隐式编码平滑的输出路径。这一范式的转变消除了拼接和排序预测离散航点的脆弱后处理步骤的需要。特别是，我们的方法在预测高性能方面优于近期提出的基于学习的方法，并且即使在仅提供有限数量（70个）专家样本的实际工业环境中也能实现泛化能力。我们通过在现实的仿真环境中进行全面的实验并引入新的严格评估指标来验证FoldPath，从而推动OCMG任务向实用成熟迈进。 

---
# Embodied Cognition Augmented End2End Autonomous Driving 

**Title (ZH)**: 具身认知增强端到端自动驾驶 

**Authors**: Ling Niu, Xiaoji Zheng, Han Wang, Chen Zheng, Ziyuan Yang, Bokui Chen, Jiangtao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2511.01334)  

**Abstract**: In recent years, vision-based end-to-end autonomous driving has emerged as a new paradigm. However, popular end-to-end approaches typically rely on visual feature extraction networks trained under label supervision. This limited supervision framework restricts the generality and applicability of driving models. In this paper, we propose a novel paradigm termed $E^{3}AD$, which advocates for comparative learning between visual feature extraction networks and the general EEG large model, in order to learn latent human driving cognition for enhancing end-to-end planning. In this work, we collected a cognitive dataset for the mentioned contrastive learning process. Subsequently, we investigated the methods and potential mechanisms for enhancing end-to-end planning with human driving cognition, using popular driving models as baselines on publicly available autonomous driving datasets. Both open-loop and closed-loop tests are conducted for a comprehensive evaluation of planning performance. Experimental results demonstrate that the $E^{3}AD$ paradigm significantly enhances the end-to-end planning performance of baseline models. Ablation studies further validate the contribution of driving cognition and the effectiveness of comparative learning process. To the best of our knowledge, this is the first work to integrate human driving cognition for improving end-to-end autonomous driving planning. It represents an initial attempt to incorporate embodied cognitive data into end-to-end autonomous driving, providing valuable insights for future brain-inspired autonomous driving systems. Our code will be made available at Github 

**Abstract (ZH)**: 基于视觉的端到端自主驾驶新兴范式：E³AD框架 

---
# RobustVLA: Robustness-Aware Reinforcement Post-Training for Vision-Language-Action Models 

**Title (ZH)**: RobustVLA：面向鲁棒性的视觉-语言-行动模型后训练强化方法 

**Authors**: Hongyin Zhang, Shuo Zhang, Junxi Jin, Qixin Zeng, Runze Li, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01331)  

**Abstract**: Vision-Language-Action (VLA) models have recently emerged as powerful general-purpose policies for robotic manipulation, benefiting from large-scale multi-modal pre-training. However, they often fail to generalize reliably in out-of-distribution deployments, where unavoidable disturbances such as observation noise, sensor errors, or actuation perturbations become prevalent. While recent Reinforcement Learning (RL)-based post-training provides a practical means to adapt pre-trained VLA models, existing methods mainly emphasize reward maximization and overlook robustness to environmental uncertainty. In this work, we introduce RobustVLA, a lightweight online RL post-training method designed to explicitly enhance the resilience of VLA models. Through a systematic robustness analysis, we identify two key regularizations: Jacobian regularization, which mitigates sensitivity to observation noise, and smoothness regularization, which stabilizes policies under action perturbations. Extensive experiments across diverse robotic environments demonstrate that RobustVLA significantly outperforms prior state-of-the-art methods in robustness and reliability. Our results highlight the importance of principled robustness-aware RL post-training as a key step toward improving the reliability and robustness of VLA models. 

**Abstract (ZH)**: RobustVLA：一种用于增强Vision-Language-Action模型鲁棒性的轻量级在线RL后训练方法 

---
# Kinematify: Open-Vocabulary Synthesis of High-DoF Articulated Objects 

**Title (ZH)**: Kinematify: 开放词汇高自由度articulated对象合成 

**Authors**: Jiawei Wang, Dingyou Wang, Jiaming Hu, Qixuan Zhang, Jingyi Yu, Lan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01294)  

**Abstract**: A deep understanding of kinematic structures and movable components is essential for enabling robots to manipulate objects and model their own articulated forms. Such understanding is captured through articulated objects, which are essential for tasks such as physical simulation, motion planning, and policy learning. However, creating these models, particularly for complex systems like robots or objects with high degrees of freedom (DoF), remains a significant challenge. Existing methods typically rely on motion sequences or strong assumptions from hand-curated datasets, which hinders scalability. In this paper, we introduce Kinematify, an automated framework that synthesizes articulated objects directly from arbitrary RGB images or text prompts. Our method addresses two core challenges: (i) inferring kinematic topologies for high-DoF objects and (ii) estimating joint parameters from static geometry. To achieve this, we combine MCTS search for structural inference with geometry-driven optimization for joint reasoning, producing physically consistent and functionally valid descriptions. We evaluate Kinematify on diverse inputs from both synthetic and real-world environments, demonstrating improvements in registration and kinematic topology accuracy over prior work. 

**Abstract (ZH)**: 深入理解运动结构和可动组件对于使机器人能够操作物体并建模其自身 articulated 形态至关重要。这种理解通过articulated对象来捕捉，这些对象对于物理仿真、运动规划和策略学习等任务至关重要。然而，特别是对于机器人或具有高自由度（DoF）的复杂系统，创建这些模型仍然是一个重大挑战。现有方法通常依赖于运动序列或来自手工制作数据集的强假设，这限制了其扩展性。在这篇论文中，我们介绍了Kinematify，这是一个自动框架，可以直接从任意的RGB图像或文本提示中合成articulated对象。我们的方法解决了两个核心挑战：（i）推断高自由度对象的运动学拓扑结构，（ii）从静态几何结构估计关节参数。为了实现这一目标，我们将基于MCTS的结构推理与基于几何的优化相结合，进行关节推理，生成物理上一致且功能有效的描述。我们在来自合成和真实环境的多种输入上评估了Kinematify，表明与先前工作相比，在注册和运动学拓扑结构准确性方面取得了改进。 

---
# Contact Map Transfer with Conditional Diffusion Model for Generalizable Dexterous Grasp Generation 

**Title (ZH)**: 基于条件扩散模型的手部接触图转移以生成泛化 Dexterous 抓取 

**Authors**: Yiyao Ma, Kai Chen, Kexin Zheng, Qi Dou  

**Link**: [PDF](https://arxiv.org/pdf/2511.01276)  

**Abstract**: Dexterous grasp generation is a fundamental challenge in robotics, requiring both grasp stability and adaptability across diverse objects and tasks. Analytical methods ensure stable grasps but are inefficient and lack task adaptability, while generative approaches improve efficiency and task integration but generalize poorly to unseen objects and tasks due to data limitations. In this paper, we propose a transfer-based framework for dexterous grasp generation, leveraging a conditional diffusion model to transfer high-quality grasps from shape templates to novel objects within the same category. Specifically, we reformulate the grasp transfer problem as the generation of an object contact map, incorporating object shape similarity and task specifications into the diffusion process. To handle complex shape variations, we introduce a dual mapping mechanism, capturing intricate geometric relationship between shape templates and novel objects. Beyond the contact map, we derive two additional object-centric maps, the part map and direction map, to encode finer contact details for more stable grasps. We then develop a cascaded conditional diffusion model framework to jointly transfer these three maps, ensuring their intra-consistency. Finally, we introduce a robust grasp recovery mechanism, identifying reliable contact points and optimizing grasp configurations efficiently. Extensive experiments demonstrate the superiority of our proposed method. Our approach effectively balances grasp quality, generation efficiency, and generalization performance across various tasks. Project homepage: this https URL 

**Abstract (ZH)**: 基于迁移的灵巧抓取生成：通过条件扩散模型在类别内从形状模板转移高质量抓取 

---
# Don't Just Search, Understand: Semantic Path Planning Agent for Spherical Tensegrity Robots in Unknown Environments 

**Title (ZH)**: 不只是搜索，还要理解：未知环境中文紧张柔性机器人语义路径规划代理 

**Authors**: Junwen Zhang, Changyue Liu, Pengqi Fu, Xiang Guo, Ye Shi, Xudong Liang, Zhijian Wang, Hanzhi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.01236)  

**Abstract**: Endowed with inherent dynamical properties that grant them remarkable ruggedness and adaptability, spherical tensegrity robots stand as prototypical examples of hybrid softrigid designs and excellent mobile platforms. However, path planning for these robots in unknown environments presents a significant challenge, requiring a delicate balance between efficient exploration and robust planning. Traditional path planners, which treat the environment as a geometric grid, often suffer from redundant searches and are prone to failure in complex scenarios due to their lack of semantic understanding. To overcome these limitations, we reframe path planning in unknown environments as a semantic reasoning task. We introduce a Semantic Agent for Tensegrity robots (SATPlanner) driven by a Large Language Model (LLM). SATPlanner leverages high-level environmental comprehension to generate efficient and reliable planning this http URL the core of SATPlanner is an Adaptive Observation Window mechanism, inspired by the "fast" and "slow" thinking paradigms of LLMs. This mechanism dynamically adjusts the perceptual field of the agent: it narrows for rapid traversal of open spaces and expands to reason about complex obstacle configurations. This allows the agent to construct a semantic belief of the environment, enabling the search space to grow only linearly with the path length (O(L)) while maintaining path quality. We extensively evaluate SATPlanner in 1,000 simulation trials, where it achieves a 100% success rate, outperforming other real-time planning algorithms. Critically, SATPlanner reduces the search space by 37.2% compared to the A* algorithm while achieving comparable, near-optimal path lengths. Finally, the practical feasibility of SATPlanner is validated on a physical spherical tensegrity robot prototype. 

**Abstract (ZH)**: 具备固有的动态特性，赋予其非凡的鲁棒性和适应性，球形 tensegrity 机器人是混合软硬设计的典范实例，也是优秀的移动平台。然而，在未知环境中进行路径规划面临着重大挑战，需要在高效的探索和稳健的规划之间取得微妙平衡。传统路径规划算法将环境视为几何网格，往往遭受冗余搜索的困扰，并且在复杂场景中由于缺乏语义理解而容易失败。为了克服这些限制，我们将未知环境中路径规划重新定义为语义推理任务。我们引入了由大型语言模型（LLM）驱动的 tensegrity 机器人语义代理（SATPlanner）。SATPlanner 通过高层次的环境理解来生成高效可靠的规划路径。SATPlanner 的核心是受 LLM 的“快速”和“缓慢”思维 paradigm 启发的自适应观察窗口机制。该机制动态调整代理的感知范围：在快速穿越开阔空间时缩小感知范围，而在处理复杂障碍配置时扩大感知范围。这使得代理能够构建环境的语义信念，从而使搜索空间的增长与路径长度呈线性关系（O(L)），同时保持路径质量。我们在1,000次模拟试验中广泛评估了SATPlanner，其成功率达到100%，优于其他实时规划算法。关键地，与A*算法相比，SATPlanner 将搜索空间减少了37.2%，同时获得接近最优路径长度。最后，我们在物理球形 tensegrity 机器人原型上验证了SATPlanner 的实际可行性。 

---
# Embodiment Transfer Learning for Vision-Language-Action Models 

**Title (ZH)**: 视觉-语言-行动模型中的躯体化迁移学习 

**Authors**: Chengmeng Li, Yaxin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2511.01224)  

**Abstract**: Vision-language-action (VLA) models have significantly advanced robotic learning, enabling training on large-scale, cross-embodiment data and fine-tuning for specific robots. However, state-of-the-art autoregressive VLAs struggle with multi-robot collaboration. We introduce embodiment transfer learning, denoted as ET-VLA, a novel framework for efficient and effective transfer of pre-trained VLAs to multi-robot. ET-VLA's core is Synthetic Continued Pretraining (SCP), which uses synthetically generated data to warm up the model for the new embodiment, bypassing the need for real human demonstrations and reducing data collection costs. SCP enables the model to learn correct actions and precise action token numbers. Following SCP, the model is fine-tuned on target embodiment data. To further enhance the model performance on multi-embodiment, we present the Embodied Graph-of-Thought technique, a novel approach that formulates each sub-task as a node, that allows the VLA model to distinguish the functionalities and roles of each embodiment during task execution. Our work considers bimanual robots, a simple version of multi-robot to verify our approaches. We validate the effectiveness of our method on both simulation benchmarks and real robots covering three different bimanual embodiments. In particular, our proposed ET-VLA \space can outperform OpenVLA on six real-world tasks over 53.2%. We will open-source all codes to support the community in advancing VLA models for robot learning. 

**Abstract (ZH)**: 基于视觉-语言-行动的多机器人协作学习框架：ET-VLA 

---
# Scaling Cross-Embodiment World Models for Dexterous Manipulation 

**Title (ZH)**: 扩展跨躯体世界模型以实现灵巧 manipulation 

**Authors**: Zihao He, Bo Ai, Tongzhou Mu, Yulin Liu, Weikang Wan, Jiawei Fu, Yilun Du, Henrik I. Christensen, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2511.01177)  

**Abstract**: Cross-embodiment learning seeks to build generalist robots that operate across diverse morphologies, but differences in action spaces and kinematics hinder data sharing and policy transfer. This raises a central question: Is there any invariance that allows actions to transfer across embodiments? We conjecture that environment dynamics are embodiment-invariant, and that world models capturing these dynamics can provide a unified interface across embodiments. To learn such a unified world model, the crucial step is to design state and action representations that abstract away embodiment-specific details while preserving control relevance. To this end, we represent different embodiments (e.g., human hands and robot hands) as sets of 3D particles and define actions as particle displacements, creating a shared representation for heterogeneous data and control problems. A graph-based world model is then trained on exploration data from diverse simulated robot hands and real human hands, and integrated with model-based planning for deployment on novel hardware. Experiments on rigid and deformable manipulation tasks reveal three findings: (i) scaling to more training embodiments improves generalization to unseen ones, (ii) co-training on both simulated and real data outperforms training on either alone, and (iii) the learned models enable effective control on robots with varied degrees of freedom. These results establish world models as a promising interface for cross-embodiment dexterous manipulation. 

**Abstract (ZH)**: 跨体态学习旨在构建能够在多种形态下操作的一般型机器人，但动作空间和运动学差异阻碍了数据共享和策略转移。这提出了一个中心问题：是否存在能使动作在不同体态间转移的不变性？我们推测环境动力学是体态不变的，并认为能够捕获这些动力学的环境模型可以提供统一接口跨体态使用。为了学习这样的统一环境模型，关键步骤是设计能抽象掉体态特定细节但保留控制相关性的状态和动作表示。为此，我们将不同的体态（例如，人类手和机器人手）表示为3D粒子集，并将动作定义为粒子位移，从而为异构数据和控制问题创建共享表示。基于图的环境模型在多种模拟机器人手和真实人类手的探索数据上进行训练，并与基于模型的规划集成用于新硬件的部署。实验表明：(i) 增加训练体态的数量可以提高对未见体态的泛化能力；(ii) 在模拟数据和真实数据上联合训练优于单独使用任一类数据进行训练；(iii) 学习到的模型能够有效控制具有不同自由度的机器人。这些结果确立了环境模型作为跨体态灵巧操控的有前途接口的地位。 

---
# SLAP: Shortcut Learning for Abstract Planning 

**Title (ZH)**: SLAP: Shortcut Learning for Abstract Planning 

**Authors**: Y. Isabel Liu, Bowen Li, Benjamin Eysenbach, Tom Silver  

**Link**: [PDF](https://arxiv.org/pdf/2511.01107)  

**Abstract**: Long-horizon decision-making with sparse rewards and continuous states and actions remains a fundamental challenge in AI and robotics. Task and motion planning (TAMP) is a model-based framework that addresses this challenge by planning hierarchically with abstract actions (options). These options are manually defined, limiting the agent to behaviors that we as human engineers know how to program (pick, place, move). In this work, we propose Shortcut Learning for Abstract Planning (SLAP), a method that leverages existing TAMP options to automatically discover new ones. Our key idea is to use model-free reinforcement learning (RL) to learn shortcuts in the abstract planning graph induced by the existing options in TAMP. Without any additional assumptions or inputs, shortcut learning leads to shorter solutions than pure planning, and higher task success rates than flat and hierarchical RL. Qualitatively, SLAP discovers dynamic physical improvisations (e.g., slap, wiggle, wipe) that differ significantly from the manually-defined ones. In experiments in four simulated robotic environments, we show that SLAP solves and generalizes to a wide range of tasks, reducing overall plan lengths by over 50% and consistently outperforming planning and RL baselines. 

**Abstract (ZH)**: 长时限决策问题中的稀疏奖励与连续状态和动作仍是对AI和机器人领域的一项基本挑战。基于任务和运动规划（TAMP）是一种通过使用抽象动作（选项）进行分层规划的模型化框架，以应对这一挑战。这些选项由人工定义，限制了代理只能执行我们作为人类工程师能够编程的行为（例如，抓取、放置、移动）。在本项工作中，我们提出了一种名为Shortcut Learning for Abstract Planning（SLAP）的方法，该方法利用现有TAMP选项自动发现新的选项。我们的核心思想是使用无模型的强化学习（RL）来学习在现有TAMP选项诱导的抽象规划图中的捷径。在没有任何额外假设或输入的情况下，捷径学习可以得到比纯粹规划更短的解决方案，并且在任务成功率上优于平面RL和分层RL。定性上，SLAP发现了一种动态的物理即兴行为（例如，轻拍、摇动、擦拭），这些行为与人工定义的行为有显著不同。在对四种模拟机器人环境进行的实验中，我们展示了SLAP能够解决并泛化到多种任务，总体规划长度减少了超过50%，并且在所有基线方法上表现优异。 

---
# GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies 

**Title (ZH)**: GauDP：通过高斯图像协同作用重 invent 多智能体合作在扩散策略中的实现 

**Authors**: Ziye Wang, Li Kang, Yiran Qin, Jiahua Ma, Zhanglin Peng, Lei Bai, Ruimao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00998)  

**Abstract**: Recently, effective coordination in embodied multi-agent systems has remained a fundamental challenge, particularly in scenarios where agents must balance individual perspectives with global environmental awareness. Existing approaches often struggle to balance fine-grained local control with comprehensive scene understanding, resulting in limited scalability and compromised collaboration quality. In this paper, we present GauDP, a novel Gaussian-image synergistic representation that facilitates scalable, perception-aware imitation learning in multi-agent collaborative systems. Specifically, GauDP constructs a globally consistent 3D Gaussian field from decentralized RGB observations, then dynamically redistributes 3D Gaussian attributes to each agent's local perspective. This enables all agents to adaptively query task-critical features from the shared scene representation while maintaining their individual viewpoints. This design facilitates both fine-grained control and globally coherent behavior without requiring additional sensing modalities (e.g., 3D point cloud). We evaluate GauDP on the RoboFactory benchmark, which includes diverse multi-arm manipulation tasks. Our method achieves superior performance over existing image-based methods and approaches the effectiveness of point-cloud-driven methods, while maintaining strong scalability as the number of agents increases. 

**Abstract (ZH)**: 最近，具身多智能体系统的有效协调依然是一个基本挑战，特别是在智能体必须平衡个体视角与全局环境意识的场景中。现有方法往往难以平衡精细的局部控制与全面的场景理解，导致有限的扩展性和协作质量受损。本文介绍了GauDP，一种新颖的高斯图像协同表示，它促进了多智能体协作系统中具有感知意识的模仿学习的高效性。具体而言，GauDP 从去中心化的RGB观测中构建全局一致的3D高斯场，然后动态重新分配3D高斯属性到每个智能体的局部视角。这使所有智能体能够适应性地从共享的场景表示中查询任务关键特征，同时保持各自的视角。此设计在不需额外感知模态（例如，3D点云）的情况下，实现了精细控制和全局一致行为的结合。我们已在RoboFactory基准测试上评估了GauDP，该基准包含多样化的多臂操作任务。我们的方法在图像基方法中表现更优，并接近基于点云的方法的效果，同时随着智能体数量增加保持了强大的扩展性。 

---
# URDF-Anything: Constructing Articulated Objects with 3D Multimodal Language Model 

**Title (ZH)**: URDF-Anything: 使用3D多模态语言模型构建 articulated 对象 

**Authors**: Zhe Li, Xiang Bai, Jieyu Zhang, Zhuangzhe Wu, Che Xu, Ying Li, Chengkai Hou, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00940)  

**Abstract**: Constructing accurate digital twins of articulated objects is essential for robotic simulation training and embodied AI world model building, yet historically requires painstaking manual modeling or multi-stage pipelines. In this work, we propose \textbf{URDF-Anything}, an end-to-end automatic reconstruction framework based on a 3D multimodal large language model (MLLM). URDF-Anything utilizes an autoregressive prediction framework based on point-cloud and text multimodal input to jointly optimize geometric segmentation and kinematic parameter prediction. It implements a specialized $[SEG]$ token mechanism that interacts directly with point cloud features, enabling fine-grained part-level segmentation while maintaining consistency with the kinematic parameter predictions. Experiments on both simulated and real-world datasets demonstrate that our method significantly outperforms existing approaches regarding geometric segmentation (mIoU 17\% improvement), kinematic parameter prediction (average error reduction of 29\%), and physical executability (surpassing baselines by 50\%). Notably, our method exhibits excellent generalization ability, performing well even on objects outside the training set. This work provides an efficient solution for constructing digital twins for robotic simulation, significantly enhancing the sim-to-real transfer capability. 

**Abstract (ZH)**: URDF-Anything：基于3D多模态大语言模型的端到端自动重建框架 

---
# Maestro: Orchestrating Robotics Modules with Vision-Language Models for Zero-Shot Generalist Robots 

**Title (ZH)**: Maestro: 通过视觉语言模型 orchestrating 机器人模块的零样本通用机器人 

**Authors**: Junyao Shi, Rujia Yang, Kaitian Chao, Selina Bingqing Wan, Yifei Shao, Jiahui Lei, Jianing Qian, Long Le, Pratik Chaudhari, Kostas Daniilidis, Chuan Wen, Dinesh Jayaraman  

**Link**: [PDF](https://arxiv.org/pdf/2511.00917)  

**Abstract**: Today's best-explored routes towards generalist robots center on collecting ever larger "observations-in actions-out" robotics datasets to train large end-to-end models, copying a recipe that has worked for vision-language models (VLMs). We pursue a road less traveled: building generalist policies directly around VLMs by augmenting their general capabilities with specific robot capabilities encapsulated in a carefully curated set of perception, planning, and control modules. In Maestro, a VLM coding agent dynamically composes these modules into a programmatic policy for the current task and scenario. Maestro's architecture benefits from a streamlined closed-loop interface without many manually imposed structural constraints, and a comprehensive and diverse tool repertoire. As a result, it largely surpasses today's VLA models for zero-shot performance on challenging manipulation skills. Further, Maestro is easily extensible to incorporate new modules, easily editable to suit new embodiments such as a quadruped-mounted arm, and even easily adapts from minimal real-world experiences through local code edits. 

**Abstract (ZH)**: 今天的通用机器人研究主要集中在收集越来越大的“观察-动作”机器人数据集以训练端到端模型，借鉴视觉语言模型的成功配方。我们探索了一条较少走过的道路：通过将特定机器人能力封装在仔细筛选的感知、规划和控制模块中，直接在视觉语言模型周围构建通用政策。在Maestro中，一个视觉语言模型编码代理动态组合这些模块，为当前任务和场景生成程序化的策略。Maestro的架构得益于简化了的闭环接口，缺乏许多手动施加的结构约束，并且具有全面且多样的工具库。结果，它在零样本操作技能表现上显著超越当今的VA一审模。此外，Maestro容易扩展以整合新模块，容易编辑以适应新的躯体配置，如四足机器臂，甚至可以通过局部代码编辑从少量的现实世界经验中适应。 

---
# Heuristic Step Planning for Learning Dynamic Bipedal Locomotion: A Comparative Study of Model-Based and Model-Free Approaches 

**Title (ZH)**: 基于模型和无模型方法的动态双足行走学习启发式步骤规划比较研究 

**Authors**: William Suliman, Ekaterina Chaikovskaia, Egor Davydenko, Roman Gorbachev  

**Link**: [PDF](https://arxiv.org/pdf/2511.00840)  

**Abstract**: This work presents an extended framework for learning-based bipedal locomotion that incorporates a heuristic step-planning strategy guided by desired torso velocity tracking. The framework enables precise interaction between a humanoid robot and its environment, supporting tasks such as crossing gaps and accurately approaching target objects. Unlike approaches based on full or simplified dynamics, the proposed method avoids complex step planners and analytical models. Step planning is primarily driven by heuristic commands, while a Raibert-type controller modulates the foot placement length based on the error between desired and actual torso velocity. We compare our method with a model-based step-planning approach -- the Linear Inverted Pendulum Model (LIPM) controller. Experimental results demonstrate that our approach attains comparable or superior accuracy in maintaining target velocity (up to 80%), significantly greater robustness on uneven terrain (over 50% improvement), and improved energy efficiency. These results suggest that incorporating complex analytical, model-based components into the training architecture may be unnecessary for achieving stable and robust bipedal walking, even in unstructured environments. 

**Abstract (ZH)**: 基于启发式步态规划的人形机器人两足步行扩展框架： torso姿态追踪指导的步态规划策略及其应用 

---
# When Semantics Connect the Swarm: LLM-Driven Fuzzy Control for Cooperative Multi-Robot Underwater Coverage 

**Title (ZH)**: 当语义连接群体：由大规模语言模型驱动的模糊控制在协同水下覆盖机器人中的应用 

**Authors**: Jingzehua Xu, Weihang Zhang, Yangyang Li, Hongmiaoyi Zhang, Guanwen Xie, Jiwei Tang, Shuai Zhang, Yi Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.00783)  

**Abstract**: Underwater multi-robot cooperative coverage remains challenging due to partial observability, limited communication, environmental uncertainty, and the lack of access to global localization. To address these issues, this paper presents a semantics-guided fuzzy control framework that couples Large Language Models (LLMs) with interpretable control and lightweight coordination. Raw multimodal observations are compressed by the LLM into compact, human-interpretable semantic tokens that summarize obstacles, unexplored regions, and Objects Of Interest (OOIs) under uncertain perception. A fuzzy inference system with pre-defined membership functions then maps these tokens into smooth and stable steering and gait commands, enabling reliable navigation without relying on global positioning. Then, we further coordinate multiple robots by introducing semantic communication that shares intent and local context in linguistic form, enabling agreement on who explores where while avoiding redundant revisits. Extensive simulations in unknown reef-like environments show that, under limited sensing and communication, the proposed framework achieves robust OOI-oriented navigation and cooperative coverage with improved efficiency and adaptability, narrowing the gap between semantic cognition and distributed underwater control in GPS-denied, map-free conditions. 

**Abstract (ZH)**: underwater 多机器人协同覆盖仍因部分可观测性、有限通信、环境不确定性以及缺少全局定位访问而具有挑战性。为解决这些问题，本文提出了一种语义引导的模糊控制框架，结合了大型语言模型（LLMs）与可解释控制和轻量级协调。原始多模态观测由LLM压缩为紧凑且人类可解释的语义标记，这些标记在不确定感知下总结障碍物、未探索区域和感兴趣对象（OOIs）。随后利用预定义的隶属函数的模糊推理系统将这些标记映射为平滑且稳定的转向和步态指令，使导航可靠而不依赖于全球定位。进一步通过引入语义通信协调多机器人，以语言形式共享意图和局部上下文，从而达成对探索区域的共识并避免重复访问。在未知礁石状环境中的 extensive 模拟表明，在有限的感知和通信条件下，所提出框架实现了以感兴趣对象为导向的稳健导航和协同覆盖，并提高了效率和适应性，缩小了在无 GPS、无地图条件下语义认知与分布式水下控制之间的差距。 

---
# Improving Robustness to Out-of-Distribution States in Imitation Learning via Deep Koopman-Boosted Diffusion Policy 

**Title (ZH)**: 通过深Koopman增强扩散策略提高 imitation learning 对离分布状态的鲁棒性 

**Authors**: Dianye Huang, Nassir Navab, Zhongliang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00555)  

**Abstract**: Integrating generative models with action chunking has shown significant promise in imitation learning for robotic manipulation. However, the existing diffusion-based paradigm often struggles to capture strong temporal dependencies across multiple steps, particularly when incorporating proprioceptive input. This limitation can lead to task failures, where the policy overfits to proprioceptive cues at the expense of capturing the visually derived features of the task. To overcome this challenge, we propose the Deep Koopman-boosted Dual-branch Diffusion Policy (D3P) algorithm. D3P introduces a dual-branch architecture to decouple the roles of different sensory modality combinations. The visual branch encodes the visual observations to indicate task progression, while the fused branch integrates both visual and proprioceptive inputs for precise manipulation. Within this architecture, when the robot fails to accomplish intermediate goals, such as grasping a drawer handle, the policy can dynamically switch to execute action chunks generated by the visual branch, allowing recovery to previously observed states and facilitating retrial of the task. To further enhance visual representation learning, we incorporate a Deep Koopman Operator module that captures structured temporal dynamics from visual inputs. During inference, we use the test-time loss of the generative model as a confidence signal to guide the aggregation of the temporally overlapping predicted action chunks, thereby enhancing the reliability of policy execution. In simulation experiments across six RLBench tabletop tasks, D3P outperforms the state-of-the-art diffusion policy by an average of 14.6\%. On three real-world robotic manipulation tasks, it achieves a 15.0\% improvement. Code: this https URL. 

**Abstract (ZH)**: 将生成模型与动作切片结合用于机器人操作的模仿学习显示出显著潜力。然而，现有的基于扩散的过程往往难以捕捉跨多个步骤的强时间依赖关系，特别是在结合本体感受输入时。这一限制可能导致任务失败，即策略过度拟合本体感受线索，而忽略了视觉特征的捕捉。为克服这一挑战，我们提出了一种Deep Koopman-提振的双支道路扩散策略（D3P）算法。D3P引入了双支路径架构，以解耦不同感觉模态组合的角色。视觉支路编码视觉观察以指示任务进度，而融合支路则结合视觉和本体感受输入以实现精确操作。在此架构中，当机器人未能完成如抽屉把手抓取等中间目标时，策略可以动态切换到执行视觉支路生成的动作切片，从而恢复到先前观测的状态并促进任务重试。为增强视觉表示学习，我们引入了一个Deep Koopman算子模块，用于从视觉输入中捕获结构化的时间动态。在推理过程中，我们使用生成模型的测试时间损失作为置信信号来引导时间重叠预测动作切片的聚合，以增强策略执行的可靠性。在针对六项RLBench桌面任务的模拟实验中，D3P平均优于现有最先进的扩散策略14.6%。在三项真实世界的机器人操作任务中，它实现了15.0%的改进。代码：请参见链接。 

---
# Descriptive Model-based Learning and Control for Bipedal Locomotion 

**Title (ZH)**: 基于描述性模型的学习与控制方法在 bipedal 行走中的应用 

**Authors**: Suraj Kumar, Andy Ruina  

**Link**: [PDF](https://arxiv.org/pdf/2511.00512)  

**Abstract**: Bipedal balance is challenging due to its multi-phase, hybrid nature and high-dimensional state space. Traditional balance control approaches for bipedal robots rely on low-dimensional models for locomotion planning and reactive control, constraining the full robot to behave like these simplified models. This involves tracking preset reference paths for the Center of Mass and upper body obtained through low-dimensional models, often resulting in inefficient walking patterns with bent knees. However, we observe that bipedal balance is inherently low-dimensional and can be effectively described with simple state and action descriptors in a low-dimensional state space. This allows the robot's motion to evolve freely in its high-dimensional state space, only constraining its projection in the low-dimensional state space. In this work, we propose a novel control approach that avoids prescribing a low-dimensional model to the full model. Instead, our control framework uses a descriptive model with the minimum degrees of freedom necessary to maintain balance, allowing the remaining degrees of freedom to evolve freely in the high-dimensional space. This results in an efficient human-like walking gait and improved robustness. 

**Abstract (ZH)**: 双足平衡控制由于其多阶段、混合性质和高维状态空间而具有挑战性。传统的双足机器人平衡控制方法依赖于低维模型来进行运动规划和反应控制，这限制了整个机器人只能像这些简化模型那样行为。这通常涉及通过低维模型获得的质心和上体的预设参考路径跟踪，导致了不高效的行走模式，膝关节弯曲。然而，我们观察到双足平衡本质上是低维的，并且可以用简单的状态和动作描述符在低维状态空间中有效地描述。这使得机器人的运动可以在高维状态空间中自由演化，仅在其在低维状态空间的投影中受到约束。在本文中，我们提出了一种新方法，避免将低维模型强加于完整模型。相反，我们的控制框架使用一个描述性模型，其中包含维持平衡所需的最小自由度，从而使其余自由度可以在高维空间中自由演化。这导致了高效的人类般的行走模式并提高了系统的鲁棒性。 

---
# EgoMI: Learning Active Vision and Whole-Body Manipulation from Egocentric Human Demonstrations 

**Title (ZH)**: EgoMI: 从第一人称人类示范中学习主动视觉和全身 manipulation 

**Authors**: Justin Yu, Yide Shentu, Di Wu, Pieter Abbeel, Ken Goldberg, Philipp Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00153)  

**Abstract**: Imitation learning from human demonstrations offers a promising approach for robot skill acquisition, but egocentric human data introduces fundamental challenges due to the embodiment gap. During manipulation, humans actively coordinate head and hand movements, continuously reposition their viewpoint and use pre-action visual fixation search strategies to locate relevant objects. These behaviors create dynamic, task-driven head motions that static robot sensing systems cannot replicate, leading to a significant distribution shift that degrades policy performance. We present EgoMI (Egocentric Manipulation Interface), a framework that captures synchronized end-effector and active head trajectories during manipulation tasks, resulting in data that can be retargeted to compatible semi-humanoid robot embodiments. To handle rapid and wide-spanning head viewpoint changes, we introduce a memory-augmented policy that selectively incorporates historical observations. We evaluate our approach on a bimanual robot equipped with an actuated camera head and find that policies with explicit head-motion modeling consistently outperform baseline methods. Results suggest that coordinated hand-eye learning with EgoMI effectively bridges the human-robot embodiment gap for robust imitation learning on semi-humanoid embodiments. Project page: this https URL 

**Abstract (ZH)**: 基于人类演示的模仿学习为机器人技能获取提供了有前景的方法，但第一人称人类数据由于存在体感差距引入了根本性的挑战。在操作过程中，人类会主动协调头部和手部动作，持续重新定位视角，并在动作前使用视觉固定搜索策略来定位相关物体。这些行为产生了由任务驱动的动态头部运动，而静态的机器人感知系统无法复制，导致数据分布转移，从而降低了策略性能。我们提出了EgoMI（第一人称操纵界面）框架，该框架在操作任务中捕捉同步的末端执行器和主动头部轨迹，生成可重新定向到兼容的半类人机器人体感的数据。为了处理快速且广泛的头部视角变化，我们引入了一个增强记忆的策略，该策略选择性地结合了历史观测。我们评估了该方法在配备可动作摄像头头部的双臂机器人上的效果，发现具有明确头部运动建模的策略始终优于基线方法。结果表明，通过EgoMI进行协调的手眼学习有效缩小了人类与机器人之间的体感差距，提高了半类人机器人上稳健的模仿学习。项目页面：this https URL 

---
# End-to-End Dexterous Arm-Hand VLA Policies via Shared Autonomy: VR Teleoperation Augmented by Autonomous Hand VLA Policy for Efficient Data Collection 

**Title (ZH)**: 基于共享自治的端到端手动灵活手臂数据收集策略：通过自主手部数据收集策略增强的虚拟现实遥操作 

**Authors**: Yu Cui, Yujian Zhang, Lina Tao, Yang Li, Xinyu Yi, Zhibin Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.00139)  

**Abstract**: Achieving human-like dexterous manipulation remains a major challenge for general-purpose robots. While Vision-Language-Action (VLA) models show potential in learning skills from demonstrations, their scalability is limited by scarce high-quality training data. Existing data collection methods face inherent constraints: manual teleoperation overloads human operators, while automated planning often produces unnatural motions. We propose a Shared Autonomy framework that divides control between macro and micro motions. A human operator guides the robot's arm pose through intuitive VR teleoperation, while an autonomous DexGrasp-VLA policy handles fine-grained hand control using real-time tactile and visual feedback. This division significantly reduces cognitive load and enables efficient collection of high-quality coordinated arm-hand demonstrations. Using this data, we train an end-to-end VLA policy enhanced with our novel Arm-Hand Feature Enhancement module, which captures both distinct and shared representations of macro and micro movements for more natural coordination. Our Corrective Teleoperation system enables continuous policy improvement through human-in-the-loop failure recovery. Experiments demonstrate that our framework generates high-quality data with minimal manpower and achieves a 90% success rate across diverse objects, including unseen instances. Comprehensive evaluations validate the system's effectiveness in developing dexterous manipulation capabilities. 

**Abstract (ZH)**: 实现类人的灵巧操作仍然是通用机器人面临的重大挑战。视觉-语言-动作（VLA）模型在通过演示学习技能方面展现潜力，但其可扩展性受限于稀缺的高质量训练数据。现有数据收集方法存在固有约束：手动遥操作增加了人类操作员的负担，而自动化规划常常产生不自然的运动。我们提出了一种协同自治框架，将控制分为宏观和微观运动。人类操作员通过直观的VR遥操作引导机器人的手臂姿态，而自主的DexGrasp-VLA策略则利用实时的触觉和视觉反馈处理精细的手部控制。这一划分显著减轻了认知负担，并使高保真协调的手臂-手演示的高效收集成为可能。使用这些数据，我们训练了一个端到端的VLA策略，该策略结合了我们提出的新型手臂-手特征增强模块，该模块捕捉宏观和微观运动的独特性和共性表示，以实现更自然的协调。我们的纠正遥操作系统通过人类在回路中的失败恢复实现持续的策略改进。实验表明，我们的框架以最少的人力生成高质量数据，并在各种物体上实现了90%的成功率，包括未见过的实例。全面评估验证了该系统在开发灵巧操作能力方面的有效性。 

---
# Real-DRL: Teach and Learn in Reality 

**Title (ZH)**: Real-DRL: 在现实中教与学 

**Authors**: Yanbing Mao, Yihao Cai, Lui Sha  

**Link**: [PDF](https://arxiv.org/pdf/2511.00112)  

**Abstract**: This paper introduces the Real-DRL framework for safety-critical autonomous systems, enabling runtime learning of a deep reinforcement learning (DRL) agent to develop safe and high-performance action policies in real plants (i.e., real physical systems to be controlled), while prioritizing safety! The Real-DRL consists of three interactive components: a DRL-Student, a PHY-Teacher, and a Trigger. The DRL-Student is a DRL agent that innovates in the dual self-learning and teaching-to-learn paradigm and the real-time safety-informed batch sampling. On the other hand, PHY-Teacher is a physics-model-based design of action policies that focuses solely on safety-critical functions. PHY-Teacher is novel in its real-time patch for two key missions: i) fostering the teaching-to-learn paradigm for DRL-Student and ii) backing up the safety of real plants. The Trigger manages the interaction between the DRL-Student and the PHY-Teacher. Powered by the three interactive components, the Real-DRL can effectively address safety challenges that arise from the unknown unknowns and the Sim2Real gap. Additionally, Real-DRL notably features i) assured safety, ii) automatic hierarchy learning (i.e., safety-first learning and then high-performance learning), and iii) safety-informed batch sampling to address the learning experience imbalance caused by corner cases. Experiments with a real quadruped robot, a quadruped robot in NVIDIA Isaac Gym, and a cart-pole system, along with comparisons and ablation studies, demonstrate the Real-DRL's effectiveness and unique features. 

**Abstract (ZH)**: 基于实时深度强化学习的实时安全框架（Real-DRL）：在安全关键自主系统中实现运行时安全导向的学习 

---
# Digital Twin based Automatic Reconfiguration of Robotic Systems in Smart Environments 

**Title (ZH)**: 基于数字孪生的智能环境中机器人系统自动重构方法 

**Authors**: Angelos Alexopoulos, Agorakis Bompotas, Nikitas Rigas Kalogeropoulos, Panagiotis Kechagias, Athanasios P. Kalogeras, Christos Alexakos  

**Link**: [PDF](https://arxiv.org/pdf/2511.00094)  

**Abstract**: Robotic systems have become integral to smart environments, enabling applications ranging from urban surveillance and automated agriculture to industrial automation. However, their effective operation in dynamic settings - such as smart cities and precision farming - is challenged by continuously evolving topographies and environmental conditions. Traditional control systems often struggle to adapt quickly, leading to inefficiencies or operational failures. To address this limitation, we propose a novel framework for autonomous and dynamic reconfiguration of robotic controllers using Digital Twin technology. Our approach leverages a virtual replica of the robot's operational environment to simulate and optimize movement trajectories in response to real-world changes. By recalculating paths and control parameters in the Digital Twin and deploying the updated code to the physical robot, our method ensures rapid and reliable adaptation without manual intervention. This work advances the integration of Digital Twins in robotics, offering a scalable solution for enhancing autonomy in smart, dynamic environments. 

**Abstract (ZH)**: 基于数字孪生的自主动态机器人控制器重构框架 

---
# Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail 

**Title (ZH)**: Alpamayo-R1: 跨越推理与动作预测，实现长尾通用自主驾驶 

**Authors**: NVIDIA, Yan Wang, Wenjie Luo, Junjie Bai, Yulong Cao, Tong Che, Ke Chen, Yuxiao Chen, Jenna Diamond, Yifan Ding, Wenhao Ding, Liang Feng, Greg Heinrich, Jack Huang, Peter Karkus, Boyi Li, Pinyi Li, Tsung-Yi Lin, Dongran Liu, Ming-Yu Liu, Langechuan Liu, Zhijian Liu, Jason Lu, Yunxiang Mao, Pavlo Molchanov, Lindsey Pavao, Zhenghao Peng, Mike Ranzinger, Ed Schmerling, Shida Shen, Yunfei Shi, Sarah Tariq, Ran Tian, Tilman Wekel, Xinshuo Weng, Tianjun Xiao, Eric Yang, Xiaodong Yang, Yurong You, Xiaohui Zeng, Wenyuan Zhang, Boris Ivanovic, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2511.00088)  

**Abstract**: End-to-end architectures trained via imitation learning have advanced autonomous driving by scaling model size and data, yet performance remains brittle in safety-critical long-tail scenarios where supervision is sparse and causal understanding is limited. To address this, we introduce Alpamayo-R1 (AR1), a vision-language-action model (VLA) that integrates Chain of Causation reasoning with trajectory planning to enhance decision-making in complex driving scenarios. Our approach features three key innovations: (1) the Chain of Causation (CoC) dataset, built through a hybrid auto-labeling and human-in-the-loop pipeline producing decision-grounded, causally linked reasoning traces aligned with driving behaviors; (2) a modular VLA architecture combining Cosmos-Reason, a Vision-Language Model pre-trained for Physical AI applications, with a diffusion-based trajectory decoder that generates dynamically feasible plans in real time; (3) a multi-stage training strategy using supervised fine-tuning to elicit reasoning and reinforcement learning (RL) to optimize reasoning quality via large reasoning model feedback and enforce reasoning-action consistency. Evaluation shows AR1 achieves up to a 12% improvement in planning accuracy on challenging cases compared to a trajectory-only baseline, with a 35% reduction in off-road rate and 25% reduction in close encounter rate in closed-loop simulation. RL post-training improves reasoning quality by 45% as measured by a large reasoning model critic and reasoning-action consistency by 37%. Model scaling from 0.5B to 7B parameters shows consistent improvements. On-vehicle road tests confirm real-time performance (99 ms latency) and successful urban deployment. By bridging interpretable reasoning with precise control, AR1 demonstrates a practical path towards Level 4 autonomous driving. We plan to release AR1 models and a subset of the CoC in a future update. 

**Abstract (ZH)**: 基于模仿学习训练的端到端架构通过扩大模型规模和数据促进了自动驾驶的发展，但在监督稀少且因果理解有限的安全关键长尾场景中，性能仍然脆弱。为解决这一问题，我们引入了Alpamayo-R1 (AR1) 视觉-语言-动作模型（VLA），该模型将因果链推理与轨迹规划相结合，以增强在复杂驾驶场景中的决策能力。我们的方法包含三个关键创新点：（1）因果链（CoC）数据集，通过混合自动标注和人工介入管道构建，生成与驾驶行为相关的、因果关联的推理轨迹；（2）模块化的VLA架构，结合了为物理AI应用预训练的Cosmos-Reason视觉-语言模型，与基于扩散的轨迹解码器，实时生成动态可行的计划；（3）多阶段训练策略，通过监督微调激发推理能力，并通过大规模推理模型反馈和推理-行动一致性来优化推理质量，以及通过强化学习（RL）加强推理质量的优化和一致性保证。评估结果显示，与仅基于轨迹的基线相比，AR1在具有挑战性的案例中规划准确率提高了12%，越野率降低了35%，近距离相遇率降低了25%。闭环模拟后，训练后强化学习将推理质量提高了45%，推理-行动一致性提高了37%。参数从0.5B增加到7B，显示了持续的改进。车顶实测证实了实时性能（99 ms延迟）和成功的城市部署。通过将可解释的推理与精确控制结合，AR1展示了通往L4级自动驾驶的实用路径。未来我们将发布AR1模型和因果链（CoC）的部分数据。 

---
# Endowing GPT-4 with a Humanoid Body: Building the Bridge Between Off-the-Shelf VLMs and the Physical World 

**Title (ZH)**: 赋予GPT-4人型身体：构建现成多模态模型与物理世界之间的桥梁 

**Authors**: Yingzhao Jian, Zhongan Wang, Yi Yang, Hehe Fan  

**Link**: [PDF](https://arxiv.org/pdf/2511.00041)  

**Abstract**: Humanoid agents often struggle to handle flexible and diverse interactions in open environments. A common solution is to collect massive datasets to train a highly capable model, but this approach can be prohibitively expensive. In this paper, we explore an alternative solution: empowering off-the-shelf Vision-Language Models (VLMs, such as GPT-4) to control humanoid agents, thereby leveraging their strong open-world generalization to mitigate the need for extensive data collection. To this end, we present \textbf{BiBo} (\textbf{B}uilding humano\textbf{I}d agent \textbf{B}y \textbf{O}ff-the-shelf VLMs). It consists of two key components: (1) an \textbf{embodied instruction compiler}, which enables the VLM to perceive the environment and precisely translate high-level user instructions (e.g., {\small\itshape ``have a rest''}) into low-level primitive commands with control parameters (e.g., {\small\itshape ``sit casually, location: (1, 2), facing: 90$^\circ$''}); and (2) a diffusion-based \textbf{motion executor}, which generates human-like motions from these commands, while dynamically adapting to physical feedback from the environment. In this way, BiBo is capable of handling not only basic interactions but also diverse and complex motions. Experiments demonstrate that BiBo achieves an interaction task success rate of 90.2\% in open environments, and improves the precision of text-guided motion execution by 16.3\% over prior methods. The code will be made publicly available. 

**Abstract (ZH)**: 使用即用型视觉-语言模型赋能 humanoId 代理的 BiBo 方法 

---
# STRIDER: Navigation via Instruction-Aligned Structural Decision Space Optimization 

**Title (ZH)**: STRIDER：通过指令对齐结构决策空间优化实现导航 

**Authors**: Diqi He, Xuehao Gao, Hao Li, Junwei Han, Dingwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00033)  

**Abstract**: The Zero-shot Vision-and-Language Navigation in Continuous Environments (VLN-CE) task requires agents to navigate previously unseen 3D environments using natural language instructions, without any scene-specific training. A critical challenge in this setting lies in ensuring agents' actions align with both spatial structure and task intent over long-horizon execution. Existing methods often fail to achieve robust navigation due to a lack of structured decision-making and insufficient integration of feedback from previous actions. To address these challenges, we propose STRIDER (Instruction-Aligned Structural Decision Space Optimization), a novel framework that systematically optimizes the agent's decision space by integrating spatial layout priors and dynamic task feedback. Our approach introduces two key innovations: 1) a Structured Waypoint Generator that constrains the action space through spatial structure, and 2) a Task-Alignment Regulator that adjusts behavior based on task progress, ensuring semantic alignment throughout navigation. Extensive experiments on the R2R-CE and RxR-CE benchmarks demonstrate that STRIDER significantly outperforms strong SOTA across key metrics; in particular, it improves Success Rate (SR) from 29% to 35%, a relative gain of 20.7%. Such results highlight the importance of spatially constrained decision-making and feedback-guided execution in improving navigation fidelity for zero-shot VLN-CE. 

**Abstract (ZH)**: 零样本连续环境中的视觉语言导航（VLN-CE）任务要求代理使用自然语言指令导航之前未见过的3D环境，无需进行场景特定训练。在这个设定中，确保代理的行动在长时序执行中与空间结构和任务意图保持一致是一个关键挑战。现有方法由于缺乏结构化的决策机制和前行动作反馈的不足，往往无法实现稳健的导航。为了解决这些挑战，我们提出了一种名为STRIDER（指令对齐结构决策空间优化）的新框架，该框架通过整合空间布局先验和动态任务反馈系统地优化代理的决策空间。我们的方法引入了两个关键创新：1）结构化航点生成器，通过空间结构约束动作空间；2）任务对齐调节器，根据任务进度调整行为，确保导航过程中的语义对齐。在R2R-CE和RxR-CE基准上的大量实验表明，STRIDER在关键指标上显著优于当前最强的方法；特别是在成功率为29%提高到35%（相对增益为20.7%）方面表现尤为突出。这些结果强调了在零样本VLN-CE中通过空间约束决策和反馈引导执行提高导航准确性的必要性。 

---
# Gen AI in Automotive: Applications, Challenges, and Opportunities with a Case study on In-Vehicle Experience 

**Title (ZH)**: 汽车领域的生成型AI：应用、挑战与机遇——以车内体验为例 

**Authors**: Chaitanya Shinde, Divya Garikapati  

**Link**: [PDF](https://arxiv.org/pdf/2511.00026)  

**Abstract**: Generative Artificial Intelligence is emerging as a transformative force in the automotive industry, enabling novel applications across vehicle design, manufacturing, autonomous driving, predictive maintenance, and in vehicle user experience. This paper provides a comprehensive review of the current state of GenAI in automotive, highlighting enabling technologies such as Generative Adversarial Networks and Variational Autoencoders. Key opportunities include accelerating autonomous driving validation through synthetic data generation, optimizing component design, and enhancing human machine interaction via personalized and adaptive interfaces. At the same time, the paper identifies significant technical, ethical, and safety challenges, including computational demands, bias, intellectual property concerns, and adversarial robustness, that must be addressed for responsible deployment. A case study on Mercedes Benzs MBUX Virtual Assistant illustrates how GenAI powered voice systems deliver more natural, proactive, and personalized in car interactions compared to legacy rule based assistants. Through this review and case study, the paper outlines both the promise and limitations of GenAI integration in the automotive sector and presents directions for future research and development aimed at achieving safer, more efficient, and user centric mobility. Unlike prior reviews that focus solely on perception or manufacturing, this paper emphasizes generative AI in voice based HMI, bridging safety and user experience perspectives. 

**Abstract (ZH)**: 生成式人工智能正成为推动汽车行业的变革力量，使其能够在车辆设计、制造、自动驾驶、预测性维护和车内用户体验等方面实现新颖的应用。本文对当前生成式人工智能在汽车领域的研究进行全面综述，强调生成对抗网络和变分自动编码器等使能技术。关键机会包括通过合成数据生成加速自动驾驶验证、优化组件设计以及通过个性化和自适应界面提升人机交互。同时，本文指出了负责任部署过程中的重大技术、伦理和安全挑战，包括计算需求、偏见、知识产权问题以及对抗性鲁棒性。梅赛德斯奔驰MBUX虚拟助手的案例研究表明，生成式人工智能驱动的语音系统与基于规则的遗留系统相比，能够实现更自然、主动和个性化的车内交互。通过这一综述和案例研究，本文概述了生成式人工智能在汽车领域的前景与限制，并指出了旨在实现更安全、更高效和用户中心化移动性的未来研究和发展方向。与此前仅关注感知或制造的综述不同，本文强调了基于语音的人机交互中的生成AI，结合了安全性和用户体验的视角。 

---
# 3EED: Ground Everything Everywhere in 3D 

**Title (ZH)**: 3EED: 在三维中处处ground一切 

**Authors**: Rong Li, Yuhao Dong, Tianshuai Hu, Ao Liang, Youquan Liu, Dongyue Lu, Liang Pan, Lingdong Kong, Junwei Liang, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01755)  

**Abstract**: Visual grounding in 3D is the key for embodied agents to localize language-referred objects in open-world environments. However, existing benchmarks are limited to indoor focus, single-platform constraints, and small scale. We introduce 3EED, a multi-platform, multi-modal 3D grounding benchmark featuring RGB and LiDAR data from vehicle, drone, and quadruped platforms. We provide over 128,000 objects and 22,000 validated referring expressions across diverse outdoor scenes -- 10x larger than existing datasets. We develop a scalable annotation pipeline combining vision-language model prompting with human verification to ensure high-quality spatial grounding. To support cross-platform learning, we propose platform-aware normalization and cross-modal alignment techniques, and establish benchmark protocols for in-domain and cross-platform evaluations. Our findings reveal significant performance gaps, highlighting the challenges and opportunities of generalizable 3D grounding. The 3EED dataset and benchmark toolkit are released to advance future research in language-driven 3D embodied perception. 

**Abstract (ZH)**: 3EED：多平台多模态3D语义接地基准istiketi，在开放世界环境中，3D视觉定位是使体现代理能够定位语言所指对象的关键。然而，现有的基准测试主要局限于室内场景、单一平台约束和小规模数据。我们引入了3EED，这是一个多平台、多模态的3D语义接地基准测试，包含来自车辆、无人机和四足机器人平台的RGB和LiDAR数据。我们提供了超过128,000个对象和22,000个验证的引用表达式，涵盖多种户外场景——规模比现有数据集大10倍。我们开发了一种可扩展的注释流程，结合视觉-语言模型提示与人工验证，以确保高质量的空间定位。为支持跨平台学习，我们提出了平台感知规范化和跨模态对齐技术，并建立了域内和跨平台评估基准。我们的研究结果揭示了显著的性能差距，突显了通用3D语义接地的挑战与机遇。3EED数据集和基准工具包的发布旨在推动未来由语言驱动的3D体现感知研究。 

---
# EREBUS: End-to-end Robust Event Based Underwater Simulation 

**Title (ZH)**: EREBUS:端到端鲁棒基于事件的水下模拟 

**Authors**: Hitesh Kyatham, Arjun Suresh, Aadi Palnitkar, Yiannis Aloimonos  

**Link**: [PDF](https://arxiv.org/pdf/2511.01381)  

**Abstract**: The underwater domain presents a vast array of challenges for roboticists and computer vision researchers alike, such as poor lighting conditions and high dynamic range scenes. In these adverse conditions, traditional vision techniques struggle to adapt and lead to suboptimal performance. Event-based cameras present an attractive solution to this problem, mitigating the issues of traditional cameras by tracking changes in the footage on a frame-by-frame basis. In this paper, we introduce a pipeline which can be used to generate realistic synthetic data of an event-based camera mounted to an AUV (Autonomous Underwater Vehicle) in an underwater environment for training vision models. We demonstrate the effectiveness of our pipeline using the task of rock detection with poor visibility and suspended particulate matter, but the approach can be generalized to other underwater tasks. 

**Abstract (ZH)**: 水下环境为机器人学家和计算机视觉研究人员带来了诸多挑战，如光线条件差和高动态范围场景。在这些不利条件下，传统的视觉技术难以适应并导致性能不佳。基于事件的相机通过逐帧跟踪视频中的变化提供了一种有吸引力的解决方案，从而缓解了传统相机的问题。本文介绍了一种管道，可用于生成安装在自主水下车辆(AUV)上的基于事件的相机在水下环境中具有真实感的合成数据，以训练视觉模型。我们使用贫视域和悬浮颗粒物情况下的岩石检测任务来证明我们管道的有效性，但该方法可以 generalized 至其他水下任务。 

---
# Bootstrap Off-policy with World Model 

**Title (ZH)**: 基于模型的离策 Bootstrap Off-policy with World Model 

**Authors**: Guojian Zhan, Likun Wang, Xiangteng Zhang, Jiaxin Gao, Masayoshi Tomizuka, Shengbo Eben Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.00423)  

**Abstract**: Online planning has proven effective in reinforcement learning (RL) for improving sample efficiency and final performance. However, using planning for environment interaction inevitably introduces a divergence between the collected data and the policy's actual behaviors, degrading both model learning and policy improvement. To address this, we propose BOOM (Bootstrap Off-policy with WOrld Model), a framework that tightly integrates planning and off-policy learning through a bootstrap loop: the policy initializes the planner, and the planner refines actions to bootstrap the policy through behavior alignment. This loop is supported by a jointly learned world model, which enables the planner to simulate future trajectories and provides value targets to facilitate policy improvement. The core of BOOM is a likelihood-free alignment loss that bootstraps the policy using the planner's non-parametric action distribution, combined with a soft value-weighted mechanism that prioritizes high-return behaviors and mitigates variability in the planner's action quality within the replay buffer. Experiments on the high-dimensional DeepMind Control Suite and Humanoid-Bench show that BOOM achieves state-of-the-art results in both training stability and final performance. The code is accessible at this https URL. 

**Abstract (ZH)**: 基于世界模型的Bootstrap离策略规划方法（BOOM）：提高训练稳定性和最终性能 

---
# Pelican-VL 1.0: A Foundation Brain Model for Embodied Intelligence 

**Title (ZH)**: Pelican-VL 1.0: 一个基础脑模型用于体现智能 

**Authors**: Yi Zhang, Che Liu, Xiancong Ren, Hanchu Ni, Shuai Zhang, Zeyuan Ding, Jiayu Hu, Hanzhe Shan, Zhenwei Niu, Zhaoyang Liu, Yue Zhao, Junbo Qi, Qinfan Zhang, Dengjie Li, Yidong Wang, Jiachen Luo, Yong Dai, Jian Tang, Xiaozhu Ju  

**Link**: [PDF](https://arxiv.org/pdf/2511.00108)  

**Abstract**: This report presents Pelican-VL 1.0, a new family of open-source embodied brain models with parameter scales ranging from 7 billion to 72 billion. Our explicit mission is clearly stated as: To embed powerful intelligence into various embodiments. Pelican-VL 1.0 is currently the largest-scale open-source embodied multimodal brain model. Its core advantage lies in the in-depth integration of data power and intelligent adaptive learning mechanisms. Specifically, metaloop distilled a high-quality dataset from a raw dataset containing 4+ billion tokens. Pelican-VL 1.0 is trained on a large-scale cluster of 1000+ A800 GPUs, consuming over 50k+ A800 GPU-hours per checkpoint. This translates to a 20.3% performance uplift from its base model and outperforms 100B-level open-source counterparts by 10.6%, placing it on par with leading proprietary systems on well-known embodied benchmarks. We establish a novel framework, DPPO (Deliberate Practice Policy Optimization), inspired by human metacognition to train Pelican-VL 1.0. We operationalize this as a metaloop that teaches the AI to practice deliberately, which is a RL-Refine-Diagnose-SFT loop. 

**Abstract (ZH)**: Pelican-VL 1.0: 一种新型开源 embodiable 大型脑模型系列，参数规模从 70 亿到 720 亿 

---
# Self-Improving Vision-Language-Action Models with Data Generation via Residual RL 

**Title (ZH)**: 基于残差强化学习的数据生成改进视觉-语言-行动模型 

**Authors**: Wenli Xiao, Haotian Lin, Andy Peng, Haoru Xue, Tairan He, Yuqi Xie, Fengyuan Hu, Jimmy Wu, Zhengyi Luo, Linxi "Jim" Fan, Guanya Shi, Yuke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00091)  

**Abstract**: Supervised fine-tuning (SFT) has become the de facto post-training strategy for large vision-language-action (VLA) models, but its reliance on costly human demonstrations limits scalability and generalization. We propose Probe, Learn, Distill (PLD), a three-stage plug-and-play framework that improves VLAs through residual reinforcement learning (RL) and distribution-aware data collection. In Stage 1, we train lightweight residual actors to probe failure regions of the VLA generalist. In Stage 2, we use a hybrid rollout scheme that aligns collected trajectories with the generalist's deployment distribution while capturing recovery behaviors. In Stage 3, we distill the curated trajectories back into the generalist with standard SFT. PLD achieves near-saturated 99% task success on LIBERO, over 50% gains in SimplerEnv, and 100% success on real-world Franka and YAM arm manipulation tasks. Ablations show that residual probing and distribution-aware replay are key to collecting deployment-aligned data that improves both seen and unseen tasks, offering a scalable path toward self-improving VLA models. 

**Abstract (ZH)**: Probe, Learn, Distill (PLD): 一种通过残差强化学习和分布感知数据收集改进多模态视觉-语言-动作模型的三阶段插件式框架 

---
# World Simulation with Video Foundation Models for Physical AI 

**Title (ZH)**: 基于视频基础模型的物理AI世界仿真 

**Authors**: NVIDIA, Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, Prithvijit Chattopadhyay, Mike Chen, Yongxin Chen, Yu Chen, Shuai Cheng, Yin Cui, Jenna Diamond, Yifan Ding, Jiaojiao Fan, Linxi Fan, Liang Feng, Francesco Ferroni, Sanja Fidler, Xiao Fu, Ruiyuan Gao, Yunhao Ge, Jinwei Gu, Aryaman Gupta, Siddharth Gururani, Imad El Hanafi, Ali Hassani, Zekun Hao, Jacob Huffman, Joel Jang, Pooya Jannaty, Jan Kautz, Grace Lam, Xuan Li, Zhaoshuo Li, Maosheng Liao, Chen-Hsuan Lin, Tsung-Yi Lin, Yen-Chen Lin, Huan Ling, Ming-Yu Liu, Xian Liu, Yifan Lu, Alice Luo, Qianli Ma, Hanzi Mao, Kaichun Mo, Seungjun Nah, Yashraj Narang, Abhijeet Panaskar, Lindsey Pavao, Trung Pham, Morteza Ramezanali, Fitsum Reda, Scott Reed, Xuanchi Ren, Haonan Shao, Yue Shen, Stella Shi, Shuran Song, Bartosz Stefaniak, Shangkun Sun, Shitao Tang, Sameena Tasmeen, Lyne Tchapmi, Wei-Cheng Tseng, Jibin Varghese, Andrew Z. Wang, Hao Wang, Haoxiang Wang, Heng Wang, Ting-Chun Wang, Fangyin Wei, Jiashu Xu, Dinghao Yang, Xiaodong Yang, Haotian Ye, Seonghyeon Ye, Xiaohui Zeng, Jing Zhang, Qinsheng Zhang, Kaiwen Zheng, Andrew Zhu, Yuke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00062)  

**Abstract**: We introduce [Cosmos-Predict2.5], the latest generation of the Cosmos World Foundation Models for Physical AI. Built on a flow-based architecture, [Cosmos-Predict2.5] unifies Text2World, Image2World, and Video2World generation in a single model and leverages [Cosmos-Reason1], a Physical AI vision-language model, to provide richer text grounding and finer control of world simulation. Trained on 200M curated video clips and refined with reinforcement learning-based post-training, [Cosmos-Predict2.5] achieves substantial improvements over [Cosmos-Predict1] in video quality and instruction alignment, with models released at 2B and 14B scales. These capabilities enable more reliable synthetic data generation, policy evaluation, and closed-loop simulation for robotics and autonomous systems. We further extend the family with [Cosmos-Transfer2.5], a control-net style framework for Sim2Real and Real2Real world translation. Despite being 3.5$\times$ smaller than [Cosmos-Transfer1], it delivers higher fidelity and robust long-horizon video generation. Together, these advances establish [Cosmos-Predict2.5] and [Cosmos-Transfer2.5] as versatile tools for scaling embodied intelligence. To accelerate research and deployment in Physical AI, we release source code, pretrained checkpoints, and curated benchmarks under the NVIDIA Open Model License at this https URL and this https URL. We hope these open resources lower the barrier to adoption and foster innovation in building the next generation of embodied intelligence. 

**Abstract (ZH)**: 我们介绍[Cosmos-Predict2.5]，这是Cosmos World基金会模型的最新一代，适用于物理AI。基于流化架构，[Cosmos-Predict2.5]将Text2World、Image2World和Video2World生成统一在一个模型中，并利用[Cosmos-Reason1]物理AI视觉语言模型，提供更丰富的文本语义关联和更精细的世界模拟控制。该模型在200M精编视频片段上进行训练，并通过基于强化学习的后训练 refinement 进行优化，相比于[Cosmos-Predict1]在视频质量和指令对齐方面取得了显著改进，模型规模分别为2B和14B。这些能力使得合成数据生成、政策评估和闭环仿真方法在机器人技术和自主系统中更加可靠。我们还推出了[Cosmos-Transfer2.5]，这是一种适用于Sim2Real和Real2Real世界转换的控制网风格框架。尽管其大小仅为[Cosmos-Transfer1]的3.5倍，却提供了更高的保真度和更 robust 的长时序视频生成能力。这些进步确立了[Cosmos-Predict2.5]和[Cosmos-Transfer2.5]作为扩展具身智能的多功能工具的地位。为了加速物理AI领域的研究和部署，我们在以下链接发布源代码、预训练权重和精心整理的基准测试：此https URL和此https URL。我们希望这些开放资源能够降低采用门槛，并促进构建下一代具身智能的创新。 

---
# Simulating Environments with Reasoning Models for Agent Training 

**Title (ZH)**: 基于推理模型的环境模拟agent训练 

**Authors**: Yuetai Li, Huseyin A Inan, Xiang Yue, Wei-Ning Chen, Lukas Wutschitz, Janardhan Kulkarni, Radha Poovendran, Robert Sim, Saravan Rajmohan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01824)  

**Abstract**: LLM agents excel in compact environments requiring deep reasoning but remain brittle when operating in broader, more complex contexts that demand robustness across diverse tools and schemas. Building bespoke environments for training is heavy, brittle, and limits progress. In this paper, we demonstrate that LLMs can simulate realistic environment feedback without access to actual testbed data or APIs. Inspired by this capability, we propose two frameworks: Simia-SFT, a pipeline that synthesizes SFT data by amplifying small seed sets into diverse trajectories in an environment-agnostic manner, and Simia-RL, a framework that enables RL training without real environment implementations through LLM-simulated feedback. Fine-tuning open models yields consistent improvements across multiple benchmarks, surpassing GPT-4o and approaching o4-mini on $\tau^2$-Bench. Together, Simia-SFT and Simia-RL enable scalable agent training without environment engineering, replacing heavy and brittle implementations with flexible LLM-based simulation. 

**Abstract (ZH)**: LLM代理在紧凑的环境要求深入推理时表现出色，但在更广泛、更复杂、要求跨多种工具和模式的鲁棒性的情境下则显得脆弱。构建定制的训练环境既费时又脆弱，限制了进展。在本文中，我们证明LLM可以在不访问实际测试床数据或API的情况下模拟现实环境反馈。受此能力启发，我们提出了两种框架：Simia-SFT，一种通过在环境无关的方式中放大种子集以生成多种轨迹来合成SFT数据的流水线；以及Simia-RL，一种通过LLM模拟反馈实现无需真实环境实现的强化学习培训的框架。对开源模型进行微调在多个基准测试上表现出一致的改进，超越了GPT-4o，并接近o4-mini在$\tau^2$-Bench上的表现。通过Simia-SFT和Simia-RL，可以实现无需环境工程的可扩展代理训练，用灵活的基于LLM的模拟替代了繁重且脆弱的实现。 

---
# Learning to Seek Evidence: A Verifiable Reasoning Agent with Causal Faithfulness Analysis 

**Title (ZH)**: 学习寻找证据：一种基于因果忠实性分析的可验证推理代理 

**Authors**: Yuhang Huang, Zekai Lin, Fan Zhong, Lei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01425)  

**Abstract**: Explanations for AI models in high-stakes domains like medicine often lack verifiability, which can hinder trust. To address this, we propose an interactive agent that produces explanations through an auditable sequence of actions. The agent learns a policy to strategically seek external visual evidence to support its diagnostic reasoning. This policy is optimized using reinforcement learning, resulting in a model that is both efficient and generalizable. Our experiments show that this action-based reasoning process significantly improves calibrated accuracy, reducing the Brier score by 18\% compared to a non-interactive baseline. To validate the faithfulness of the agent's explanations, we introduce a causal intervention method. By masking the visual evidence the agent chooses to use, we observe a measurable degradation in its performance ($\Delta$Brier=+0.029), confirming that the evidence is integral to its decision-making process. Our work provides a practical framework for building AI systems with verifiable and faithful reasoning capabilities. 

**Abstract (ZH)**: 高风险领域如医疗中AI模型的解释往往缺乏可验证性，这可能妨碍信任。为此，我们提出一个交互式代理，通过可审计的操作序列生成解释。该代理学习一个策略以战略性地寻求外部视觉证据来支持其诊断推理。此策略使用强化学习进行优化，从而生成一个既高效又可泛化的模型。我们的实验表明，基于行动的推理过程显著提高了校准准确性，与非交互式基线相比，Brier评分降低了18%。为了验证代理解释的忠实性，我们引入了一种因果干预方法。通过遮蔽代理选择使用的视觉证据，我们观察到其性能出现可衡量的下降（$\Delta$Brier=+0.029），这确认了证据对于其决策过程的重要性。我们的工作提供了一个实用的框架，用于构建具有可验证和忠实推理能力的AI系统。 

---
# Modulation of temporal decision-making in a deep reinforcement learning agent under the dual-task paradigm 

**Title (ZH)**: 双任务 paradigm 下深度强化学习代理的时间决策调控 

**Authors**: Amrapali Pednekar, Álvaro Garrido-Pérez, Yara Khaluf, Pieter Simoens  

**Link**: [PDF](https://arxiv.org/pdf/2511.01415)  

**Abstract**: This study explores the interference in temporal processing within a dual-task paradigm from an artificial intelligence (AI) perspective. In this context, the dual-task setup is implemented as a simplified version of the Overcooked environment with two variations, single task (T) and dual task (T+N). Both variations involve an embedded time production task, but the dual task (T+N) additionally involves a concurrent number comparison task. Two deep reinforcement learning (DRL) agents were separately trained for each of these tasks. These agents exhibited emergent behavior consistent with human timing research. Specifically, the dual task (T+N) agent exhibited significant overproduction of time relative to its single task (T) counterpart. This result was consistent across four target durations. Preliminary analysis of neural dynamics in the agents' LSTM layers did not reveal any clear evidence of a dedicated or intrinsic timer. Hence, further investigation is needed to better understand the underlying time-keeping mechanisms of the agents and to provide insights into the observed behavioral patterns. This study is a small step towards exploring parallels between emergent DRL behavior and behavior observed in biological systems in order to facilitate a better understanding of both. 

**Abstract (ZH)**: 本研究从人工智能视角探索双重任务范式中时间处理的干扰现象。在此背景下，双重任务设置被实现为Overcooked环境的简化版本，包括单任务（T）和双重任务（T+N）两种变体。两种变体均包含嵌入的时间生产任务，而双重任务（T+N）还额外包含一个并发的数字比较任务。分别为这两种任务分别训练了两个深度强化学习（DRL）代理。这些代理表现出与人类时间感知研究一致的新兴行为。具体而言，双重任务（T+N）代理相对于单任务（T）的对应者表现出显著的时间过长现象。该结果在四个目标持续时间中均一致。对代理LSTM层中的神经动力学初步分析未发现明确的专用计时器证据。因此，需要进一步研究以更好地理解代理的时间保持机制，并提供对观察到的行为模式的见解。本研究是探索新兴DRL行为与生物系统中观察到的行为之间类比关系的一个小步骤，旨在促进对两者的更好地理解。 

---
# Active Thinking Model: A Goal-Directed Self-Improving Framework for Real-World Adaptive Intelligence 

**Title (ZH)**: 主动思考模型：一种目标导向的自我提升框架，用于现实世界中的自适应智能 

**Authors**: Hong Su  

**Link**: [PDF](https://arxiv.org/pdf/2511.00758)  

**Abstract**: Real-world artificial intelligence (AI) systems are increasingly required to operate autonomously in dynamic, uncertain, and continuously changing environments. However, most existing AI models rely on predefined objectives, static training data, and externally supplied feedback, which restrict their ability to adapt, reflect, and improve independently. In this paper, we propose the Active Thinking Model (ATM)- a unified cognitive framework that integrates goal reasoning, dynamic task generation, and self-reflective learning into an adaptive architecture. Unlike conventional systems that passively execute fixed procedures, ATM actively evaluates its performance through logical reasoning and environmental indicators, reuses effective methods to solve new problems, and generates novel strategies for unseen situations via a continuous self-improvement loop. A mathematically grounded theoretical analysis demonstrates that ATM can autonomously evolve from suboptimal to optimal behavior without external supervision and maintain bounded tracking regret under changing environmental conditions. 

**Abstract (ZH)**: 人工智能系统在动态、不确定且持续变化环境中的自主运行及其自适应认知框架：Active Thinking Model 

---
# How Far Are Surgeons from Surgical World Models? A Pilot Study on Zero-shot Surgical Video Generation with Expert Assessment 

**Title (ZH)**: 外科医生与手术世界模型的距离有多远？一项基于专家评估的零样本手术视频生成 pilot 研究 

**Authors**: Zhen Chen, Qing Xu, Jinlin Wu, Biao Yang, Yuhao Zhai, Geng Guo, Jing Zhang, Yinlu Ding, Nassir Navab, Jiebo Luo  

**Link**: [PDF](https://arxiv.org/pdf/2511.01775)  

**Abstract**: Foundation models in video generation are demonstrating remarkable capabilities as potential world models for simulating the physical world. However, their application in high-stakes domains like surgery, which demand deep, specialized causal knowledge rather than general physical rules, remains a critical unexplored gap. To systematically address this challenge, we present SurgVeo, the first expert-curated benchmark for video generation model evaluation in surgery, and the Surgical Plausibility Pyramid (SPP), a novel, four-tiered framework tailored to assess model outputs from basic appearance to complex surgical strategy. On the basis of the SurgVeo benchmark, we task the advanced Veo-3 model with a zero-shot prediction task on surgical clips from laparoscopic and neurosurgical procedures. A panel of four board-certified surgeons evaluates the generated videos according to the SPP. Our results reveal a distinct "plausibility gap": while Veo-3 achieves exceptional Visual Perceptual Plausibility, it fails critically at higher levels of the SPP, including Instrument Operation Plausibility, Environment Feedback Plausibility, and Surgical Intent Plausibility. This work provides the first quantitative evidence of the chasm between visually convincing mimicry and causal understanding in surgical AI. Our findings from SurgVeo and the SPP establish a crucial foundation and roadmap for developing future models capable of navigating the complexities of specialized, real-world healthcare domains. 

**Abstract (ZH)**: 基于视频生成的基石模型在手术领域的潜在世界模型评估中表现出色，但在高风险领域如手术所需的深刻专业因果知识方面仍存在关键未开发的缺口。为系统地应对这一挑战，我们提出了SurgVeo——首个针对手术视频生成模型评估的专家 compilene 基准，并提出了手术合理性金字塔（Surgical Plausibility Pyramid，SPP）——一种新颖的四层次框架，用于从基本外观到复杂手术策略评估模型输出。基于SurgVeo基准，我们让先进的Veo-3模型在内窥镜手术和神经外科手术片段上进行零样本预测任务。四位认证外科医生根据SPP评估生成的视频。我们的结果显示一个显著的“合理性缺口”：尽管Veo-3在视觉感知合理性方面表现出色，但在SPP的较高层次上却表现不佳，包括器械操作合理性、环境反馈合理性和手术意图合理性。本文提供了手术AI在视觉逼真模仿与因果理解之间差距的首个定量证据。基于SurgVeo和SPP的研究结果建立了未来模型发展的关键基础和路线图，以应对专业且实际的医疗保健领域中的复杂性。 

---
# Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI 

**Title (ZH)**: 开放角色训练：通过宪法AI塑造AI助手的人设 

**Authors**: Sharan Maiya, Henning Bartsch, Nathan Lambert, Evan Hubinger  

**Link**: [PDF](https://arxiv.org/pdf/2511.01689)  

**Abstract**: The character of the "AI assistant" persona generated by modern chatbot large language models influences both surface-level behavior and apparent values, beliefs, and ethics. These all affect interaction quality, perceived intelligence, and alignment with both developer and user intentions. The shaping of this persona, known as character training, is a critical component of industry post-training, yet remains effectively unstudied in the academic literature. We introduce the first open implementation of character training, leveraging Constitutional AI and a new data pipeline using synthetic introspective data to shape the assistant persona in a more effective and controlled manner than alternatives such as constraining system prompts or activation steering. Specifically, we fine-tune three popular open-weights models using 11 example personas, such as humorous, deeply caring, or even malevolent. To track the effects of our approach, we introduce a method which analyzes revealed preferences, uncovering clear and holistic changes in character. We find these changes are more robust to adversarial prompting than the above two alternatives, while also leading to more coherent and realistic generations. Finally, we demonstrate this fine-tuning has little to no effect on general capabilities as measured by common benchmarks. We describe and open-source our full post-training method, the implementation of which can be found at this https URL. 

**Abstract (ZH)**: 现代聊天机器人大型语言模型生成的“AI助手”人设特征影响表面行为和显性价值观、信念和伦理观点，这些因素都会影响交互质量、感知智能水平以及与开发人员和用户意图的契合度。人设塑造，即角色训练，是行业后培训中的一个关键组成部分，但在学术文献中仍基本未被研究。我们介绍了第一个开放的角色训练实现，利用宪法AI和一种新的数据管道，使用合成的内省数据来更有效地、更可控地塑造助手人设，这比通过约束系统提示或激活引导等替代方法更为有效。具体来说，我们针对幽默的、深切关怀的或甚至恶意的人设进行了三个流行开源模型的微调，共使用了11个人设示例。为了跟踪我们方法的效果，我们提出了一种分析揭示偏好（revealed preferences）的方法，揭示了人设的明显而全面的变化。我们发现，这些变化比上述两种替代方法更能抵抗对抗性提示，同时也导致了更连贯和真实的生成结果。最后，我们展示了这种微调在通用能力方面基本没有影响，通过常见基准进行测量。我们描述并开源了我们的整个后培训方法，相关信息可在以下链接获取：this https URL。 

---
# The Ghost in the Keys: A Disklavier Demo for Human-AI Musical Co-Creativity 

**Title (ZH)**: 琴键中的幽灵：人类-人工智能音乐共创演示 

**Authors**: Louis Bradshaw, Alexander Spangher, Stella Biderman, Simon Colton  

**Link**: [PDF](https://arxiv.org/pdf/2511.01663)  

**Abstract**: While generative models for music composition are increasingly capable, their adoption by musicians is hindered by text-prompting, an asynchronous workflow disconnected from the embodied, responsive nature of instrumental performance. To address this, we introduce Aria-Duet, an interactive system facilitating a real-time musical duet between a human pianist and Aria, a state-of-the-art generative model, using a Yamaha Disklavier as a shared physical interface. The framework enables a turn-taking collaboration: the user performs, signals a handover, and the model generates a coherent continuation performed acoustically on the piano. Beyond describing the technical architecture enabling this low-latency interaction, we analyze the system's output from a musicological perspective, finding the model can maintain stylistic semantics and develop coherent phrasal ideas, demonstrating that such embodied systems can engage in musically sophisticated dialogue and open a promising new path for human-AI co-creation. 

**Abstract (ZH)**: 基于Aria-Duet的实时人机即兴合奏系统：一种将先进生成模型融入乐器表演的交互方式 

---
# DINO-MX: A Modular & Flexible Framework for Self-Supervised Learning 

**Title (ZH)**: DINO-MX：一种模块化且灵活的自监督学习框架 

**Authors**: Mahmut Selman Gokmen, Cody Bumgardner  

**Link**: [PDF](https://arxiv.org/pdf/2511.01610)  

**Abstract**: Vision Foundation Models (VFMs) have advanced representation learning through self-supervised methods. However, existing training pipelines are often inflexible, domain-specific, or computationally expensive, which limits their usability across different domains and resource settings. DINO-MX is a modular and extensible training framework that combines the core principles of DINO, DINOv2 and DINOv3 within a unified configuration-driven system. It supports a variety of transformer-based architectures and is fully compatible with the Hugging Face ecosystem. The framework includes multiple training strategies such as low-rank adaptation (LoRA), layer freezing, and knowledge distillation, along with support for distributed training through both Distributed Data Parallel (DDP) and Fully Sharded Data Parallel (FSDP). DINO-MX is designed to work with both natural and specialized data types, including single- and multi-channel images. Experimental results on diverse datasets show that DINO-MX achieves competitive performance while significantly reducing computational costs. Additionally, it offers interpretability tools and a label-guided data augmentation method that improves attention-based localization without the need for extra detection or segmentation heads. DINO-MX provides a reproducible and scalable foundation for developing, adapting, and benchmarking self-supervised vision models across a range of research and real-world applications. 

**Abstract (ZH)**: DINO-MX：模块化可扩展的自监督视觉模型训练框架 

---
# Driving scenario generation and evaluation using a structured layer representation and foundational models 

**Title (ZH)**: 基于结构层表示和基础模型的驾驶场景生成与评估 

**Authors**: Arthur Hubert, Gamal Elghazaly, Raphaël Frank  

**Link**: [PDF](https://arxiv.org/pdf/2511.01541)  

**Abstract**: Rare and challenging driving scenarios are critical for autonomous vehicle development. Since they are difficult to encounter, simulating or generating them using generative models is a popular approach. Following previous efforts to structure driving scenario representations in a layer model, we propose a structured five-layer model to improve the evaluation and generation of rare scenarios. We use this model alongside large foundational models to generate new driving scenarios using a data augmentation strategy. Unlike previous representations, our structure introduces subclasses and characteristics for every agent of the scenario, allowing us to compare them using an embedding specific to our layer-model. We study and adapt two metrics to evaluate the relevance of a synthetic dataset in the context of a structured representation: the diversity score estimates how different the scenarios of a dataset are from one another, while the originality score calculates how similar a synthetic dataset is from a real reference set. This paper showcases both metrics in different generation setup, as well as a qualitative evaluation of synthetic videos generated from structured scenario descriptions. The code and extended results can be found at this https URL. 

**Abstract (ZH)**: 罕见且具有挑战性的驾驶场景对于自动驾驶车辆的发展至关重要。由于这些场景难以遇到，使用生成模型进行模拟或生成是一种流行的方法。在先前尝试将驾驶场景表示为分层模型的努力基础上，我们提出了一个结构化的五层模型，以改进稀有场景的评估和生成。我们使用此模型和大型基础模型，基于数据扩增策略生成新的驾驶场景。与之前的表示不同，我们的结构为场景中的每个代理引入了子类和特性，从而使我们可以使用特定于分层模型的嵌入进行比较。我们研究并适应了两种指标来评估合成数据集在结构化表示下的相关性：多样性分数估计数据集中场景的差异性，而原创性分数计算合成数据集与真实参考集的相似性。本文展示了这两种指标在不同生成设置下的应用，并对从结构化场景描述生成的合成视频进行了定性评估。代码和扩展结果可在如下链接找到：this https URL。 

---
# Continual Learning, Not Training: Online Adaptation For Agents 

**Title (ZH)**: 不间断学习，而非训练：代理的在线适应 

**Authors**: Aman Jaglan, Jarrod Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2511.01093)  

**Abstract**: Continual Learning (CL) methods have traditionally focused on mitigating catastrophic forgetting through gradient-based retraining, an approach ill-suited for deployed agents that must adapt in real time. We introduce our Adaptive Teaching and Learning System (ATLAS), a dual-agent architecture that decouples reasoning (Teacher) from execution (Student) and incorporates a persistent learning memory that stores distilled guidance from experience. This informs the orchestration layer, enabling the system to dynamically adjust its operational strategies, such as supervision level or initial plan selection, at inference time. In doing so, ATLAS achieves gradient-free continual learning, shifting the locus of adaptation from model parameters to system-level orchestration. We formulate this as a system-centric paradigm for continual learning, where the objective is adaptive efficiency: maximizing task success while minimizing computational cost through inference-time orchestration rather than parameter updates. Evaluated on Microsoft's ExCyTIn-Bench, an open-source benchmark simulating complex cyberthreat investigation, ATLAS achieves 54.1% success with GPT-5-mini as its Student, outperforming the larger GPT-5 (High) by 13% while reducing cost by 86%. Cross-incident validation demonstrates generalization: frozen pamphlets from Incident #5 improve accuracy from 28% to 41% with zero retraining, while shifting output composition from verbose exploration to structured reasoning. Together, these findings establish gradient-free continual learning as a viable path toward adaptive, deployable AI systems and provide causally annotated traces valuable for training explicit world models. 

**Abstract (ZH)**: 持续学习（CL）方法传统上侧重于通过基于梯度的重新训练缓解灾难性遗忘，但这种方式不适合必须实时适应的部署代理。我们介绍了自适应教学与学习系统（ATLAS），这是一种双代理架构，将推理（教师）与执行（学生）脱钩，并集成了持久学习记忆，存储了经验提炼的指导。这为交响 orchestration 层提供了信息，使系统能够动态调整其操作策略，如监督级别或初始计划选择，在推断时。通过这种方式，ATLAS 实现了无需梯度的持续学习，将适应的重心从模型参数转移到系统级交响。我们将其表述为以系统为中心的持续学习范式，其中目标是适应性效率：通过对推断时的交响控制而非参数更新来最大化任务成功并最小化计算成本。在微软的 ExCyTIn-Bench 上评估，ATLAS 使用 GPT-5-mini 作为学生模型，成功率达到 54.1%，比更大的 GPT-5（High）高出 13%，成本降低 86%。跨事件验证显示了泛化能力：冻结的第 5 起事件手册将准确率从 28% 提高到 41%，而无需重新训练，同时从冗长的探索转向结构化推理。这些发现共同确立了无需梯度的持续学习是实现适应性、可部署 AI 系统的一种可行路径，并提供因果注释跟踪，对于训练显式世界模型非常有价值。 

---
# Learning with Category-Equivariant Representations for Human Activity Recognition 

**Title (ZH)**: 基于类别等变表示的人体活动识别学习 

**Authors**: Yoshihiro Maruyama  

**Link**: [PDF](https://arxiv.org/pdf/2511.00900)  

**Abstract**: Human activity recognition is challenging because sensor signals shift with context, motion, and environment; effective models must therefore remain stable as the world around them changes. We introduce a categorical symmetry-aware learning framework that captures how signals vary over time, scale, and sensor hierarchy. We build these factors into the structure of feature representations, yielding models that automatically preserve the relationships between sensors and remain stable under realistic distortions such as time shifts, amplitude drift, and device orientation changes. On the UCI Human Activity Recognition benchmark, this categorical symmetry-driven design improves out-of-distribution accuracy by approx. 46 percentage points (approx. 3.6x over the baseline), demonstrating that abstract symmetry principles can translate into concrete performance gains in everyday sensing tasks via category-equivariant representation theory. 

**Abstract (ZH)**: 基于类别对称性的活动识别学习框架 

---
# KFCPO: Kronecker-Factored Approximated Constrained Policy Optimization 

**Title (ZH)**: KFCPO：克罗内克分解近似约束策略优化 

**Authors**: Joonyoung Lim, Younghwan Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2511.00880)  

**Abstract**: We propose KFCPO, a novel Safe Reinforcement Learning (Safe RL) algorithm that combines scalable Kronecker-Factored Approximate Curvature (K-FAC) based second-order policy optimization with safety-aware gradient manipulation. KFCPO leverages K-FAC to perform efficient and stable natural gradient updates by approximating the Fisher Information Matrix (FIM) in a layerwise, closed form manner, avoiding iterative approximation overheads. To address the tradeoff between reward maximization and constraint satisfaction, we introduce a margin aware gradient manipulation mechanism that adaptively adjusts the influence of reward and cost gradients based on the agent's proximity to safety boundaries. This method blends gradients using a direction sensitive projection, eliminating harmful interference and avoiding abrupt changes caused by fixed hard thresholds. Additionally, a minibatch level KL rollback strategy is adopted to ensure trust region compliance and to prevent destabilizing policy shifts. Experiments on Safety Gymnasium using OmniSafe show that KFCPO achieves 10.3% to 50.2% higher average return across environments compared to the best baseline that respected the safety constraint, demonstrating superior balance of safety and performance. 

**Abstract (ZH)**: 我们提出了一种新型安全强化学习算法KFCPO，该算法结合了基于Kronecker-Factored Approximate Curvature (K-FAC)的大规模二阶策略优化方法，并且包含一种安全意识梯度操纵机制。KFCPO 利用 K-FAC 通过逐层闭式近似费歇信息矩阵 （FIM） 来执行高效稳定的自然梯度更新，避免了迭代近似带来的开销。为了解决奖励最大化与约束满足之间的权衡问题，我们引入了一种基于智能边界的梯度操纵机制，该机制根据智能体接近安全边界的程度，自适应地调整奖励梯度和成本梯度的影响。该方法通过方向敏感的投影来混合梯度，消除了有害干扰，并避免了由固定硬阈值引起的急剧变化。此外，采用小批量级的 KL 回滚策略以确保信任区域合规，并防止不稳定策略变化。在使用 OmniSafe 对 Safety Gymnasium 进行的实验中，KFCPO 在各环境中的平均回报比严格遵守安全约束的最佳基线算法高出 10.3% 至 50.2%，展示了其在安全与性能方面的优越平衡。 

---
# Logic-informed reinforcement learning for cross-domain optimization of large-scale cyber-physical systems 

**Title (ZH)**: 逻辑驱动的强化学习在大规模网络物理系统跨域优化中的应用 

**Authors**: Guangxi Wan, Peng Zeng, Xiaoting Dong, Chunhe Song, Shijie Cui, Dong Li, Qingwei Dong, Yiyang Liu, Hongfei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2511.00806)  

**Abstract**: Cyber-physical systems (CPS) require the joint optimization of discrete cyber actions and continuous physical parameters under stringent safety logic constraints. However, existing hierarchical approaches often compromise global optimality, whereas reinforcement learning (RL) in hybrid action spaces often relies on brittle reward penalties, masking, or shielding and struggles to guarantee constraint satisfaction. We present logic-informed reinforcement learning (LIRL), which equips standard policy-gradient algorithms with projection that maps a low-dimensional latent action onto the admissible hybrid manifold defined on-the-fly by first-order logic. This guarantees feasibility of every exploratory step without penalty tuning. Experimental evaluations have been conducted across multiple scenarios, including industrial manufacturing, electric vehicle charging stations, and traffic signal control, in all of which the proposed method outperforms existing hierarchical optimization approaches. Taking a robotic reducer assembly system in industrial manufacturing as an example, LIRL achieves a 36.47\% to 44.33\% reduction at most in the combined makespan-energy objective compared to conventional industrial hierarchical scheduling methods. Meanwhile, it consistently maintains zero constraint violations and significantly surpasses state-of-the-art hybrid-action reinforcement learning baselines. Thanks to its declarative logic-based constraint formulation, the framework can be seamlessly transferred to other domains such as smart transportation and smart grid, thereby paving the way for safe and real-time optimization in large-scale CPS. 

**Abstract (ZH)**: 基于逻辑指导的强化学习（LIRL）：融合离散网络行动与连续物理参数的优化 

---
# Robust Single-Agent Reinforcement Learning for Regional Traffic Signal Control Under Demand Fluctuations 

**Title (ZH)**: 具有需求波动鲁棒性的单Agent强化学习区域交通信号控制 

**Authors**: Qiang Li, Jin Niu, Lina Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00549)  

**Abstract**: Traffic congestion, primarily driven by intersection queuing, significantly impacts urban living standards, safety, environmental quality, and economic efficiency. While Traffic Signal Control (TSC) systems hold potential for congestion mitigation, traditional optimization models often fail to capture real-world traffic complexity and dynamics. This study introduces a novel single-agent reinforcement learning (RL) framework for regional adaptive TSC, circumventing the coordination complexities inherent in multi-agent systems through a centralized decision-making paradigm. The model employs an adjacency matrix to unify the encoding of road network topology, real-time queue states derived from probe vehicle data, and current signal timing parameters. Leveraging the efficient learning capabilities of the DreamerV3 world model, the agent learns control policies where actions sequentially select intersections and adjust their signal phase splits to regulate traffic inflow/outflow, analogous to a feedback control system. Reward design prioritizes queue dissipation, directly linking congestion metrics (queue length) to control actions. Simulation experiments conducted in SUMO demonstrate the model's effectiveness: under inference scenarios with multi-level (10%, 20%, 30%) Origin-Destination (OD) demand fluctuations, the framework exhibits robust anti-fluctuation capability and significantly reduces queue lengths. This work establishes a new paradigm for intelligent traffic control compatible with probe vehicle technology. Future research will focus on enhancing practical applicability by incorporating stochastic OD demand fluctuations during training and exploring regional optimization mechanisms for contingency events. 

**Abstract (ZH)**: 基于单智能体强化学习的区域自适应交通信号控制框架：兼容探针车辆技术的新范式 

---
# On Improvisation and Open-Endedness: Insights for Experiential AI 

**Title (ZH)**: 即兴创作与开放性：体验型AI的洞见 

**Authors**: Botao 'Amber' Hu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00529)  

**Abstract**: Improvisation-the art of spontaneous creation that unfolds moment-to-moment without a scripted outcome-requires practitioners to continuously sense, adapt, and create anew. It is a fundamental mode of human creativity spanning music, dance, and everyday life. The open-ended nature of improvisation produces a stream of novel, unrepeatable moments-an aspect highly valued in artistic creativity. In parallel, open-endedness (OE)-a system's capacity for unbounded novelty and endless "interestingness"-is exemplified in natural or cultural evolution and has been considered "the last grand challenge" in artificial life (ALife). The rise of generative AI now raises the question in computational creativity (CC) research: What makes a "good" improvisation for AI? Can AI learn to improvise in a genuinely open-ended way? In this work-in-progress paper, we report insights from in-depth interviews with 6 experts in improvisation across dance, music, and contact improvisation. We draw systemic connections between human improvisational arts and the design of future experiential AI agents that could improvise alone or alongside humans-or even with other AI agents-embodying qualities of improvisation drawn from practice: active listening (umwelt and awareness), being in the time (mindfulness and ephemerality), embracing the unknown (source of randomness and serendipity), non-judgmental flow (acceptance and dynamical stability, balancing structure and surprise (unpredictable criticality at edge of chaos), imaginative metaphor (synaesthesia and planning), empathy, trust, boundary, and care (mutual theory of mind), and playfulness and intrinsic motivation (maintaining interestingness). 

**Abstract (ZH)**: 即兴创作——一种不依赖预设结果、不断展开的自发创造艺术——要求实践者不断感知、适应并不断重新创造。它是跨越音乐、舞蹈和日常生活的根本性人类创造力模式。即兴创作的开放性特征产生了众多新颖且不可重复的时刻——这是高度受艺术创造力重视的方面。与此同时，开放性的能力在自然或文化进化中得到了体现，并被认为是人工生命领域的“最后一个重大挑战”。随着生成式AI的崛起，现在在计算创造力研究中提出了一个问题：什么样的即兴创作对于AI是“好的”？AI能否以真正的开放性方式进行即兴创作？在本文中，我们通过深度访谈6位舞蹈、音乐及接触即兴领域的专家，报告了关于人类即兴艺术与其未来能够单独或与人类一起即兴的体验式AI代理设计之间系统联系的见解。这些即兴创作的品质包括：积极参与（环境感知和意识）、当下体验（觉知和暂態）、拥抱未知（随机性和偶然性的来源）、非评判性流动（接纳和动态稳定性）、平衡结构与惊喜（混沌边缘的不可预测性）、富有创意的隐喻（联觉与规划）、同理心、信任、边界和关怀（相互的心灵理论）、以及趣味性和内在动机（保持有趣性）。 

---
# Consistently Simulating Human Personas with Multi-Turn Reinforcement Learning 

**Title (ZH)**: 一致地使用多轮强化学习模拟人类个性 

**Authors**: Marwa Abdulhai, Ryan Cheng, Donovan Clay, Tim Althoff, Sergey Levine, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2511.00222)  

**Abstract**: Large Language Models (LLMs) are increasingly used to simulate human users in interactive settings such as therapy, education, and social role-play. While these simulations enable scalable training and evaluation of AI agents, off-the-shelf LLMs often drift from their assigned personas, contradict earlier statements, or abandon role-appropriate behavior. We introduce a unified framework for evaluating and improving persona consistency in LLM-generated dialogue. We define three automatic metrics: prompt-to-line consistency, line-to-line consistency, and Q&A consistency, that capture different types of persona drift and validate each against human annotations. Using these metrics as reward signals, we apply multi-turn reinforcement learning to fine-tune LLMs for three user roles: a patient, a student, and a social chat partner. Our method reduces inconsistency by over 55%, resulting in more coherent and faithful simulated users. 

**Abstract (ZH)**: 大型语言模型（LLMs）在治疗、教育和社会角色扮演等交互设置中越来越多地被用于模拟人类用户。虽然这些模拟使AI代理的大规模训练和评估成为可能，但现成的LLMs往往会偏离其分配的人格，前后矛盾，或放弃适当的角色行为。我们提出了一种统一框架，用于评估和改进LLM生成对话中的人格一致性。我们定义了三个自动度量标准：提示到行的一致性、行到行的一致性和问答一致性，这些度量标准捕捉不同类型的人格漂移，并通过人类注释进行验证。利用这些度量作为奖励信号，我们应用多轮强化学习微调LLM，以模拟三种用户角色：患者、学生和社会聊天伙伴。我们的方法将不一致性降低了超过55%，导致模拟用户更加连贯和忠实。 

---
# LC-Opt: Benchmarking Reinforcement Learning and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers 

**Title (ZH)**: LC-Opt: 评价面向数据中心液体冷却端到端优化的强化学习与自主智能体技术 

**Authors**: Avisek Naug, Antonio Guillen, Vineet Kumar, Scott Greenwood, Wesley Brewer, Sahand Ghorbanpour, Ashwin Ramesh Babu, Vineet Gundecha, Ricardo Luna Gutierrez, Soumyendu Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2511.00116)  

**Abstract**: Liquid cooling is critical for thermal management in high-density data centers with the rising AI workloads. However, machine learning-based controllers are essential to unlock greater energy efficiency and reliability, promoting sustainability. We present LC-Opt, a Sustainable Liquid Cooling (LC) benchmark environment, for reinforcement learning (RL) control strategies in energy-efficient liquid cooling of high-performance computing (HPC) systems. Built on the baseline of a high-fidelity digital twin of Oak Ridge National Lab's Frontier Supercomputer cooling system, LC-Opt provides detailed Modelica-based end-to-end models spanning site-level cooling towers to data center cabinets and server blade groups. RL agents optimize critical thermal controls like liquid supply temperature, flow rate, and granular valve actuation at the IT cabinet level, as well as cooling tower (CT) setpoints through a Gymnasium interface, with dynamic changes in workloads. This environment creates a multi-objective real-time optimization challenge balancing local thermal regulation and global energy efficiency, and also supports additional components like a heat recovery unit (HRU). We benchmark centralized and decentralized multi-agent RL approaches, demonstrate policy distillation into decision and regression trees for interpretable control, and explore LLM-based methods that explain control actions in natural language through an agentic mesh architecture designed to foster user trust and simplify system management. LC-Opt democratizes access to detailed, customizable liquid cooling models, enabling the ML community, operators, and vendors to develop sustainable data center liquid cooling control solutions. 

**Abstract (ZH)**: 可持续液体冷却基准环境LC-Opt：面向高效液体冷却的强化学习控制策略 

---
# Cognitive Alignment in Personality Reasoning: Leveraging Prototype Theory for MBTI Inference 

**Title (ZH)**: 人格推理中的认知对齐：利用原型理论进行MBTI推断 

**Authors**: Haoyuan Li, Yuanbo Tong, Yuchen Li, Zirui Wang, Chunhou Liu, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00115)  

**Abstract**: Personality recognition from text is typically cast as hard-label classification, which obscures the graded, prototype-like nature of human personality judgments. We present ProtoMBTI, a cognitively aligned framework for MBTI inference that operationalizes prototype theory within an LLM-based pipeline. First, we construct a balanced, quality-controlled corpus via LLM-guided multi-dimensional augmentation (semantic, linguistic, sentiment). Next, we LoRA-fine-tune a lightweight (<=2B) encoder to learn discriminative embeddings and to standardize a bank of personality prototypes. At inference, we retrieve top-k prototypes for a query post and perform a retrieve--reuse--revise--retain cycle: the model aggregates prototype evidence via prompt-based voting, revises when inconsistencies arise, and, upon correct prediction, retains the sample to continually enrich the prototype library. Across Kaggle and Pandora benchmarks, ProtoMBTI improves over baselines on both the four MBTI dichotomies and the full 16-type task, and exhibits robust cross-dataset generalization. Our results indicate that aligning the inference process with psychological prototype reasoning yields gains in accuracy, interpretability, and transfer for text-based personality modeling. 

**Abstract (ZH)**: 基于原型理论的认知对齐MBTI推断框架 

---
# End-to-End Framework Integrating Generative AI and Deep Reinforcement Learning for Autonomous Ultrasound Scanning 

**Title (ZH)**: 集成生成式AI和深度强化学习的端到端自主超声扫描框架 

**Authors**: Hanae Elmekki, Amanda Spilkin, Ehsan Zakeri, Antonela Mariel Zanuttini, Ahmed Alagha, Hani Sami, Jamal Bentahar, Lyes Kadem, Wen-Fang Xie, Philippe Pibarot, Rabeb Mizouni, Hadi Otrok, Azzam Mourad, Sami Muhaidat  

**Link**: [PDF](https://arxiv.org/pdf/2511.00114)  

**Abstract**: Cardiac ultrasound (US) is among the most widely used diagnostic tools in cardiology for assessing heart health, but its effectiveness is limited by operator dependence, time constraints, and human error. The shortage of trained professionals, especially in remote areas, further restricts access. These issues underscore the need for automated solutions that can ensure consistent, and accessible cardiac imaging regardless of operator skill or location. Recent progress in artificial intelligence (AI), especially in deep reinforcement learning (DRL), has gained attention for enabling autonomous decision-making. However, existing DRL-based approaches to cardiac US scanning lack reproducibility, rely on proprietary data, and use simplified models. Motivated by these gaps, we present the first end-to-end framework that integrates generative AI and DRL to enable autonomous and reproducible cardiac US scanning. The framework comprises two components: (i) a conditional generative simulator combining Generative Adversarial Networks (GANs) with Variational Autoencoders (VAEs), that models the cardiac US environment producing realistic action-conditioned images; and (ii) a DRL module that leverages this simulator to learn autonomous, accurate scanning policies. The proposed framework delivers AI-driven guidance through expert-validated models that classify image type and assess quality, supports conditional generation of realistic US images, and establishes a reproducible foundation extendable to other organs. To ensure reproducibility, a publicly available dataset of real cardiac US scans is released. The solution is validated through several experiments. The VAE-GAN is benchmarked against existing GAN variants, with performance assessed using qualitative and quantitative approaches, while the DRL-based scanning system is evaluated under varying configurations to demonstrate effectiveness. 

**Abstract (ZH)**: 基于生成AI和深度强化学习的全程自主 cardiac ultrasound 扫描框架 

---
# A generative adversarial network optimization method for damage detection and digital twinning by deep AI fault learning: Z24 Bridge structural health monitoring benchmark validation 

**Title (ZH)**: 基于深度AI故障学习的生成对抗网络优化方法：Z24桥梁结构健康监测基准验证 

**Authors**: Marios Impraimakis, Evangelia Nektaria Palkanoglou  

**Link**: [PDF](https://arxiv.org/pdf/2511.00099)  

**Abstract**: The optimization-based damage detection and damage state digital twinning capabilities are examined here of a novel conditional-labeled generative adversarial network methodology. The framework outperforms current approaches for fault anomaly detection as no prior information is required for the health state of the system: a topic of high significance for real-world applications. Specifically, current artificial intelligence-based digital twinning approaches suffer from the uncertainty related to obtaining poor predictions when a low number of measurements is available, physics knowledge is missing, or when the damage state is unknown. To this end, an unsupervised framework is examined and validated rigorously on the benchmark structural health monitoring measurements of Z24 Bridge: a post-tensioned concrete highway bridge in Switzerland. In implementing the approach, firstly, different same damage-level measurements are used as inputs, while the model is forced to converge conditionally to two different damage states. Secondly, the process is repeated for a different group of measurements. Finally, the convergence scores are compared to identify which one belongs to a different damage state. The process for both healthy-to-healthy and damage-to-healthy input data creates, simultaneously, measurements for digital twinning purposes at different damage states, capable of pattern recognition and machine learning data generation. Further to this process, a support vector machine classifier and a principal component analysis procedure is developed to assess the generated and real measurements of each damage category, serving as a secondary new dynamics learning indicator in damage scenarios. Importantly, the approach is shown to capture accurately damage over healthy measurements, providing a powerful tool for vibration-based system-level monitoring and scalable infrastructure resilience. 

**Abstract (ZH)**: 基于优化的损伤检测与损伤状态数字孪生能力研究：一种新颖的条件标记生成对抗网络方法 

---
# Generative human motion mimicking through feature extraction in denoising diffusion settings 

**Title (ZH)**: 去噪扩散设置中基于特征提取的人体动作生成性模仿 

**Authors**: Alexander Okupnik, Johannes Schneider, Kyriakos Flouris  

**Link**: [PDF](https://arxiv.org/pdf/2511.00011)  

**Abstract**: Recent success with large language models has sparked a new wave of verbal human-AI interaction. While such models support users in a variety of creative tasks, they lack the embodied nature of human interaction. Dance, as a primal form of human expression, is predestined to complement this experience. To explore creative human-AI interaction exemplified by dance, we build an interactive model based on motion capture (MoCap) data. It generates an artificial other by partially mimicking and also "creatively" enhancing an incoming sequence of movement data. It is the first model, which leverages single-person motion data and high level features in order to do so and, thus, it does not rely on low level human-human interaction data. It combines ideas of two diffusion models, motion inpainting, and motion style transfer to generate movement representations that are both temporally coherent and responsive to a chosen movement reference. The success of the model is demonstrated by quantitatively assessing the convergence of the feature distribution of the generated samples and the test set which serves as simulating the human performer. We show that our generations are first steps to creative dancing with AI as they are both diverse showing various deviations from the human partner while appearing realistic. 

**Abstract (ZH)**: 近期大规模语言模型的success激发了新的口头人机交互浪潮。尽管这类模型支持用户完成各种创意任务，但缺乏人类交互的具身特性。作为人类表达的原始形式，舞蹈注定要补充这种体验。为了探索由舞蹈体现的创意人机交互，我们基于运动捕捉（MoCap）数据构建了一个交互模型。该模型通过部分模仿并“创造性”增强传入的动作数据序列生成一个虚拟的“他者”。这是首个利用单人动作数据和高级特征进行此项工作的模型，因此它不依赖于低级的人际交互数据。该模型结合了两种扩散模型、运动填补和运动風格转移的思想，生成既时序连贯又能响应所选动作参考的动作表示。通过定量评估生成样本和测试集中特征分布的收敛性，展示了模型的成功。结果表明，我们的生成成果是与AI共舞的第一步，这些成果既多样又能体现出各种与人类伙伴的不同，同时保持了逼真性。 

---
