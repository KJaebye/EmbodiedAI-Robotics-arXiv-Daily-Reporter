# IFG: Internet-Scale Guidance for Functional Grasping Generation 

**Title (ZH)**: IFG: 面向功能抓取生成的互联网规模指导 

**Authors**: Ray Muxin Liu, Mingxuan Li, Kenneth Shaw, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2511.09558)  

**Abstract**: Large Vision Models trained on internet-scale data have demonstrated strong capabilities in segmenting and semantically understanding object parts, even in cluttered, crowded scenes. However, while these models can direct a robot toward the general region of an object, they lack the geometric understanding required to precisely control dexterous robotic hands for 3D grasping. To overcome this, our key insight is to leverage simulation with a force-closure grasping generation pipeline that understands local geometries of the hand and object in the scene. Because this pipeline is slow and requires ground-truth observations, the resulting data is distilled into a diffusion model that operates in real-time on camera point clouds. By combining the global semantic understanding of internet-scale models with the geometric precision of a simulation-based locally-aware force-closure, \our achieves high-performance semantic grasping without any manually collected training data. For visualizations of this please visit our website at this https URL 

**Abstract (ZH)**: 大规模训练于互联网规模数据的视觉模型在分割和语义理解物体部分方面展示了强大的能力，即使在杂乱拥挤的场景中也是如此。然而，尽管这些模型可以引导机器人朝物体的大致位置移动，但它们缺乏精确控制灵巧机器人手进行3D抓取所需的几何理解能力。为克服这一问题，我们的关键洞察是利用模拟与力闭合抓取生成管道相结合，该管道理解场景中手和物体的局部几何结构。由于该管道运行缓慢并需要 ground-truth 观测，结果数据被精简成一个实时操作于相机点云上的扩散模型。通过结合互联网规模模型的全局语义理解和基于模拟的局部意识力闭合的几何精度，我们的方法实现了高性能的语义抓取，而无需任何手动收集的训练数据。有关此方法的可视化，请访问我们的网站：this https URL。 

---
# MAP-VLA: Memory-Augmented Prompting for Vision-Language-Action Model in Robotic Manipulation 

**Title (ZH)**: MAP-VLA: 增强记忆提示在机器人操作中视觉语言行动模型的研究 

**Authors**: Runhao Li, Wenkai Guo, Zhenyu Wu, Changyuan Wang, Haoyuan Deng, Zhenyu Weng, Yap-Peng Tan, Ziwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.09516)  

**Abstract**: Pre-trained Vision-Language-Action (VLA) models have achieved remarkable success in improving robustness and generalization for end-to-end robotic manipulation. However, these models struggle with long-horizon tasks due to their lack of memory and reliance solely on immediate sensory inputs. To address this limitation, we propose Memory-Augmented Prompting for Vision-Language-Action model (MAP-VLA), a novel framework that empowers pre-trained VLA models with demonstration-derived memory prompts to augment action generation for long-horizon robotic manipulation tasks. To achieve this, MAP-VLA first constructs a memory library from historical demonstrations, where each memory unit captures information about a specific stage of a task. These memory units are implemented as learnable soft prompts optimized through prompt tuning. Then, during real-time task execution, MAP-VLA retrieves relevant memory through trajectory similarity matching and dynamically integrates it into the VLA model for augmented action generation. Importantly, this prompt tuning and retrieval augmentation approach operates as a plug-and-play module for a frozen VLA model, offering a lightweight and flexible solution to improve task performance. Experimental results show that MAP-VLA delivers up to 7.0% absolute performance gains in the simulation benchmark and 25.0% on real robot evaluations for long-horizon tasks, surpassing the current state-of-the-art methods. 

**Abstract (ZH)**: 基于记忆增强提示的预训练视觉-语言-行动模型（MAP-VLA）在长时 horizon 机器人 manipulation 任务中的应用 

---
# WMPO: World Model-based Policy Optimization for Vision-Language-Action Models 

**Title (ZH)**: WMPO：基于世界模型的策略优化方法用于视觉-语言-动作模型 

**Authors**: Fangqi Zhu, Zhengyang Yan, Zicong Hong, Quanxin Shou, Xiao Ma, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.09515)  

**Abstract**: Vision-Language-Action (VLA) models have shown strong potential for general-purpose robotic manipulation, but their reliance on expert demonstrations limits their ability to learn from failures and perform self-corrections. Reinforcement learning (RL) addresses these through self-improving interactions with the physical environment, but suffers from high sample complexity on real robots. We introduce World-Model-based Policy Optimization (WMPO), a principled framework for on-policy VLA RL without interacting with the real environment. In contrast to widely used latent world models, WMPO focuses on pixel-based predictions that align the "imagined" trajectories with the VLA features pretrained with web-scale images. Crucially, WMPO enables the policy to perform on-policy GRPO that provides stronger performance than the often-used off-policy methods. Extensive experiments in both simulation and real-robot settings demonstrate that WMPO (i) substantially improves sample efficiency, (ii) achieves stronger overall performance, (iii) exhibits emergent behaviors such as self-correction, and (iv) demonstrates robust generalization and lifelong learning capabilities. 

**Abstract (ZH)**: 基于世界模型的策略优化（WMPO）：一种无需与真实环境交互的Vision-Language-Action（VLA）强化学习框架 

---
# SPIDER: Scalable Physics-Informed Dexterous Retargeting 

**Title (ZH)**: SPIDER: 可扩展的物理驱动的灵巧动作转移 

**Authors**: Chaoyi Pan, Changhao Wang, Haozhi Qi, Zixi Liu, Homanga Bharadhwaj, Akash Sharma, Tingfan Wu, Guanya Shi, Jitendra Malik, Francois Hogan  

**Link**: [PDF](https://arxiv.org/pdf/2511.09484)  

**Abstract**: Learning dexterous and agile policy for humanoid and dexterous hand control requires large-scale demonstrations, but collecting robot-specific data is prohibitively expensive. In contrast, abundant human motion data is readily available from motion capture, videos, and virtual reality, which could help address the data scarcity problem. However, due to the embodiment gap and missing dynamic information like force and torque, these demonstrations cannot be directly executed on robots. To bridge this gap, we propose Scalable Physics-Informed DExterous Retargeting (SPIDER), a physics-based retargeting framework to transform and augment kinematic-only human demonstrations to dynamically feasible robot trajectories at scale. Our key insight is that human demonstrations should provide global task structure and objective, while large-scale physics-based sampling with curriculum-style virtual contact guidance should refine trajectories to ensure dynamical feasibility and correct contact sequences. SPIDER scales across diverse 9 humanoid/dexterous hand embodiments and 6 datasets, improving success rates by 18% compared to standard sampling, while being 10X faster than reinforcement learning (RL) baselines, and enabling the generation of a 2.4M frames dynamic-feasible robot dataset for policy learning. As a universal physics-based retargeting method, SPIDER can work with diverse quality data and generate diverse and high-quality data to enable efficient policy learning with methods like RL. 

**Abstract (ZH)**: 基于物理的可扩展灵巧再现（SPIDER）：大规模生成动态可行的机器人轨迹以解决灵巧和敏捷政策学习的数据稀缺问题 

---
# CoRL-MPPI: Enhancing MPPI With Learnable Behaviours For Efficient And Provably-Safe Multi-Robot Collision Avoidance 

**Title (ZH)**: CoRL-MPPI: 通过可学习行为提高MPPI以实现高效且证明安全的多机器人避碰 

**Authors**: Stepan Dergachev, Artem Pshenitsyn, Aleksandr Panov, Alexey Skrynnik, Konstantin Yakovlev  

**Link**: [PDF](https://arxiv.org/pdf/2511.09331)  

**Abstract**: Decentralized collision avoidance remains a core challenge for scalable multi-robot systems. One of the promising approaches to tackle this problem is Model Predictive Path Integral (MPPI) -- a framework that is naturally suited to handle any robot motion model and provides strong theoretical guarantees. Still, in practice MPPI-based controller may provide suboptimal trajectories as its performance relies heavily on uninformed random sampling. In this work, we introduce CoRL-MPPI, a novel fusion of Cooperative Reinforcement Learning and MPPI to address this limitation. We train an action policy (approximated as deep neural network) in simulation that learns local cooperative collision avoidance behaviors. This learned policy is then embedded into the MPPI framework to guide its sampling distribution, biasing it towards more intelligent and cooperative actions. Notably, CoRL-MPPI preserves all the theoretical guarantees of regular MPPI. We evaluate our approach in dense, dynamic simulation environments against state-of-the-art baselines, including ORCA, BVC, and a multi-agent MPPI implementation. Our results demonstrate that CoRL-MPPI significantly improves navigation efficiency (measured by success rate and makespan) and safety, enabling agile and robust multi-robot navigation. 

**Abstract (ZH)**: 去中心化避碰仍然是可扩展多机器人系统的核心挑战。一种有前景的解决方案是模型预测路径积分（MPPI）——这是一种天然适用于处理任何机器人运动模型并提供强大理论保证的框架。然而，基于MPPI的控制器在实践中可能提供次优轨迹，其性能高度依赖于未启发式的随机采样。在本工作中，我们提出了CoRL-MPPI，这是一种将合作强化学习与MPPI相结合的新颖融合方法，以解决这一限制。我们在仿真中训练一个动作策略（近似为深度神经网络），使其学习局部合作避碰行为。然后将此学习策略嵌入到MPPI框架中，指导其采样分布，使其偏向更智能和合作的动作。值得注意的是，CoRL-MPPI 保留了常规MPPI的所有理论保证。我们使用密集动态仿真环境对我们的方法与最先进的基线方法（包括ORCA、BVC以及多Agent的MPPI实现）进行评估。我们的结果表明，CoRL-MPPI 显著提高了导航效率（通过成功率和耗时度量）并增强了安全性，从而实现了灵活且可靠的多机器人导航。 

---
# UMIGen: A Unified Framework for Egocentric Point Cloud Generation and Cross-Embodiment Robotic Imitation Learning 

**Title (ZH)**: UMIGen：统一的自视点点云生成与跨躯体机器人模仿学习框架 

**Authors**: Yan Huang, Shoujie Li, Xingting Li, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2511.09302)  

**Abstract**: Data-driven robotic learning faces an obvious dilemma: robust policies demand large-scale, high-quality demonstration data, yet collecting such data remains a major challenge owing to high operational costs, dependence on specialized hardware, and the limited spatial generalization capability of current methods. The Universal Manipulation Interface (UMI) relaxes the strict hardware requirements for data collection, but it is restricted to capturing only RGB images of a scene and omits the 3D geometric information on which many tasks rely. Inspired by DemoGen, we propose UMIGen, a unified framework that consists of two key components: (1) Cloud-UMI, a handheld data collection device that requires no visual SLAM and simultaneously records point cloud observation-action pairs; and (2) a visibility-aware optimization mechanism that extends the DemoGen pipeline to egocentric 3D observations by generating only points within the camera's field of view. These two components enable efficient data generation that aligns with real egocentric observations and can be directly transferred across different robot embodiments without any post-processing. Experiments in both simulated and real-world settings demonstrate that UMIGen supports strong cross-embodiment generalization and accelerates data collection in diverse manipulation tasks. 

**Abstract (ZH)**: 数据驱动的机器人学习面临一个明显的困境：鲁棒策略需要大量的高质量示范数据，然而由于操作成本高、依赖专门硬件以及当前方法在空间泛化能力上的局限，收集此类数据仍然是一个主要挑战。通用操作接口（UMI）放松了数据收集的严格硬件要求，但仅能捕捉场景的RGB图像，而不包含许多任务依赖的3D几何信息。受到DemoGen的启发，我们提出了UMIGen统一框架，该框架包含两个关键组件：（1）Cloud-UMI，一种无需视觉SLAM的手持数据采集设备，同时记录点云观测-动作对；（2）一种基于可见性优化机制，通过生成摄像机视野内的点来扩展DemoGen流水线，以进行第一人称3D观察。这两个组件使数据生成更加高效，并且可以实时地在不同机器人身体形态之间直接传输而无需任何后处理。实验结果表明，UMIGen支持强大的跨身体形态泛化，并加速了在多种操作任务中的数据收集。 

---
# Unveiling the Impact of Data and Model Scaling on High-Level Control for Humanoid Robots 

**Title (ZH)**: 揭示数据和模型扩展对类人机器人高层次控制影响的研究 

**Authors**: Yuxi Wei, Zirui Wang, Kangning Yin, Yue Hu, Jingbo Wang, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.09241)  

**Abstract**: Data scaling has long remained a critical bottleneck in robot learning. For humanoid robots, human videos and motion data are abundant and widely available, offering a free and large-scale data source. Besides, the semantics related to the motions enable modality alignment and high-level robot control learning. However, how to effectively mine raw video, extract robot-learnable representations, and leverage them for scalable learning remains an open problem. To address this, we introduce Humanoid-Union, a large-scale dataset generated through an autonomous pipeline, comprising over 260 hours of diverse, high-quality humanoid robot motion data with semantic annotations derived from human motion videos. The dataset can be further expanded via the same pipeline. Building on this data resource, we propose SCHUR, a scalable learning framework designed to explore the impact of large-scale data on high-level control in humanoid robots. Experimental results demonstrate that SCHUR achieves high robot motion generation quality and strong text-motion alignment under data and model scaling, with 37\% reconstruction improvement under MPJPE and 25\% alignment improvement under FID comparing with previous methods. Its effectiveness is further validated through deployment in real-world humanoid robot. 

**Abstract (ZH)**: 机器人学习中的数据缩放一直是关键瓶颈。对于类人机器人而言，人类的视频和运动数据丰富且广泛可用，提供了免费且规模庞大的数据源。此外，与运动相关的语义信息有助于模态对齐和高阶机器人控制学习。然而，如何有效挖掘原始视频、提取可学习的机器人表示，并利用它们进行可扩展学习仍是一个开放问题。为解决这一问题，我们引入了Humanoid-Union，这是一种通过自主管道生成的大规模数据集，包含了超过260小时的多样化、高质量的类人机器人运动数据，这些数据的语义注释源自人类运动视频。该数据集可通过相同管道进一步扩展。基于这一数据资源，我们提出了一种可扩展的学习框架SCHUR，旨在探索大规模数据对类人机器人高阶控制的影响。实验结果表明，SCHUR 在数据和模型缩放下实现了高质量的机器人运动生成和强大的文本-运动对齐，MPJPE 下重构性能提升了 37%，FID 下对齐性能提升了 25%，并在真实世界类人机器人部署中得到了进一步验证。 

---
# RGMP: Recurrent Geometric-prior Multimodal Policy for Generalizable Humanoid Robot Manipulation 

**Title (ZH)**: RGMP：循环几何先验多模态政策用于通用 humanoid 机器人操作 

**Authors**: Xuetao Li, Wenke Huang, Nengyuan Pan, Kaiyan Zhao, Songhua Yang, Yiming Wang, Mengde Li, Mang Ye, Jifeng Xuan, Miao Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.09141)  

**Abstract**: Humanoid robots exhibit significant potential in executing diverse human-level skills. However, current research predominantly relies on data-driven approaches that necessitate extensive training datasets to achieve robust multimodal decision-making capabilities and generalizable visuomotor control. These methods raise concerns due to the neglect of geometric reasoning in unseen scenarios and the inefficient modeling of robot-target relationships within the training data, resulting in significant waste of training resources. To address these limitations, we present the Recurrent Geometric-prior Multimodal Policy (RGMP), an end-to-end framework that unifies geometric-semantic skill reasoning with data-efficient visuomotor control. For perception capabilities, we propose the Geometric-prior Skill Selector, which infuses geometric inductive biases into a vision language model, producing adaptive skill sequences for unseen scenes with minimal spatial common sense tuning. To achieve data-efficient robotic motion synthesis, we introduce the Adaptive Recursive Gaussian Network, which parameterizes robot-object interactions as a compact hierarchy of Gaussian processes that recursively encode multi-scale spatial relationships, yielding dexterous, data-efficient motion synthesis even from sparse demonstrations. Evaluated on both our humanoid robot and desktop dual-arm robot, the RGMP framework achieves 87% task success in generalization tests and exhibits 5x greater data efficiency than the state-of-the-art model. This performance underscores its superior cross-domain generalization, enabled by geometric-semantic reasoning and recursive-Gaussion adaptation. 

**Abstract (ZH)**: 类人机器人在执行多种人类水平技能方面展现出显著潜力。然而，当前研究主要依赖于数据驱动的方法，需要大量训练数据才能实现稳健的多模态决策能力和可泛化的视觉-运动控制。这些方法因忽视未见场景中的几何推理以及训练数据中机器人-目标关系的低效建模而受到质疑，导致大量训练资源浪费。为解决这些问题，我们提出了循环几何先验多模态策略（RGMP）框架，该框架将几何语义技能推理与数据高效视觉-运动控制统一起来。在感知能力方面，我们提出了几何先验技能选择器，将几何归纳偏差注入视觉语言模型中，产生适用于未见场景的自适应技能序列，对空间常识的调整降到最低。为实现数据高效的机器人运动合成，我们引入了自适应递归高斯网络，将机器人-物体相互作用参数化为紧凑的高斯过程层次结构，递归地编码多尺度空间关系，即使在稀疏演示情况下也能生成灵巧的数据高效的运动合成。在我们的人形机器人和桌面双臂机器人上进行评估，RGMP框架在泛化测试中的任务成功率达到了87%，数据效率比当前最先进的模型高出5倍。这一性能突显了其在跨域泛化中的优越性，得益于几何语义推理和递归高斯适应。 

---
# Data Assessment for Embodied Intelligence 

**Title (ZH)**: 数据评估与体域智能 

**Authors**: Jiahao Xiao, Bowen Yan, Jianbo Zhang, Jia Wang, Chunyi Li, Zhengxue Cheng, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2511.09119)  

**Abstract**: In embodied intelligence, datasets play a pivotal role, serving as both a knowledge repository and a conduit for information transfer. The two most critical attributes of a dataset are the amount of information it provides and how easily this information can be learned by models. However, the multimodal nature of embodied data makes evaluating these properties particularly challenging. Prior work has largely focused on diversity, typically counting tasks and scenes or evaluating isolated modalities, which fails to provide a comprehensive picture of dataset diversity. On the other hand, the learnability of datasets has received little attention and is usually assessed post-hoc through model training, an expensive, time-consuming process that also lacks interpretability, offering little guidance on how to improve a dataset. In this work, we address both challenges by introducing two principled, data-driven tools. First, we construct a unified multimodal representation for each data sample and, based on it, propose diversity entropy, a continuous measure that characterizes the amount of information contained in a dataset. Second, we introduce the first interpretable, data-driven algorithm to efficiently quantify dataset learnability without training, enabling researchers to assess a dataset's learnability immediately upon its release. We validate our algorithm on both simulated and real-world embodied datasets, demonstrating that it yields faithful, actionable insights that enable researchers to jointly improve diversity and learnability. We hope this work provides a foundation for designing higher-quality datasets that advance the development of embodied intelligence. 

**Abstract (ZH)**: 在具身智能中，数据集在知识存储和信息传递中扮演着关键角色。数据集的两个最核心属性是其提供的信息量以及模型学习这些信息的难易程度。然而，具身数据的多模态特性使得评估这些属性尤为具有挑战性。此前的工作主要集中在多样性上，通常通过计数任务和场景或评估孤立的模态来进行，未能提供数据集多样性的全面视角。另一方面，数据集的可学习性受到的关注较少，通常通过模型训练后进行评估，这不仅耗费时间和资源，且缺乏可解释性，难以为改进数据集提供指导。在这项工作中，我们通过引入两个原理性和数据驱动的工具来解决这些问题。首先，我们为每个数据样本构建了一个统一的多模态表示，并在此基础上提出了多样性熵，这是一个连续量度，用来表征数据集中的信息量。其次，我们引入了第一个可解释的数据驱动算法，能够在不进行模型训练的情况下高效地量化数据集的可学习性，使研究人员能够在数据集发布后立即对其可学习性进行评估。我们分别在模拟和实际的具身数据集上验证了该算法，结果表明它能提供忠实和可行的洞察，帮助研究人员同时提升多样性和可学习性。我们希望这项工作能够为设计更高质量的数据集并推动具身智能的发展提供基础。 

---
# APEX: Action Priors Enable Efficient Exploration for Robust Motion Tracking on Legged Robots 

**Title (ZH)**: APEX: 行动先验助力腿足机器人稳健运动追踪的高效探索 

**Authors**: Shivam Sood, Laukik Nakhwa, Sun Ge, Yuhong Cao, Jin Cheng, Fatemah Zargarbashi, Taerim Yoon, Sungjoon Choi, Stelian Coros, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2511.09091)  

**Abstract**: Learning natural, animal-like locomotion from demonstrations has become a core paradigm in legged robotics. Despite the recent advancements in motion tracking, most existing methods demand extensive tuning and rely on reference data during deployment, limiting adaptability. We present APEX (Action Priors enable Efficient Exploration), a plug-and-play extension to state-of-the-art motion tracking algorithms that eliminates any dependence on reference data during deployment, improves sample efficiency, and reduces parameter tuning effort. APEX integrates expert demonstrations directly into reinforcement learning (RL) by incorporating decaying action priors, which initially bias exploration toward expert demonstrations but gradually allow the policy to explore independently. This is combined with a multi-critic framework that balances task performance with motion style. Moreover, APEX enables a single policy to learn diverse motions and transfer reference-like styles across different terrains and velocities, while remaining robust to variations in reward design. We validate the effectiveness of our method through extensive experiments in both simulation and on a Unitree Go2 robot. By leveraging demonstrations to guide exploration during RL training, without imposing explicit bias toward them, APEX enables legged robots to learn with greater stability, efficiency, and generalization. We believe this approach paves the way for guidance-driven RL to boost natural skill acquisition in a wide array of robotic tasks, from locomotion to manipulation. Website and code: this https URL. 

**Abstract (ZH)**: 基于演示学习自然的仿动物运动成为腿足机器人研究的核心 paradigm。尽管最近在运动跟踪方面取得了进展，但大多数现有方法在部署时仍需要大量调优并依赖参考数据，限制了适应性。我们提出 APEX（动作先验促进高效探索）——一种可插拔扩展，作为最先进的运动跟踪算法的扩展，能够在部署时消除对参考数据的依赖，提高样本效率，减少参数调优工作量。APEX 通过引入衰减的动作先验将专家演示直接整合到强化学习（RL）中，初始时偏向于专家演示的探索，但逐渐允许策略独立探索。这与多 Critic 框架相结合，平衡任务性能与运动风格。此外，APEX 使得单个策略能够学习多种运动，并在不同地形和速度下传递参考样式的风格，同时对奖励设计的变化具有鲁棒性。我们通过在仿真和 Unitree Go2 机器人上的广泛实验验证了该方法的有效性。通过在 RL 训练期间利用演示来引导探索，而不对其施加显式的偏见，APEX 使腿足机器人能够更加稳定、高效和泛化地学习。我们相信这种方法为通过引导式 RL 在广泛机器人任务中提升自然技能获取开辟了道路，从移动到操作。网站和代码：this https URL。 

---
# Think, Remember, Navigate: Zero-Shot Object-Goal Navigation with VLM-Powered Reasoning 

**Title (ZH)**: 思考、记忆、导航：基于VLM增强推理的零样本物体目标导航 

**Authors**: Mobin Habibpour, Fatemeh Afghah  

**Link**: [PDF](https://arxiv.org/pdf/2511.08942)  

**Abstract**: While Vision-Language Models (VLMs) are set to transform robotic navigation, existing methods often underutilize their reasoning capabilities. To unlock the full potential of VLMs in robotics, we shift their role from passive observers to active strategists in the navigation process. Our framework outsources high-level planning to a VLM, which leverages its contextual understanding to guide a frontier-based exploration agent. This intelligent guidance is achieved through a trio of techniques: structured chain-of-thought prompting that elicits logical, step-by-step reasoning; dynamic inclusion of the agent's recent action history to prevent getting stuck in loops; and a novel capability that enables the VLM to interpret top-down obstacle maps alongside first-person views, thereby enhancing spatial awareness. When tested on challenging benchmarks like HM3D, Gibson, and MP3D, this method produces exceptionally direct and logical trajectories, marking a substantial improvement in navigation efficiency over existing approaches and charting a path toward more capable embodied agents. 

**Abstract (ZH)**: 视觉-语言模型在机器人导航中的潜力解锁：从被动观察者到主动策略师 

---
# Expand Your SCOPE: Semantic Cognition over Potential-Based Exploration for Embodied Visual Navigation 

**Title (ZH)**: 扩展您的SCOPE：基于潜力场的语义认知在体感视觉导航中的应用 

**Authors**: Ningnan Wang, Weihuang Chen, Liming Chen, Haoxuan Ji, Zhongyu Guo, Xuchong Zhang, Hongbin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.08935)  

**Abstract**: Embodied visual navigation remains a challenging task, as agents must explore unknown environments with limited knowledge. Existing zero-shot studies have shown that incorporating memory mechanisms to support goal-directed behavior can improve long-horizon planning performance. However, they overlook visual frontier boundaries, which fundamentally dictate future trajectories and observations, and fall short of inferring the relationship between partial visual observations and navigation goals. In this paper, we propose Semantic Cognition Over Potential-based Exploration (SCOPE), a zero-shot framework that explicitly leverages frontier information to drive potential-based exploration, enabling more informed and goal-relevant decisions. SCOPE estimates exploration potential with a Vision-Language Model and organizes it into a spatio-temporal potential graph, capturing boundary dynamics to support long-horizon planning. In addition, SCOPE incorporates a self-reconsideration mechanism that revisits and refines prior decisions, enhancing reliability and reducing overconfident errors. Experimental results on two diverse embodied navigation tasks show that SCOPE outperforms state-of-the-art baselines by 4.6\% in accuracy. Further analysis demonstrates that its core components lead to improved calibration, stronger generalization, and higher decision quality. 

**Abstract (ZH)**: 基于潜在空间的语义认知探索（SCOPE）：零样本环境导航中的前沿信息利用 

---
# ATOM-CBF: Adaptive Safe Perception-Based Control under Out-of-Distribution Measurements 

**Title (ZH)**: ATOM-CBF: 基于自适应安全感知的异分布测量控制 

**Authors**: Kai S. Yun, Navid Azizan  

**Link**: [PDF](https://arxiv.org/pdf/2511.08741)  

**Abstract**: Ensuring the safety of real-world systems is challenging, especially when they rely on learned perception modules to infer the system state from high-dimensional sensor data. These perception modules are vulnerable to epistemic uncertainty, often failing when encountering out-of-distribution (OoD) measurements not seen during training. To address this gap, we introduce ATOM-CBF (Adaptive-To-OoD-Measurement Control Barrier Function), a novel safe control framework that explicitly computes and adapts to the epistemic uncertainty from OoD measurements, without the need for ground-truth labels or information on distribution shifts. Our approach features two key components: (1) an OoD-aware adaptive perception error margin and (2) a safety filter that integrates this adaptive error margin, enabling the filter to adjust its conservatism in real-time. We provide empirical validation in simulations, demonstrating that ATOM-CBF maintains safety for an F1Tenth vehicle with LiDAR scans and a quadruped robot with RGB images. 

**Abstract (ZH)**: 确保真实世界系统的安全性具有挑战性，尤其是在它们依赖于从高维传感器数据中推断系统状态的学习感知模块时。这些感知模块对epistemic不确定性敏感，经常在遇到训练中未见过的(out-of-distribution, OoD)测量值时失效。为解决这一问题，我们引入了ATOM-CBF（Adaptive-To-OoD-Measurement Control Barrier Function）这一新颖的安全控制框架，该框架能够明确计算并适应从OoD测量值中获得的epistemic不确定性，而无需 ground-truth标签或分布变化信息。我们的方法包含两个关键组件：(1) OoD感知误差容限的自适应调整，以及(2) 结合该自适应误差容限的安全过滤器，使过滤器能够实时调整其保守程度。我们在仿真中提供了 empirical验证，表明ATOM-CBF能够确保使用LiDAR扫描的F1Tenth车辆和使用RGB图像的四足机器人的安全性。 

---
# Intuitive Programming, Adaptive Task Planning, and Dynamic Role Allocation in Human-Robot Collaboration 

**Title (ZH)**: 直观编程、自适应任务规划与动态角色分配在人机协作中的应用 

**Authors**: Marta Lagomarsino, Elena Merlo, Andrea Pupa, Timo Birr, Franziska Krebs, Cristian Secchi, Tamim Asfour, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2511.08732)  

**Abstract**: Remarkable capabilities have been achieved by robotics and AI, mastering complex tasks and environments. Yet, humans often remain passive observers, fascinated but uncertain how to engage. Robots, in turn, cannot reach their full potential in human-populated environments without effectively modeling human states and intentions and adapting their behavior. To achieve a synergistic human-robot collaboration (HRC), a continuous information flow should be established: humans must intuitively communicate instructions, share expertise, and express needs. In parallel, robots must clearly convey their internal state and forthcoming actions to keep users informed, comfortable, and in control. This review identifies and connects key components enabling intuitive information exchange and skill transfer between humans and robots. We examine the full interaction pipeline: from the human-to-robot communication bridge translating multimodal inputs into robot-understandable representations, through adaptive planning and role allocation, to the control layer and feedback mechanisms to close the loop. Finally, we highlight trends and promising directions toward more adaptive, accessible HRC. 

**Abstract (ZH)**: 机器人和AI在掌握复杂任务和环境方面取得了显著能力，但人类往往仍处于被动观察者的位置，对如何参与感到困惑。为了实现人类与机器人协同合作（HRC），需要建立持续的信息流：人类必须直观地传达指令、共享专业知识并表达需求。同时，机器人需要清晰地传达其内部状态和即将采取的行动，以使用户保持被告知、安心和处于控制之中。本文回顾并连接了关键组件，使人类与机器人之间能够直观地进行信息交流和技能转移。我们探讨了完整的交互管道：从人类到机器人的沟通桥梁，将多模态输入转化为机器人可理解的表示，再到适应性规划和角色分配，以及控制层和反馈机制以形成闭环。最后，我们强调了更适应性、更可访问的HRC的发展趋势和有前景的方向。 

---
# Diffusion Policies with Value-Conditional Optimization for Offline Reinforcement Learning 

**Title (ZH)**: 基于价值条件优化的离线强化学习扩散策略 

**Authors**: Yunchang Ma, Tenglong Liu, Yixing Lan, Xin Yin, Changxin Zhang, Xinglong Zhang, Xin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.08922)  

**Abstract**: In offline reinforcement learning, value overestimation caused by out-of-distribution (OOD) actions significantly limits policy performance. Recently, diffusion models have been leveraged for their strong distribution-matching capabilities, enforcing conservatism through behavior policy constraints. However, existing methods often apply indiscriminate regularization to redundant actions in low-quality datasets, resulting in excessive conservatism and an imbalance between the expressiveness and efficiency of diffusion modeling. To address these issues, we propose DIffusion policies with Value-conditional Optimization (DIVO), a novel approach that leverages diffusion models to generate high-quality, broadly covered in-distribution state-action samples while facilitating efficient policy improvement. Specifically, DIVO introduces a binary-weighted mechanism that utilizes the advantage values of actions in the offline dataset to guide diffusion model training. This enables a more precise alignment with the dataset's distribution while selectively expanding the boundaries of high-advantage actions. During policy improvement, DIVO dynamically filters high-return-potential actions from the diffusion model, effectively guiding the learned policy toward better performance. This approach achieves a critical balance between conservatism and explorability in offline RL. We evaluate DIVO on the D4RL benchmark and compare it against state-of-the-art baselines. Empirical results demonstrate that DIVO achieves superior performance, delivering significant improvements in average returns across locomotion tasks and outperforming existing methods in the challenging AntMaze domain, where sparse rewards pose a major difficulty. 

**Abstract (ZH)**: 离线强化学习中，由分布外（OOD）动作引起的价值高估显著限制了策略性能。为了解决这一问题，我们提出了以值条件优化为特征的扩散模型策略（DIVO），该方法利用了扩散模型生成高质量、广泛覆盖的在分布状态-动作样本的同时，促进高效的策略改进。具体地，DIVO 引入了一种二值加权机制，利用离线数据集中动作的优势值来指导扩散模型的训练。这使得扩散模型更加精确地匹配数据集的分布，并有选择性地扩展高优势动作的边界。在策略改进过程中，DIVO 动态过滤出具有高回报潜力的动作，有效地引导学习到的策略向更好的性能方向发展。该方法在离线 RL 中实现了保守性和探索性的关键平衡。我们在 D4RL 基准上评估了 DIVO，并将其与最先进的基线方法进行了比较。实验结果表明，DIVO 达到了优越的性能，在步行任务中显著提高了平均回报，并且在具有稀疏回报的挑战性 AntMaze 领域中优于现有方法。 

---
# Fundamentals of Physical AI 

**Title (ZH)**: 物理驱动人工智能基础 

**Authors**: Vahid Salehi  

**Link**: [PDF](https://arxiv.org/pdf/2511.09497)  

**Abstract**: This work will elaborate the fundamental principles of physical artificial intelligence (Physical AI) from a scientific and systemic perspective. The aim is to create a theoretical foundation that describes the physical embodiment, sensory perception, ability to act, learning processes, and context sensitivity of intelligent systems within a coherent framework. While classical AI approaches rely on symbolic processing and data driven models, Physical AI understands intelligence as an emergent phenomenon of real interaction between body, environment, and experience. The six fundamentals presented here are embodiment, sensory perception, motor action, learning, autonomy, and context sensitivity, and form the conceptual basis for designing and evaluating physically intelligent systems. Theoretically, it is shown that these six principles do not represent loose functional modules but rather act as a closed control loop in which energy, information, control, and context are in constant interaction. This circular interaction enables a system to generate meaning not from databases, but from physical experience, a paradigm shift that understands intelligence as an physical embodied process. Physical AI understands learning not as parameter adjustment, but as a change in the structural coupling between agents and the environment. To illustrate this, the theoretical model is explained using a practical scenario: An adaptive assistant robot supports patients in a rehabilitation clinic. This example illustrates that physical intelligence does not arise from abstract calculation, but from immediate, embodied experience. It shows how the six fundamentals interact in a real system: embodiment as a prerequisite, perception as input, movement as expression, learning as adaptation, autonomy as regulation, and context as orientation. 

**Abstract (ZH)**: 本研究将从科学和系统化的角度阐述物理人工智能（Physical AI）的基本原理，旨在构建一个理论基础，该基础能够描述智能系统在一致框架内的物理体现、感官感知、行动能力、学习过程和情境敏感性。虽然经典人工智能依赖于符号处理和数据驱动模型，物理人工智能则理解智能为身体、环境和体验之间真实互动中涌现的现象。本文提出的六大基本原理为实体性、感官感知、运动行动、学习、自主性和情境敏感性，它们构成了设计和评估物理智能系统概念基础。理论上，这六大原则并非松散的功能模块，而是一个闭环控制回路，在该回路中，能量、信息、控制和情境持续交互。这种闭环交互使系统能够从物理体验中生成意义，而非仅从数据库中获取。这代表了智能理解为具身物理过程的范式转变。物理人工智能将学习理解为代理与环境之间结构耦合的变化，而非参数调整。为了说明这一点，本文通过一个实际场景解释了理论模型：一个自适应辅助机器人在康复诊所支持患者。这个例子展示了物理智能不是源自抽象计算，而是源自即时的、具身的经验。它说明了六大基本原理在实际系统中如何交互：实体性作为先决条件，感知作为输入，运动作为表达，学习作为适应，自主性作为调节，情境作为定向。 

---
# History-Aware Reasoning for GUI Agents 

**Title (ZH)**: 基于历史的推理方法用于GUI代理 

**Authors**: Ziwei Wang, Leyang Yang, Xiaoxuan Tang, Sheng Zhou, Dajun Chen, Wei Jiang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.09127)  

**Abstract**: Advances in Multimodal Large Language Models have significantly enhanced Graphical User Interface (GUI) automation. Equipping GUI agents with reliable episodic reasoning capabilities is essential for bridging the gap between users' concise task descriptions and the complexities of real-world execution. Current methods integrate Reinforcement Learning (RL) with System-2 Chain-of-Thought, yielding notable gains in reasoning enhancement. For long-horizon GUI tasks, historical interactions connect each screen to the goal-oriented episode chain, and effectively leveraging these clues is crucial for the current decision. However, existing native GUI agents exhibit weak short-term memory in their explicit reasoning, interpreting the chained interactions as discrete screen understanding, i.e., unawareness of the historical interactions within the episode. This history-agnostic reasoning challenges their performance in GUI automation. To alleviate this weakness, we propose a History-Aware Reasoning (HAR) framework, which encourages an agent to reflect on its own errors and acquire episodic reasoning knowledge from them via tailored strategies that enhance short-term memory in long-horizon interaction. The framework mainly comprises constructing a reflective learning scenario, synthesizing tailored correction guidelines, and designing a hybrid RL reward function. Using the HAR framework, we develop a native end-to-end model, HAR-GUI-3B, which alters the inherent reasoning mode from history-agnostic to history-aware, equipping the GUI agent with stable short-term memory and reliable perception of screen details. Comprehensive evaluations across a range of GUI-related benchmarks demonstrate the effectiveness and generalization of our method. 

**Abstract (ZH)**: 多模态大型语言模型的进步显著增强了图形用户界面（GUI）自动化。为了弥合用户简洁的任务描述与现实世界执行复杂性之间的差距，为GUI代理装备可靠的瞬时推理能力至关重要。当前方法将强化学习（RL）与系统2链式思考相结合，显著提升了推理能力。对于长期目标导向的GUI任务，历史交互将每个屏幕连接到目标导向的序列链中，有效地利用这些线索对于当前决策至关重要。然而，现有的原生GUI代理在显式推理中表现出较弱的短期记忆，将链式交互视为离散的屏幕理解，即不了解序列链中的历史交互。这种历史无意识的推理对其在GUI自动化中的表现构成了挑战。为了缓解这一弱点，我们提出了一个历史意识推理（HAR）框架，该框架鼓励代理反思自己的错误，并通过定制策略增强长期交互中的短期记忆，从而从其中获取 episodic 推理知识。该框架主要包含构建反思学习场景、合成定制的校正指南和设计混合RL奖励函数。利用HAR框架，我们开发了一个原生端到端模型HAR-GUI-3B，使其推理模式从历史无意识转变为历史意识，为GUI代理提供了稳定的短期记忆和可靠的屏幕细节感知能力。广泛的跨GUI相关基准测试表明了我们方法的有效性和泛化能力。 

---
# OR-R1: Automating Modeling and Solving of Operations Research Optimization Problem via Test-Time Reinforcement Learning 

**Title (ZH)**: OR-R1: 通过测试时强化学习自动建模和求解运筹优化问题 

**Authors**: Zezhen Ding, Zhen Tan, Jiheng Zhang, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.09092)  

**Abstract**: Optimization modeling and solving are fundamental to the application of Operations Research (OR) in real-world decision making, yet the process of translating natural language problem descriptions into formal models and solver code remains highly expertise intensive. While recent advances in large language models (LLMs) have opened new opportunities for automation, the generalization ability and data efficiency of existing LLM-based methods are still limited, asmost require vast amounts of annotated or synthetic data, resulting in high costs and scalability barriers. In this work, we present OR-R1, a data-efficient training framework for automated optimization modeling and solving. OR-R1 first employs supervised fine-tuning (SFT) to help the model acquire the essential reasoning patterns for problem formulation and code generation from limited labeled data. In addition, it improves the capability and consistency through Test-Time Group Relative Policy Optimization (TGRPO). This two-stage design enables OR-R1 to leverage both scarce labeled and abundant unlabeled data for effective learning. Experiments show that OR-R1 achieves state-of-the-art performance with an average solving accuracy of $67.7\%$, using only $1/10$ the synthetic data required by prior methods such as ORLM, exceeding ORLM's solving accuracy by up to $4.2\%$. Remarkably, OR-R1 outperforms ORLM by over $2.4\%$ with just $100$ synthetic samples. Furthermore, TGRPO contributes an additional $3.1\%-6.4\%$ improvement in accuracy, significantly narrowing the gap between single-attempt (Pass@1) and multi-attempt (Pass@8) performance from $13\%$ to $7\%$. Extensive evaluations across diverse real-world benchmarks demonstrate that OR-R1 provides a robust, scalable, and cost-effective solution for automated OR optimization problem modeling and solving, lowering the expertise and data barriers for industrial OR applications. 

**Abstract (ZH)**: 一种数据效率高的自动化优化建模与求解训练框架：OR-R1 

---
# Lumine: An Open Recipe for Building Generalist Agents in 3D Open Worlds 

**Title (ZH)**: Lumine：构建三维开放世界通用智能体的开源方案 

**Authors**: Weihao Tan, Xiangyang Li, Yunhao Fang, Heyuan Yao, Shi Yan, Hao Luo, Tenglong Ao, Huihui Li, Hongbin Ren, Bairen Yi, Yujia Qin, Bo An, Libin Liu, Guang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2511.08892)  

**Abstract**: We introduce Lumine, the first open recipe for developing generalist agents capable of completing hours-long complex missions in real time within challenging 3D open-world environments. Lumine adopts a human-like interaction paradigm that unifies perception, reasoning, and action in an end-to-end manner, powered by a vision-language model. It processes raw pixels at 5 Hz to produce precise 30 Hz keyboard-mouse actions and adaptively invokes reasoning only when necessary. Trained in Genshin Impact, Lumine successfully completes the entire five-hour Mondstadt main storyline on par with human-level efficiency and follows natural language instructions to perform a broad spectrum of tasks in both 3D open-world exploration and 2D GUI manipulation across collection, combat, puzzle-solving, and NPC interaction. In addition to its in-domain performance, Lumine demonstrates strong zero-shot cross-game generalization. Without any fine-tuning, it accomplishes 100-minute missions in Wuthering Waves and the full five-hour first chapter of Honkai: Star Rail. These promising results highlight Lumine's effectiveness across distinct worlds and interaction dynamics, marking a concrete step toward generalist agents in open-ended environments. 

**Abstract (ZH)**: Lumine：首个用于在挑战性3D开放世界环境中实时完成长时间复杂任务的一般型智能体开源框架 

---
# Interpretable by Design: Query-Specific Neural Modules for Explainable Reinforcement Learning 

**Title (ZH)**: 设计可解释的：面向查询的神经模块在可解释 reinforcement 学习中的应用 

**Authors**: Mehrdad Zakershahrak  

**Link**: [PDF](https://arxiv.org/pdf/2511.08749)  

**Abstract**: Reinforcement learning has traditionally focused on a singular objective: learning policies that select actions to maximize reward. We challenge this paradigm by asking: what if we explicitly architected RL systems as inference engines that can answer diverse queries about their environment? In deterministic settings, trained agents implicitly encode rich knowledge about reachability, distances, values, and dynamics - yet current architectures are not designed to expose this information efficiently. We introduce Query Conditioned Deterministic Inference Networks (QDIN), a unified architecture that treats different types of queries (policy, reachability, paths, comparisons) as first-class citizens, with specialized neural modules optimized for each inference pattern. Our key empirical finding reveals a fundamental decoupling: inference accuracy can reach near-perfect levels (99% reachability IoU) even when control performance remains suboptimal (31% return), suggesting that the representations needed for accurate world knowledge differ from those required for optimal control. Experiments demonstrate that query specialized architectures outperform both unified models and post-hoc extraction methods, while maintaining competitive control performance. This work establishes a research agenda for RL systems designed from inception as queryable knowledge bases, with implications for interpretability, verification, and human-AI collaboration. 

**Abstract (ZH)**: 强化学习传统上专注于单一目标：学习选择行动以最大化奖励的策略。我们通过提出一个问题来挑战这一范式：如果我们明确将RL系统设计为可以回答其环境的各种查询的推理引擎，会怎样？在确定性环境中，训练过的智能体隐含地编码了丰富的可达性、距离、值和动力学知识——但当前的架构并未设计成能够高效地暴露这些信息。我们引入了查询条件确定性推理网络（QDIN），这是一种统一架构，将不同类型的查询（策略、可达性、路径、比较）视为一等公民，为每种推理模式优化专门的神经模块。我们的关键实验证据揭示了一个根本性的解耦：即使控制性能不佳（31%的回报），推理准确性也可以达到近乎完美的水平（99%的可达性IoU），表明用于准确世界知识的表示与用于最优控制的表示是不同的。实验表明，查询专门化架构在保持竞争力的控制性能的同时，优于统一模型和事后提取方法。本工作确立了一个研究议程，即将RL系统从设计之初就作为一个可查询的知识库，这将对可解释性、验证和人机协作产生影响。 

---
# Thinking Forward and Backward: Multi-Objective Reinforcement Learning for Retrieval-Augmented Reasoning 

**Title (ZH)**: 从前往后和从后往前思考：用于检索增强推理的多目标强化学习 

**Authors**: Wenda Wei, Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Lixin Su, Shuaiqiang Wang, Dawei Yin, Maarten de Rijke, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.09109)  

**Abstract**: Retrieval-augmented generation (RAG) has proven to be effective in mitigating hallucinations in large language models, yet its effectiveness remains limited in complex, multi-step reasoning this http URL efforts have incorporated search-based interactions into RAG, enabling iterative reasoning with real-time retrieval. Most approaches rely on outcome-based supervision, offering no explicit guidance for intermediate steps. This often leads to reward hacking and degraded response quality. We propose Bi-RAR, a novel retrieval-augmented reasoning framework that evaluates each intermediate step jointly in both forward and backward directions. To assess the information completeness of each step, we introduce a bidirectional information distance grounded in Kolmogorov complexity, approximated via language model generation probabilities. This quantification measures both how far the current reasoning is from the answer and how well it addresses the question. To optimize reasoning under these bidirectional signals, we adopt a multi-objective reinforcement learning framework with a cascading reward structure that emphasizes early trajectory alignment. Empirical results on seven question answering benchmarks demonstrate that Bi-RAR surpasses previous methods and enables efficient interaction and reasoning with the search engine during training and inference. 

**Abstract (ZH)**: 检索增强生成（RAG）在减轻大型语言模型幻觉方面已被证明是有效的，但在复杂、多步推理方面其效果仍然有限。通过将基于搜索的交互融入RAG，可以实现迭代推理和实时检索。大多数方法依赖于结果导向的监督，无法为中间步骤提供明确的指导。这往往导致奖励黑客行为和响应质量下降。我们提出了双向检索增强推理（Bi-RAR）框架，这是一个新颖的检索增强推理框架，联合从正向和反向两个方向评估每一步的中间步骤。为了评估每一步的信息完整性，我们引入了一种基于kolmogorov复杂性的双向信息距离，通过语言模型生成概率进行近似计算。这种量化测量了当前推理与答案之间的差距以及对问题的解决程度。为了在这些双向信号下优化推理，我们采用了一种具有级联奖励结构的多目标强化学习框架，强调早期轨迹对齐。在七个问答基准数据集上的实验证明，Bi-RAR 超越了先前的方法，并在训练和推理期间与搜索引擎实现了高效交互和推理。 

---
# PAN: A World Model for General, Interactable, and Long-Horizon World Simulation 

**Title (ZH)**: PAN：一种用于通用、可交互和长时 horizon 世界模拟的世界模型 

**Authors**: PAN Team Institute of Foundation Models, Jiannan Xiang, Yi Gu, Zihan Liu, Zeyu Feng, Qiyue Gao, Yiyan Hu, Benhao Huang, Guangyi Liu, Yichi Yang, Kun Zhou, Davit Abrahamyan, Arif Ahmad, Ganesh Bannur, Junrong Chen, Kimi Chen, Mingkai Deng, Ruobing Han, Xinqi Huang, Haoqiang Kang, Zheqi Li, Enze Ma, Hector Ren, Yashowardhan Shinde, Rohan Shingre, Ramsundar Tanikella, Kaiming Tao, Dequan Yang, Xinle Yu, Cong Zeng, Binglin Zhou, Hector Liu, Zhiting Hu, Eric P. Xing  

**Link**: [PDF](https://arxiv.org/pdf/2511.09057)  

**Abstract**: A world model enables an intelligent agent to imagine, predict, and reason about how the world evolves in response to its actions, and accordingly to plan and strategize. While recent video generation models produce realistic visual sequences, they typically operate in the prompt-to-full-video manner without causal control, interactivity, or long-horizon consistency required for purposeful reasoning. Existing world modeling efforts, on the other hand, often focus on restricted domains (e.g., physical, game, or 3D-scene dynamics) with limited depth and controllability, and struggle to generalize across diverse environments and interaction formats. In this work, we introduce PAN, a general, interactable, and long-horizon world model that predicts future world states through high-quality video simulation conditioned on history and natural language actions. PAN employs the Generative Latent Prediction (GLP) architecture that combines an autoregressive latent dynamics backbone based on a large language model (LLM), which grounds simulation in extensive text-based knowledge and enables conditioning on language-specified actions, with a video diffusion decoder that reconstructs perceptually detailed and temporally coherent visual observations, to achieve a unification between latent space reasoning (imagination) and realizable world dynamics (reality). Trained on large-scale video-action pairs spanning diverse domains, PAN supports open-domain, action-conditioned simulation with coherent, long-term dynamics. Extensive experiments show that PAN achieves strong performance in action-conditioned world simulation, long-horizon forecasting, and simulative reasoning compared to other video generators and world models, taking a step towards general world models that enable predictive simulation of future world states for reasoning and acting. 

**Abstract (ZH)**: 一个世界模型使智能代理能够想象、预测和推理世界在其行为响应下的演变，并据此规划和制定策略。尽管近期的视频生成模型能够生成逼真的视觉序列，但它们通常以提示到完整视频的方式运行，并缺乏用于有目的推理所需的因果控制、交互性和长时一致性。现有的世界建模努力往往集中在受限制的领域（如物理、游戏或3D场景动力学）上，这些领域深度有限且可控性差，难以在多种环境和交互格式之间泛化。在本项工作中，我们引入了PAN，这是一种通用、可交互和长时未来预测的世界模型，它通过基于历史和自然语言动作的高质量视频模拟来预测未来的世 界状态。PAN采用生成潜空间预测（GLP）架构，该架构结合了一个基于大规模语言模型（LLM）的自回归潜动力学骨干网络，以广泛的文本知识为基础，使模拟得以实现并能够根据语言规定的动作进行条件化，以及一个视频扩散解码器，该解码器能够重建知觉详细且时间一致的视觉观察。PAN实现了潜空间推理（想象）与可实现的世界动力学（现实）之间的统一。PAN在多种领域的大规模视频-动作对上进行训练，支持开放领域的、根据动作条件化的仿真，具有连贯的长时动态。大量实验表明，PAN在动作条件化的世界仿真、长时预测和仿真正推理方面相较于其他视频生成器和世界模型表现优异，朝着能够进行预测仿真实现未来世界状态推理和行动的通用世界模型迈出了一步。 

---
# Causally-Grounded Dual-Path Attention Intervention for Object Hallucination Mitigation in LVLMs 

**Title (ZH)**: 因果导向的双路径注意力干预方法以减轻低级视觉语言模型中的物体错幻覺 

**Authors**: Liu Yu, Zhonghao Chen, Ping Kuang, Zhikun Feng, Fan Zhou, Lan Wang, Gillian Dobbie  

**Link**: [PDF](https://arxiv.org/pdf/2511.09018)  

**Abstract**: Object hallucination remains a critical challenge in Large Vision-Language Models (LVLMs), where models generate content inconsistent with visual inputs. Existing language-decoder based mitigation approaches often regulate visual or textual attention independently, overlooking their interaction as two key causal factors. To address this, we propose Owl (Bi-mOdal attention reWeighting for Layer-wise hallucination mitigation), a causally-grounded framework that models hallucination process via a structural causal graph, treating decomposed visual and textual attentions as mediators. We introduce VTACR (Visual-to-Textual Attention Contribution Ratio), a novel metric that quantifies the modality contribution imbalance during decoding. Our analysis reveals that hallucinations frequently occur in low-VTACR scenarios, where textual priors dominate and visual grounding is weakened. To mitigate this, we design a fine-grained attention intervention mechanism that dynamically adjusts token- and layer-wise attention guided by VTACR signals. Finally, we propose a dual-path contrastive decoding strategy: one path emphasizes visually grounded predictions, while the other amplifies hallucinated ones -- letting visual truth shine and hallucination collapse. Experimental results on the POPE and CHAIR benchmarks show that Owl achieves significant hallucination reduction, setting a new SOTA in faithfulness while preserving vision-language understanding capability. Our code is available at this https URL 

**Abstract (ZH)**: Bi模态注意力重加权以减少层级幻觉：Owl框架 

---
# Convergence dynamics of Agent-to-Agent Interactions with Misaligned objectives 

**Title (ZH)**: 代理间交互的动力学收敛性与目标不一致 

**Authors**: Romain Cosentino, Sarath Shekkizhar, Adam Earle  

**Link**: [PDF](https://arxiv.org/pdf/2511.08710)  

**Abstract**: We develop a theoretical framework for agent-to-agent interactions in multi-agent scenarios. We consider the setup in which two language model based agents perform iterative gradient updates toward their respective objectives in-context, using the output of the other agent as input. We characterize the generation dynamics associated with the interaction when the agents have misaligned objectives, and show that this results in a biased equilibrium where neither agent reaches its target - with the residual errors predictable from the objective gap and the geometry induced by the prompt of each agent. We establish the conditions for asymmetric convergence and provide an algorithm that provably achieves an adversarial result, producing one-sided success. Experiments with trained transformer models as well as GPT$5$ for the task of in-context linear regression validate the theory. Our framework presents a setup to study, predict, and defend multi-agent systems; explicitly linking prompt design and interaction setup to stability, bias, and robustness. 

**Abstract (ZH)**: 我们开发了一个代理到代理交互的理论框架，应用于多代理场景。我们考虑基于语言模型的两个代理在上下文中迭代更新其目标的过程，其中一个代理的输出作为另一个代理的输入。当代理的目标不一致时，我们描述了生成动态，并证明这导致一个有偏的平衡点，其中没有任何一个代理达到其目标，剩余误差可以预测来自目标差距和每个代理提示诱导的几何结构。我们建立了非对称收敛的条件，并提供了一个能证明达成对抗结果的算法，产生单方面成功。使用训练好的变换器模型以及GPT$5$进行上下文线性回归的任务验证了该理论。我们的框架提供了一个研究、预测和防御多代理系统的设置，明确地将提示设计和交互设置与稳定性、偏差和鲁棒性联系起来。 

---
