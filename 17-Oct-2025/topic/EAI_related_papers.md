# CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions 

**Title (ZH)**: 基于控制屏障函数的训练安全性滤波强化学习(CBF-RL) 

**Authors**: Lizhi Yang, Blake Werner, Massimiliano de Sa Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2510.14959)  

**Abstract**: Reinforcement learning (RL), while powerful and expressive, can often prioritize performance at the expense of safety. Yet safety violations can lead to catastrophic outcomes in real-world deployments. Control Barrier Functions (CBFs) offer a principled method to enforce dynamic safety -- traditionally deployed \emph{online} via safety filters. While the result is safe behavior, the fact that the RL policy does not have knowledge of the CBF can lead to conservative behaviors. This paper proposes CBF-RL, a framework for generating safe behaviors with RL by enforcing CBFs \emph{in training}. CBF-RL has two key attributes: (1) minimally modifying a nominal RL policy to encode safety constraints via a CBF term, (2) and safety filtering of the policy rollouts in training. Theoretically, we prove that continuous-time safety filters can be deployed via closed-form expressions on discrete-time roll-outs. Practically, we demonstrate that CBF-RL internalizes the safety constraints in the learned policy -- both enforcing safer actions and biasing towards safer rewards -- enabling safe deployment without the need for an online safety filter. We validate our framework through ablation studies on navigation tasks and on the Unitree G1 humanoid robot, where CBF-RL enables safer exploration, faster convergence, and robust performance under uncertainty, enabling the humanoid robot to avoid obstacles and climb stairs safely in real-world settings without a runtime safety filter. 

**Abstract (ZH)**: 基于控制障碍函数的强化学习：通过在训练中强制执行控制障碍函数生成安全行为 

---
# From Language to Locomotion: Retargeting-free Humanoid Control via Motion Latent Guidance 

**Title (ZH)**: 从语言到运动：基于运动潜在指导的无靶向人体控制 

**Authors**: Zhe Li, Cheng Chi, Yangyang Wei, Boan Zhu, Yibo Peng, Tao Huang, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14952)  

**Abstract**: Natural language offers a natural interface for humanoid robots, but existing language-guided humanoid locomotion pipelines remain cumbersome and unreliable. They typically decode human motion, retarget it to robot morphology, and then track it with a physics-based controller. However, this multi-stage process is prone to cumulative errors, introduces high latency, and yields weak coupling between semantics and control. These limitations call for a more direct pathway from language to action, one that eliminates fragile intermediate stages. Therefore, we present RoboGhost, a retargeting-free framework that directly conditions humanoid policies on language-grounded motion latents. By bypassing explicit motion decoding and retargeting, RoboGhost enables a diffusion-based policy to denoise executable actions directly from noise, preserving semantic intent and supporting fast, reactive control. A hybrid causal transformer-diffusion motion generator further ensures long-horizon consistency while maintaining stability and diversity, yielding rich latent representations for precise humanoid behavior. Extensive experiments demonstrate that RoboGhost substantially reduces deployment latency, improves success rates and tracking accuracy, and produces smooth, semantically aligned locomotion on real humanoids. Beyond text, the framework naturally extends to other modalities such as images, audio, and music, providing a general foundation for vision-language-action humanoid systems. 

**Abstract (ZH)**: 无需重定位的基于语言的类人机器人运动框架：RoboGhost 

---
# Architecture Is All You Need: Diversity-Enabled Sweet Spots for Robust Humanoid Locomotion 

**Title (ZH)**: Architecture Is All You Need: 基于多样性的稳健人形机器人行走优化nellested 

**Authors**: Blake Werner, Lizhi Yang, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2510.14947)  

**Abstract**: Robust humanoid locomotion in unstructured environments requires architectures that balance fast low-level stabilization with slower perceptual decision-making. We show that a simple layered control architecture (LCA), a proprioceptive stabilizer running at high rate, coupled with a compact low-rate perceptual policy, enables substantially more robust performance than monolithic end-to-end designs, even when using minimal perception encoders. Through a two-stage training curriculum (blind stabilizer pretraining followed by perceptual fine-tuning), we demonstrate that layered policies consistently outperform one-stage alternatives in both simulation and hardware. On a Unitree G1 humanoid, our approach succeeds across stair and ledge tasks where one-stage perceptual policies fail. These results highlight that architectural separation of timescales, rather than network scale or complexity, is the key enabler for robust perception-conditioned locomotion. 

**Abstract (ZH)**: 无结构环境中鲁棒的人形机器人运动需要平衡快速低层级稳定与缓慢感知决策的架构。我们展示了简单分层控制架构（LCA）、高频率的本体感受稳定器与低频的感知策略相结合，即使使用最小的感知编码器，也能实现比端到端单一架构更鲁棒的表现。通过两阶段培训课程（盲稳定器预训练后进行感知微调），我们证明了分层策略在仿真和硬件中的一致性表现优于单阶段替代方案。在Unitree G1人形机器人上，我们的方法在一台阶感知策略失败的楼梯和凸起任务中取得了成功。这些结果强调了时间尺度的架构分离而非网络规模或复杂性是实现鲁棒感知条件下的运动的关键。 

---
# VLA^2: Empowering Vision-Language-Action Models with an Agentic Framework for Unseen Concept Manipulation 

**Title (ZH)**: VLA^2：为不可见概念操控赋能的代理框架视觉-语言-动作模型 

**Authors**: Han Zhao, Jiaxuan Zhang, Wenxuan Song, Pengxiang Ding, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14902)  

**Abstract**: Current vision-language-action (VLA) models, pre-trained on large-scale robotic data, exhibit strong multi-task capabilities and generalize well to variations in visual and language instructions for manipulation. However, their success rate drops significantly when faced with object concepts outside the training data, such as unseen object descriptions and textures in the dataset. To address this, we propose a novel agentic framework, VLA^2, which leverages OpenVLA as the execution backbone and effectively leverages external modules such as web retrieval and object detection to provide visual and textual knowledge about target objects to the VLA. This approach mitigates generalization failure when handling out-of-distribution objects. Based on the LIBERO simulation environment, we introduced novel objects and object descriptions to construct a new evaluation benchmark with three difficulty levels to test the effectiveness of our method. Our framework successfully outperformed the current state-of-the-art models on our designed hard-level generalization benchmark. Compared to the standalone OpenVLA baseline, VLA^2 achieves a 44.2% improvement in the success rate in the hard-level benchmark and an average improvement of 20.2% in all customized environments without any performance degradation on in-domain tasks. Project website: this https URL. 

**Abstract (ZH)**: 当前的视觉-语言-动作（VLA）模型，基于大规模机器人数据预训练，展示了强大的多任务能力，并且在操作任务中能够很好地泛化到视觉和语言指令的变化。然而，当面对训练数据之外的对象概念，如未见过的对象描述和数据集中未见的纹理时，其成功率显著下降。为了解决这一问题，我们提出了一种新的自主框架VLA^2，该框架以OpenVLA作为执行骨干，并有效利用网页检索和目标检测等外部模块，向VLA提供目标对象的视觉和文本知识，从而缓解处理分布外对象时的泛化失败问题。基于LIBERO模拟环境，我们引入了新的对象和对象描述，构建了一个具有三个难度级别的新评估基准，以测试我们方法的有效性。我们的框架在我们设计的高难度泛化基准上成功超越了当前的最先进模型。与独立的OpenVLA基线相比，VLA^2在高难度基准上的成功率提高了44.2%，在所有定制环境中平均提高了20.2%，且不影响领域内任务的性能。项目网址：this https URL。 

---
# SkyDreamer: Interpretable End-to-End Vision-Based Drone Racing with Model-Based Reinforcement Learning 

**Title (ZH)**: SkyDreamer：基于模型的强化学习端到端可解释无人机竞速 

**Authors**: Aderik Verraest, Stavrow Bahnam, Robin Ferede, Guido de Croon, Christophe De Wagter  

**Link**: [PDF](https://arxiv.org/pdf/2510.14783)  

**Abstract**: Autonomous drone racing (ADR) systems have recently achieved champion-level performance, yet remain highly specific to drone racing. While end-to-end vision-based methods promise broader applicability, no system to date simultaneously achieves full sim-to-real transfer, onboard execution, and champion-level performance. In this work, we present SkyDreamer, to the best of our knowledge, the first end-to-end vision-based ADR policy that maps directly from pixel-level representations to motor commands. SkyDreamer builds on informed Dreamer, a model-based reinforcement learning approach where the world model decodes to privileged information only available during training. By extending this concept to end-to-end vision-based ADR, the world model effectively functions as an implicit state and parameter estimator, greatly improving interpretability. SkyDreamer runs fully onboard without external aid, resolves visual ambiguities by tracking progress using the state decoded from the world model's hidden state, and requires no extrinsic camera calibration, enabling rapid deployment across different drones without retraining. Real-world experiments show that SkyDreamer achieves robust, high-speed flight, executing tight maneuvers such as an inverted loop, a split-S and a ladder, reaching speeds of up to 21 m/s and accelerations of up to 6 g. It further demonstrates a non-trivial visual sim-to-real transfer by operating on poor-quality segmentation masks, and exhibits robustness to battery depletion by accurately estimating the maximum attainable motor RPM and adjusting its flight path in real-time. These results highlight SkyDreamer's adaptability to important aspects of the reality gap, bringing robustness while still achieving extremely high-speed, agile flight. 

**Abstract (ZH)**: 自主无人机竞速（ADR）系统 recently 已经达到了冠军级别的性能，但仍高度特化于无人机竞速。虽然端到端基于视觉的方法有望实现更广泛的应用，但至今为止还没有一个系统能够同时实现完整的仿真到真实世界的转移、板载执行和冠军级性能。在本工作中，我们提出了 SkyDreamer，据我们所知，这是第一个端到端基于视觉的 ADR 策略，能够直接从像素级表示映射到电机命令。SkyDreamer 以有信息量的 Dreamer 为基础，这是一种基于模型的强化学习方法，其中世界模型仅在训练期间才解码特权信息。通过将这一概念扩展到端到端基于视觉的 ADR 中，世界模型有效地充当了隐式状态和参数估算器，极大地提高了可解释性。SkyDreamer 完全在板载运行，不依赖外部帮助，通过跟踪由世界模型隐状态解码的状态来解决视觉模糊性，且无需额外的相机校准，从而可以在不同的无人机上快速部署而无需重新训练。实验证明，SkyDreamer 实现了稳健而高速的飞行，执行诸如倒立环、分割 S 和梯子等精确机动，达到了最大速度 21 m/s 和最大加速度 6 g。此外，SkyDreamer 进一步展示了在质量较差的分割掩模上进行非平凡的仿真到现实世界转换的能力，并通过准确估计最大可实现的电机 RPM 并实时调整飞行路径展示了对电池耗尽的鲁棒性。这些结果突显了 SkyDreamer 在现实差距关键方面的适应性，同时仍然实现了极其高速和敏捷的飞行。 

---
# Leveraging Neural Descriptor Fields for Learning Contact-Aware Dynamic Recovery 

**Title (ZH)**: 利用神经描述符场学习接触感知动态恢复 

**Authors**: Fan Yang, Zixuan Huang, Abhinav Kumar, Sergio Aguilera Marinovic, Soshi Iba, Rana Soltani Zarrin, Dmitry Berenson  

**Link**: [PDF](https://arxiv.org/pdf/2510.14768)  

**Abstract**: Real-world dexterous manipulation often encounters unexpected errors and disturbances, which can lead to catastrophic failures, such as dropping the manipulated object. To address this challenge, we focus on the problem of catching a falling object while it remains within grasping range and, importantly, resetting the system to a configuration favorable for resuming the primary manipulation task. We propose Contact-Aware Dynamic Recovery (CADRE), a reinforcement learning framework that incorporates a Neural Descriptor Field (NDF)-inspired module to extract implicit contact features. Compared to methods that rely solely on object pose or point cloud input, NDFs can directly reason about finger-object correspondence and adapt to different object geometries. Our experiments show that incorporating contact features improves training efficiency, enhances convergence performance for RL training, and ultimately leads to more successful recoveries. Additionally, we demonstrate that CADRE can generalize zero-shot to unseen objects with different geometries. 

**Abstract (ZH)**: Real-world灵巧操作经常遇到意外的错误和干扰，这可能导致灾难性的失败，比如抓取对象时将其掉落。为应对这一挑战，我们关注在对象仍处于抓取范围内时捕捉掉落对象的问题，并且重要的是，将系统重置到有利于恢复主要操作任务的配置。我们提出了接触感知动态恢复（CADRE），这是一种 reinforcement learning 框架，结合了受 Neural Descriptor Field (NDF) 启发的模块以提取隐式接触特征。与仅依赖对象姿态或点云输入的方法相比，NDF们可以直接推理指对象对应关系，并适应不同对象的几何形状。我们的实验表明，集成接触特征可以提高训练效率，增强 RL 训练的收敛性能，并最终实现更成功的恢复。此外，我们展示了 CADRE 能够零样本泛化到不同几何形状的未见过的对象。 

---
# When Planners Meet Reality: How Learned, Reactive Traffic Agents Shift nuPlan Benchmarks 

**Title (ZH)**: 规划者遇现实：学习到的反应式交通代理如何影响nuPlan基准测试 

**Authors**: Steffen Hagedorn, Luka Donkov, Aron Distelzweig, Alexandru P. Condurache  

**Link**: [PDF](https://arxiv.org/pdf/2510.14677)  

**Abstract**: Planner evaluation in closed-loop simulation often uses rule-based traffic agents, whose simplistic and passive behavior can hide planner deficiencies and bias rankings. Widely used IDM agents simply follow a lead vehicle and cannot react to vehicles in adjacent lanes, hindering tests of complex interaction capabilities. We address this issue by integrating the state-of-the-art learned traffic agent model SMART into nuPlan. Thus, we are the first to evaluate planners under more realistic conditions and quantify how conclusions shift when narrowing the sim-to-real gap. Our analysis covers 14 recent planners and established baselines and shows that IDM-based simulation overestimates planning performance: nearly all scores deteriorate. In contrast, many planners interact better than previously assumed and even improve in multi-lane, interaction-heavy scenarios like lane changes or turns. Methods trained in closed-loop demonstrate the best and most stable driving performance. However, when reaching their limits in augmented edge-case scenarios, all learned planners degrade abruptly, whereas rule-based planners maintain reasonable basic behavior. Based on our results, we suggest SMART-reactive simulation as a new standard closed-loop benchmark in nuPlan and release the SMART agents as a drop-in alternative to IDM at this https URL. 

**Abstract (ZH)**: 闭环仿真中计划器评估通常使用基于规则的交通代理，其简单且被动的行为可能掩盖计划器缺陷并偏倚排名。广泛使用的 IDM 代理 merely 仅跟随前车，不能响应相邻车道的车辆，阻碍了对复杂交互能力的测试。我们通过将最先进的学习交通代理模型 SMART 集成到 nuPlan 中解决了这一问题，从而成为第一个在更现实条件下评估计划器并在缩小仿真实际差距时量化结论变化的研究。我们的分析涵盖14个近期计划器和基准，表明基于 IDM 的仿真高估了计划性能：几乎所有分数都下降了。相反，许多计划器的交互能力优于之前假设的，在多车道、交互密集的场景（如变道或转弯）中甚至有所提升。在闭环中训练的方法展示出了最佳且最稳定的驾驶性能。然而，当在扩展的边缘情况下达到极限时，所有学习的计划器会突然退化，而基于规则的计划器则保持基本的合理行为。基于我们的结果，我们建议将 SMART 反应式仿真作为 nuPlan 中新的标准闭环基准，并在此 <https://> 释放 SMART 代理作为 IDM 的现成替代品。 

---
# Generative Models From and For Sampling-Based MPC: A Bootstrapped Approach For Adaptive Contact-Rich Manipulation 

**Title (ZH)**: 基于采样 MPC 的生成模型：一种适应性接触丰富操作的-bootstrap 方法 

**Authors**: Lara Brudermüller, Brandon Hung, Xinghao Zhu, Jiuguang Wang, Nick Hawes, Preston Culbertson, Simon Le Cleac'h  

**Link**: [PDF](https://arxiv.org/pdf/2510.14643)  

**Abstract**: We present a generative predictive control (GPC) framework that amortizes sampling-based Model Predictive Control (SPC) by bootstrapping it with conditional flow-matching models trained on SPC control sequences collected in simulation. Unlike prior work relying on iterative refinement or gradient-based solvers, we show that meaningful proposal distributions can be learned directly from noisy SPC data, enabling more efficient and informed sampling during online planning. We further demonstrate, for the first time, the application of this approach to real-world contact-rich loco-manipulation with a quadruped robot. Extensive experiments in simulation and on hardware show that our method improves sample efficiency, reduces planning horizon requirements, and generalizes robustly across task variations. 

**Abstract (ZH)**: 我们提出了一种生成预测控制(GPC)框架，通过使用基于模拟收集的SPC控制序列训练的条件流匹配模型来加速基于采样的模型预测控制(SPC)。我们展示了可以直接从噪声SPC数据中学习有意义的提案分布，从而在在线规划期间实现更高效的、更有信息量的采样。此外，我们首次展示了这种方法在四足机器人进行接触丰富型移动操作中的应用。大量的模拟和硬件实验表明，我们的方法提高了采样效率，减少了规划 horizon 的要求，并且能够稳健地泛化到任务变化。 

---
# GOPLA: Generalizable Object Placement Learning via Synthetic Augmentation of Human Arrangement 

**Title (ZH)**: GOPLA: 通过人工排列的合成增强实现可泛化的物体放置学习 

**Authors**: Yao Zhong, Hanzhi Chen, Simon Schaefer, Anran Zhang, Stefan Leutenegger  

**Link**: [PDF](https://arxiv.org/pdf/2510.14627)  

**Abstract**: Robots are expected to serve as intelligent assistants, helping humans with everyday household organization. A central challenge in this setting is the task of object placement, which requires reasoning about both semantic preferences (e.g., common-sense object relations) and geometric feasibility (e.g., collision avoidance). We present GOPLA, a hierarchical framework that learns generalizable object placement from augmented human demonstrations. A multi-modal large language model translates human instructions and visual inputs into structured plans that specify pairwise object relationships. These plans are then converted into 3D affordance maps with geometric common sense by a spatial mapper, while a diffusion-based planner generates placement poses guided by test-time costs, considering multi-plan distributions and collision avoidance. To overcome data scarcity, we introduce a scalable pipeline that expands human placement demonstrations into diverse synthetic training data. Extensive experiments show that our approach improves placement success rates by 30.04 percentage points over the runner-up, evaluated on positioning accuracy and physical plausibility, demonstrating strong generalization across a wide range of real-world robotic placement scenarios. 

**Abstract (ZH)**: 机器人预期作为智能助手，帮助人类进行日常家庭组织。在这个场景下的一个核心挑战是物体放置task，这要求同时考虑语义偏好（例如，常识性物体关系）和几何可行性（例如，碰撞避免）。我们提出了GOPLA，一种分级框架，通过增强的人类示范学习可泛化的物体放置。多模态大语言模型将人类指令和视觉输入转化为结构化计划，指定物体对之间的关系。这些计划随后通过空间映射转换为包含几何常识的3D可利用性地图，而基于扩散的计划生成放置姿态，考虑多计划分布和碰撞避免。为了克服数据稀缺性，我们引入了一种可扩展的流水线，将人类的物体放置示范扩展为多样化的合成训练数据。广泛的实验结果显示，我们的方法在定位准确性和物理可 plausibility 方面的置信成功率提高了30.04个百分点，展示了在广泛的真实世界机器人放置场景中的强大泛化能力。 

---
# Accelerated Multi-Modal Motion Planning Using Context-Conditioned Diffusion Models 

**Title (ZH)**: 基于上下文条件化的扩散模型加速多模态运动规划 

**Authors**: Edward Sandra, Lander Vanroye, Dries Dirckx, Ruben Cartuyvels, Jan Swevers, Wilm Decré  

**Link**: [PDF](https://arxiv.org/pdf/2510.14615)  

**Abstract**: Classical methods in robot motion planning, such as sampling-based and optimization-based methods, often struggle with scalability towards higher-dimensional state spaces and complex environments. Diffusion models, known for their capability to learn complex, high-dimensional and multi-modal data distributions, provide a promising alternative when applied to motion planning problems and have already shown interesting results. However, most of the current approaches train their model for a single environment, limiting their generalization to environments not seen during training. The techniques that do train a model for multiple environments rely on a specific camera to provide the model with the necessary environmental information and therefore always require that sensor. To effectively adapt to diverse scenarios without the need for retraining, this research proposes Context-Aware Motion Planning Diffusion (CAMPD). CAMPD leverages a classifier-free denoising probabilistic diffusion model, conditioned on sensor-agnostic contextual information. An attention mechanism, integrated in the well-known U-Net architecture, conditions the model on an arbitrary number of contextual parameters. CAMPD is evaluated on a 7-DoF robot manipulator and benchmarked against state-of-the-art approaches on real-world tasks, showing its ability to generalize to unseen environments and generate high-quality, multi-modal trajectories, at a fraction of the time required by existing methods. 

**Abstract (ZH)**: 基于扩散模型的Context-Aware运动规划方法：无需重新训练的有效适应多样化场景 

---
# Proprioceptive Image: An Image Representation of Proprioceptive Data from Quadruped Robots for Contact Estimation Learning 

**Title (ZH)**: 本体感觉图像： quadruped 机器人本体感觉数据的图像表示及其接触估计学习。 

**Authors**: Gabriel Fischer Abati, João Carlos Virgolino Soares, Giulio Turrisi, Victor Barasuol, Claudio Semini  

**Link**: [PDF](https://arxiv.org/pdf/2510.14612)  

**Abstract**: This paper presents a novel approach for representing proprioceptive time-series data from quadruped robots as structured two-dimensional images, enabling the use of convolutional neural networks for learning locomotion-related tasks. The proposed method encodes temporal dynamics from multiple proprioceptive signals, such as joint positions, IMU readings, and foot velocities, while preserving the robot's morphological structure in the spatial arrangement of the image. This transformation captures inter-signal correlations and gait-dependent patterns, providing a richer feature space than direct time-series processing. We apply this concept in the problem of contact estimation, a key capability for stable and adaptive locomotion on diverse terrains. Experimental evaluations on both real-world datasets and simulated environments show that our image-based representation consistently enhances prediction accuracy and generalization over conventional sequence-based models, underscoring the potential of cross-modal encoding strategies for robotic state learning. Our method achieves superior performance on the contact dataset, improving contact state accuracy from 87.7% to 94.5% over the recently proposed MI-HGNN method, using a 15 times shorter window size. 

**Abstract (ZH)**: 本文提出了一种新颖的方法，将四足机器人 proprioceptive 时间序列数据表示为结构化的二维图像，从而能够使用卷积神经网络学习与步态相关的任务。所提出的方法编码了多个 proprioceptive 信号（如关节位置、IMU 读数和足部速度）的时间动态，同时在图像的空间排列中保留了机器人的形态结构。这种转换捕捉了信号间的相关性和步态依赖的模式，提供了比直接时间序列处理更丰富的特征空间。我们将这一概念应用于接触估计问题，这是在多种地形上实现稳定和自适应步态的关键能力。我们在现实世界数据集和模拟环境中进行的实验评估表明，与传统的基于序列的模型相比，我们的图像表示方法在预测准确性和泛化能力上均得到了提升，突显了跨模态编码策略在机器人状态学习中的潜在价值。我们的方法在接触数据集上表现出更优的性能，在比最近提出的 MI-HGNN 方法短15倍的窗口大小下，接触状态准确率从87.7%提高到了94.5%。 

---
# QuASH: Using Natural-Language Heuristics to Query Visual-Language Robotic Maps 

**Title (ZH)**: QuASH: 使用自然语言启发式查询视觉语言机器人地图 

**Authors**: Matti Pekkanen, Francesco Verdoja, Ville Kyrki  

**Link**: [PDF](https://arxiv.org/pdf/2510.14546)  

**Abstract**: Embeddings from Visual-Language Models are increasingly utilized to represent semantics in robotic maps, offering an open-vocabulary scene understanding that surpasses traditional, limited labels. Embeddings enable on-demand querying by comparing embedded user text prompts to map embeddings via a similarity metric. The key challenge in performing the task indicated in a query is that the robot must determine the parts of the environment relevant to the query.
This paper proposes a solution to this challenge. We leverage natural-language synonyms and antonyms associated with the query within the embedding space, applying heuristics to estimate the language space relevant to the query, and use that to train a classifier to partition the environment into matches and non-matches. We evaluate our method through extensive experiments, querying both maps and standard image benchmarks. The results demonstrate increased queryability of maps and images. Our querying technique is agnostic to the representation and encoder used, and requires limited training. 

**Abstract (ZH)**: 视觉语言模型嵌入在机器人地图中的应用：基于嵌入空间的自然语言同义词与反义词扩展的环境查询方法 

---
# Towards Adaptable Humanoid Control via Adaptive Motion Tracking 

**Title (ZH)**: 基于自适应运动跟踪的可适应 humanoid 控制研究 

**Authors**: Tao Huang, Huayi Wang, Junli Ren, Kangning Yin, Zirui Wang, Xiao Chen, Feiyu Jia, Wentao Zhang, Junfeng Long, Jingbo Wang, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14454)  

**Abstract**: Humanoid robots are envisioned to adapt demonstrated motions to diverse real-world conditions while accurately preserving motion patterns. Existing motion prior approaches enable well adaptability with a few motions but often sacrifice imitation accuracy, whereas motion-tracking methods achieve accurate imitation yet require many training motions and a test-time target motion to adapt. To combine their strengths, we introduce AdaMimic, a novel motion tracking algorithm that enables adaptable humanoid control from a single reference motion. To reduce data dependence while ensuring adaptability, our method first creates an augmented dataset by sparsifying the single reference motion into keyframes and applying light editing with minimal physical assumptions. A policy is then initialized by tracking these sparse keyframes to generate dense intermediate motions, and adapters are subsequently trained to adjust tracking speed and refine low-level actions based on the adjustment, enabling flexible time warping that further improves imitation accuracy and adaptability. We validate these significant improvements in our approach in both simulation and the real-world Unitree G1 humanoid robot in multiple tasks across a wide range of adaptation conditions. Videos and code are available at this https URL. 

**Abstract (ZH)**: 类人机器人通过单个参考动作实现适应性运动模仿并保留运动模式。 

---
# SUM-AgriVLN: Spatial Understanding Memory for Agricultural Vision-and-Language Navigation 

**Title (ZH)**: SUM-AgriVLN: 空间理解记忆在农业视觉与语言导航中的应用 

**Authors**: Xiaobei Zhao, Xingqi Lyu, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14357)  

**Abstract**: Agricultural robots are emerging as powerful assistants across a wide range of agricultural tasks, nevertheless, still heavily rely on manual operation or fixed rail systems for movement. The AgriVLN method and the A2A benchmark pioneeringly extend Vision-and-Language Navigation (VLN) to the agricultural domain, enabling robots to navigate to the target positions following the natural language instructions. In practical agricultural scenarios, navigation instructions often repeatedly occur, yet AgriVLN treat each instruction as an independent episode, overlooking the potential of past experiences to provide spatial context for subsequent ones. To bridge this gap, we propose the method of Spatial Understanding Memory for Agricultural Vision-and-Language Navigation (SUM-AgriVLN), in which the SUM module employs spatial understanding and save spatial memory through 3D reconstruction and representation. When evaluated on the A2A benchmark, our SUM-AgriVLN effectively improves Success Rate from 0.47 to 0.54 with slight sacrifice on Navigation Error from 2.91m to 2.93m, demonstrating the state-of-the-art performance in the agricultural domain. Code: this https URL. 

**Abstract (ZH)**: 农业机器人正在成为各种农业任务的强大助手，但仍主要依赖手动操作或固定轨道系统进行移动。AgriVLN方法和A2A基准首次将视觉语言导航（VLN）扩展到农业领域，使机器人能够根据自然语言指令导航到目标位置。在实际 agricultural 场景中，导航指令经常重复出现，但 AgriVLN 将每条指令视为独立的场景，忽视了过往经验可能为后续指令提供的空间上下文。为解决这一问题，我们提出了一种农业视觉语言导航中的空间理解记忆方法（SUM-AgriVLN），其中 SUM 模块通过三维重建和表示来实现空间理解并保存空间记忆。在 A2A 基准上的评估表明，我们的 SUM-AgriVLN 能将成功率从 0.47 提高到 0.54，同时导航误差略有增加，从 2.91m 增加到 2.93m，展示了农业领域的先进性能。代码：this https URL。 

---
# Risk-Aware Reinforcement Learning with Bandit-Based Adaptation for Quadrupedal Locomotion 

**Title (ZH)**: 具有臂端适配的风险意识强化学习在四足行走中的应用 

**Authors**: Yuanhong Zeng, Anushri Dixit  

**Link**: [PDF](https://arxiv.org/pdf/2510.14338)  

**Abstract**: In this work, we study risk-aware reinforcement learning for quadrupedal locomotion. Our approach trains a family of risk-conditioned policies using a Conditional Value-at-Risk (CVaR) constrained policy optimization technique that provides improved stability and sample efficiency. At deployment, we adaptively select the best performing policy from the family of policies using a multi-armed bandit framework that uses only observed episodic returns, without any privileged environment information, and adapts to unknown conditions on the fly. Hence, we train quadrupedal locomotion policies at various levels of robustness using CVaR and adaptively select the desired level of robustness online to ensure performance in unknown environments. We evaluate our method in simulation across eight unseen settings (by changing dynamics, contacts, sensing noise, and terrain) and on a Unitree Go2 robot in previously unseen terrains. Our risk-aware policy attains nearly twice the mean and tail performance in unseen environments compared to other baselines and our bandit-based adaptation selects the best-performing risk-aware policy in unknown terrain within two minutes of operation. 

**Abstract (ZH)**: 基于CVaR的风险感知强化学习在四足机器人运动中的研究与应用 

---
# Expertise need not monopolize: Action-Specialized Mixture of Experts for Vision-Language-Action Learning 

**Title (ZH)**: 专家知识不必垄断：面向视觉-语言-行动学习的动作专业化专家混合模型 

**Authors**: Weijie Shen, Yitian Liu, Yuhao Wu, Zhixuan Liang, Sijia Gu, Dehui Wang, Tian Nian, Lei Xu, Yusen Qin, Jiangmiao Pang, Xinping Guan, Xiaokang Yang, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14300)  

**Abstract**: Vision-Language-Action (VLA) models are experiencing rapid development and demonstrating promising capabilities in robotic manipulation tasks. However, scaling up VLA models presents several critical challenges: (1) Training new VLA models from scratch demands substantial computational resources and extensive datasets. Given the current scarcity of robot data, it becomes particularly valuable to fully leverage well-pretrained VLA model weights during the scaling process. (2) Real-time control requires carefully balancing model capacity with computational efficiency. To address these challenges, We propose AdaMoE, a Mixture-of-Experts (MoE) architecture that inherits pretrained weights from dense VLA models, and scales up the action expert by substituting the feedforward layers into sparsely activated MoE layers. AdaMoE employs a decoupling technique that decouples expert selection from expert weighting through an independent scale adapter working alongside the traditional router. This enables experts to be selected based on task relevance while contributing with independently controlled weights, allowing collaborative expert utilization rather than winner-takes-all dynamics. Our approach demonstrates that expertise need not monopolize. Instead, through collaborative expert utilization, we can achieve superior performance while maintaining computational efficiency. AdaMoE consistently outperforms the baseline model across key benchmarks, delivering performance gains of 1.8% on LIBERO and 9.3% on RoboTwin. Most importantly, a substantial 21.5% improvement in real-world experiments validates its practical effectiveness for robotic manipulation tasks. 

**Abstract (ZH)**: 基于视觉-语言-动作（VLA）模型在机器人操作任务中的快速发展和前景能力，面对扩大VLA模型规模的若干关键挑战，提出了一种混合专家（MoE）架构AdaMoE，该架构通过稀疏激活的MoE层替代密集VLA模型中的前馈层，继承预训练权重并扩展动作专家。AdaMoE通过解耦专家选择与权重分配，使专家可以根据任务相关性进行选择并独立控制权重，从而实现协作专家利用而非胜者全拿的动态。该方法表明，专家无需垄断，通过协作利用专家，可以在保持计算效率的同时实现卓越的性能。实验结果表明，AdaMoE在关键基准上始终优于基线模型，在LIBERO上性能提升1.8%，在RoboTwin上提升9.3%。最重要的是，实际实验中的显著改进（21.5%）验证了其在机器人操作任务中的实际有效性。 

---
# Learning Human-Humanoid Coordination for Collaborative Object Carrying 

**Title (ZH)**: 学习人类-类人机器人协作搬运物体的合作协调 

**Authors**: Yushi Du, Yixuan Li, Baoxiong Jia, Yutang Lin, Pei Zhou, Wei Liang, Yanchao Yang, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14293)  

**Abstract**: Human-humanoid collaboration shows significant promise for applications in healthcare, domestic assistance, and manufacturing. While compliant robot-human collaboration has been extensively developed for robotic arms, enabling compliant human-humanoid collaboration remains largely unexplored due to humanoids' complex whole-body dynamics. In this paper, we propose a proprioception-only reinforcement learning approach, COLA, that combines leader and follower behaviors within a single policy. The model is trained in a closed-loop environment with dynamic object interactions to predict object motion patterns and human intentions implicitly, enabling compliant collaboration to maintain load balance through coordinated trajectory planning. We evaluate our approach through comprehensive simulator and real-world experiments on collaborative carrying tasks, demonstrating the effectiveness, generalization, and robustness of our model across various terrains and objects. Simulation experiments demonstrate that our model reduces human effort by 24.7%. compared to baseline approaches while maintaining object stability. Real-world experiments validate robust collaborative carrying across different object types (boxes, desks, stretchers, etc.) and movement patterns (straight-line, turning, slope climbing). Human user studies with 23 participants confirm an average improvement of 27.4% compared to baseline models. Our method enables compliant human-humanoid collaborative carrying without requiring external sensors or complex interaction models, offering a practical solution for real-world deployment. 

**Abstract (ZH)**: 人形机器人与人类协作在医疗、家庭辅助和制造领域的应用展现出显著潜力。尽管具备顺应性的机器人臂人机协作已得到广泛开发，但如何实现具备顺应性的人类与人形机器人协作还未得到充分探索，主要归因于人形机器人复杂的全身动力学。在本文中，我们提出了一种仅依靠本体感受的强化学习方法COLA，该方法结合了领导行为和跟随行为于单一策略中。该模型在包含动态物体交互的闭环环境中训练，以隐式预测物体运动模式和人类意图，从而通过协调轨迹规划维持负载平衡。我们通过全面的模拟器和真实世界实验评估了该方法在协作承载任务中的有效性、泛化能力和鲁棒性，结果表明，在不同地形和物体类型下，模型的效能均表现出良好的适用性。模拟实验结果显示，相比基线方法，该模型在保持物体稳定性的前提下，能将人类的努力降低24.7%。真实世界实验验证了该方法在不同物体类型（箱子、桌子、担架等）和运动模式（直线、转弯、坡度攀爬）下的鲁棒性协作承载能力。23名参与者的用户研究结果表明，与基线模型相比，该方法能平均提高27.4%的协作承载效果。本方法无需外部传感器或复杂的交互模型，为实际部署提供了可行的解决方案。 

---
# ViTacGen: Robotic Pushing with Vision-to-Touch Generation 

**Title (ZH)**: ViTacGen: 视觉到触觉的生成在机器人推举中 

**Authors**: Zhiyuan Wu, Yijiong Lin, Yongqiang Zhao, Xuyang Zhang, Zhuo Chen, Nathan Lepora, Shan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.14117)  

**Abstract**: Robotic pushing is a fundamental manipulation task that requires tactile feedback to capture subtle contact forces and dynamics between the end-effector and the object. However, real tactile sensors often face hardware limitations such as high costs and fragility, and deployment challenges involving calibration and variations between different sensors, while vision-only policies struggle with satisfactory performance. Inspired by humans' ability to infer tactile states from vision, we propose ViTacGen, a novel robot manipulation framework designed for visual robotic pushing with vision-to-touch generation in reinforcement learning to eliminate the reliance on high-resolution real tactile sensors, enabling effective zero-shot deployment on visual-only robotic systems. Specifically, ViTacGen consists of an encoder-decoder vision-to-touch generation network that generates contact depth images, a standardized tactile representation, directly from visual image sequence, followed by a reinforcement learning policy that fuses visual-tactile data with contrastive learning based on visual and generated tactile observations. We validate the effectiveness of our approach in both simulation and real world experiments, demonstrating its superior performance and achieving a success rate of up to 86\%. 

**Abstract (ZH)**: 视觉触觉生成的强化学习机器人推动物理框架 

---
# Optimistic Reinforcement Learning-Based Skill Insertions for Task and Motion Planning 

**Title (ZH)**: 基于乐观强化学习的技能插入的任务与运动规划 

**Authors**: Gaoyuan Liu, Joris de Winter, Yuri Durodie, Denis Steckelmacher, Ann Nowe, Bram Vanderborght  

**Link**: [PDF](https://arxiv.org/pdf/2510.14065)  

**Abstract**: Task and motion planning (TAMP) for robotics manipulation necessitates long-horizon reasoning involving versatile actions and skills. While deterministic actions can be crafted by sampling or optimizing with certain constraints, planning actions with uncertainty, i.e., probabilistic actions, remains a challenge for TAMP. On the contrary, Reinforcement Learning (RL) excels in acquiring versatile, yet short-horizon, manipulation skills that are robust with uncertainties. In this letter, we design a method that integrates RL skills into TAMP pipelines. Besides the policy, a RL skill is defined with data-driven logical components that enable the skill to be deployed by symbolic planning. A plan refinement sub-routine is designed to further tackle the inevitable effect uncertainties. In the experiments, we compare our method with baseline hierarchical planning from both TAMP and RL fields and illustrate the strength of the method. The results show that by embedding RL skills, we extend the capability of TAMP to domains with probabilistic skills, and improve the planning efficiency compared to the previous methods. 

**Abstract (ZH)**: 基于强化学习的技巧集成到任务与动作规划中以处理概率性技能的任务与动作规划 

---
# Adaptive Obstacle-Aware Task Assignment and Planning for Heterogeneous Robot Teaming 

**Title (ZH)**: 异构机器人团队的自适应障碍aware任务分配与规划 

**Authors**: Nan Li, Jiming Ren, Haris Miller, Samuel Coogan, Karen M. Feigh, Ye Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.14063)  

**Abstract**: Multi-Agent Task Assignment and Planning (MATP) has attracted growing attention but remains challenging in terms of scalability, spatial reasoning, and adaptability in obstacle-rich environments. To address these challenges, we propose OATH: Adaptive Obstacle-Aware Task Assignment and Planning for Heterogeneous Robot Teaming, which advances MATP by introducing a novel obstacle-aware strategy for task assignment. First, we develop an adaptive Halton sequence map, the first known application of Halton sampling with obstacle-aware adaptation in MATP, which adjusts sampling density based on obstacle distribution. Second, we propose a cluster-auction-selection framework that integrates obstacle-aware clustering with weighted auctions and intra-cluster task selection. These mechanisms jointly enable effective coordination among heterogeneous robots while maintaining scalability and near-optimal allocation performance. In addition, our framework leverages an LLM to interpret human instructions and directly guide the planner in real time. We validate OATH in NVIDIA Isaac Sim, showing substantial improvements in task assignment quality, scalability, adaptability to dynamic changes, and overall execution performance compared to state-of-the-art MATP baselines. A project website is available at this https URL. 

**Abstract (ZH)**: 多机器人任务分配与规划（MATP）在处理规模性、空间推理以及多障碍环境中的适应性方面仍然具有挑战性。为了解决这些挑战，我们提出了OATH：适应性障碍感知任务分配与规划，该方法通过引入新的障碍感知策略来推进MATP。首先，我们开发了一种自适应Halton序列图，这是首次将Halton采样与障碍感知调整应用于MATP中，基于障碍分布调整采样密度。其次，我们提出了一种聚类拍卖选择框架，该框架结合了障碍感知聚类、加权拍卖以及内部簇内任务选择。这些机制共同实现了异构机器人之间的有效协调，同时保持可扩展性和接近最优的分配性能。此外，我们的框架利用大语言模型来解释人类指令并实时指导规划器。我们通过在NVIDIA Isaac Sim中的验证，展示了与最先进的MATP基线相比，在任务分配质量、可扩展性、对动态变化的适应性以及整体执行性能方面的显著改进。项目网站可通过该链接访问。 

---
# A Diffusion-Refined Planner with Reinforcement Learning Priors for Confined-Space Parking 

**Title (ZH)**: 受强化学习先验约束的扩散精化规划算法用于受限空间泊车 

**Authors**: Mingyang Jiang, Yueyuan Li, Jiaru Zhang, Songan Zhang, Ming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14000)  

**Abstract**: The growing demand for parking has increased the need for automated parking planning methods that can operate reliably in confined spaces. In restricted and complex environments, high-precision maneuvers are required to achieve a high success rate in planning, yet existing approaches often rely on explicit action modeling, which faces challenges when accurately modeling the optimal action distribution. In this paper, we propose DRIP, a diffusion-refined planner anchored in reinforcement learning (RL) prior action distribution, in which an RL-pretrained policy provides prior action distributions to regularize the diffusion training process. During the inference phase the denoising process refines these coarse priors into more precise action distributions. By steering the denoising trajectory through the reinforcement learning prior distribution during training, the diffusion model inherits a well-informed initialization, resulting in more accurate action modeling, a higher planning success rate, and reduced inference steps. We evaluate our approach across parking scenarios with varying degrees of spatial constraints. Experimental results demonstrate that our method significantly improves planning performance in confined-space parking environments while maintaining strong generalization in common scenarios. 

**Abstract (ZH)**: Growing停车需求增加了在受限空间中可靠实现自动化停车规划方法的需求。在受限和复杂环境中，高精度操作对于规划的成功率至关重要，但现有方法往往依赖于显式动作建模，这在准确建模最优动作分布时面临挑战。本文提出了一种名为DRIP的扩散精炼规划器，该规划器基于强化学习（RL）先验动作分布，其中预训练的RL策略为扩散训练过程提供先验动作分布以正则化训练过程。在推理阶段，去噪过程将这些粗糙的先验转换为更精确的动作分布。通过在训练过程中引导去噪路径通过强化学习先验分布，扩散模型继承了良好的初始化，从而实现更准确的动作建模、更高的规划成功率和较少的推理步骤。我们在具有不同空间约束的停车场景中评估了该方法。实验结果表明，我们的方法在受限空间停车环境中显著提高了规划性能，同时在常见场景中保持了强大的泛化能力。 

---
# EdgeNavMamba: Mamba Optimized Object Detection for Energy Efficient Edge Devices 

**Title (ZH)**: EdgeNavMamba: 优化对象检测以提高边缘设备能量效率的Mamba算法 

**Authors**: Romina Aalishah, Mozhgan Navardi, Tinoosh Mohsenin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14946)  

**Abstract**: Deployment of efficient and accurate Deep Learning models has long been a challenge in autonomous navigation, particularly for real-time applications on resource-constrained edge devices. Edge devices are limited in computing power and memory, making model efficiency and compression essential. In this work, we propose EdgeNavMamba, a reinforcement learning-based framework for goal-directed navigation using an efficient Mamba object detection model. To train and evaluate the detector, we introduce a custom shape detection dataset collected in diverse indoor settings, reflecting visual cues common in real-world navigation. The object detector serves as a pre-processing module, extracting bounding boxes (BBOX) from visual input, which are then passed to an RL policy to control goal-oriented navigation. Experimental results show that the student model achieved a reduction of 67% in size, and up to 73% in energy per inference on edge devices of NVIDIA Jetson Orin Nano and Raspberry Pi 5, while keeping the same performance as the teacher model. EdgeNavMamba also maintains high detection accuracy in MiniWorld and IsaacLab simulators while reducing parameters by 31% compared to the baseline. In the MiniWorld simulator, the navigation policy achieves over 90% success across environments of varying complexity. 

**Abstract (ZH)**: 基于强化学习的EdgeNavMamba：高效准确的目标导向导航框架 

---
# RoboGPT-R1: Enhancing Robot Planning with Reinforcement Learning 

**Title (ZH)**: RoboGPT-R1: 通过强化学习增强机器人规划 

**Authors**: Jinrui Liu, Bingyan Nie, Boyu Li, Yaran Chen, Yuze Wang, Shunsen He, Haoran Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14828)  

**Abstract**: Improving the reasoning capabilities of embodied agents is crucial for robots to complete complex human instructions in long-view manipulation tasks successfully. Despite the success of large language models and vision language models based on Supervised Fine-Tuning (SFT) in planning tasks, they continue facing challenges in performing long-horizon manipulation tasks in complex real-world environments, owing to their restricted common sense and reasoning capabilities. Considering that aligning general-purpose vision language models to robotic planning tasks via supervised fine-tuning suffers from poor generalization and insufficient physical understanding, we propose RoboGPT-R1, a two-stage fine-tuning framework for embodied planning. In this framework, supervised training acquires foundational knowledge through expert sequences, followed by RL to address the model's shortcomings in visual-spatial understanding and reasoning. To achieve physical understanding and action sequence consistency in multi-step reasoning tasks, we design a rule-based reward function that simultaneously considers long-horizon performance and action constraint in the environment. The reasoning model, trained on Qwen2.5-VL-3B, significantly outperforms the larger-scale model, GPT-4o-mini, by 21.33% and surpasses other work trained on Qwen2.5-VL-7B by 20.33% on the EmbodiedBench benchmark. 

**Abstract (ZH)**: 提升具身代理的推理能力对于机器人成功完成复杂的长期操作任务至关重要。尽管基于监督微调（SFT）的大语言模型和视觉语言模型在规划任务中取得了成功，但在复杂现实环境中的长期操作任务中，它们仍然面临挑战，主要是由于它们受限的常识和推理能力。鉴于通过监督微调将通用视觉语言模型对齐到机器人规划任务在泛化能力和物理理解方面存在不足，我们提出了一种双重微调框架RoboGPT-R1，用于具身规划。在该框架中，监督训练通过专家序列获取基础知识，随后通过RL解决模型在视觉空间理解和推理方面的不足。为了在多步推理任务中实现物理理解和动作序列一致性，我们设计了一种基于规则的奖励函数，同时考虑环境中的长期性能和动作约束。在Qwen2.5-VL-3B上训练的推理模型在EmbodiedBench基准测试中显著优于更大规模的模型GPT-4o-mini，高出21.33%，并在Qwen2.5-VL-7B上训练的其他工作中高出20.33%。 

---
# Choreographing Trash Cans: On Speculative Futures of Weak Robots in Public Spaces 

**Title (ZH)**: choreographing 垃圾桶：关于公共空间中弱机器人 speculate 性未来的思考 

**Authors**: Minja Axelsson, Lea Luka Sikau  

**Link**: [PDF](https://arxiv.org/pdf/2510.13810)  

**Abstract**: Delivering groceries or cleaning airports, mobile robots exist in public spaces. While these examples showcase robots that execute tasks, this paper explores mobile robots that encourage posthuman collaboration rather than managing environments independently. With feigned fragility, cuteness and incomplete functionalities, the so-called "weak robots" invite passersby to engage not only on a utilitarian level, but also through imaginative and emotional responses. After examining the workings of "weak robots" by queering notions of function and ability, we introduce two speculative design fiction vignettes that describe choreographies of such robots in future urban spaces -- one exploring a utopian weak robot and the other a dystopian weak robot. We introduce these speculations in order to discuss how different values may drive design decisions, and how such decisions may shape and drive different socio-technical futures in which robots and humans share public spaces that incentivise collaboration. 

**Abstract (ZH)**: 递送 groceries 或清洁机场，移动机器人存在于公共空间。虽然这些例子展示了执行任务的机器人，本文探讨的是鼓励后人类协作而非独立管理环境的移动机器人。通过虚构的脆弱性、可爱性和不完整的功能，所谓的“弱机器人”邀请路人不仅从实用层面参与，还通过想象和情感反应进行参与。通过质疑功能和能力的概念，我们介绍了两个关于未来城市空间中此类机器人编舞的 speculative 设计幻想片段——一个探讨乌托邦弱机器人，另一个探讨反乌托邦弱机器人。我们提出这些设想是为了讨论不同的价值观如何驱动设计决策，并探讨这些决策如何塑造和驱动机器人与人类共享的公共空间中的不同社会技术未来，这些未来鼓励合作。 

---
# ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning 

**Title (ZH)**: ExoPredicator: 学习动态世界的抽象模型以进行机器人规划 

**Authors**: Yichao Liang, Dat Nguyen, Cambridge Yang, Tianyang Li, Joshua B. Tenenbaum, Carl Edward Rasmussen, Adrian Weller, Zenna Tavares, Tom Silver, Kevin Ellis  

**Link**: [PDF](https://arxiv.org/pdf/2509.26255)  

**Abstract**: Long-horizon embodied planning is challenging because the world does not only change through an agent's actions: exogenous processes (e.g., water heating, dominoes cascading) unfold concurrently with the agent's actions. We propose a framework for abstract world models that jointly learns (i) symbolic state representations and (ii) causal processes for both endogenous actions and exogenous mechanisms. Each causal process models the time course of a stochastic cause-effect relation. We learn these world models from limited data via variational Bayesian inference combined with LLM proposals. Across five simulated tabletop robotics environments, the learned models enable fast planning that generalizes to held-out tasks with more objects and more complex goals, outperforming a range of baselines. 

**Abstract (ZH)**: 长时程体态规划具有挑战性，因为世界不仅通过代理的动作发生变化：外生过程（如热水加热、多米诺骨牌连锁反应）与代理的动作同时展开。我们提出了一种抽象世界模型框架，该框架联合学习（i）符号状态表示和（ii）因果过程，包括内生动作和外生机制。每个因果过程模型了一种随机因果关系的时间进程。我们通过结合变分贝叶斯推断和大语言模型提案的方法，从有限数据中学习这些世界模型。在五个模拟桌面机器人环境中，所学模型能够实现快速规划，并能在包含更多物体和更复杂目标的保留任务上泛化，优于多种基线方法。 

---
# Agentic Design of Compositional Machines 

**Title (ZH)**: 代理设计组成的机器 

**Authors**: Wenqian Zhang, Weiyang Liu, Zhen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14980)  

**Abstract**: The design of complex machines stands as both a marker of human intelligence and a foundation of engineering practice. Given recent advances in large language models (LLMs), we ask whether they, too, can learn to create. We approach this question through the lens of compositional machine design: a task in which machines are assembled from standardized components to meet functional demands like locomotion or manipulation in a simulated physical environment. To support this investigation, we introduce BesiegeField, a testbed built on the machine-building game Besiege, which enables part-based construction, physical simulation and reward-driven evaluation. Using BesiegeField, we benchmark state-of-the-art LLMs with agentic workflows and identify key capabilities required for success, including spatial reasoning, strategic assembly, and instruction-following. As current open-source models fall short, we explore reinforcement learning (RL) as a path to improvement: we curate a cold-start dataset, conduct RL finetuning experiments, and highlight open challenges at the intersection of language, machine design, and physical reasoning. 

**Abstract (ZH)**: 复杂机器的设计既体现了人类智能，也是工程实践的基础。鉴于大型语言模型（LLMs）的 recent 进展，我们询问它们是否也能学会创造。我们通过组合式机器设计这一视角来探讨这一问题：这是一种将标准化组件组装起来以满足功能性需求（如移动或操纵）的任务，并在模拟物理环境中进行。为了支持这一调查，我们引入了基于《围城》（Besiege）机器建造游戏构建的 BesiegeField 测试平台，该平台支持基于部件的构建、物理模拟和奖励驱动的评估。使用 BesiegeField，我们对最先进的 LLMs 进行基准测试，并使用代理工作流识别成功所需的关键能力，包括空间推理、战略组装和指令遵循。由于当前的开源模型尚不理想，我们探讨了强化学习（RL）作为改进的途径，进行了冷启动数据集的整理、RL 微调实验，并指出了语言、机器设计和物理推理交叉领域中的开放挑战。 

---
# Mapping Smarter, Not Harder: A Test-Time Reinforcement Learning Agent That Improves Without Labels or Model Updates 

**Title (ZH)**: 不靠更努力，靠更聪明：一种在测试时增强学习的代理，无需标签或模型更新即可改进 

**Authors**: Wen-Kwang Tsao, Yao-Ching Yu, Chien-Ming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14900)  

**Abstract**: The Enterprise Intelligence Platform must integrate logs from numerous third-party vendors in order to perform various downstream tasks. However, vendor documentation is often unavailable at test time. It is either misplaced, mismatched, poorly formatted, or incomplete, which makes schema mapping challenging. We introduce a reinforcement learning agent that can self-improve without labeled examples or model weight updates. During inference, the agent: 1) Identifies ambiguous field-mapping attempts. 2) Generates targeted web-search queries to gather external evidence. 3) Applies a confidence-based reward to iteratively refine its mappings. To demonstrate this concept, we converted Microsoft Defender for Endpoint logs into a common schema. Our method increased mapping accuracy from 56.4\%(LLM-only) to 72.73\%(RAG) to 93.94\% over 100 iterations using GPT-4o. At the same time, it reduced the number of low-confidence mappings requiring expert review by 85\%. This new approach provides an evidence-driven, transparent method for solving future industry problems, paving the way for more robust, accountable, scalable, efficient, flexible, adaptable, and collaborative solutions. 

**Abstract (ZH)**: 企业智能平台必须整合众多第三方供应商的日志以执行各种下游任务。然而，在测试时，供应商文档往往不可用，要么丢失，要么不匹配，要么格式不佳，要么不完整，这使得模式映射变得困难。我们引入了一个无需标记样本或模型权重更新即可自我改进的强化学习代理。在推理过程中，代理：1）识别模糊的字段映射尝试。2）生成针对性的网络搜索查询以收集外部证据。3）应用基于置信度的奖励以迭代精化其映射。为了验证这一概念，我们将Microsoft Defender for Endpoint日志转换为通用模式。经过100次迭代后，我们的方法将仅使用大语言模型的映射准确性从56.4%提高到使用检索增强生成式代理（RAG）的72.73%，最终使用GPT-4o提高到93.94%。同时，它还将需要专家审核的低置信度映射数量减少了85%。这一新方法提供了基于证据的、透明的解决未来行业问题的方法，为更 robust、可问责、可扩展、高效、灵活、适应性强和协作性解决方案铺平了道路。 

---
# LabOS: The AI-XR Co-Scientist That Sees and Works With Humans 

**Title (ZH)**: LabOS: 能看到并与人类协作的AI-XR 合作科学家 

**Authors**: Le Cong, Zaixi Zhang, Xiaotong Wang, Yin Di, Ruofan Jin, Michal Gerasimiuk, Yinkai Wang, Ravi K. Dinesh, David Smerkous, Alex Smerkous, Xuekun Wu, Shilong Liu, Peishan Li, Yi Zhu, Simran Serrao, Ning Zhao, Imran A. Mohammad, John B. Sunwoo, Joseph C. Wu, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14861)  

**Abstract**: Modern science advances fastest when thought meets action. LabOS represents the first AI co-scientist that unites computational reasoning with physical experimentation through multimodal perception, self-evolving agents, and Entended-Reality(XR)-enabled human-AI collaboration. By connecting multi-model AI agents, smart glasses, and human-AI collaboration, LabOS allows AI to see what scientists see, understand experimental context, and assist in real-time execution. Across applications--from cancer immunotherapy target discovery to stem-cell engineering -- LabOS shows that AI can move beyond computational design to participation, turning the laboratory into an intelligent, collaborative environment where human and machine discovery evolve together. 

**Abstract (ZH)**: 现代科学在思想与行动相遇时进步最快。LabOS代表了第一个通过多模态感知、自我进化的代理和扩展现实(XR)-enable的人机协作将计算推理与物理实验结合在一起的AI合作科学家。通过连接多模型AI代理、智能眼镜和人机协作，LabOS使AI能够看到科学家所见，理解实验背景，并在实时执行中提供协助。从癌症免疫疗法靶点发现到干细胞工程等应用中，LabOS展示了AI可以从计算设计跨越到参与，将实验室变成一个智能化、协作的环境，在人类和机器发现共同进化中发挥作用。 

---
# Hi-Agent: Hierarchical Vision-Language Agents for Mobile Device Control 

**Title (ZH)**: Hi-Agent: 分层级的视觉-语言代理用于移动设备控制 

**Authors**: Zhe Wu, Hongjin Lu, Junliang Xing, Changhao Zhang, Yin Zhu, Yuhao Yang, Yuheng Jing, Kai Li, Kun Shao, Jianye Hao, Jun Wang, Yuanchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14388)  

**Abstract**: Building agents that autonomously operate mobile devices has attracted increasing attention. While Vision-Language Models (VLMs) show promise, most existing approaches rely on direct state-to-action mappings, which lack structured reasoning and planning, and thus generalize poorly to novel tasks or unseen UI layouts. We introduce Hi-Agent, a trainable hierarchical vision-language agent for mobile control, featuring a high-level reasoning model and a low-level action model that are jointly optimized. For efficient training, we reformulate multi-step decision-making as a sequence of single-step subgoals and propose a foresight advantage function, which leverages execution feedback from the low-level model to guide high-level optimization. This design alleviates the path explosion issue encountered by Group Relative Policy Optimization (GRPO) in long-horizon tasks and enables stable, critic-free joint training. Hi-Agent achieves a new State-Of-The-Art (SOTA) 87.9% task success rate on the Android-in-the-Wild (AitW) benchmark, significantly outperforming prior methods across three paradigms: prompt-based (AppAgent: 17.7%), supervised (Filtered BC: 54.5%), and reinforcement learning-based (DigiRL: 71.9%). It also demonstrates competitive zero-shot generalization on the ScreenSpot-v2 benchmark. On the more challenging AndroidWorld benchmark, Hi-Agent also scales effectively with larger backbones, showing strong adaptability in high-complexity mobile control scenarios. 

**Abstract (ZH)**: 自主操作移动设备的代理构建引起了越来越多的关注。虽然视觉-语言模型（VLMs）表现出色，但大多数现有方法依赖于直接的状态到动作映射，缺乏结构化的推理和规划，因此在处理新型任务或未见的UI布局时泛化能力较差。我们提出了Hi-Agent，一个用于移动控制的可训练层次视觉-语言代理，该代理包含一个高层推理模型和一个低层动作模型，并且两者是联合优化的。为实现高效的训练，我们将多步决策问题重新表述为一系列单一步骤的子目标，并提出了一种前瞻优势函数，该函数利用低层模型的执行反馈来指导高层优化。这一设计缓解了长时序任务中基于组相对策略优化（GRPO）方法遇到的路径爆炸问题，并使稳定、无批评家的联合训练成为可能。Hi-Agent 在Android-in-the-Wild（AitW）基准测试中达到了新的最佳表现，任务成功率为87.9%，显著优于先前的方法，在三种范式中均表现出更优异的表现：提示驱动（AppAgent：17.7%）、监督学习（过滤后的BC：54.5%）和强化学习驱动（DigiRL：71.9%）。它还在ScreenSpot-v2基准测试中展示了较强的零-shot泛化能力。在更具挑战性的AndroidWorld基准测试中，Hi-Agent 也随着更大模型规模的有效扩展，在高复杂度的移动控制场景中表现出强烈的适应能力。 

---
# AI for Service: Proactive Assistance with AI Glasses 

**Title (ZH)**: AI 服务：AI 眼镜下的主动协助 

**Authors**: Zichen Wen, Yiyu Wang, Chenfei Liao, Boxue Yang, Junxian Li, Weifeng Liu, Haocong He, Bolong Feng, Xuyang Liu, Yuanhuiyi Lyu, Xu Zheng, Xuming Hu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14359)  

**Abstract**: In an era where AI is evolving from a passive tool into an active and adaptive companion, we introduce AI for Service (AI4Service), a new paradigm that enables proactive and real-time assistance in daily life. Existing AI services remain largely reactive, responding only to explicit user commands. We argue that a truly intelligent and helpful assistant should be capable of anticipating user needs and taking actions proactively when appropriate. To realize this vision, we propose Alpha-Service, a unified framework that addresses two fundamental challenges: Know When to intervene by detecting service opportunities from egocentric video streams, and Know How to provide both generalized and personalized services. Inspired by the von Neumann computer architecture and based on AI glasses, Alpha-Service consists of five key components: an Input Unit for perception, a Central Processing Unit for task scheduling, an Arithmetic Logic Unit for tool utilization, a Memory Unit for long-term personalization, and an Output Unit for natural human interaction. As an initial exploration, we implement Alpha-Service through a multi-agent system deployed on AI glasses. Case studies, including a real-time Blackjack advisor, a museum tour guide, and a shopping fit assistant, demonstrate its ability to seamlessly perceive the environment, infer user intent, and provide timely and useful assistance without explicit prompts. 

**Abstract (ZH)**: 在人工智能从被动工具演化为主动适应伴侣的时代，我们提出了AI服务（AI4Service）这一新范式，以实现日常生活中的主动和实时辅助。现有的AI服务主要反应式地响应用户的显式命令。我们主张，真正智能且有帮助的助手应该能够预测用户需求并在适当的时候主动采取行动。为实现这一愿景，我们提出了一种统一框架Alpha-Service，该框架解决了两个基本挑战：知晓何时干预，通过检测自我中心视频流中的服务机会；知晓如何提供，提供通用和个性化服务。Alpha-Service借鉴了冯·诺依曼计算机架构并基于AI眼镜，由五个关键组件组成：感知单元、中央处理单元、算术逻辑单元、长期个性化存储单元以及自然人机交互单元。作为初步探索，我们通过部署在AI眼镜上的多智能体系统实现Alpha-Service。案例研究，包括实时黑 jack 导师、博物馆导游和购物搭配助手，展示了其无缝感知环境、推断用户意图并在无需明确提示的情况下提供及时有用辅助的能力。 

---
# From Pixels to Words -- Towards Native Vision-Language Primitives at Scale 

**Title (ZH)**: 从像素到文字——迈向大规模的本源多模态感知语言基础单元 

**Authors**: Haiwen Diao, Mingxuan Li, Silei Wu, Linjun Dai, Xiaohua Wang, Hanming Deng, Lewei Lu, Dahua Lin, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14979)  

**Abstract**: The edifice of native Vision-Language Models (VLMs) has emerged as a rising contender to typical modular VLMs, shaped by evolving model architectures and training paradigms. Yet, two lingering clouds cast shadows over its widespread exploration and promotion: (-) What fundamental constraints set native VLMs apart from modular ones, and to what extent can these barriers be overcome? (-) How to make research in native VLMs more accessible and democratized, thereby accelerating progress in the field. In this paper, we clarify these challenges and outline guiding principles for constructing native VLMs. Specifically, one native VLM primitive should: (i) effectively align pixel and word representations within a shared semantic space; (ii) seamlessly integrate the strengths of formerly separate vision and language modules; (iii) inherently embody various cross-modal properties that support unified vision-language encoding, aligning, and reasoning. Hence, we launch NEO, a novel family of native VLMs built from first principles, capable of rivaling top-tier modular counterparts across diverse real-world scenarios. With only 390M image-text examples, NEO efficiently develops visual perception from scratch while mitigating vision-language conflicts inside a dense and monolithic model crafted from our elaborate primitives. We position NEO as a cornerstone for scalable and powerful native VLMs, paired with a rich set of reusable components that foster a cost-effective and extensible ecosystem. Our code and models are publicly available at: this https URL. 

**Abstract (ZH)**: 本土视觉-语言模型的基石：从基本原则构建本土视觉-语言模型及NEO家族的提出 

---
# Terra: Explorable Native 3D World Model with Point Latents 

**Title (ZH)**: Terra：可探索的原生3D世界模型与点潜在表示 

**Authors**: Yuanhui Huang, Weiliang Chen, Wenzhao Zheng, Xin Tao, Pengfei Wan, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14977)  

**Abstract**: World models have garnered increasing attention for comprehensive modeling of the real world. However, most existing methods still rely on pixel-aligned representations as the basis for world evolution, neglecting the inherent 3D nature of the physical world. This could undermine the 3D consistency and diminish the modeling efficiency of world models. In this paper, we present Terra, a native 3D world model that represents and generates explorable environments in an intrinsic 3D latent space. Specifically, we propose a novel point-to-Gaussian variational autoencoder (P2G-VAE) that encodes 3D inputs into a latent point representation, which is subsequently decoded as 3D Gaussian primitives to jointly model geometry and appearance. We then introduce a sparse point flow matching network (SPFlow) for generating the latent point representation, which simultaneously denoises the positions and features of the point latents. Our Terra enables exact multi-view consistency with native 3D representation and architecture, and supports flexible rendering from any viewpoint with only a single generation process. Furthermore, Terra achieves explorable world modeling through progressive generation in the point latent space. We conduct extensive experiments on the challenging indoor scenes from ScanNet v2. Terra achieves state-of-the-art performance in both reconstruction and generation with high 3D consistency. 

**Abstract (ZH)**: 一种原生3D世界模型：Terra 

---
# An Active Inference Model of Mouse Point-and-Click Behaviour 

**Title (ZH)**: 小鼠点选行为的主动推断模型 

**Authors**: Markus Klar, Sebastian Stein, Fraser Paterson, John H. Williamson, Roderick Murray-Smith  

**Link**: [PDF](https://arxiv.org/pdf/2510.14611)  

**Abstract**: We explore the use of Active Inference (AIF) as a computational user model for spatial pointing, a key problem in Human-Computer Interaction (HCI). We present an AIF agent with continuous state, action, and observation spaces, performing one-dimensional mouse pointing and clicking. We use a simple underlying dynamic system to model the mouse cursor dynamics with realistic perceptual delay. In contrast to previous optimal feedback control-based models, the agent's actions are selected by minimizing Expected Free Energy, solely based on preference distributions over percepts, such as observing clicking a button correctly. Our results show that the agent creates plausible pointing movements and clicks when the cursor is over the target, with similar end-point variance to human users. In contrast to other models of pointing, we incorporate fully probabilistic, predictive delay compensation into the agent. The agent shows distinct behaviour for differing target difficulties without the need to retune system parameters, as done in other approaches. We discuss the simulation results and emphasize the challenges in identifying the correct configuration of an AIF agent interacting with continuous systems. 

**Abstract (ZH)**: 我们探索将主动推断（AIF）作为计算用户模型应用于空间指点的问题，这是人机交互（HCI）中的一个关键问题。我们呈现了一个拥有连续状态、动作和观测空间的AIF代理，执行一维鼠标指点和点击任务。我们使用简单的动态系统来模拟鼠标光标的动力学，并考虑了现实的感知延迟。与基于最优反馈控制的模型不同，代理的动作是通过最小化预期自由能来选择的，仅基于对感知（如正确点击按钮）的偏好分布。我们的结果显示，当鼠标光标位于目标上时，代理能够产生合理的指点运动和点击，其端点变异量与人类用户类似。与其它指点模型不同，我们为代理引入了完整的概率预测延迟补偿。代理能够表现出不同的行为特征以应对不同的目标难度，无需重新调整系统参数。我们讨论了模拟结果，并强调了确定与连续系统交互的AIF代理正确配置所面临的挑战。 

---
# Agentic Entropy-Balanced Policy Optimization 

**Title (ZH)**: 代理熵平衡策略优化 

**Authors**: Guanting Dong, Licheng Bao, Zhongyuan Wang, Kangzhi Zhao, Xiaoxi Li, Jiajie Jin, Jinghan Yang, Hangyu Mao, Fuzheng Zhang, Kun Gai, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2510.14545)  

**Abstract**: Recently, Agentic Reinforcement Learning (Agentic RL) has made significant progress in incentivizing the multi-turn, long-horizon tool-use capabilities of web agents. While mainstream agentic RL algorithms autonomously explore high-uncertainty tool-call steps under the guidance of entropy, excessive reliance on entropy signals can impose further constraints, leading to the training collapse. In this paper, we delve into the challenges caused by entropy and propose the Agentic Entropy-Balanced Policy Optimization (AEPO), an agentic RL algorithm designed to balance entropy in both the rollout and policy update phases. AEPO comprises two core components: (1) a dynamic entropy-balanced rollout mechanism that adaptively allocate global and branch sampling budget through entropy pre-monitoring, while imposing a branch penalty on consecutive high-entropy tool-call steps to prevent over-branching issues; and (2) Entropy-Balanced Policy Optimization that inserts a stop-gradient operation into the high-entropy clipping term to preserve and properly rescale gradients on high-entropy tokens, while incorporating entropy-aware advantage estimation to prioritize learning on high-uncertainty tokens. Results across 14 challenging datasets show that AEPO consistently outperforms 7 mainstream RL algorithms. With just 1K RL samples, Qwen3-14B with AEPO achieves impressive results: 47.6% on GAIA, 11.2% on Humanity's Last Exam, and 43.0% on WebWalker for Pass@1; 65.0% on GAIA, 26.0% on Humanity's Last Exam, and 70.0% on WebWalker for Pass@5. Further analysis reveals that AEPO improves rollout sampling diversity while maintaining stable policy entropy, facilitating scalable web agent training. 

**Abstract (ZH)**: 近来，代理强化学习（Agentic RL）在激励多轮、长_horizon_工具使用能力方面取得了显著进展。然而，主流代理强化学习算法在熵的指导下自主探索高不确定性工具调用步骤时，过度依赖熵信号可能导致进一步的约束，从而导致训练崩溃。本文探讨了熵带来的挑战，并提出代理平衡熵策略优化（AEPO），这是一种设计用于在展开和策略更新阶段平衡熵的代理强化学习算法。AEPO 包含两个核心组件：（1）动态平衡熵展开机制，通过熵预监测动态分配全局和分支采样预算，同时对连续的高熵工具调用步骤施加分支惩罚以防止过度分支问题；（2）平衡熵策略优化，通过在高熵剪辑项中插入止梯度操作来保留和适当缩放高熵标记的梯度，并结合熵感知的优势估计来优先学习高不确定性标记。在14个具有挑战性的数据集上的结果表明，AEPO 始终优于7种主流RL算法。仅使用1K RL样本，应用AEPO的Qwen3-14B在GAIA、Humanity's Last Exam和WebWalker上的性能分别为47.6%、11.2%和43.0%（Pass@1）；分别为65.0%、26.0%和70.0%（Pass@5）。进一步的分析表明，AEPO 在保持策略熵稳定的同时提高了展开采样的多样性，促进了可扩展的网页代理训练。 

---
# Watermarking for Factuality: Guiding Vision-Language Models Toward Truth via Tri-layer Contrastive Decoding 

**Title (ZH)**: 事实水印：通过三层对比解码引导视觉-语言模型追求真理 

**Authors**: Kyungryul Back, Seongbeom Park, Milim Kim, Mincheol Kwon, SangHyeok Lee, Hyunyoung Lee, Junhee Cho, Seunghyun Park, Jinkyu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.14304)  

**Abstract**: Large Vision-Language Models (LVLMs) have recently shown promising results on various multimodal tasks, even achieving human-comparable performance in certain cases. Nevertheless, LVLMs remain prone to hallucinations -- they often rely heavily on a single modality or memorize training data without properly grounding their outputs. To address this, we propose a training-free, tri-layer contrastive decoding with watermarking, which proceeds in three steps: (1) select a mature layer and an amateur layer among the decoding layers, (2) identify a pivot layer using a watermark-related question to assess whether the layer is visually well-grounded, and (3) apply tri-layer contrastive decoding to generate the final output. Experiments on public benchmarks such as POPE, MME and AMBER demonstrate that our method achieves state-of-the-art performance in reducing hallucinations in LVLMs and generates more visually grounded responses. 

**Abstract (ZH)**: 大型多模态语言视觉模型（LVLMs）在多种多模态任务中展现出了有前途的结果，即使在某些情况下达到了与人类相当的性能。然而，LVLMs仍然容易产生幻觉——它们经常过度依赖单一模态或记忆训练数据，而未能适当地将输出进行 grounded。为解决这一问题，我们提出了一个无需训练的三层对比解码方法，并结合水印技术，该方法分为三步：（1）选择一个成熟的解码层和一个新手层，（2）使用与水印相关的问题识别一个枢纽层，评估该层是否在视觉上良好grounded，（3）应用三层对比解码生成最终输出。在POPE、MME和AMBER等公开基准上的实验表明，我们的方法在减少LVLMs中的幻觉方面取得了最优性能，并生成了更多视觉上grounded的响应。 

---
# DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans 

**Title (ZH)**: DPRF：一种优化个性化LLM角色扮演代理与人类行为对齐的可泛化动态人设精炼框架 

**Authors**: Bingsheng Yao, Bo Sun, Yuanzhe Dong, Yuxuan Lu, Dakuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14205)  

**Abstract**: The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF).DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these this http URL evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie this http URL can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and this http URL work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI. 

**Abstract (ZH)**: 新兴的大语言模型角色扮演代理（LLM RPAs）旨在模拟个体人类行为，但个性保真度往往因缺乏验证的手工创建的人物档案（例如，挑中的信息和个人特质）而受损。为解决这一局限，我们提出了一种动态个性细化框架（DPRF）。DPRF通过迭代地识别生成行为与人类真实行为之间的认知差异（无论是自由形式的还是理论支撑的结构化分析），来优化LLM RPAs的行为与目标个体行为的一致性，并细化人物档案以减轻这些差异。我们使用五种LLM在四种不同行为预测场景（正式辩论、涉及心理健康问题的社交媒体帖子、公开采访和电影）上对DPRF进行了评估，结果表明DPRF可以显著提高行为一致性，并在不同模型和任务之间具有泛化能力。本项工作提供了一种稳健的方法来创建高保真人物档案，并增强下游应用（如用户仿真、社会研究和个人化AI）的有效性。 

---
# GenCellAgent: Generalizable, Training-Free Cellular Image Segmentation via Large Language Model Agents 

**Title (ZH)**: GenCellAgent: 基于大型语言模型代理的通用无训练细胞图像分割 

**Authors**: Xi Yu, Yang Yang, Qun Liu, Yonghua Du, Sean McSweeney, Yuewei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.13896)  

**Abstract**: Cellular image segmentation is essential for quantitative biology yet remains difficult due to heterogeneous modalities, morphological variability, and limited annotations. We present GenCellAgent, a training-free multi-agent framework that orchestrates specialist segmenters and generalist vision-language models via a planner-executor-evaluator loop (choose tool $\rightarrow$ run $\rightarrow$ quality-check) with long-term memory. The system (i) automatically routes images to the best tool, (ii) adapts on the fly using a few reference images when imaging conditions differ from what a tool expects, (iii) supports text-guided segmentation of organelles not covered by existing models, and (iv) commits expert edits to memory, enabling self-evolution and personalized workflows. Across four cell-segmentation benchmarks, this routing yields a 15.7\% mean accuracy gain over state-of-the-art baselines. On endoplasmic reticulum and mitochondria from new datasets, GenCellAgent improves average IoU by 37.6\% over specialist models. It also segments novel objects such as the Golgi apparatus via iterative text-guided refinement, with light human correction further boosting performance. Together, these capabilities provide a practical path to robust, adaptable cellular image segmentation without retraining, while reducing annotation burden and matching user preferences. 

**Abstract (ZH)**: 细胞图像分割对于定量生物学至关重要，但由于异质模态、形态变异性和有限的注释，这一任务依然颇具挑战。我们提出了一种无需训练的多代理框架GenCellAgent，该框架通过规划者-执行者-评估者循环（选择工具 → 运行 → 质量检查）并利用长期记忆，协调专业分割器和通用的视觉语言模型。该系统（i）自动将图像路由到最佳工具，（ii）在成像条件与工具预期不符时，能够即刻调整，（iii）支持文本引导的现有模型未涵盖的细胞器分割，（iv）将专家编辑记忆化，从而实现自我进化和个人化的工作流程。在四个细胞分割基准测试中，这种路由方法相较于最先进的基线提高了15.7%的平均准确率。在内质网和线粒体等新数据集上，GenCellAgent相较于专门模型平均提升了37.6%的IoU。此外，通过迭代的文本引导细化，该系统还能够分割新的细胞结构如高尔基体，适度的人工校正进一步提升了性能。这些能力为在无需重新训练的情况下实现鲁棒且适应性强的细胞图像分割提供了一条实际路径，同时减少了注释负担并匹配用户偏好。 

---
# Multimodal Retrieval-Augmented Generation with Large Language Models for Medical VQA 

**Title (ZH)**: 大规模语言模型增强的多模态检索生成医疗VQA 

**Authors**: A H M Rezaul Karim, Ozlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2510.13856)  

**Abstract**: Medical Visual Question Answering (MedVQA) enables natural language queries over medical images to support clinical decision-making and patient care. The MEDIQA-WV 2025 shared task addressed wound-care VQA, requiring systems to generate free-text responses and structured wound attributes from images and patient queries. We present the MasonNLP system, which employs a general-domain, instruction-tuned large language model with a retrieval-augmented generation (RAG) framework that incorporates textual and visual examples from in-domain data. This approach grounds outputs in clinically relevant exemplars, improving reasoning, schema adherence, and response quality across dBLEU, ROUGE, BERTScore, and LLM-based metrics. Our best-performing system ranked 3rd among 19 teams and 51 submissions with an average score of 41.37%, demonstrating that lightweight RAG with general-purpose LLMs -- a minimal inference-time layer that adds a few relevant exemplars via simple indexing and fusion, with no extra training or complex re-ranking -- provides a simple and effective baseline for multimodal clinical NLP tasks. 

**Abstract (ZH)**: 医学视觉问答（MedVQA）使自然语言查询能够应用于医疗图像，以支持临床决策和患者护理。MEDIQA-WV 2025 共享任务关注伤口护理的视觉问答任务，要求系统从图像和患者查询中生成自由文本回答和结构化的伤口属性。我们介绍了MasonNLP系统，该系统采用了一种通用领域、指令微调的大规模语言模型，并结合了检索增强生成（RAG）框架，该框架包含领域内的文本和视觉示例。该方法将输出与临床相关示例联系起来，提高了推理、模式遵从性和响应质量，在dBLEU、ROUGE、BERTScore和基于LLM的指标上均有所提升。我们的最佳系统在19支团队和51个提交中排名第3，得分为41.37%，表明轻量级RAG与通用目的的语言模型（通过简单的索引和融合添加少量相关示例，无需额外训练或复杂重新排序，在推理时间上增加了一层最小的计算）为多模态临床自然语言处理任务提供了一个简单而有效的基本方案。 

---
# Towards Neurocognitive-Inspired Intelligence: From AI's Structural Mimicry to Human-Like Functional Cognition 

**Title (ZH)**: 面向神经认知启发的智能：从AI的结构模拟到类似人类的功能认知 

**Authors**: Noorbakhsh Amiri Golilarz, Hassan S. Al Khatib, Shahram Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2510.13826)  

**Abstract**: Artificial intelligence has advanced significantly through deep learning, reinforcement learning, and large language and vision models. However, these systems often remain task specific, struggle to adapt to changing conditions, and cannot generalize in ways similar to human cognition. Additionally, they mainly focus on mimicking brain structures, which often leads to black-box models with limited transparency and adaptability. Inspired by the structure and function of biological cognition, this paper introduces the concept of "Neurocognitive-Inspired Intelligence (NII)," a hybrid approach that combines neuroscience, cognitive science, computer vision, and AI to develop more general, adaptive, and robust intelligent systems capable of rapid learning, learning from less data, and leveraging prior experience. These systems aim to emulate the human brain's ability to flexibly learn, reason, remember, perceive, and act in real-world settings with minimal supervision. We review the limitations of current AI methods, define core principles of neurocognitive-inspired intelligence, and propose a modular, biologically inspired architecture that emphasizes integration, embodiment, and adaptability. We also discuss potential implementation strategies and outline various real-world applications, from robotics to education and healthcare. Importantly, this paper offers a hybrid roadmap for future research, laying the groundwork for building AI systems that more closely resemble human cognition. 

**Abstract (ZH)**: 基于神经认知灵感的人工智能（NII）：一种结合神经科学、认知科学、计算机视觉和AI的综合方法 

---
