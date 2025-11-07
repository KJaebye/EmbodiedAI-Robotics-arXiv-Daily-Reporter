# GentleHumanoid: Learning Upper-body Compliance for Contact-rich Human and Object Interaction 

**Title (ZH)**: GentleHumanoid：学习丰富接触条件下上身柔顺性的人机与物交互 

**Authors**: Qingzhou Lu, Yao Feng, Baiyu Shi, Michael Piseno, Zhenan Bao, C. Karen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04679)  

**Abstract**: Humanoid robots are expected to operate in human-centered environments where safe and natural physical interaction is essential. However, most recent reinforcement learning (RL) policies emphasize rigid tracking and suppress external forces. Existing impedance-augmented approaches are typically restricted to base or end-effector control and focus on resisting extreme forces rather than enabling compliance. We introduce GentleHumanoid, a framework that integrates impedance control into a whole-body motion tracking policy to achieve upper-body compliance. At its core is a unified spring-based formulation that models both resistive contacts (restoring forces when pressing against surfaces) and guiding contacts (pushes or pulls sampled from human motion data). This formulation ensures kinematically consistent forces across the shoulder, elbow, and wrist, while exposing the policy to diverse interaction scenarios. Safety is further supported through task-adjustable force thresholds. We evaluate our approach in both simulation and on the Unitree G1 humanoid across tasks requiring different levels of compliance, including gentle hugging, sit-to-stand assistance, and safe object manipulation. Compared to baselines, our policy consistently reduces peak contact forces while maintaining task success, resulting in smoother and more natural interactions. These results highlight a step toward humanoid robots that can safely and effectively collaborate with humans and handle objects in real-world environments. 

**Abstract (ZH)**: 人形机器人期望在以人类为中心的环境中操作，其中安全自然的物理交互是必不可少的。然而，现有的大部分强化学习（RL）策略强调刚性跟踪并抑制外部力。现有的阻抗增强方法通常仅限于基座或末端执行器控制，并侧重于抵抗极端力而不是实现柔顺性。我们提出了GentleHumanoid框架，将阻抗控制整合到全身运动追踪策略中，以实现上身柔顺性。其核心是一种统一的弹簧模型，同时表征阻力接触（对表面施压时的恢复力）和引导接触（来自人类运动数据的推力或拉力）。该模型确保肩部、肘部和腕部的机械一致力，同时使策略暴露在多种交互场景中。通过任务可调的力阈值进一步支持安全性。我们在模拟和Unitree G1人形机器人上评估了该方法，在不同柔顺性要求的任务中包括温柔拥抱、坐下站立辅助和安全物体操纵，与基线方法相比，我们的策略始终降低峰值接触力的同时维持任务成功，从而实现更顺畅和自然的交互。这些结果突显了朝向能在实际环境中安全有效地与人类协作并处理物体的人形机器人迈出的一步。 

---
# X-Diffusion: Training Diffusion Policies on Cross-Embodiment Human Demonstrations 

**Title (ZH)**: X-扩散：在跨体型人类示范上训练扩散策略 

**Authors**: Maximus A. Pace, Prithwish Dan, Chuanruo Ning, Atiksh Bhardwaj, Audrey Du, Edward W. Duan, Wei-Chiu Ma, Kushal Kedia  

**Link**: [PDF](https://arxiv.org/pdf/2511.04671)  

**Abstract**: Human videos can be recorded quickly and at scale, making them an appealing source of training data for robot learning. However, humans and robots differ fundamentally in embodiment, resulting in mismatched action execution. Direct kinematic retargeting of human hand motion can therefore produce actions that are physically infeasible for robots. Despite these low-level differences, human demonstrations provide valuable motion cues about how to manipulate and interact with objects. Our key idea is to exploit the forward diffusion process: as noise is added to actions, low-level execution differences fade while high-level task guidance is preserved. We present X-Diffusion, a principled framework for training diffusion policies that maximally leverages human data without learning dynamically infeasible motions. X-Diffusion first trains a classifier to predict whether a noisy action is executed by a human or robot. Then, a human action is incorporated into policy training only after adding sufficient noise such that the classifier cannot discern its embodiment. Actions consistent with robot execution supervise fine-grained denoising at low noise levels, while mismatched human actions provide only coarse guidance at higher noise levels. Our experiments show that naive co-training under execution mismatches degrades policy performance, while X-Diffusion consistently improves it. Across five manipulation tasks, X-Diffusion achieves a 16% higher average success rate than the best baseline. The project website is available at this https URL. 

**Abstract (ZH)**: 人类视频可以快速大规模录制，成为机器人学习训练数据的诱人来源。然而，人类和机器人在基本体态上存在根本差异，导致动作执行不匹配。直接运动学重定位人类手部动作因此会产生机器人无法执行的物理动作。尽管存在这些低级差异，人类示范提供了关于如何操作和与物体交互的重要运动提示。我们的关键思想是利用前向扩散过程：随着噪声被添加到动作中，低级执行差异会减弱而高级任务指导得以保留。我们提出了X-Diffusion，这是一种原则性的框架，用于训练最大限度利用人类数据而不学习动态不可行动作的扩散策略。X-Diffusion首先训练一个分类器预测一个带噪声的动作是由人类还是机器人执行的。然后，在添加足够的噪声使得分类器无法区分其体态后，人类动作才被纳入策略训练。一致执行的动作在低噪声水平下监督精细去噪，而与机器人执行不符的人类动作仅在高噪声水平下提供粗略指导。我们的实验表明，执行不匹配下的天真共训练会降低策略性能，而X-Diffusion始终能够提升性能。在五个操作任务中，X-Diffusion的平均成功率高于最佳基线16%。项目网站可在以下链接访问。 

---
# Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions 

**Title (ZH)**: 基于高斯点云模拟软体物相互作用的实物到模拟机器人策略评估 

**Authors**: Kaifeng Zhang, Shuo Sha, Hanxiao Jiang, Matthew Loper, Hyunjong Song, Guangyan Cai, Zhuo Xu, Xiaochen Hu, Changxi Zheng, Yunzhu Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.04665)  

**Abstract**: Robotic manipulation policies are advancing rapidly, but their direct evaluation in the real world remains costly, time-consuming, and difficult to reproduce, particularly for tasks involving deformable objects. Simulation provides a scalable and systematic alternative, yet existing simulators often fail to capture the coupled visual and physical complexity of soft-body interactions. We present a real-to-sim policy evaluation framework that constructs soft-body digital twins from real-world videos and renders robots, objects, and environments with photorealistic fidelity using 3D Gaussian Splatting. We validate our approach on representative deformable manipulation tasks, including plush toy packing, rope routing, and T-block pushing, demonstrating that simulated rollouts correlate strongly with real-world execution performance and reveal key behavioral patterns of learned policies. Our results suggest that combining physics-informed reconstruction with high-quality rendering enables reproducible, scalable, and accurate evaluation of robotic manipulation policies. Website: this https URL 

**Abstract (ZH)**: 机器人操纵策略的发展正在加速，但它们在真实世界中的直接评估仍然成本高、耗时且难以再现，尤其是对于涉及变形物体的任务。仿真提供了一种可扩展且系统化的替代方案，但现有仿真器往往无法捕捉软体交互的耦合视觉和物理复杂性。我们提出了一种真实世界到仿真的策略评估框架，该框架从真实世界视频中构建软体数字双胞胎，并使用3D 高斯点绘制技术以高保真度渲染机器人、物体和环境。我们在代表性变形物体操作任务上验证了该方法，包括布偶玩具打包、绳索布线和T块推拉，结果显示模拟滚动轨迹与实际执行性能相关性很强，并揭示了学习策略的关键行为模式。我们的结果表明，结合物理信息重建与高质量渲染能够实现可再现、可扩展且准确的机器人操作策略评估。Website: https://this.url 

---
# SAFe-Copilot: Unified Shared Autonomy Framework 

**Title (ZH)**: SAFe-Copilot: 统一共享自主框架 

**Authors**: Phat Nguyen, Erfan Aasi, Shiva Sreeram, Guy Rosman, Andrew Silva, Sertac Karaman, Daniela Rus  

**Link**: [PDF](https://arxiv.org/pdf/2511.04664)  

**Abstract**: Autonomous driving systems remain brittle in rare, ambiguous, and out-of-distribution scenarios, where human driver succeed through contextual reasoning. Shared autonomy has emerged as a promising approach to mitigate such failures by incorporating human input when autonomy is uncertain. However, most existing methods restrict arbitration to low-level trajectories, which represent only geometric paths and therefore fail to preserve the underlying driving intent. We propose a unified shared autonomy framework that integrates human input and autonomous planners at a higher level of abstraction. Our method leverages Vision Language Models (VLMs) to infer driver intent from multi-modal cues -- such as driver actions and environmental context -- and to synthesize coherent strategies that mediate between human and autonomous control. We first study the framework in a mock-human setting, where it achieves perfect recall alongside high accuracy and precision. A human-subject survey further shows strong alignment, with participants agreeing with arbitration outcomes in 92% of cases. Finally, evaluation on the Bench2Drive benchmark demonstrates a substantial reduction in collision rate and improvement in overall performance compared to pure autonomy. Arbitration at the level of semantic, language-based representations emerges as a design principle for shared autonomy, enabling systems to exercise common-sense reasoning and maintain continuity with human intent. 

**Abstract (ZH)**: 自主驾驶系统在罕见的、模糊的和分布外的情景中仍然脆弱，这时人类驾驶员通过情景推理成功驾驶。共享自主作为一种将人类输入纳入不确定情况下的自主规划中的有前景的方法已 Emerges 作为一种减轻此类失败的有前景的方法。然而，大多数现有方法将裁决限制在低级轨迹上，这仅表示几何路径，因此无法保留驾驶意图。我们提出了一种统一的共享自主框架，将人类输入和自主规划集成到更高的抽象级别。我们的方法利用视觉语言模型（VLM）从多模态线索（例如驾驶员行为和环境上下文）中推断驾驶意图，并综合出协调的人类和自主控制之间的策略。首先，我们在一个模拟人类设置中研究了该框架，实现了完美的召回率并具有高准确性和精确度。人类被试调查进一步表明强烈的一致性，参与者在92%的情况下同意裁决结果。最后，在 Bench2Drive 标准上进行的评估显示与纯自主相比，碰撞率显著降低，整体性能也有所提高。在语义和基于语言的表示级别上的裁决成为共享自主的设计原则，使系统能够行使常识推理并保持与人类意图的一致性。 

---
# Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment 

**Title (ZH)**: Evo-1: 轻量级视觉-语言-动作模型，保持语义对齐 

**Authors**: Tao Lin, Yilei Zhong, Yuxin Du, Jingjing Zhang, Jiting Liu, Yinxinyu Chen, Encheng Gu, Ziyan Liu, Hongyi Cai, Yanwen Zou, Lixing Zou, Zhaoye Zhou, Gen Li, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.04555)  

**Abstract**: Vision-Language-Action (VLA) models have emerged as a powerful framework that unifies perception, language, and control, enabling robots to perform diverse tasks through multimodal understanding. However, current VLA models typically contain massive parameters and rely heavily on large-scale robot data pretraining, leading to high computational costs during training, as well as limited deployability for real-time inference. Moreover, most training paradigms often degrade the perceptual representations of the vision-language backbone, resulting in overfitting and poor generalization to downstream tasks. In this work, we present Evo-1, a lightweight VLA model that reduces computation and improves deployment efficiency, while maintaining strong performance without pretraining on robot data. Evo-1 builds on a native multimodal Vision-Language model (VLM), incorporating a novel cross-modulated diffusion transformer along with an optimized integration module, together forming an effective architecture. We further introduce a two-stage training paradigm that progressively aligns action with perception, preserving the representations of the VLM. Notably, with only 0.77 billion parameters, Evo-1 achieves state-of-the-art results on the Meta-World and RoboTwin suite, surpassing the previous best models by 12.4% and 6.9%, respectively, and also attains a competitive result of 94.8% on LIBERO. In real-world evaluations, Evo-1 attains a 78% success rate with high inference frequency and low memory overhead, outperforming all baseline methods. We release code, data, and model weights to facilitate future research on lightweight and efficient VLA models. 

**Abstract (ZH)**: 基于视觉-语言-行动的轻量级模型Evo-1：高效部署与强性能的统一框架 

---
# ForeRobo: Unlocking Infinite Simulation Data for 3D Goal-driven Robotic Manipulation 

**Title (ZH)**: ForeRobo: 解锁无限模拟数据以实现3D目标驱动的机器人操作 

**Authors**: Dexin wang, Faliang Chang, Chunsheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04381)  

**Abstract**: Efficiently leveraging simulation to acquire advanced manipulation skills is both challenging and highly significant. We introduce \textit{ForeRobo}, a generative robotic agent that utilizes generative simulations to autonomously acquire manipulation skills driven by envisioned goal states. Instead of directly learning low-level policies, we advocate integrating generative paradigms with classical control. Our approach equips a robotic agent with a self-guided \textit{propose-generate-learn-actuate} cycle. The agent first proposes the skills to be acquired and constructs the corresponding simulation environments; it then configures objects into appropriate arrangements to generate skill-consistent goal states (\textit{ForeGen}). Subsequently, the virtually infinite data produced by ForeGen are used to train the proposed state generation model (\textit{ForeFormer}), which establishes point-wise correspondences by predicting the 3D goal position of every point in the current state, based on the scene state and task instructions. Finally, classical control algorithms are employed to drive the robot in real-world environments to execute actions based on the envisioned goal states. Compared with end-to-end policy learning methods, ForeFormer offers superior interpretability and execution efficiency. We train and benchmark ForeFormer across a variety of rigid-body and articulated-object manipulation tasks, and observe an average improvement of 56.32\% over the state-of-the-art state generation models, demonstrating strong generality across different manipulation patterns. Moreover, in real-world evaluations involving more than 20 robotic tasks, ForeRobo achieves zero-shot sim-to-real transfer and exhibits remarkable generalization capabilities, attaining an average success rate of 79.28\%. 

**Abstract (ZH)**: 高效利用模拟来获取高级操作技能既具有挑战性又极为重要。我们介绍了一种名为ForeRobo的生成型机器人代理，该代理利用生成模拟自主学习由预想目标状态驱动的操作技能。我们主张将生成范式与经典控制相结合，而不是直接学习低级策略。我们的方法为机器人代理配备了一个自我引导的“提出-生成-学习-执行”循环。代理首先提出要学习的技能，并构建相应的模拟环境；然后，通过生成符合技能的目标状态（ForeGen），配置对象以形成适当的排列。随后，由ForeGen产生的虚拟无限数据被用于训练所提出的状态生成模型（ForeFormer），该模型基于当前场景状态和任务指令，预测每个点的3D目标位置，建立逐点对应关系。最后，采用经典控制算法在真实环境中驱动机器人执行基于预想目标状态的动作。与端到端策略学习方法相比，ForeFormer提供了更优越的可解释性和执行效率。我们在各种刚体和 articulated-object 操作任务上训练和基准测试了ForeFormer，相对于最先进的状态生成模型，观察到平均改进56.32％，证明了其在不同操作模式下强大的泛化能力。此外，在涉及超过20个机器人任务的真实世界评估中，ForeRobo实现了零样本的模拟到实际的转移，并展示了出色的泛化能力，平均成功率达到了79.28％。 

---
# MacroNav: Multi-Task Context Representation Learning Enables Efficient Navigation in Unknown Environments 

**Title (ZH)**: MacroNav: 多任务上下文表示学习实现未知环境中的高效导航 

**Authors**: Kuankuan Sima, Longbin Tang, Haozhe Ma, Lin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.04320)  

**Abstract**: Autonomous navigation in unknown environments requires compact yet expressive spatial understanding under partial observability to support high-level decision making. Existing approaches struggle to balance rich contextual representation with navigation efficiency. We present MacroNav, a learning-based navigation framework featuring two key components: (1) a lightweight context encoder trained via multi-task self-supervised learning to capture multi-scale, navigation-centric spatial representations; and (2) a reinforcement learning policy that seamlessly integrates these representations with graph-based reasoning for efficient action selection. Extensive experiments demonstrate the context encoder's efficient and robust environmental understanding. Real-world deployments further validate MacroNav's effectiveness, yielding significant gains over state-of-the-art navigation methods in both Success Rate (SR) and Success weighted by Path Length (SPL), while maintaining low computational cost. Code will be released upon acceptance. 

**Abstract (ZH)**: 自主在未知环境中导航需要在部分可观测性条件下具备紧凑且表达能力强的空间理解能力，以支持高级决策制定。现有方法难以在丰富的上下文表示与导航效率之间取得平衡。我们提出了一种基于学习的导航框架MacroNav，该框架包含两个关键组件：（1）一种通过多任务自我监督学习训练的轻量级上下文编码器，用于捕获多尺度、以导航为中心的空间表示；以及（2）一种将这些表示与图基推理无缝结合的强化学习策略，用于高效的动作选择。广泛的实验表明上下文编码器具有高效的且鲁棒的环境理解能力。实际部署进一步验证了MacroNav的有效性，在成功率（SR）和路径长度加权成功率（SPL）方面显著优于现有最先进的导航方法，同时保持较低的计算成本。在接受后将发布代码。 

---
# Can Context Bridge the Reality Gap? Sim-to-Real Transfer of Context-Aware Policies 

**Title (ZH)**: 情境能否弥合现实差距？从模拟到现实的基于情境政策转移 

**Authors**: Marco Iannotta, Yuxuan Yang, Johannes A. Stork, Erik Schaffernicht, Todor Stoyanov  

**Link**: [PDF](https://arxiv.org/pdf/2511.04249)  

**Abstract**: Sim-to-real transfer remains a major challenge in reinforcement learning (RL) for robotics, as policies trained in simulation often fail to generalize to the real world due to discrepancies in environment dynamics. Domain Randomization (DR) mitigates this issue by exposing the policy to a wide range of randomized dynamics during training, yet leading to a reduction in performance. While standard approaches typically train policies agnostic to these variations, we investigate whether sim-to-real transfer can be improved by conditioning the policy on an estimate of the dynamics parameters -- referred to as context. To this end, we integrate a context estimation module into a DR-based RL framework and systematically compare SOTA supervision strategies. We evaluate the resulting context-aware policies in both a canonical control benchmark and a real-world pushing task using a Franka Emika Panda robot. Results show that context-aware policies outperform the context-agnostic baseline across all settings, although the best supervision strategy depends on the task. 

**Abstract (ZH)**: 模拟到现实的迁移仍然是机器人强化学习中的一个主要挑战，因为模拟中训练的策略往往由于环境动力学的差异无法在现实世界中有效泛化。领域随机化（DR）通过在训练过程中让策略暴露在广泛随机化的动力学中来缓解这一问题，但可能导致性能下降。尽管标准方法通常在忽略这些变化的情况下训练策略，我们研究了通过让策略依赖于动力学参数的估计（称为上下文）是否可以改善模拟到现实的迁移。为此，我们将一个上下文估计模块整合到基于领域随机化的RL框架中，系统地比较了现有的最佳监督策略。我们在一个经典的控制基准和使用Franka Emika Panda机器人的实际推物任务中评估了结果的上下文感知策略。结果显示，在所有设置下，上下文感知策略均优于上下文无关的基线，尽管最佳监督策略取决于具体任务。 

---
# PUL-SLAM: Path-Uncertainty Co-Optimization with Lightweight Stagnation Detection for Efficient Robotic Exploration 

**Title (ZH)**: PUL-SLAM：路径不确定性协同优化与轻量级停滞检测以实现高效的机器人探索 

**Authors**: Yizhen Yin, Dapeng Feng, Hongbo Chen, Yuhua Qi  

**Link**: [PDF](https://arxiv.org/pdf/2511.04180)  

**Abstract**: Existing Active SLAM methodologies face issues such as slow exploration speed and suboptimal paths. To address these limitations, we propose a hybrid framework combining a Path-Uncertainty Co-Optimization Deep Reinforcement Learning framework and a Lightweight Stagnation Detection mechanism. The Path-Uncertainty Co-Optimization framework jointly optimizes travel distance and map uncertainty through a dual-objective reward function, balancing exploration and exploitation. The Lightweight Stagnation Detection reduces redundant exploration through Lidar Static Anomaly Detection and Map Update Stagnation Detection, terminating episodes on low expansion rates. Experimental results show that compared with the frontier-based method and RRT method, our approach shortens exploration time by up to 65% and reduces path distance by up to 42%, significantly improving exploration efficiency in complex environments while maintaining reliable map completeness. Ablation studies confirm that the collaborative mechanism accelerates training convergence. Empirical validation on a physical robotic platform demonstrates the algorithm's practical applicability and its successful transferability from simulation to real-world environments. 

**Abstract (ZH)**: 现有的主动SLAM方法面临探索速度慢和路径次优的问题。为了解决这些限制，我们提出了一种结合路径-不确定性协同优化深度强化学习框架和轻量级停滞检测机制的混合框架。路径-不确定性协同优化框架通过双目标奖励函数联合优化出行距离和地图不确定性，平衡探索与开发。轻量级停滞检测机制通过激光雷达静态异常检测和地图更新停滞检测减少冗余探索，并在低扩展率时终止episode。实验结果表明，与基于前沿的方法和RRT方法相比，我们的方法将探索时间缩短最多65%，路径距离减少最多42%，显著提高了在复杂环境中的探索效率，同时保持了可靠的地图完整性。消融研究证实了协作机制加速了训练收敛。在物理机器人平台上的经验验证表明该算法的实际适用性和从仿真环境到真实环境的可迁移性。 

---
# BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning 

**Title (ZH)**: BFM-Zero：一种使用无监督强化学习的可提示行为基础模型用于类人控制 

**Authors**: Yitang Li, Zhengyi Luo, Tonghe Zhang, Cunxi Dai, Anssi Kanervisto, Andrea Tirinzoni, Haoyang Weng, Kris Kitani, Mateusz Guzek, Ahmed Touati, Alessandro Lazaric, Matteo Pirotta, Guanya Shi  

**Link**: [PDF](https://arxiv.org/pdf/2511.04131)  

**Abstract**: Building Behavioral Foundation Models (BFMs) for humanoid robots has the potential to unify diverse control tasks under a single, promptable generalist policy. However, existing approaches are either exclusively deployed on simulated humanoid characters, or specialized to specific tasks such as tracking. We propose BFM-Zero, a framework that learns an effective shared latent representation that embeds motions, goals, and rewards into a common space, enabling a single policy to be prompted for multiple downstream tasks without retraining. This well-structured latent space in BFM-Zero enables versatile and robust whole-body skills on a Unitree G1 humanoid in the real world, via diverse inference methods, including zero-shot motion tracking, goal reaching, and reward optimization, and few-shot optimization-based adaptation. Unlike prior on-policy reinforcement learning (RL) frameworks, BFM-Zero builds upon recent advancements in unsupervised RL and Forward-Backward (FB) models, which offer an objective-centric, explainable, and smooth latent representation of whole-body motions. We further extend BFM-Zero with critical reward shaping, domain randomization, and history-dependent asymmetric learning to bridge the sim-to-real gap. Those key design choices are quantitatively ablated in simulation. A first-of-its-kind model, BFM-Zero establishes a step toward scalable, promptable behavioral foundation models for whole-body humanoid control. 

**Abstract (ZH)**: 基于行为基础模型（BFMs）的人形机器人构建具有潜力统一多种控制任务于单一可提示通用策略之下。然而，现有方法要么仅部署在模拟人形角色上，要么专门针对特定任务如跟踪。我们提出BFM-Zero框架，该框架学习一种有效的共享潜在表示，将动作、目标和奖励嵌入到一个共同空间中，使得单一策略能够在无需重新训练的情况下被提示执行多个下游任务。BFM-Zero中的结构良好潜在空间通过多种推断方法，在真实世界中的人形机器人（如Unitree G1）上实现了灵活且稳健的全身技能，包括零样本运动跟踪、目标逼近和奖励优化，以及少量样本基于优化的适应。与先前的在线策略强化学习（RL）框架不同，BFM-Zero基于无监督RL和前向-后向（FB）模型的最新进展，提供以目标为中心、可解释且平滑的全身运动潜在表示。我们进一步通过关键奖励塑造、领域随机化和历史依赖的非对称学习扩展BFM-Zero，以弥合模拟到现实的差距。这些关键设计选择在仿真中进行了定量分析。BFM-Zero作为首个此类模型，为全身人形控制的可扩展和可提示行为基础模型铺平了道路。 

---
# Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots 

**Title (ZH)**: 基于视觉驱动的反应式足球技能学习for类人机器人 

**Authors**: Yushi Wang, Changsheng Luo, Penghui Chen, Jianran Liu, Weijian Sun, Tong Guo, Kechang Yang, Biao Hu, Yangang Zhang, Mingguo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.03996)  

**Abstract**: Humanoid soccer poses a representative challenge for embodied intelligence, requiring robots to operate within a tightly coupled perception-action loop. However, existing systems typically rely on decoupled modules, resulting in delayed responses and incoherent behaviors in dynamic environments, while real-world perceptual limitations further exacerbate these issues. In this work, we present a unified reinforcement learning-based controller that enables humanoid robots to acquire reactive soccer skills through the direct integration of visual perception and motion control. Our approach extends Adversarial Motion Priors to perceptual settings in real-world dynamic environments, bridging motion imitation and visually grounded dynamic control. We introduce an encoder-decoder architecture combined with a virtual perception system that models real-world visual characteristics, allowing the policy to recover privileged states from imperfect observations and establish active coordination between perception and action. The resulting controller demonstrates strong reactivity, consistently executing coherent and robust soccer behaviors across various scenarios, including real RoboCup matches. 

**Abstract (ZH)**: 人形足球为 embodied 智能提出了一个富有代表性的挑战，要求机器人在 Perception-Action 紧密耦合循环中操作。然而，现有系统通常依赖于解耦模块，导致在动态环境中产生延迟响应和不协调行为，而现实世界的感知限制进一步加剧了这些问题。在本文中，我们提出了一种统一的基于强化学习的控制器，通过直接整合视觉感知与运动控制使类人机器人获得反应型足球技能。我们扩展了 Competing Motion Priors 方法，将其应用于现实世界动态环境中的感知场景，实现了运动模仿与视觉接地动态控制的桥梁。我们引入了一种编码器-解码器架构结合虚拟感知系统，该系统可以建模现实世界的视觉特性，使策略能够从不完美的观测中恢复特权状态，并在感知与动作之间建立主动协调。所得到的控制器表现出强大的反应性，能够在各种场景中，包括实际的 RoboCup 比赛中，一致地执行协调且稳健的足球行为。 

---
# Investigating Robot Control Policy Learning for Autonomous X-ray-guided Spine Procedures 

**Title (ZH)**: 探究自主X射线引导脊柱手术中的机器人控制策略学习 

**Authors**: Florence Klitzner, Blanca Inigo, Benjamin D. Killeen, Lalithkumar Seenivasan, Michelle Song, Axel Krieger, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2511.03882)  

**Abstract**: Imitation learning-based robot control policies are enjoying renewed interest in video-based robotics. However, it remains unclear whether this approach applies to X-ray-guided procedures, such as spine instrumentation. This is because interpretation of multi-view X-rays is complex. We examine opportunities and challenges for imitation policy learning in bi-plane-guided cannula insertion. We develop an in silico sandbox for scalable, automated simulation of X-ray-guided spine procedures with a high degree of realism. We curate a dataset of correct trajectories and corresponding bi-planar X-ray sequences that emulate the stepwise alignment of providers. We then train imitation learning policies for planning and open-loop control that iteratively align a cannula solely based on visual information. This precisely controlled setup offers insights into limitations and capabilities of this method. Our policy succeeded on the first attempt in 68.5% of cases, maintaining safe intra-pedicular trajectories across diverse vertebral levels. The policy generalized to complex anatomy, including fractures, and remained robust to varied initializations. Rollouts on real bi-planar X-rays further suggest that the model can produce plausible trajectories, despite training exclusively in simulation. While these preliminary results are promising, we also identify limitations, especially in entry point precision. Full closed-look control will require additional considerations around how to provide sufficiently frequent feedback. With more robust priors and domain knowledge, such models may provide a foundation for future efforts toward lightweight and CT-free robotic intra-operative spinal navigation. 

**Abstract (ZH)**: 基于模仿学习的机器人控制策略在基于视频的机器人技术中正重新引起关注。然而，这种方法是否适用于X射线引导的操作，如脊柱内固定术尚不明确，因为多视角X射线的解释复杂。我们探讨了双平面引导穿刺针插入中模仿策略学习的机会与挑战。我们开发了一个在现实感极高的在硅沙盒，用于大规模、自动模拟X射线引导下的脊柱手术。我们构建了一个正确轨迹及其对应的双平面X射线序列数据集，模拟医疗提供者的逐步对齐过程。然后，我们训练基于模仿学习的规划和开环控制策略，仅依据视觉信息迭代地对准穿刺针。这一精确受控的设置提供了对该方法的限制与能力的洞见。我们的策略在68.5%的情况下首次尝试即成功，并且在整个不同椎体水平上保持了安全的椎体内路径。该策略在复杂解剖结构下，包括骨折情况，以及面对各种初始条件时依然表现出高度稳健性。在实际双平面X射线上的进一步模拟表明，即使仅在仿真中训练，模型也可以生成合理的轨迹。尽管初步结果令人鼓舞，我们还识别出了局限性，特别是在进入点精度方面。要实现完全闭环控制，还需要考虑如何提供足够的反馈频率。通过更稳健的先验知识和领域知识，此类模型可能为未来基于轻量级、无需CT的机器人术中脊柱导航工作提供基础。 

---
# DR. WELL: Dynamic Reasoning and Learning with Symbolic World Model for Embodied LLM-Based Multi-Agent Collaboration 

**Title (ZH)**: DR. WELL: 基于符号世界模型的动态推理与学习的体态化多智能体协同研究 

**Authors**: Narjes Nourzad, Hanqing Yang, Shiyu Chen, Carlee Joe-Wong  

**Link**: [PDF](https://arxiv.org/pdf/2511.04646)  

**Abstract**: Cooperative multi-agent planning requires agents to make joint decisions with partial information and limited communication. Coordination at the trajectory level often fails, as small deviations in timing or movement cascade into conflicts. Symbolic planning mitigates this challenge by raising the level of abstraction and providing a minimal vocabulary of actions that enable synchronization and collective progress. We present DR. WELL, a decentralized neurosymbolic framework for cooperative multi-agent planning. Cooperation unfolds through a two-phase negotiation protocol: agents first propose candidate roles with reasoning and then commit to a joint allocation under consensus and environment constraints. After commitment, each agent independently generates and executes a symbolic plan for its role without revealing detailed trajectories. Plans are grounded in execution outcomes via a shared world model that encodes the current state and is updated as agents act. By reasoning over symbolic plans rather than raw trajectories, DR. WELL avoids brittle step-level alignment and enables higher-level operations that are reusable, synchronizable, and interpretable. Experiments on cooperative block-push tasks show that agents adapt across episodes, with the dynamic world model capturing reusable patterns and improving task completion rates and efficiency. Experiments on cooperative block-push tasks show that our dynamic world model improves task completion and efficiency through negotiation and self-refinement, trading a time overhead for evolving, more efficient collaboration strategies. 

**Abstract (ZH)**: 协作多智能体规划要求智能体在部分信息和有限通信的情况下做出联合决策。轨迹层面的协调往往失败，因为时间节点或动作的细微偏差会引发冲突。符号化规划通过提高抽象水平并提供可用于同步和集体进步的最小动作用 vocabulary 来缓解这一挑战。我们提出 DR. WELL，一种分布式神经符号框架，用于协作多智能体规划。合作通过两阶段谈判协议展开：智能体首先提出候选角色并进行推理，然后在一致性和环境约束下承诺联合分配。在承诺之后，每个智能体独立生成并执行其角色的符号计划而不泄露详细的轨迹。计划通过共享世界模型与执行结果相关联，该模型编码当前状态并在智能体行动时更新。通过在符号计划而非原始轨迹级别进行推理，DR. WELL 避免了脆弱的逐步骤对齐，允许可重用、可同步和可解释的高层操作。在协作块推任务实验中，智能体在各集会中表现出适应性，动态世界模型捕获可重用的模式并提高任务完成率和效率。在协作块推任务实验中，我们的动态世界模型通过谈判和自我优化提高任务完成和效率，以时间开销为代价获得更高效的协作策略。 

---
# Jr. AI Scientist and Its Risk Report: Autonomous Scientific Exploration from a Baseline Paper 

**Title (ZH)**: Jr. AI科学家及其风险报告：基于基准论文的自主科学探索 

**Authors**: Atsuyuki Miyai, Mashiro Toyooka, Takashi Otonari, Zaiying Zhao, Kiyoharu Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2511.04583)  

**Abstract**: Understanding the current capabilities and risks of AI Scientist systems is essential for ensuring trustworthy and sustainable AI-driven scientific progress while preserving the integrity of the academic ecosystem. To this end, we develop Jr. AI Scientist, a state-of-the-art autonomous AI scientist system that mimics the core research workflow of a novice student researcher: Given the baseline paper from the human mentor, it analyzes its limitations, formulates novel hypotheses for improvement, validates them through rigorous experimentation, and writes a paper with the results. Unlike previous approaches that assume full automation or operate on small-scale code, Jr. AI Scientist follows a well-defined research workflow and leverages modern coding agents to handle complex, multi-file implementations, leading to scientifically valuable contributions. For evaluation, we conducted automated assessments using AI Reviewers, author-led evaluations, and submissions to Agents4Science, a venue dedicated to AI-driven scientific contributions. The findings demonstrate that Jr. AI Scientist generates papers receiving higher review scores than existing fully automated systems. Nevertheless, we identify important limitations from both the author evaluation and the Agents4Science reviews, indicating the potential risks of directly applying current AI Scientist systems and key challenges for future research. Finally, we comprehensively report various risks identified during development. We hope these insights will deepen understanding of current progress and risks in AI Scientist development. 

**Abstract (ZH)**: 理解当前AI科学家系统的能力和风险对于确保可信赖和可持续的AI驱动科学进步，同时保持学术生态系统完整性至关重要。为此，我们开发了Jr. AI科学家，这是一个先进的自主AI科学家系统，模拟了初级学生研究者的核心研究工作流程：给定人类导师的基础论文，它分析其局限性，提出改进的新假设，通过严格的实验验证它们，并撰写包含结果的论文。与之前假设完全自动化或仅操作小规模代码的方法不同，Jr. AI科学家遵循明确的研究工作流程，并利用现代编码代理处理复杂、多文件的实现，从而产生科学上有价值的贡献。为了评估，我们使用AI审稿人进行自动化评估，由作者主导的评估，以及向专门接收AI驱动科学贡献的 Agents4Science 投稿。研究发现表明，Jr. AI科学家生成的论文在审稿评分上高于现有完全自动化的系统。然而，我们从作者评估和Agents4Science评审中识别出重要的局限性，这表明当前AI科学家系统的潜在风险以及未来研究的关键挑战。最后，我们全面报告了在开发过程中识别出的各种风险。我们希望通过这些见解加深对AI科学家当前进展和风险的理解。 

---
# Shared Spatial Memory Through Predictive Coding 

**Title (ZH)**: 通过预测编码实现共享空间记忆 

**Authors**: Zhengru Fang, Yu Guo, Jingjing Wang, Yuang Zhang, Haonan An, Yinhai Wang, Yuguang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04235)  

**Abstract**: Sharing and reconstructing a consistent spatial memory is a critical challenge in multi-agent systems, where partial observability and limited bandwidth often lead to catastrophic failures in coordination. We introduce a multi-agent predictive coding framework that formulate coordination as the minimization of mutual uncertainty among agents. Instantiated as an information bottleneck objective, it prompts agents to learn not only who and what to communicate but also when. At the foundation of this framework lies a grid-cell-like metric as internal spatial coding for self-localization, emerging spontaneously from self-supervised motion prediction. Building upon this internal spatial code, agents gradually develop a bandwidth-efficient communication mechanism and specialized neural populations that encode partners' locations: an artificial analogue of hippocampal social place cells (SPCs). These social representations are further enacted by a hierarchical reinforcement learning policy that actively explores to reduce joint uncertainty. On the Memory-Maze benchmark, our approach shows exceptional resilience to bandwidth constraints: success degrades gracefully from 73.5% to 64.4% as bandwidth shrinks from 128 to 4 bits/step, whereas a full-broadcast baseline collapses from 67.6% to 28.6%. Our findings establish a theoretically principled and biologically plausible basis for how complex social representations emerge from a unified predictive drive, leading to social collective intelligence. 

**Abstract (ZH)**: 共享和重构一致的空间记忆是多agent系统中的一个重要挑战，其中的部分可观测性和有限带宽往往导致协调中的灾难性失败。我们提出了一种多agent预测编码框架，将协调公式化为代理间相互不确定性最小化的问题。该框架以信息瓶颈目标的形式呈现，促使代理不仅学习谁和什么需要沟通，还学习何时沟通。该框架的基础是一种类似于网格细胞的内部空间编码机制，这种机制通过自我监督的运动预测自发涌现，用于自我定位。在此内部空间编码的基础上，代理逐渐发展出一种高效的通信机制和专门的神经群体，用于编码同伴的位置：一种人工版的海马体社交位置细胞（SPCs）。这些社交表征进一步通过层次强化学习策略实现，该策略主动探索以减少联合不确定性。在Memory-Maze基准测试中，我们的方法在带宽限制下表现出色：随着带宽从每步128位减少到4位，成功率从73.5%平滑下降到64.4%，而一个全广播基线则从67.6%下降到28.6%。我们的研究为复杂社交表征如何源自统一的预测驱动提供了理论上的基础，并可能导致社交集体智能。 

---
# Detecting Silent Failures in Multi-Agentic AI Trajectories 

**Title (ZH)**: 检测多代理AI轨迹中的隐性失败 

**Authors**: Divya Pathak, Harshit Kumar, Anuska Roy, Felix George, Mudit Verma, Pratibha Moogi  

**Link**: [PDF](https://arxiv.org/pdf/2511.04032)  

**Abstract**: Multi-Agentic AI systems, powered by large language models (LLMs), are inherently non-deterministic and prone to silent failures such as drift, cycles, and missing details in outputs, which are difficult to detect. We introduce the task of anomaly detection in agentic trajectories to identify these failures and present a dataset curation pipeline that captures user behavior, agent non-determinism, and LLM variation. Using this pipeline, we curate and label two benchmark datasets comprising \textbf{4,275 and 894} trajectories from Multi-Agentic AI systems. Benchmarking anomaly detection methods on these datasets, we show that supervised (XGBoost) and semi-supervised (SVDD) approaches perform comparably, achieving accuracies up to 98% and 96%, respectively. This work provides the first systematic study of anomaly detection in Multi-Agentic AI systems, offering datasets, benchmarks, and insights to guide future research. 

**Abstract (ZH)**: 多智能体AI系统由大语言模型驱动，本质上是非确定性的，并且容易出现漂移、循环和输出中缺少细节等难以检测的隐性故障。我们提出了智能体轨迹异常检测任务以识别这些故障，并提出了一种数据集编目流程来捕获用户行为、智能体的非确定性和大语言模型的变化。使用此流程，我们编目并标注了两个基准数据集，包含来自多智能体AI系统的\textbf{4,275和894}条轨迹。在这些数据集上对异常检测方法进行基准测试，结果显示监督（XGBoost）和半监督（SVDD）方法表现相当，准确率分别达到98%和96%。本研究提供了首个多智能体AI系统中异常检测的系统性研究，提供了数据集、基准测试和指导未来研究的见解。 

---
# Deep Koopman Economic Model Predictive Control of a Pasteurisation Unit 

**Title (ZH)**: 深度科普曼经济模型预测控制在巴氏杀菌单元中的应用 

**Authors**: Patrik Valábek, Michaela Horváthová, Martin Klaučo  

**Link**: [PDF](https://arxiv.org/pdf/2511.04437)  

**Abstract**: This paper presents a deep Koopman-based Economic Model Predictive Control (EMPC) for efficient operation of a laboratory-scale pasteurization unit (PU). The method uses Koopman operator theory to transform the complex, nonlinear system dynamics into a linear representation, enabling the application of convex optimization while representing the complex PU accurately. The deep Koopman model utilizes neural networks to learn the linear dynamics from experimental data, achieving a 45% improvement in open-loop prediction accuracy over conventional N4SID subspace identification. Both analyzed models were employed in the EMPC formulation that includes interpretable economic costs, such as energy consumption, material losses due to inadequate pasteurization, and actuator wear. The feasibility of EMPC is ensured using slack variables. The deep Koopman EMPC and N4SID EMPC are numerically validated on a nonlinear model of multivariable PU under external disturbance. The disturbances include feed pump fail-to-close scenario and the introduction of a cold batch to be pastuerized. These results demonstrate that the deep Koopmand EMPC achieves a 32% reduction in total economic cost compared to the N4SID baseline. This improvement is mainly due to the reductions in material losses and energy consumption. Furthermore, the steady-state operation via Koopman-based EMPC requires 10.2% less electrical energy. The results highlight the practical advantages of integrating deep Koopman representations with economic optimization to achieve resource-efficient control of thermal-intensive plants. 

**Abstract (ZH)**: 基于深度Koopman的经济模型预测控制在实验室规模巴氏杀菌单元高效运行中的应用研究 

---
# Efficient Reinforcement Learning from Human Feedback via Bayesian Preference Inference 

**Title (ZH)**: 通过贝叶斯偏好推理实现高效的人工反馈强化学习 

**Authors**: Matteo Cercola, Valeria Capretti, Simone Formentin  

**Link**: [PDF](https://arxiv.org/pdf/2511.04286)  

**Abstract**: Learning from human preferences is a cornerstone of aligning machine learning models with subjective human judgments. Yet, collecting such preference data is often costly and time-consuming, motivating the need for more efficient learning paradigms. Two established approaches offer complementary advantages: RLHF scales effectively to high-dimensional tasks such as LLM fine-tuning, while PBO achieves greater sample efficiency through active querying. We propose a hybrid framework that unifies RLHF's scalability with PBO's query efficiency by integrating an acquisition-driven module into the RLHF pipeline, thereby enabling active and sample-efficient preference gathering. We validate the proposed approach on two representative domains: (i) high-dimensional preference optimization and (ii) LLM fine-tuning. Experimental results demonstrate consistent improvements in both sample efficiency and overall performance across these tasks. 

**Abstract (ZH)**: 从人类偏好中学习是使机器学习模型与主观人类判断相一致的基石。然而，收集这样的偏好数据往往成本高且耗时，推动了更高效学习范式的需要。两种已建立的方法分别具有互补的优势：RLHF能够有效地扩展到高维任务，如大型语言模型微调，而PBO则通过主动查询实现了更高的样本效率。我们提出了一种混合框架，将RLHF的扩展性与PBO的查询效率结合起来，通过将一个基于获取的模块集成到RLHF流程中，从而实现主动且样本高效的偏好收集。我们在此类应用的两个代表性领域中验证了所提出的方法：(i) 高维偏好优化和(ii) 大型语言模型微调。实验结果表明，在这些任务中，该方法在样本效率和整体性能上都表现出一致的改进。 

---
# Not All Explanations are Created Equal: Investigating the Pitfalls of Current XAI Evaluation 

**Title (ZH)**: 并非所有解释都平等：探究当前解释可解释性评估中的局限性 

**Authors**: Joe Shymanski, Jacob Brue, Sandip Sen  

**Link**: [PDF](https://arxiv.org/pdf/2511.03730)  

**Abstract**: Explainable Artificial Intelligence (XAI) aims to create transparency in modern AI models by offering explanations of the models to human users. There are many ways in which researchers have attempted to evaluate the quality of these XAI models, such as user studies or proposed objective metrics like "fidelity". However, these current XAI evaluation techniques are ad hoc at best and not generalizable. Thus, most studies done within this field conduct simple user surveys to analyze the difference between no explanations and those generated by their proposed solution. We do not find this to provide adequate evidence that the explanations generated are of good quality since we believe any kind of explanation will be "better" in most metrics when compared to none at all. Thus, our study looks to highlight this pitfall: most explanations, regardless of quality or correctness, will increase user satisfaction. We also propose that emphasis should be placed on actionable explanations. We demonstrate the validity of both of our claims using an agent assistant to teach chess concepts to users. The results of this chapter will act as a call to action in the field of XAI for more comprehensive evaluation techniques for future research in order to prove explanation quality beyond user satisfaction. Additionally, we present an analysis of the scenarios in which placebic or actionable explanations would be most useful. 

**Abstract (ZH)**: 可解释的人工智能（XAI）旨在通过向人类用户提供模型解释来增强现代AI模型的透明度。尽管研究人员已经尝试通过用户研究或类似于“保真度”的客观指标来评估XAI模型的质量，但现有的XAI评估技术最多只能说是临时性的，并不具备普适性。因此，该领域内的大多数研究仅通过简单的用户调查来分析未提供解释与根据提出的解决方案提供的解释之间的差异。我们并不认为这些研究提供了足够的证据来证明生成的解释是高质量的，因为我们认为在大多数评价标准下，任何类型的解释都比完全没有解释更“好”。因此，我们的研究旨在指出这一缺陷：大多数解释，无论质量还是正确性如何，都会提高用户满意度。我们也提出应强调可操作性解释的重要性。我们通过一个智能助手教授棋盘游戏概念的实验证明了上述两个观点的有效性。本章的结果将促使XAI领域的研究人员采用更全面的评估技术以证明解释质量超出用户满意度。此外，我们还分析了疗效性或可操作性解释最适用的情景。 

---
# Efficient On-Device Agents via Adaptive Context Management 

**Title (ZH)**: 基于适应性上下文管理的高效设备端代理 

**Authors**: Sanidhya Vijayvargiya, Rahul Lokesh  

**Link**: [PDF](https://arxiv.org/pdf/2511.03728)  

**Abstract**: On-device AI agents offer the potential for personalized, low-latency assistance, but their deployment is fundamentally constrained by limited memory capacity, which restricts usable context. This reduced practical context window creates a trade-off between supporting rich, stateful interactions with complex tool capabilities and maintaining on-device feasibility. We break this trade-off with a framework for context-efficient on-device agents, driven by three synergistic optimizations (1) a dynamic memory system using specialized LoRA adapters to distill conversational history into a compressed, and structured Context State Object; (2) a minimalist serialization format for tool schemas to minimize token overhead per tool; and (3) a just-in-time schema-passing mechanism that loads full tool definitions only upon tool selection. We instantiate this framework by adapting a 3B parameter SLM to context-efficient trajectories and rigorously evaluate it against a conventional baseline on complex user tasks. Our agent matches, or exceeds, the performance of a conventional baseline while dramatically compressing context, achieving more than a 6-fold reduction in initial system prompt context and a 10- to 25-fold reduction in context growth rate based on the interaction verbosity, demonstrating that strategic context management is key to unlocking capable and persistent on-device AI. 

**Abstract (ZH)**: 设备端AI代理提供了个性化、低延迟辅助的潜力，但其部署受到有限内存容量的基本约束，这限制了可用上下文。减小的实际上下文窗口在支持丰富且状态相关的交互（涉及复杂工具功能）和保持设备端可行性之间创建了权衡。我们通过一种高效上下文管理的设备端代理框架打破了这一权衡，该框架由三个协同优化驱动（1）一种动态内存系统，使用专门的LoRA适配器将对话历史精简为压缩且结构化的内容状态对象；（2）一种简约的序列化格式来最小化每个工具的标记开销；（3）一种即时方案传递机制，仅在选择工具时加载完整工具定义。通过将一个3B参数SLM模型调整为高效上下文轨迹，并针对复杂用户任务与传统基线进行严格评估，我们的代理在显著压缩上下文的同时表现出与传统基线相当或更好的性能，分别实现了初始系统提示上下文超过6倍的压缩和基于交互详细程度上下文增长速率10到25倍的压缩，表明战略性上下文管理是解锁功能强大且持久的设备端AI的关键。 

---
