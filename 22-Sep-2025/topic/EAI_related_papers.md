# Agentic Aerial Cinematography: From Dialogue Cues to Cinematic Trajectories 

**Title (ZH)**: 代理无人机摄影：从对话线索到电影轨迹 

**Authors**: Yifan Lin, Sophie Ziyu Liu, Ran Qi, George Z. Xue, Xinping Song, Chao Qin, Hugh H.-T. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.16176)  

**Abstract**: We present Agentic Aerial Cinematography: From Dialogue Cues to Cinematic Trajectories (ACDC), an autonomous drone cinematography system driven by natural language communication between human directors and drones. The main limitation of previous drone cinematography workflows is that they require manual selection of waypoints and view angles based on predefined human intent, which is labor-intensive and yields inconsistent performance. In this paper, we propose employing large language models (LLMs) and vision foundation models (VFMs) to convert free-form natural language prompts directly into executable indoor UAV video tours. Specifically, our method comprises a vision-language retrieval pipeline for initial waypoint selection, a preference-based Bayesian optimization framework that refines poses using aesthetic feedback, and a motion planner that generates safe quadrotor trajectories. We validate ACDC through both simulation and hardware-in-the-loop experiments, demonstrating that it robustly produces professional-quality footage across diverse indoor scenes without requiring expertise in robotics or cinematography. These results highlight the potential of embodied AI agents to close the loop from open-vocabulary dialogue to real-world autonomous aerial cinematography. 

**Abstract (ZH)**: 基于对话提示到影视航迹的自主无人机 cinematography 系统：从对话线索到 Cinematic 航迹 (ACDC) 

---
# I-FailSense: Towards General Robotic Failure Detection with Vision-Language Models 

**Title (ZH)**: I-FailSense: 向通用机器人故障检测的语言视觉模型方向探索 

**Authors**: Clemence Grislain, Hamed Rahimi, Olivier Sigaud, Mohamed Chetouani  

**Link**: [PDF](https://arxiv.org/pdf/2509.16072)  

**Abstract**: Language-conditioned robotic manipulation in open-world settings requires not only accurate task execution but also the ability to detect failures for robust deployment in real-world environments. Although recent advances in vision-language models (VLMs) have significantly improved the spatial reasoning and task-planning capabilities of robots, they remain limited in their ability to recognize their own failures. In particular, a critical yet underexplored challenge lies in detecting semantic misalignment errors, where the robot executes a task that is semantically meaningful but inconsistent with the given instruction. To address this, we propose a method for building datasets targeting Semantic Misalignment Failures detection, from existing language-conditioned manipulation datasets. We also present I-FailSense, an open-source VLM framework with grounded arbitration designed specifically for failure detection. Our approach relies on post-training a base VLM, followed by training lightweight classification heads, called FS blocks, attached to different internal layers of the VLM and whose predictions are aggregated using an ensembling mechanism. Experiments show that I-FailSense outperforms state-of-the-art VLMs, both comparable in size and larger, in detecting semantic misalignment errors. Notably, despite being trained only on semantic misalignment detection, I-FailSense generalizes to broader robotic failure categories and effectively transfers to other simulation environments and real-world with zero-shot or minimal post-training. The datasets and models are publicly released on HuggingFace (Webpage: this https URL). 

**Abstract (ZH)**: 开放场景中基于语言条件的机器人操作需要精确的任务执行能力和故障检测能力，以确保在现实环境中的稳健部署。尽管近期视觉语言模型在空间推理和任务规划能力方面取得了显著进步，但仍限于无法识别自身的故障。特别地，发现语义对齐错误这一关键但未充分探索的挑战尤为突出，这是一种机器人执行虽然具有语义意义但与给定指令不一致的任务情况。为解决这一问题，我们提出了一种用于构建针对语义对齐故障检测的数据集的方法，源自现有语言条件操作数据集。我们还介绍了I-FailSense，一个具备基于事实仲裁机制的开源视觉语言模型框架，专门用于故障检测。该方法基于对基础视觉语言模型进行后训练，随后训练附接在模型不同内部层的轻量级分类头FS块，并通过集成机制聚合预测结果。实验结果显示，I-FailSense 在检测语义对齐错误方面优于最新的视觉语言模型，无论是大小相当的还是更大的模型。值得注意的是，尽管仅用于语义对齐检测的训练，I-FailSense 在更广泛的机器人故障类别中表现出良好的泛化能力，并且能够零样本或少量后训练有效转移到其他模拟环境和现实世界。数据集和模型已在HuggingFace上公开发布（网页: this https URL）。 

---
# DSPv2: Improved Dense Policy for Effective and Generalizable Whole-body Mobile Manipulation 

**Title (ZH)**: DSPv2: 提升的密集策略以实现有效和泛化的全身移动 manipulation 

**Authors**: Yue Su, Chubin Zhang, Sijin Chen, Liufan Tan, Yansong Tang, Jianan Wang, Xihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.16063)  

**Abstract**: Learning whole-body mobile manipulation via imitation is essential for generalizing robotic skills to diverse environments and complex tasks. However, this goal is hindered by significant challenges, particularly in effectively processing complex observation, achieving robust generalization, and generating coherent actions. To address these issues, we propose DSPv2, a novel policy architecture. DSPv2 introduces an effective encoding scheme that aligns 3D spatial features with multi-view 2D semantic features. This fusion enables the policy to achieve broad generalization while retaining the fine-grained perception necessary for precise control. Furthermore, we extend the Dense Policy paradigm to the whole-body mobile manipulation domain, demonstrating its effectiveness in generating coherent and precise actions for the whole-body robotic platform. Extensive experiments show that our method significantly outperforms existing approaches in both task performance and generalization ability. Project page is available at: this https URL. 

**Abstract (ZH)**: 通过模仿学习全身移动操作对于将机器人技能泛化到多样环境和复杂任务至关重要。然而，这一目标受制于重大挑战，特别是有效处理复杂观测、实现坚稳健性泛化以及生成连贯动作。为解决这些问题，我们提出DSPv2，一种新型策略架构。DSPv2引入了一种有效的编码方案，将3D空间特征与多视图2D语义特征对齐，从而实现广泛的泛化同时保留精细感知以进行精确控制。此外，我们将密集策略范式扩展到全身移动操作领域，并在全身机器人平台中展示了其生成连贯和精确动作的有效性。大量实验表明，我们的方法在任务性能和泛化能力上显著优于现有方法。项目页面链接为：this https URL。 

---
# Latent Conditioned Loco-Manipulation Using Motion Priors 

**Title (ZH)**: 基于运动先验的隐含条件局部操作 

**Authors**: Maciej Stępień, Rafael Kourdis, Constant Roux, Olivier Stasse  

**Link**: [PDF](https://arxiv.org/pdf/2509.16061)  

**Abstract**: Although humanoid and quadruped robots provide a wide range of capabilities, current control methods, such as Deep Reinforcement Learning, focus mainly on single skills. This approach is inefficient for solving more complicated tasks where high-level goals, physical robot limitations and desired motion style might all need to be taken into account. A more effective approach is to first train a multipurpose motion policy that acquires low-level skills through imitation, while providing latent space control over skill execution. Then, this policy can be used to efficiently solve downstream tasks. This method has already been successful for controlling characters in computer graphics. In this work, we apply the approach to humanoid and quadrupedal loco-manipulation by imitating either simple synthetic motions or kinematically retargeted dog motions. We extend the original formulation to handle constraints, ensuring deployment safety, and use a diffusion discriminator for better imitation quality. We verify our methods by performing loco-manipulation in simulation for the H1 humanoid and Solo12 quadruped, as well as deploying policies on Solo12 hardware. Videos and code are available at this https URL 

**Abstract (ZH)**: 尽管类人机器人和四足机器人的能力范围广泛，当前的控制方法，如深度强化学习，主要关注单一技能。对于需要高级目标、物理机器人限制和期望运动风格综合考量的复杂任务，这种做法效率较低。一种更有效的做法是首先训练一个多功能运动策略，通过模仿获取低级技能，同时提供技能执行的潜在空间控制。然后，该策略可用于高效解决下游任务。这种方法已经在控制计算机图形中的角色方面取得了成功。在本文中，我们将该方法应用于类人机器人和四足机器人以进行移动操作与操纵，模仿简单的合成动作或动力学适配的狗动作。我们将原始模型扩展以处理约束，确保部署安全性，并使用扩散判别器以提高模仿质量。我们通过在模拟中进行移动操作来验证方法，用于H1类人机器人和Solo12四足机器人，并在Solo12硬件上部署策略。相关视频和代码可在以下链接获取。 

---
# Compose by Focus: Scene Graph-based Atomic Skills 

**Title (ZH)**: 焦点驱动的生成：基于场景图的原子技能 

**Authors**: Han Qi, Changhe Chen, Heng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16053)  

**Abstract**: A key requirement for generalist robots is compositional generalization - the ability to combine atomic skills to solve complex, long-horizon tasks. While prior work has primarily focused on synthesizing a planner that sequences pre-learned skills, robust execution of the individual skills themselves remains challenging, as visuomotor policies often fail under distribution shifts induced by scene composition. To address this, we introduce a scene graph-based representation that focuses on task-relevant objects and relations, thereby mitigating sensitivity to irrelevant variation. Building on this idea, we develop a scene-graph skill learning framework that integrates graph neural networks with diffusion-based imitation learning, and further combine "focused" scene-graph skills with a vision-language model (VLM) based task planner. Experiments in both simulation and real-world manipulation tasks demonstrate substantially higher success rates than state-of-the-art baselines, highlighting improved robustness and compositional generalization in long-horizon tasks. 

**Abstract (ZH)**: 通用机器人的一个关键要求是组合泛化能力——即结合原子技能以解决复杂的、长期任务的能力。虽然先前的工作主要集中在合成一个规划器来序列化预先学习的技能，但个体技能的稳健执行仍然具有挑战性，因为视觉-运动策略往往会在场景组合诱导的分布变化下失效。为了解决这一问题，我们引入了一种基于场景图的表示，该表示侧重于与任务相关的对象和关系，从而减轻对无关变异的敏感性。在此基础上，我们开发了一种结合图神经网络与基于扩散的模仿学习的场景图技能学习框架，并进一步将“聚焦”的场景图技能与基于视觉-语言模型的任务规划器相结合。在模拟和实际操作任务中的实验结果表明，与最先进的基线相比，成功率显著提高，突出了在长期任务中改进的稳健性和组合泛化能力。 

---
# Defining and Monitoring Complex Robot Activities via LLMs and Symbolic Reasoning 

**Title (ZH)**: 通过大语言模型和符号推理定义与监控复杂机器人活动 

**Authors**: Francesco Argenziano, Elena Umili, Francesco Leotta, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2509.16006)  

**Abstract**: Recent years have witnessed a growing interest in automating labor-intensive and complex activities, i.e., those consisting of multiple atomic tasks, by deploying robots in dynamic and unpredictable environments such as industrial and agricultural settings. A key characteristic of these contexts is that activities are not predefined: while they involve a limited set of possible tasks, their combinations may vary depending on the situation. Moreover, despite recent advances in robotics, the ability for humans to monitor the progress of high-level activities - in terms of past, present, and future actions - remains fundamental to ensure the correct execution of safety-critical processes. In this paper, we introduce a general architecture that integrates Large Language Models (LLMs) with automated planning, enabling humans to specify high-level activities (also referred to as processes) using natural language, and to monitor their execution by querying a robot. We also present an implementation of this architecture using state-of-the-art components and quantitatively evaluate the approach in a real-world precision agriculture scenario. 

**Abstract (ZH)**: 近年来，人们越来越关注通过在动态和不可预测的环境中部署机器人来自动化劳动密集型和复杂的活动，这些活动由多个原子任务组成。这些环境的一个关键特点是活动不是预先定义的：虽然涉及一组有限的任务，但其组合可能会根据具体情况而变化。此外，尽管机器人技术取得了近期进展，人类继续监控高层次活动（包括过去、现在和未来的行动）以确保关键安全过程的正确执行仍然是基本需求。本文提出了一种通用架构，将大型语言模型（LLMs）与自动规划相结合，使人类能够使用自然语言指定高层次活动（也称为过程），并通过查询机器人来监控其执行。我们还使用最先进的组件实现了该架构，并在实际的精准农业场景中定量评估了该方法。 

---
# CoReVLA: A Dual-Stage End-to-End Autonomous Driving Framework for Long-Tail Scenarios via Collect-and-Refine 

**Title (ZH)**: CoReVLA：一种用于长尾场景的端到端自主驾驶框架（基于收集与精炼的双阶段方法） 

**Authors**: Shiyu Fang, Yiming Cui, Haoyang Liang, Chen Lv, Peng Hang, Jian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.15968)  

**Abstract**: Autonomous Driving (AD) systems have made notable progress, but their performance in long-tail, safety-critical scenarios remains limited. These rare cases contribute a disproportionate number of accidents. Vision-Language Action (VLA) models have strong reasoning abilities and offer a potential solution, but their effectiveness is limited by the lack of high-quality data and inefficient learning in such conditions. To address these challenges, we propose CoReVLA, a continual learning end-to-end autonomous driving framework that improves the performance in long-tail scenarios through a dual-stage process of data Collection and behavior Refinement. First, the model is jointly fine-tuned on a mixture of open-source driving QA datasets, allowing it to acquire a foundational understanding of driving scenarios. Next, CoReVLA is deployed within the Cave Automatic Virtual Environment (CAVE) simulation platform, where driver takeover data is collected from real-time interactions. Each takeover indicates a long-tail scenario that CoReVLA fails to handle reliably. Finally, the model is refined via Direct Preference Optimization (DPO), allowing it to learn directly from human preferences and thereby avoid reward hacking caused by manually designed rewards. Extensive open-loop and closed-loop experiments demonstrate that the proposed CoReVLA model can accurately perceive driving scenarios and make appropriate decisions. On the Bench2Drive benchmark, CoReVLA achieves a Driving Score (DS) of 72.18 and a Success Rate (SR) of 50%, outperforming state-of-the-art methods by 7.96 DS and 15% SR under long-tail, safety-critical scenarios. Furthermore, case studies demonstrate the model's ability to continually improve its performance in similar failure-prone scenarios by leveraging past takeover experiences. All codea and preprocessed datasets are available at: this https URL 

**Abstract (ZH)**: 自主驾驶（AD）系统取得了显著进展，但在长尾、安全关键场景中的表现仍然有限。这些罕见情况导致了不成比例的事故。视觉-语言-动作（VLA）模型具备强大的推理能力，可能提供解决方案，但由于缺乏高质量数据以及在这些条件下学习效率低下，其效果受限。为解决这些挑战，我们提出了一种持续学习端到端自主驾驶框架CoReVLA，通过数据收集和行为细化的双重过程，提高长尾场景中的性能。首先，模型在开源驾驶QA数据集中联合微调，使其获得驾驶场景的基本理解。接着，CoReVLA在Cave自动虚拟环境（CAVE）模拟平台上部署，收集实时交互中的驾驶接管数据。每次接管都表示一个CoReVLA无法可靠处理的长尾场景。最后，模型通过直接偏好优化（DPO）进行细化，使其可以直接从人类偏好中学习，从而避免手动设计奖励引起的奖励作弊。大量开环和闭环实验表明，所提出的CoReVLA模型能够准确感知驾驶场景并做出适当决策。在Bench2Drive基准测试中，CoReVLA的驾驶得分为72.18，成功率50%，在长尾、安全关键场景中分别优于现有最佳方法7.96分和15%的成功率。此外，案例研究显示该模型能够通过利用过去的接管经验，持续改进类似可能出现故障的场景中的表现。所有代码和预处理数据集可在以下链接获取：this https URL。 

---
# Swarm Oracle: Trustless Blockchain Agreements through Robot Swarms 

**Title (ZH)**: swarm oracle: 通过机器人蜂群实现的无信任区块链协议 

**Authors**: Alexandre Pacheco, Hanqing Zhao, Volker Strobel, Tarik Roukny, Gregory Dudek, Andreagiovanni Reina, Marco Dorigo  

**Link**: [PDF](https://arxiv.org/pdf/2509.15956)  

**Abstract**: Blockchain consensus, rooted in the principle ``don't trust, verify'', limits access to real-world data, which may be ambiguous or inaccessible to some participants. Oracles address this limitation by supplying data to blockchains, but existing solutions may reduce autonomy, transparency, or reintroduce the need for trust. We propose Swarm Oracle: a decentralized network of autonomous robots -- that is, a robot swarm -- that use onboard sensors and peer-to-peer communication to collectively verify real-world data and provide it to smart contracts on public blockchains. Swarm Oracle leverages the built-in decentralization, fault tolerance and mobility of robot swarms, which can flexibly adapt to meet information requests on-demand, even in remote locations. Unlike typical cooperative robot swarms, Swarm Oracle integrates robots from multiple stakeholders, protecting the system from single-party biases but also introducing potential adversarial behavior. To ensure the secure, trustless and global consensus required by blockchains, we employ a Byzantine fault-tolerant protocol that enables robots from different stakeholders to operate together, reaching social agreements of higher quality than the estimates of individual robots. Through extensive experiments using both real and simulated robots, we showcase how consensus on uncertain environmental information can be achieved, despite several types of attacks orchestrated by large proportions of the robots, and how a reputation system based on blockchain tokens lets Swarm Oracle autonomously recover from faults and attacks, a requirement for long-term operation. 

**Abstract (ZH)**: 基于自治机器人蜂群的 Swarm Oracle：一种分布式数据验证系统 

---
# Right-Side-Out: Learning Zero-Shot Sim-to-Real Garment Reversal 

**Title (ZH)**: 从右翻转：学习零样本模拟到现实的服装翻转 

**Authors**: Chang Yu, Siyu Ma, Wenxin Du, Zeshun Zong, Han Xue, Wendi Chen, Cewu Lu, Yin Yang, Xuchen Han, Joseph Masterjohn, Alejandro Castro, Chenfanfu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15953)  

**Abstract**: Turning garments right-side out is a challenging manipulation task: it is highly dynamic, entails rapid contact changes, and is subject to severe visual occlusion. We introduce Right-Side-Out, a zero-shot sim-to-real framework that effectively solves this challenge by exploiting task structures. We decompose the task into Drag/Fling to create and stabilize an access opening, followed by Insert&Pull to invert the garment. Each step uses a depth-inferred, keypoint-parameterized bimanual primitive that sharply reduces the action space while preserving robustness. Efficient data generation is enabled by our custom-built, high-fidelity, GPU-parallel Material Point Method (MPM) simulator that models thin-shell deformation and provides robust and efficient contact handling for batched rollouts. Built on the simulator, our fully automated pipeline scales data generation by randomizing garment geometry, material parameters, and viewpoints, producing depth, masks, and per-primitive keypoint labels without any human annotations. With a single depth camera, policies trained entirely in simulation deploy zero-shot on real hardware, achieving up to 81.3% success rate. By employing task decomposition and high fidelity simulation, our framework enables tackling highly dynamic, severely occluded tasks without laborious human demonstrations. 

**Abstract (ZH)**: 正反面翻转衣物是一项具有挑战性的操作任务：它高度动态，涉及快速的接触变化，并且受到严重的视觉遮挡。我们提出了Right-Side-Out，一种通过利用任务结构有效解决这一挑战的零样本模拟到现实的框架。 

---
# A Vision-Language-Action-Critic Model for Robotic Real-World Reinforcement Learning 

**Title (ZH)**: 一种用于机器人现实世界强化学习的视觉-语言-动作-价值模型 

**Authors**: Shaopeng Zhai, Qi Zhang, Tianyi Zhang, Fuxian Huang, Haoran Zhang, Ming Zhou, Shengzhe Zhang, Litao Liu, Sixu Lin, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15937)  

**Abstract**: Robotic real-world reinforcement learning (RL) with vision-language-action (VLA) models is bottlenecked by sparse, handcrafted rewards and inefficient exploration. We introduce VLAC, a general process reward model built upon InternVL and trained on large scale heterogeneous datasets. Given pairwise observations and a language goal, it outputs dense progress delta and done signal, eliminating task-specific reward engineering, and supports one-shot in-context transfer to unseen tasks and environments. VLAC is trained on vision-language datasets to strengthen perception, dialogic and reasoning capabilities, together with robot and human trajectories data that ground action generation and progress estimation, and additionally strengthened to reject irrelevant prompts as well as detect regression or stagnation by constructing large numbers of negative and semantically mismatched samples. With prompt control, a single VLAC model alternately generating reward and action tokens, unifying critic and policy. Deployed inside an asynchronous real-world RL loop, we layer a graded human-in-the-loop protocol (offline demonstration replay, return and explore, human guided explore) that accelerates exploration and stabilizes early learning. Across four distinct real-world manipulation tasks, VLAC lifts success rates from about 30\% to about 90\% within 200 real-world interaction episodes; incorporating human-in-the-loop interventions yields a further 50% improvement in sample efficiency and achieves up to 100% final success. 

**Abstract (ZH)**: 基于视觉-语言-动作模型的机器人现实世界强化学习中的瓶颈在于稀疏的手工设计奖励和探索效率低下。我们引入了VLAC，这是一种基于InternVL构建的一般过程奖励模型，并在大规模异构数据集上进行训练。给定成对的观察和语言目标，它输出密集的进步变化和完成信号，消除了任务特定的奖励工程，并支持一次性在上下文环境中转移未知任务和环境。VLAC在视觉语言数据集上进行训练以加强感知、对话和推理能力，同时结合机器人和人类轨迹数据以强化动作生成和进度评估，并通过构造大量负样本和语义不匹配样本进一步增强，以拒绝无关提示并检测退化或停滞。通过提示控制，一个VLAC模型交替生成奖励和动作标记，统一批评家和策略。在异步现实世界强化学习循环内部部署时，我们叠加了一种分等级的人在回路协议（离线示范重放、返回和探索、人类引导探索），以加速探索并稳定早期学习。在四个不同的现实世界操作任务中，VLAC在200个现实世界交互回合内将成功率从约30%提升到约90%；包含人机交互干预措施进一步提高了样本效率50%，并在最终实现了100%的成功率。 

---
# FlyKites: Human-centric Interactive Exploration and Assistance under Limited Communication 

**Title (ZH)**: FlyKites: 以人为本的有限通信条件下的交互探索与辅助 

**Authors**: Yuyang Zhang, Zhuoli Tian, Jinsheng Wei, Meng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.15807)  

**Abstract**: Fleets of autonomous robots have been deployed for exploration of unknown scenes for features of interest, e.g., subterranean exploration, reconnaissance, search and rescue missions. During exploration, the robots may encounter un-identified targets, blocked passages, interactive objects, temporary failure, or other unexpected events, all of which require consistent human assistance with reliable communication for a time period. This however can be particularly challenging if the communication among the robots is severely restricted to only close-range exchange via ad-hoc networks, especially in extreme environments like caves and underground tunnels. This paper presents a novel human-centric interactive exploration and assistance framework called FlyKites, for multi-robot systems under limited communication. It consists of three interleaved components: (I) the distributed exploration and intermittent communication (called the "spread mode"), where the robots collaboratively explore the environment and exchange local data among the fleet and with the operator; (II) the simultaneous optimization of the relay topology, the operator path, and the assignment of robots to relay roles (called the "relay mode"), such that all requested assistance can be provided with minimum delay; (III) the human-in-the-loop online execution, where the robots switch between different roles and interact with the operator adaptively. Extensive human-in-the-loop simulations and hardware experiments are performed over numerous challenging scenes. 

**Abstract (ZH)**: 基于有限通信的多机器人系统人本交互探索与辅助框架：FlyKites 

---
# GP3: A 3D Geometry-Aware Policy with Multi-View Images for Robotic Manipulation 

**Title (ZH)**: GP3: 一种基于三维几何的多视图图像机器人操纵策略 

**Authors**: Quanhao Qian, Guoyang Zhao, Gongjie Zhang, Jiuniu Wang, Ran Xu, Junlong Gao, Deli Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.15733)  

**Abstract**: Effective robotic manipulation relies on a precise understanding of 3D scene geometry, and one of the most straightforward ways to acquire such geometry is through multi-view observations. Motivated by this, we present GP3 -- a 3D geometry-aware robotic manipulation policy that leverages multi-view input. GP3 employs a spatial encoder to infer dense spatial features from RGB observations, which enable the estimation of depth and camera parameters, leading to a compact yet expressive 3D scene representation tailored for manipulation. This representation is fused with language instructions and translated into continuous actions via a lightweight policy head. Comprehensive experiments demonstrate that GP3 consistently outperforms state-of-the-art methods on simulated benchmarks. Furthermore, GP3 transfers effectively to real-world robots without depth sensors or pre-mapped environments, requiring only minimal fine-tuning. These results highlight GP3 as a practical, sensor-agnostic solution for geometry-aware robotic manipulation. 

**Abstract (ZH)**: GP3：一种基于多视图输入的3D几何感知机器人 manipulation 政策 

---
# Imagination at Inference: Synthesizing In-Hand Views for Robust Visuomotor Policy Inference 

**Title (ZH)**: 基于推理的想象：合成手持视角以实现鲁棒的视听运动策略推理 

**Authors**: Haoran Ding, Anqing Duan, Zezhou Sun, Dezhen Song, Yoshihiko Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2509.15717)  

**Abstract**: Visual observations from different viewpoints can significantly influence the performance of visuomotor policies in robotic manipulation. Among these, egocentric (in-hand) views often provide crucial information for precise control. However, in some applications, equipping robots with dedicated in-hand cameras may pose challenges due to hardware constraints, system complexity, and cost. In this work, we propose to endow robots with imaginative perception - enabling them to 'imagine' in-hand observations from agent views at inference time. We achieve this via novel view synthesis (NVS), leveraging a fine-tuned diffusion model conditioned on the relative pose between the agent and in-hand views cameras. Specifically, we apply LoRA-based fine-tuning to adapt a pretrained NVS model (ZeroNVS) to the robotic manipulation domain. We evaluate our approach on both simulation benchmarks (RoboMimic and MimicGen) and real-world experiments using a Unitree Z1 robotic arm for a strawberry picking task. Results show that synthesized in-hand views significantly enhance policy inference, effectively recovering the performance drop caused by the absence of real in-hand cameras. Our method offers a scalable and hardware-light solution for deploying robust visuomotor policies, highlighting the potential of imaginative visual reasoning in embodied agents. 

**Abstract (ZH)**: 不同的视角视觉观察能够显著影响机器人操作中的视运动策略性能。在这其中，第一人称（在手）视角往往提供精确控制的关键信息。然而，在某些应用中，为机器人配备专用在手摄像头可能会由于硬件限制、系统复杂性和成本问题而带来挑战。在本文中，我们提出赋予机器人想象感知——使机器人能够在推理时“想象”来自执行者视角的在手观察。我们通过新颖的观点合成（NVS）实现这一目标，利用在执行者和在手视角摄像头之间相对姿态条件下细调的扩散模型。具体而言，我们采用基于LoRA的细调方法，将预训练的NVS模型（ZeroNVS）适应到机器人操作领域。我们在模拟基准（RoboMimic和MimicGen）和使用Unitree Z1机器人手臂进行的草莓采摘真实世界实验中评估了该方法。结果表明，合成的在手视角显著提升了策略推理性能，有效弥补了缺少真实在手摄像头而导致的性能下降。我们的方法提供了一种可扩展且硬件要求较低的解决方案，用于部署稳健的视运动策略，并突显了想象视觉推理在实体代理中的潜力。 

---
# Indoor Positioning Based on Active Radar Sensing and Passive Reflectors: Reflector Placement Optimization 

**Title (ZH)**: 基于主动雷达传感和被动反射器的室内定位：反射器布置优化 

**Authors**: Sven Hinderer, Pascal Schlachter, Zhibin Yu, Xiaofeng Wu, Bin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15613)  

**Abstract**: We extend our work on a novel indoor positioning system (IPS) for autonomous mobile robots (AMRs) based on radar sensing of local, passive radar reflectors. Through the combination of simple reflectors and a single-channel frequency modulated continuous wave (FMCW) radar, high positioning accuracy at low system cost can be achieved. Further, a multi-objective (MO) particle swarm optimization (PSO) algorithm is presented that optimizes the 2D placement of radar reflectors in complex room settings. 

**Abstract (ZH)**: 基于雷达感知局部被动雷达反射器的新型室内定位系统扩展研究：低成本高精度的单通道FMCW雷达与多目标粒子 swarm 优化算法优化雷达反射器布局 

---
# PRIMT: Preference-based Reinforcement Learning with Multimodal Feedback and Trajectory Synthesis from Foundation Models 

**Title (ZH)**: PRIMT：基于偏好的强化学习与多模态反馈及路径合成 

**Authors**: Ruiqi Wang, Dezhong Zhao, Ziqin Yuan, Tianyu Shao, Guohua Chen, Dominic Kao, Sungeun Hong, Byung-Cheol Min  

**Link**: [PDF](https://arxiv.org/pdf/2509.15607)  

**Abstract**: Preference-based reinforcement learning (PbRL) has emerged as a promising paradigm for teaching robots complex behaviors without reward engineering. However, its effectiveness is often limited by two critical challenges: the reliance on extensive human input and the inherent difficulties in resolving query ambiguity and credit assignment during reward learning. In this paper, we introduce PRIMT, a PbRL framework designed to overcome these challenges by leveraging foundation models (FMs) for multimodal synthetic feedback and trajectory synthesis. Unlike prior approaches that rely on single-modality FM evaluations, PRIMT employs a hierarchical neuro-symbolic fusion strategy, integrating the complementary strengths of large language models and vision-language models in evaluating robot behaviors for more reliable and comprehensive feedback. PRIMT also incorporates foresight trajectory generation, which reduces early-stage query ambiguity by warm-starting the trajectory buffer with bootstrapped samples, and hindsight trajectory augmentation, which enables counterfactual reasoning with a causal auxiliary loss to improve credit assignment. We evaluate PRIMT on 2 locomotion and 6 manipulation tasks on various benchmarks, demonstrating superior performance over FM-based and scripted baselines. 

**Abstract (ZH)**: 基于偏好强化学习的PRIMT框架：利用基础模型克服偏好输入和奖励学习挑战 

---
# STARC: See-Through-Wall Augmented Reality Framework for Human-Robot Collaboration in Emergency Response 

**Title (ZH)**: STARC: 贯穿墙体增强现实框架在紧急响应中的人机协作 

**Authors**: Shenghai Yuan, Weixiang Guo, Tianxin Hu, Yu Yang, Jinyu Chen, Rui Qian, Zhongyuan Liu, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.15507)  

**Abstract**: In emergency response missions, first responders must navigate cluttered indoor environments where occlusions block direct line-of-sight, concealing both life-threatening hazards and victims in need of rescue. We present STARC, a see-through AR framework for human-robot collaboration that fuses mobile-robot mapping with responder-mounted LiDAR sensing. A ground robot running LiDAR-inertial odometry performs large-area exploration and 3D human detection, while helmet- or handheld-mounted LiDAR on the responder is registered to the robot's global map via relative pose estimation. This cross-LiDAR alignment enables consistent first-person projection of detected humans and their point clouds - rendered in AR with low latency - into the responder's view. By providing real-time visualization of hidden occupants and hazards, STARC enhances situational awareness and reduces operator risk. Experiments in simulation, lab setups, and tactical field trials confirm robust pose alignment, reliable detections, and stable overlays, underscoring the potential of our system for fire-fighting, disaster relief, and other safety-critical operations. Code and design will be open-sourced upon acceptance. 

**Abstract (ZH)**: 在应急响应任务中，一线救援人员必须穿越拥挤的室内环境，其中遮挡物阻挡了直接视线，隐藏了生命威胁的危险和需要救援的受害者。我们提出了一种名为STARC的透明AR框架，用于人机协作，融合了移动机器人建图与救援人员佩戴的LiDAR感测。地面机器人运行LiDAR-惯性_odometry进行大面积探索和3D人类检测，而救援人员头盔或手持设备上的LiDAR与机器人全局地图通过相对姿态估计进行配准。这种跨LiDAR对齐使得检测到的人员及其点云能够在AR中以低延迟的一致第一人称投影到救援人员的视野中。通过实时可视化隐藏的占用者和危险，STARC增强了情况意识并降低了操作者风险。在仿真、实验室配置和战术现场试验中，STARC的稳健姿态对齐、可靠的检测和稳定叠加得到了证实，彰显了其在火灾救援、灾难救助和其他关键安全操作中的潜力。代码和设计将在接受后开源。 

---
# Explainable AI-Enhanced Supervisory Control for Robust Multi-Agent Robotic Systems 

**Title (ZH)**: 可解释的AI增强监督控制以实现稳健的多agents机器人系统 

**Authors**: Reza Pirayeshshirazinezhad, Nima Fathi  

**Link**: [PDF](https://arxiv.org/pdf/2509.15491)  

**Abstract**: We present an explainable AI-enhanced supervisory control framework for multi-agent robotics that combines (i) a timed-automata supervisor for safe, auditable mode switching, (ii) robust continuous control (Lyapunov-based controller for large-angle maneuver; sliding-mode controller (SMC) with boundary layers for precision and disturbance rejection), and (iii) an explainable predictor that maps mission context to gains and expected performance (energy, error). Monte Carlo-driven optimization provides the training data, enabling transparent real-time trade-offs.
We validated the approach in two contrasting domains, spacecraft formation flying and autonomous underwater vehicles (AUVs). Despite different environments (gravity/actuator bias vs. hydrodynamic drag/currents), both share uncertain six degrees of freedom (6-DOF) rigid-body dynamics, relative motion, and tight tracking needs, making them representative of general robotic systems. In the space mission, the supervisory logic selects parameters that meet mission criteria. In AUV leader-follower tests, the same SMC structure maintains a fixed offset under stochastic currents with bounded steady error. In spacecraft validation, the SMC controller achieved submillimeter alignment with 21.7% lower tracking error and 81.4% lower energy consumption compared to Proportional-Derivative PD controller baselines. At the same time, in AUV tests, SMC maintained bounded errors under stochastic currents. These results highlight both the portability and the interpretability of the approach for safety-critical, resource-constrained multi-agent robotics. 

**Abstract (ZH)**: 一种增强的多智能体机器人监督控制框架：可解释的定时自动机监督器结合鲁棒连续控制和可解释的预测器及蒙特卡洛驱动的优化方法 

---
# Implicit Kinodynamic Motion Retargeting for Human-to-humanoid Imitation Learning 

**Title (ZH)**: 隐式动力学运动重定位用于人类到类人机器人模仿学习 

**Authors**: Xingyu Chen, Hanyu Wu, Sikai Wu, Mingliang Zhou, Diyun Xiang, Haodong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15443)  

**Abstract**: Human-to-humanoid imitation learning aims to learn a humanoid whole-body controller from human motion. Motion retargeting is a crucial step in enabling robots to acquire reference trajectories when exploring locomotion skills. However, current methods focus on motion retargeting frame by frame, which lacks scalability. Could we directly convert large-scale human motion into robot-executable motion through a more efficient approach? To address this issue, we propose Implicit Kinodynamic Motion Retargeting (IKMR), a novel efficient and scalable retargeting framework that considers both kinematics and dynamics. In kinematics, IKMR pretrains motion topology feature representation and a dual encoder-decoder architecture to learn a motion domain mapping. In dynamics, IKMR integrates imitation learning with the motion retargeting network to refine motion into physically feasible trajectories. After fine-tuning using the tracking results, IKMR can achieve large-scale physically feasible motion retargeting in real time, and a whole-body controller could be directly trained and deployed for tracking its retargeted trajectories. We conduct our experiments both in the simulator and the real robot on a full-size humanoid robot. Extensive experiments and evaluation results verify the effectiveness of our proposed framework. 

**Abstract (ZH)**: 人类到类人机器人模仿学习旨在从人类运动中学习类人机器人全身控制器。运动目标化是使机器人在探索运动技能时获取参考轨迹的关键步骤。然而，当前方法专注于逐帧进行运动目标化，缺乏扩展性。我们能否通过更高效的方法直接将大规模人类运动转换为机器人可执行的运动？为了解决这一问题，我们提出了隐式动力学运动目标化（IKMR）框架，这是一种新颖有效的、可扩展的框架，同时考虑了运动学和动力学。在运动学中，IKMR 预训练运动拓扑特征表示和双编码器-解码器架构来学习运动域映射。在动力学中，IKMR 将模仿学习与运动目标化网络集成，以使运动精炼成物理可行的轨迹。经过追踪结果微调后，IKMR 可以实现实时大规模物理可行的运动目标化，并可以直接训练和部署全身控制器以跟踪其目标化轨迹。我们在仿真实际机器人上对一个全尺寸类人机器人进行了实验。广泛的实验和评估结果验证了我们提出框架的有效性。 

---
# Trust-Aware Embodied Bayesian Persuasion for Mixed-Autonomy 

**Title (ZH)**: 具有信任意识的混合自主性体态说服技术 

**Authors**: Shaoting Peng, Katherine Driggs-Campbell, Roy Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.15404)  

**Abstract**: Safe and efficient interaction between autonomous vehicles (AVs) and human-driven vehicles (HVs) is a critical challenge for future transportation systems. While game-theoretic models capture how AVs influence HVs, they often suffer from a long-term decay of influence and can be perceived as manipulative, eroding the human's trust. This can paradoxically lead to riskier human driving behavior over repeated interactions. In this paper, we address this challenge by proposing the Trust-Aware Embodied Bayesian Persuasion (TA-EBP) framework. Our work makes three key contributions: First, we apply Bayesian persuasion to model communication at traffic intersections, offering a transparent alternative to traditional game-theoretic models. Second, we introduce a trust parameter to the persuasion framework, deriving a theorem for the minimum trust level required for influence. Finally, we ground the abstract signals of Bayesian persuasion theory into a continuous, physically meaningful action space, deriving a second theorem for the optimal signal magnitude, realized as an AV's forward nudge. Additionally, we validate our framework in a mixed-autonomy traffic simulation, demonstrating that TA-EBP successfully persuades HVs to drive more cautiously, eliminating collisions and improving traffic flow compared to baselines that either ignore trust or lack communication. Our work provides a transparent and non-strategic framework for influence in human-robot interaction, enhancing both safety and efficiency. 

**Abstract (ZH)**: 可信意识嵌入贝叶斯劝导框架（TA-EBP）：自动驾驶汽车与有人驾驶车辆安全高效互动的研究 

---
# Embodied Arena: A Comprehensive, Unified, and Evolving Evaluation Platform for Embodied AI 

**Title (ZH)**: 具身竞技场：一个全面、统一且不断演化的具身AI评估平台 

**Authors**: Fei Ni, Min Zhang, Pengyi Li, Yifu Yuan, Lingfeng Zhang, Yuecheng Liu, Peilong Han, Longxin Kou, Shaojin Ma, Jinbin Qiao, David Gamaliel Arcos Bravo, Yuening Wang, Xiao Hu, Zhanguang Zhang, Xianze Yao, Yutong Li, Zhao Zhang, Ying Wen, Ying-Cong Chen, Xiaodan Liang, Liang Lin, Bin He, Haitham Bou-Ammar, He Wang, Huazhe Xu, Jiankang Deng, Shan Luo, Shuqiang Jiang, Wei Pan, Yang Gao, Stefanos Zafeiriou, Jan Peters, Yuzheng Zhuang, Yingxue Zhang, Yan Zheng, Hongyao Tang, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2509.15273)  

**Abstract**: Embodied AI development significantly lags behind large foundation models due to three critical challenges: (1) lack of systematic understanding of core capabilities needed for Embodied AI, making research lack clear objectives; (2) absence of unified and standardized evaluation systems, rendering cross-benchmark evaluation infeasible; and (3) underdeveloped automated and scalable acquisition methods for embodied data, creating critical bottlenecks for model scaling. To address these obstacles, we present Embodied Arena, a comprehensive, unified, and evolving evaluation platform for Embodied AI. Our platform establishes a systematic embodied capability taxonomy spanning three levels (perception, reasoning, task execution), seven core capabilities, and 25 fine-grained dimensions, enabling unified evaluation with systematic research objectives. We introduce a standardized evaluation system built upon unified infrastructure supporting flexible integration of 22 diverse benchmarks across three domains (2D/3D Embodied Q&A, Navigation, Task Planning) and 30+ advanced models from 20+ worldwide institutes. Additionally, we develop a novel LLM-driven automated generation pipeline ensuring scalable embodied evaluation data with continuous evolution for diversity and comprehensiveness. Embodied Arena publishes three real-time leaderboards (Embodied Q&A, Navigation, Task Planning) with dual perspectives (benchmark view and capability view), providing comprehensive overviews of advanced model capabilities. Especially, we present nine findings summarized from the evaluation results on the leaderboards of Embodied Arena. This helps to establish clear research veins and pinpoint critical research problems, thereby driving forward progress in the field of Embodied AI. 

**Abstract (ZH)**: 体态AI的发展显著落后于大型基础模型，主要原因包括三个关键挑战：缺乏对体态AI所需核心能力的系统理解，导致研究缺乏明确的目标；缺乏统一和标准化的评估体系，使得跨基准评估不可行；以及体态数据的自动化和可扩展获取方法欠发达，成为模型扩展的关键瓶颈。为应对这些障碍，我们提出了一种全面、统一和演化的评估平台——体态竞技场，用于体态AI。该平台构建了一个跨越三个层次（感知、推理、任务执行）、七个核心能力和25个细粒度维度的系统体态能力分类体系，从而实现统一评估并明确研究目标。我们引入了一个基于统一基础设施的标准化评估系统，该系统支持22个跨三个领域（2D/3D体态问答、导航、任务规划）的基准的灵活集成，并来自全球20多个研究机构的30多种高级模型。此外，我们开发了一种新颖的基于大模型的自动化生成流水线，确保能够实现可扩展的体态评估数据，并持续进化以增强多样性和全面性。体态竞技场发布了三个实时排行榜（体态问答、导航、任务规划），从基准视角和能力视角提供高级模型能力的全面概述。特别地，我们总结了体态竞技场排行榜评估结果中的九项发现，这些发现有助于明确研究方向并指出关键研究问题，从而推动体态AI领域的发展。 

---
# GiAnt: A Bio-Inspired Hexapod for Adaptive Terrain Navigation and Object Detection 

**Title (ZH)**: GiAnt：一种生物启发的六足机器人，用于自适应地形导航和对象检测 

**Authors**: Aasfee Mosharraf Bhuiyan, Md Luban Mehda, Md. Thawhid Hasan Puspo, Jubayer Amin Pritom  

**Link**: [PDF](https://arxiv.org/pdf/2509.15264)  

**Abstract**: This paper presents the design, development and testing of GiAnt, an affordable hexapod which is inspired by the efficient motions of ants. The decision to model GiAnt after ants rather than other insects is rooted in ants' natural adaptability to a variety of terrains. This bio-inspired approach gives it a significant advantage in outdoor applications, offering terrain flexibility along with efficient energy use. It features a lightweight 3D-printed and laser cut structure weighing 1.75 kg with dimensions of 310 mm x 200 mm x 120 mm. Its legs have been designed with a simple Single Degree of Freedom (DOF) using a link and crank mechanism. It is great for conquering challenging terrains such as grass, rocks, and steep surfaces. Unlike traditional robots using four wheels for motion, its legged design gives superior adaptability to uneven and rough surfaces. GiAnt's control system is built on Arduino, allowing manual operation. An effective way of controlling the legs of GiAnt was achieved by gait analysis. It can move up to 8 cm of height easily with its advanced leg positioning system. Furthermore, equipped with machine learning and image processing technology, it can identify 81 different objects in a live monitoring system. It represents a significant step towards creating accessible hexapod robots for research, exploration, and surveying, offering unique advantages in adaptability and control simplicity. 

**Abstract (ZH)**: GiAnt：一种受蚂蚁启发的经济型六足机器人及其设计、开发与测试 

---
# DIPP: Discriminative Impact Point Predictor for Catching Diverse In-Flight Objects 

**Title (ZH)**: DIPP：用于捕获多样在飞行对象的辨别性影响点预测器 

**Authors**: Ngoc Huy Nguyen, Kazuki Shibata, Takamitsu Matsubara  

**Link**: [PDF](https://arxiv.org/pdf/2509.15254)  

**Abstract**: In this study, we address the problem of in-flight object catching using a quadruped robot with a basket. Our objective is to accurately predict the impact point, defined as the object's landing position. This task poses two key challenges: the absence of public datasets capturing diverse objects under unsteady aerodynamics, which are essential for training reliable predictors; and the difficulty of accurate early-stage impact point prediction when trajectories appear similar across objects. To overcome these issues, we construct a real-world dataset of 8,000 trajectories from 20 objects, providing a foundation for advancing in-flight object catching under complex aerodynamics. We then propose the Discriminative Impact Point Predictor (DIPP), consisting of two modules: (i) a Discriminative Feature Embedding (DFE) that separates trajectories by dynamics to enable early-stage discrimination and generalization, and (ii) an Impact Point Predictor (IPP) that estimates the impact point from these features. Two IPP variants are implemented: an Neural Acceleration Estimator (NAE)-based method that predicts trajectories and derives the impact point, and a Direct Point Estimator (DPE)-based method that directly outputs it. Experimental results show that our dataset is more diverse and complex than existing dataset, and that our method outperforms baselines on both 15 seen and 5 unseen objects. Furthermore, we show that improved early-stage prediction enhances catching success in simulation and demonstrate the effectiveness of our approach through real-world experiments. The demonstration is available at this https URL. 

**Abstract (ZH)**: 在本研究中，我们使用带有篮子的四足机器人解决了飞行中捕获物体的问题。我们的目标是对物体的着陆位置进行准确预测，定义为冲击点。这项任务面临两个关键挑战：缺乏涵盖不同物体并在不稳定空气动力学条件下进行捕获的公开数据集，这些数据集对于训练可靠的预测器是必要的；以及在物体轨迹相似时难以进行准确的早期冲击点预测。为了解决这些问题，我们构建了一个包含20个物体8,000条轨迹的真实世界数据集，为在复杂空气动力学条件下进行飞行中物体捕获的研究奠定了基础。然后，我们提出了判别冲击点预测器（DIPP），由两个模块组成：（i）判别性特征嵌入（DFE）模块，通过区分轨迹的动力学特性来实现早期阶段的判别和泛化；（ii）冲击点预测器（IPP），从这些特征中估计冲击点。我们实现了两种IPP变体：一种基于神经加速度估计器（NAE）的方法，用于预测轨迹并推导出冲击点，另一种基于直接点估计器（DPE）的方法，直接输出冲击点。实验结果表明，我们的数据集比现有数据集更具多样性和复杂性，并且我们的方法在15个已见物体和5个未见物体上均优于基线方法。此外，我们展示了早期阶段预测的改进如何在仿真中提高捕获成功率，并通过实际实验展示了我们方法的有效性。演示内容可在以下链接查看：这个 https URL 

---
# A CARLA-based Simulation of Electrically Driven Forklifts 

**Title (ZH)**: 基于CARLA的电动叉车仿真模拟 

**Authors**: David Claus, Christiane Thielemann, Hans-Georg Stark  

**Link**: [PDF](https://arxiv.org/pdf/2509.15909)  

**Abstract**: This paper presents the simulation of the operation of an electric forklift fleet within an intralogistics scenario. For this purpose, the open source simulation tool CARLA is used; according to our knowledge this is a novel approach in the context of logistics simulation. First, CARLA is used to generate and visualize a realistic 3D outdoor warehouse scenario, incorporating a number of randomly moving forklifts. In a next step, intralogistics transport tasks, such as pick-and-place, are simulated for the forklift fleet, including shortest-path finding. Furthermore, the capability to play back localization data, previously recorded from a ''real'' forklift fleet, is this http URL play back is done in the original recreated environment, thereby enabling the visualization of the forklifts movements. Finally, the energy consumption of the forklift trucks is simulated by integrating a physical battery model that generates the state of charge (SOC) of each truck as a function of load and activity. To demonstrate the wide range of possible applications for the CARLA simulation platform, we describe two use cases. The first deals with the problem of detecting regions with critically high traffic densities, the second with optimal placement of charging stations for the forklift trucks. Both use cases are calculated for an exemplary warehouse model. 

**Abstract (ZH)**: 本文介绍了一种在物流场景中使用开源仿真工具CARLA模拟电叉车车队运行的仿真方法；这是一种新颖的物流仿真方法。首先，使用CARLA生成并可视化一个真实的三维室外仓库场景，并包含一组随机移动的叉车。接着，模拟叉车车队的内物流运输任务，如拣选和放置，包括最短路径查找。此外，该仿真平台还具备回放实际叉车车队记录的定位数据的能力，并能够在原始重构的环境中进行回放，从而可视化叉车的运动。最后，通过集成一个物理电池模型来模拟叉车的能耗，该模型根据负载和活动生成每辆叉车的状态电量(SOC)。为了展示CARLA仿真平台的广泛适用性，我们描述了两个使用案例。第一个案例涉及检测交通密度极高区域的问题，第二个案例涉及优化叉车充电站的布局。这两个案例均基于一个示例仓库模型进行计算。 

---
# Hierarchical Reinforcement Learning with Low-Level MPC for Multi-Agent Control 

**Title (ZH)**: 面向多agent控制的层次强化学习与低层级MPC结合 

**Authors**: Max Studt, Georg Schildbach  

**Link**: [PDF](https://arxiv.org/pdf/2509.15799)  

**Abstract**: Achieving safe and coordinated behavior in dynamic, constraint-rich environments remains a major challenge for learning-based control. Pure end-to-end learning often suffers from poor sample efficiency and limited reliability, while model-based methods depend on predefined references and struggle to generalize. We propose a hierarchical framework that combines tactical decision-making via reinforcement learning (RL) with low-level execution through Model Predictive Control (MPC). For the case of multi-agent systems this means that high-level policies select abstract targets from structured regions of interest (ROIs), while MPC ensures dynamically feasible and safe motion. Tested on a predator-prey benchmark, our approach outperforms end-to-end and shielding-based RL baselines in terms of reward, safety, and consistency, underscoring the benefits of combining structured learning with model-based control. 

**Abstract (ZH)**: 基于层次框架实现动态复杂环境中的安全协调行为仍然是基于学习的控制中的一个主要挑战。通过强化学习进行战术决策与通过模型预测控制进行低级执行相结合的方法，可以克服端到端学习的样本效率低下和可靠性有限的问题，以及模型驱动方法对预定义参考的依赖及其泛化能力不足的问题。在多agent系统的情况下，高层策略从结构化的区域兴趣（ROIs）中选择抽象目标，而模型预测控制确保动态可行且安全的运动。我们的方法在基于捕食者-猎物基准进行测试时，在奖励、安全性和一致性方面均优于端到端学习和基于防护的强化学习基线，突显了结合结构化学习与模型驱动控制的优势。 

---
# SAMPO:Scale-wise Autoregression with Motion PrOmpt for generative world models 

**Title (ZH)**: SAMPO：尺度aware自回归与运动提示生成的世界模型 

**Authors**: Sen Wang, Jingyi Tian, Le Wang, Zhimin Liao, Jiayi Li, Huaiyi Dong, Kun Xia, Sanping Zhou, Wei Tang, Hua Gang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15536)  

**Abstract**: World models allow agents to simulate the consequences of actions in imagined environments for planning, control, and long-horizon decision-making. However, existing autoregressive world models struggle with visually coherent predictions due to disrupted spatial structure, inefficient decoding, and inadequate motion modeling. In response, we propose \textbf{S}cale-wise \textbf{A}utoregression with \textbf{M}otion \textbf{P}r\textbf{O}mpt (\textbf{SAMPO}), a hybrid framework that combines visual autoregressive modeling for intra-frame generation with causal modeling for next-frame generation. Specifically, SAMPO integrates temporal causal decoding with bidirectional spatial attention, which preserves spatial locality and supports parallel decoding within each scale. This design significantly enhances both temporal consistency and rollout efficiency. To further improve dynamic scene understanding, we devise an asymmetric multi-scale tokenizer that preserves spatial details in observed frames and extracts compact dynamic representations for future frames, optimizing both memory usage and model performance. Additionally, we introduce a trajectory-aware motion prompt module that injects spatiotemporal cues about object and robot trajectories, focusing attention on dynamic regions and improving temporal consistency and physical realism. Extensive experiments show that SAMPO achieves competitive performance in action-conditioned video prediction and model-based control, improving generation quality with 4.4$\times$ faster inference. We also evaluate SAMPO's zero-shot generalization and scaling behavior, demonstrating its ability to generalize to unseen tasks and benefit from larger model sizes. 

**Abstract (ZH)**: Scales-Aware Autoregression with Motion Prompt for Enhancing Visual Consistency and Efficiency 

---
# How Good are Foundation Models in Step-by-Step Embodied Reasoning? 

**Title (ZH)**: 大型预训练模型在分步具身推理任务中的表现如何？ 

**Authors**: Dinura Dissanayake, Ahmed Heakl, Omkar Thawakar, Noor Ahsan, Ritesh Thawkar, Ketan More, Jean Lahoud, Rao Anwer, Hisham Cholakkal, Ivan Laptev, Fahad Shahbaz Khan, Salman Khan  

**Link**: [PDF](https://arxiv.org/pdf/2509.15293)  

**Abstract**: Embodied agents operating in the physical world must make decisions that are not only effective but also safe, spatially coherent, and grounded in context. While recent advances in large multimodal models (LMMs) have shown promising capabilities in visual understanding and language generation, their ability to perform structured reasoning for real-world embodied tasks remains underexplored. In this work, we aim to understand how well foundation models can perform step-by-step reasoning in embodied environments. To this end, we propose the Foundation Model Embodied Reasoning (FoMER) benchmark, designed to evaluate the reasoning capabilities of LMMs in complex embodied decision-making scenarios. Our benchmark spans a diverse set of tasks that require agents to interpret multimodal observations, reason about physical constraints and safety, and generate valid next actions in natural language. We present (i) a large-scale, curated suite of embodied reasoning tasks, (ii) a novel evaluation framework that disentangles perceptual grounding from action reasoning, and (iii) empirical analysis of several leading LMMs under this setting. Our benchmark includes over 1.1k samples with detailed step-by-step reasoning across 10 tasks and 8 embodiments, covering three different robot types. Our results highlight both the potential and current limitations of LMMs in embodied reasoning, pointing towards key challenges and opportunities for future research in robot intelligence. Our data and code will be made publicly available. 

**Abstract (ZH)**: 基于身体代理在物理世界中的操作必须做出既有效又安全、空间上连贯且基于上下文的决策。尽管大型多模态模型在视觉理解与语言生成方面展现出有前景的能力，但它们在执行真实世界身体化任务的结构化推理方面的能力仍待探索。本文旨在理解基础模型在身体化环境中的逐步推理能力。为此，我们提出了基础模型身体化推理（FoMER）基准，该基准用于评估多模态模型在复杂身体化决策场景中的推理能力。我们的基准涵盖了多种任务，要求代理解读多模态观测结果、推理物理约束和安全性，并以自然语言生成有效的后续行动。我们展示了（i）大规模且精挑细选的身体化推理任务集，（ii）一种新的评估框架，用于分离感知接地与行动推理，以及（iii）在这一设置下几种领先多模态模型的实证分析。我们的基准包含超过1100个样本，涉及10个任务和8种不同的身体化代理，涵盖三种不同类型的机器人。我们的结果突显了多模态模型在身体化推理中的潜力与当前局限，并指出了未来机器人智能研究中的关键挑战与机遇。我们的数据和代码将公开发布。 

---
# Attention Schema-based Attention Control (ASAC): A Cognitive-Inspired Approach for Attention Management in Transformers 

**Title (ZH)**: 基于注意-schema的注意控制（ASAC）：一种受认知启发的变压器中的注意管理方法 

**Authors**: Krati Saxena, Federico Jurado Ruiz, Guido Manzi, Dianbo Liu, Alex Lamb  

**Link**: [PDF](https://arxiv.org/pdf/2509.16058)  

**Abstract**: Attention mechanisms have become integral in AI, significantly enhancing model performance and scalability by drawing inspiration from human cognition. Concurrently, the Attention Schema Theory (AST) in cognitive science posits that individuals manage their attention by creating a model of the attention itself, effectively allocating cognitive resources. Inspired by AST, we introduce ASAC (Attention Schema-based Attention Control), which integrates the attention schema concept into artificial neural networks. Our initial experiments focused on embedding the ASAC module within transformer architectures. This module employs a Vector-Quantized Variational AutoEncoder (VQVAE) as both an attention abstractor and controller, facilitating precise attention management. By explicitly modeling attention allocation, our approach aims to enhance system efficiency. We demonstrate ASAC's effectiveness in both the vision and NLP domains, highlighting its ability to improve classification accuracy and expedite the learning process. Our experiments with vision transformers across various datasets illustrate that the attention controller not only boosts classification accuracy but also accelerates learning. Furthermore, we have demonstrated the model's robustness and generalization capabilities across noisy and out-of-distribution datasets. In addition, we have showcased improved performance in multi-task settings. Quick experiments reveal that the attention schema-based module enhances resilience to adversarial attacks, optimizes attention to improve learning efficiency, and facilitates effective transfer learning and learning from fewer examples. These promising results establish a connection between cognitive science and machine learning, shedding light on the efficient utilization of attention mechanisms in AI systems. 

**Abstract (ZH)**: 基于注意模式的注意控制机制：ASAC在人工智能中的应用研究 

---
# EmoHeal: An End-to-End System for Personalized Therapeutic Music Retrieval from Fine-grained Emotions 

**Title (ZH)**: EmoHeal: 一种基于细粒度情绪的个性化治疗音乐检索端到端系统 

**Authors**: Xinchen Wan, Jinhua Liang, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15986)  

**Abstract**: Existing digital mental wellness tools often overlook the nuanced emotional states underlying everyday challenges. For example, pre-sleep anxiety affects more than 1.5 billion people worldwide, yet current approaches remain largely static and "one-size-fits-all", failing to adapt to individual needs. In this work, we present EmoHeal, an end-to-end system that delivers personalized, three-stage supportive narratives. EmoHeal detects 27 fine-grained emotions from user text with a fine-tuned XLM-RoBERTa model, mapping them to musical parameters via a knowledge graph grounded in music therapy principles (GEMS, iso-principle). EmoHeal retrieves audiovisual content using the CLAMP3 model to guide users from their current state toward a calmer one ("match-guide-target"). A within-subjects study (N=40) demonstrated significant supportive effects, with participants reporting substantial mood improvement (M=4.12, p<0.001) and high perceived emotion recognition accuracy (M=4.05, p<0.001). A strong correlation between perceived accuracy and therapeutic outcome (r=0.72, p<0.001) validates our fine-grained approach. These findings establish the viability of theory-driven, emotion-aware digital wellness tools and provides a scalable AI blueprint for operationalizing music therapy principles. 

**Abstract (ZH)**: 现有的数字心理健康工具往往忽视了日常生活挑战背后复杂的 emocional 状态。例如，睡前焦虑影响全球超过 15 亿人，但现有的方法仍然主要是静态的和“一刀切”的，未能适应个体需求。在此项研究中，我们介绍了 EmoHeal，一个端到端系统，提供个性化的三阶段支持叙述。EmoHeal 使用微调后的 XLM-RoBERTa 模型从用户文本中检测 27 种细粒度情绪，并通过基于音乐疗法原则的知识图谱（GEMS，等价原则）映射到音乐参数。EmoHeal 使用 CLAMP3 模型检索音频视觉内容，引导用户从当前状态向更平静的状态过渡（“匹配-引导-目标”）。单被试研究（N=40）表明，EmoHeal 显著提升了支持效果，参与者报告情绪显著改善（M=4.12，p<0.001），且对情绪识别准确性的感知很高（M=4.05，p<0.001）。感知准确性与治疗结果之间存在显著正相关（r=0.72，p<0.001），验证了我们的细粒度方法。这些发现确立了理论驱动、情绪感知的数字心理健康工具的可行性，并为操作化音乐疗法原则提供了可扩展的 AI 蓝图。 

---
# Uncertainty-Based Smooth Policy Regularisation for Reinforcement Learning with Few Demonstrations 

**Title (ZH)**: 基于不确定性平滑策略正则化的小样本强化学习 

**Authors**: Yujie Zhu, Charles A. Hepburn, Matthew Thorpe, Giovanni Montana  

**Link**: [PDF](https://arxiv.org/pdf/2509.15981)  

**Abstract**: In reinforcement learning with sparse rewards, demonstrations can accelerate learning, but determining when to imitate them remains challenging. We propose Smooth Policy Regularisation from Demonstrations (SPReD), a framework that addresses the fundamental question: when should an agent imitate a demonstration versus follow its own policy? SPReD uses ensemble methods to explicitly model Q-value distributions for both demonstration and policy actions, quantifying uncertainty for comparisons. We develop two complementary uncertainty-aware methods: a probabilistic approach estimating the likelihood of demonstration superiority, and an advantage-based approach scaling imitation by statistical significance. Unlike prevailing methods (e.g. Q-filter) that make binary imitation decisions, SPReD applies continuous, uncertainty-proportional regularisation weights, reducing gradient variance during training. Despite its computational simplicity, SPReD achieves remarkable gains in experiments across eight robotics tasks, outperforming existing approaches by up to a factor of 14 in complex tasks while maintaining robustness to demonstration quality and quantity. Our code is available at this https URL. 

**Abstract (ZH)**: 在稀疏奖励的强化学习中，演示可以加速学习，但确定何时模仿仍具有挑战性。我们提出了平滑政策正则化从演示（SPReD）框架，该框架解决了基本问题：代理何时应该模仿演示，何时应遵循其自己的策略？SPReD 使用集成方法明确建模演示和策略动作的 Q 值分布，量化不确定性以进行比较。我们开发了两种互补的不确定性感知方法：一种概率方法估计演示优越性的可能性，以及一种基于优势的方法，根据统计显著性调整模仿比例。不同于现有的二元模仿决策方法（如 Q-filter），SPReD 应用连续的、与不确定性成比例的正则化权重，在训练期间降低梯度方差。尽管 SPReD 计算简单，在八个机器人任务的实验中仍取得了显著效果，在复杂任务中优于现有方法多达 14 倍，同时对演示的质量和数量具有鲁棒性。我们的代码可在以下网址获取：this https URL。 

---
# Explainable AI for Maritime Autonomous Surface Ships (MASS): Adaptive Interfaces and Trustworthy Human-AI Collaboration 

**Title (ZH)**: 可解释的人工智能在海上自主水面船舶（MASS）中：适应性界面和可信赖的人机协作 

**Authors**: Zhuoyue Zhang, Haitong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.15959)  

**Abstract**: Autonomous navigation in maritime domains is accelerating alongside advances in artificial intelligence, sensing, and connectivity. Opaque decision-making and poorly calibrated human-automation interaction remain key barriers to safe adoption. This article synthesizes 100 studies on automation transparency for Maritime Autonomous Surface Ships (MASS) spanning situation awareness (SA), human factors, interface design, and regulation. We (i) map the Guidance-Navigation-Control stack to shore-based operational modes -- remote supervision (RSM) and remote control (RCM) -- and identify where human unsafe control actions (Human-UCAs) concentrate in handover and emergency loops; (ii) summarize evidence that transparency features (decision rationales, alternatives, confidence/uncertainty, and rule-compliance indicators) improve understanding and support trust calibration, though reliability and predictability often dominate trust; (iii) distill design strategies for transparency at three layers: sensor/SA acquisition and fusion, HMI/eHMI presentation (textual/graphical overlays, color coding, conversational and immersive UIs), and engineer-facing processes (resilient interaction design, validation, and standardization). We integrate methods for Human-UCA identification (STPA-Cog + IDAC), quantitative trust/SA assessment, and operator workload monitoring, and outline regulatory and rule-based implications including COLREGs formalization and route exchange. We conclude with an adaptive transparency framework that couples operator state estimation with explainable decision support to reduce cognitive overload and improve takeover timeliness. The review highlights actionable figure-of-merit displays (e.g., CPA/TCPA risk bars, robustness heatmaps), transparent model outputs (rule traceability, confidence), and training pipelines (HIL/MIL, simulation) as near-term levers for safer MASS operations. 

**Abstract (ZH)**: 自主智能船舶在海洋领域的自主导航正随着人工智能、感知技术及连接性的进步而加速发展。不透明的决策制定和人机互动不匹配依然是实现安全采用的关键障碍。本文综合了100篇关于自主智能表面船舶（MASS）自动化透明性的研究，涵盖情况意识、人类因素、界面设计和法规等方面。我们（i）将控制-制导-导航堆栈与陆基操作模式——远程监督（RSM）和远程控制（RCM）——相对应，并确定了人类不安全控制行动（Human-UCAs）在交接和应急循环中的集中点；（ii）总结了透明性特征（决策依据、可选方案、置信度/不确定性、规则遵从性指示器）如何提高理解并支持信任校准的证据，尽管可靠性和可预测性经常主导信任；（iii）提炼了三个层面的透明性设计策略：传感器/情况意识获取与融合、人机界面/增强人机界面呈现（包括文本/图形叠加、颜色编码、对话式和沉浸式用户界面）、以及工程师面向过程（鲁棒交互设计、验证和标准化）。我们整合了Human-UCA识别方法（STPA-Cog + IDAC）、定量信任/情况意识评估方法和操作员工作量监测方法，并概述了包括COLREGs的正式化和航路交换在内的监管和基于规则的含义。我们提出了一个适应性透明性框架，结合操作员状态估计与可解释决策支持，以减轻认知负担并提高接管时机。综述强调了具体可操作的评价指标展示（例如，CPA/TCPA风险条形图、鲁棒性热图）、透明的模型输出（规则可追溯性、置信度）以及训练管道（HIL/MIL、模拟）作为短期内提高自主智能船舶安全运营的关键措施。 

---
# Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds 

**Title (ZH)**: Foundation Models作为世界模型：基于文本的格子世界中的基础研究 

**Authors**: Remo Sasso, Michelangelo Conserva, Dominik Jeurissen, Paulo Rauber  

**Link**: [PDF](https://arxiv.org/pdf/2509.15915)  

**Abstract**: While reinforcement learning from scratch has shown impressive results in solving sequential decision-making tasks with efficient simulators, real-world applications with expensive interactions require more sample-efficient agents. Foundation models (FMs) are natural candidates to improve sample efficiency as they possess broad knowledge and reasoning capabilities, but it is yet unclear how to effectively integrate them into the reinforcement learning framework. In this paper, we anticipate and, most importantly, evaluate two promising strategies. First, we consider the use of foundation world models (FWMs) that exploit the prior knowledge of FMs to enable training and evaluating agents with simulated interactions. Second, we consider the use of foundation agents (FAs) that exploit the reasoning capabilities of FMs for decision-making. We evaluate both approaches empirically in a family of grid-world environments that are suitable for the current generation of large language models (LLMs). Our results suggest that improvements in LLMs already translate into better FWMs and FAs; that FAs based on current LLMs can already provide excellent policies for sufficiently simple environments; and that the coupling of FWMs and reinforcement learning agents is highly promising for more complex settings with partial observability and stochastic elements. 

**Abstract (ZH)**: 从基础模型看强化学习的样本效率提升：基于基础-world模型和基础代理的前景策略评估 

---
# Reward Hacking Mitigation using Verifiable Composite Rewards 

**Title (ZH)**: 使用可验证复合奖励减轻奖励欺诈 

**Authors**: Mirza Farhan Bin Tarek, Rahmatollah Beheshti  

**Link**: [PDF](https://arxiv.org/pdf/2509.15557)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) has recently shown that large language models (LLMs) can develop their own reasoning without direct supervision. However, applications in the medical domain, specifically for question answering, are susceptible to significant reward hacking during the reasoning phase. Our work addresses two primary forms of this behavior: i) providing a final answer without preceding reasoning, and ii) employing non-standard reasoning formats to exploit the reward mechanism. To mitigate these, we introduce a composite reward function with specific penalties for these behaviors. Our experiments show that extending RLVR with our proposed reward model leads to better-formatted reasoning with less reward hacking and good accuracy compared to the baselines. This approach marks a step toward reducing reward hacking and enhancing the reliability of models utilizing RLVR. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）最近表明大型语言模型可以在无需直接监督的情况下发展自己的推理能力。然而，在医疗领域的应用，特别是在问答方面，推理阶段容易受到显著的奖励作弊影响。我们的工作针对这两种主要行为进行了处理：i) 不进行推理直接给出最终答案，ii) 使用非标准的推理格式来利用奖励机制。为减轻这些现象，我们提出了一种复合奖励函数，并对这些行为设置了特定的惩罚。实验结果显示，将我们提出的奖励模型扩展到RLVR中，可以得到格式更规范的推理，减少奖励作弊现象，且具有良好的准确性，相比baseline方法更为优越。这种方法朝着减少奖励作弊和提高使用RLVR的模型可靠性迈出了一步。 

---
# ORCA: Agentic Reasoning For Hallucination and Adversarial Robustness in Vision-Language Models 

**Title (ZH)**: ORCA：用于幻觉和对抗鲁棒性的代理推理 

**Authors**: Chung-En Johnny Yu, Hsuan-Chih, Chen, Brian Jalaian, Nathaniel D. Bastian  

**Link**: [PDF](https://arxiv.org/pdf/2509.15435)  

**Abstract**: Large Vision-Language Models (LVLMs) exhibit strong multimodal capabilities but remain vulnerable to hallucinations from intrinsic errors and adversarial attacks from external exploitations, limiting their reliability in real-world applications. We present ORCA, an agentic reasoning framework that improves the factual accuracy and adversarial robustness of pretrained LVLMs through test-time structured inference reasoning with a suite of small vision models (less than 3B parameters). ORCA operates via an Observe--Reason--Critique--Act loop, querying multiple visual tools with evidential questions, validating cross-model inconsistencies, and refining predictions iteratively without access to model internals or retraining. ORCA also stores intermediate reasoning traces, which supports auditable decision-making. Though designed primarily to mitigate object-level hallucinations, ORCA also exhibits emergent adversarial robustness without requiring adversarial training or defense mechanisms. We evaluate ORCA across three settings: (1) clean images on hallucination benchmarks, (2) adversarially perturbed images without defense, and (3) adversarially perturbed images with defense applied. On the POPE hallucination benchmark, ORCA improves standalone LVLM performance by +3.64\% to +40.67\% across different subsets. Under adversarial perturbations on POPE, ORCA achieves an average accuracy gain of +20.11\% across LVLMs. When combined with defense techniques on adversarially perturbed AMBER images, ORCA further improves standalone LVLM performance, with gains ranging from +1.20\% to +48.00\% across evaluation metrics. These results demonstrate that ORCA offers a promising path toward building more reliable and robust multimodal systems. 

**Abstract (ZH)**: 大型多模态语言模型（LVLMs）表现出强大的跨模态能力，但仍然容易受到内在错误和外部利用的对抗攻击的影响，限制了其在实际应用中的可靠性。我们提出了ORCA，一个代理推理框架，通过使用一系列小型视觉模型（参数少于3B）进行测试时结构化推理来提高预训练LVLMs的事实准确性及对抗鲁棒性。ORCA 通过观察—推理—批评—行动循环运作，查询多个视觉工具并提出证据性问题，验证跨模型的一致性问题，并在无需访问模型内部结构或重新训练的情况下迭代地改进预测。ORCA 还存储了中间推理轨迹，支持可审计的决策制定。尽管主要设计用于减轻对象级别幻觉，但ORCA 在不使用对抗训练或防护机制的情况下也表现出新兴的对抗鲁棒性。我们在三种情况下评估了ORCA：（1）在幻觉基准上的干净图像；（2）未使用防护的对抗扰动图像；（3）使用防护措施的对抗扰动图像。在POPE幻觉基准上，ORCA 在不同子集上将LVLM性能提高了3.64%至40.67%。在POPE的对抗扰动下，ORCA 在LVLMs上实现了平均准确性提高20.11%。将ORCA 与对抗扰动下的AMBER图像的防护技术结合使用时，进一步提高了LVLMs的性能，在不同评估指标上提高了1.20%至48.00%。这些结果表明，ORCA 为构建更可靠和鲁棒的多模态系统提供了有前景的途径。 

---
# Emotion-Aware Speech Generation with Character-Specific Voices for Comics 

**Title (ZH)**: 基于角色特定声音的情感意识语音生成技术在漫画中的应用 

**Authors**: Zhiwen Qian, Jinhua Liang, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15253)  

**Abstract**: This paper presents an end-to-end pipeline for generating character-specific, emotion-aware speech from comics. The proposed system takes full comic volumes as input and produces speech aligned with each character's dialogue and emotional state. An image processing module performs character detection, text recognition, and emotion intensity recognition. A large language model performs dialogue attribution and emotion analysis by integrating visual information with the evolving plot context. Speech is synthesized through a text-to-speech model with distinct voice profiles tailored to each character and emotion. This work enables automated voiceover generation for comics, offering a step toward interactive and immersive comic reading experience. 

**Abstract (ZH)**: 本文提出了一套端到端的生成特定角色、具有情感意识的漫画语音的 pipeline。该系统以完整的漫画集为输入，生成与每个角色对话及其情感状态相匹配的语音。图像处理模块进行角色检测、文本识别和情感强度识别。大规模语言模型通过整合视觉信息和不断发展的故事情节来执行对话归属和情绪分析。语音通过针对每个角色和情感定制的声音模型进行合成。本文的工作使自动语音over生成成为可能，为进一步提供交互性和沉浸式漫画阅读体验奠定了基础。 

---
