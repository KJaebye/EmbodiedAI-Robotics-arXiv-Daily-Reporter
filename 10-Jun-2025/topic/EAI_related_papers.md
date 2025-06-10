# Versatile Loco-Manipulation through Flexible Interlimb Coordination 

**Title (ZH)**: 灵活的肢体协作实现多功能移动操作 

**Authors**: Xinghao Zhu, Yuxin Chen, Lingfeng Sun, Farzad Niroui, Simon Le CleacH, Jiuguang Wang, Kuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07876)  

**Abstract**: The ability to flexibly leverage limbs for loco-manipulation is essential for enabling autonomous robots to operate in unstructured environments. Yet, prior work on loco-manipulation is often constrained to specific tasks or predetermined limb configurations. In this work, we present Reinforcement Learning for Interlimb Coordination (ReLIC), an approach that enables versatile loco-manipulation through flexible interlimb coordination. The key to our approach is an adaptive controller that seamlessly bridges the execution of manipulation motions and the generation of stable gaits based on task demands. Through the interplay between two controller modules, ReLIC dynamically assigns each limb for manipulation or locomotion and robustly coordinates them to achieve the task success. Using efficient reinforcement learning in simulation, ReLIC learns to perform stable gaits in accordance with the manipulation goals in the real world. To solve diverse and complex tasks, we further propose to interface the learned controller with different types of task specifications, including target trajectories, contact points, and natural language instructions. Evaluated on 12 real-world tasks that require diverse and complex coordination patterns, ReLIC demonstrates its versatility and robustness by achieving a success rate of 78.9% on average. Videos and code can be found at this https URL. 

**Abstract (ZH)**: 灵活利用肢体进行运动操作的能力是使自主机器人能够在未结构化环境中操作的关键。然而，现有的运动操作研究往往局限于特定任务或预设的肢体配置。在此工作中，我们提出了一种通过灵活的肢体协调实现多样化运动操作的方法——基于强化学习的肢体间协调（ReLIC）。我们方法的关键在于一个自适应控制器，该控制器能够无缝地将操作动作的执行与基于任务需求的稳定运动模式生成相结合。通过两个控制器模块之间的相互作用，ReLIC 动态地分配每个肢体用于操作或运动，并稳健地协调它们以实现任务的成功。通过在仿真中高效地使用强化学习，ReLIC 学习在实际任务中根据操作目标生成稳定的运动模式。为了解决各种复杂任务，我们进一步提出将所学习的控制器与不同类型的任务规范（包括目标轨迹、接触点和自然语言指令）进行接口连接。在对12个需要多样化和复杂协调模式的现实任务进行评估后，ReLIC展示了其多样性和鲁棒性，平均成功率达到了78.9%。更多视频和代码请访问此链接。 

---
# Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse 

**Title (ZH)**: Fast ECoT: 通过思考重用的高效体态链式思维 

**Authors**: Zhekai Duan, Yuan Zhang, Shikai Geng, Gaowen Liu, Joschka Boedecker, Chris Xiaoxuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07639)  

**Abstract**: Embodied Chain-of-Thought (ECoT) reasoning enhances vision-language-action (VLA) models by improving performance and interpretability through intermediate reasoning steps. However, its sequential autoregressive token generation introduces significant inference latency, limiting real-time deployment. We propose Fast ECoT, an inference-time acceleration method that exploits the structured and repetitive nature of ECoT to (1) cache and reuse high-level reasoning across timesteps and (2) parallelise the generation of modular reasoning steps. Additionally, we introduce an asynchronous scheduler that decouples reasoning from action decoding, further boosting responsiveness. Fast ECoT requires no model changes or additional training and integrates easily into existing VLA pipelines. Experiments in both simulation (LIBERO) and real-world robot tasks show up to a 7.5% reduction in latency with comparable or improved task success rate and reasoning faithfulness, bringing ECoT policies closer to practical real-time deployment. 

**Abstract (ZH)**: 富含实体的链式思维(Fast ECoT)推理通过中间推理步骤提高视觉-语言-行动(VLA)模型的性能和可解释性，但其 sequential 自回归标记生成引入了显著的推理延迟，限制了实时部署。我们提出了一种在推理时加速的方法 Fast ECoT，该方法利用了 ECoT 的结构化和重复性特征，(1) 缓存并跨时间步重用高层次推理，(2) 并行生成模块化推理步骤。此外，我们引入了一个异步调度器，将推理与动作解码解耦，进一步提升了响应性。Fast ECoT 不需要对模型进行更改或额外训练，并且可以轻松集成到现有的 VLA 管道中。在模拟环境(LIBERO)和真实世界机器人任务中的实验显示，与可比或改进的任务成功率和推理正确性相比，延迟最多可减少 7.5%，使 ECoT 策略更接近实际的实时部署。 

---
# Blending Participatory Design and Artificial Awareness for Trustworthy Autonomous Vehicles 

**Title (ZH)**: 融合参与设计与人工意识以提高自主车辆的可信度 

**Authors**: Ana Tanevska, Ananthapathmanabhan Ratheesh Kumar, Arabinda Ghosh, Ernesto Casablanca, Ginevra Castellano, Sadegh Soudjani  

**Link**: [PDF](https://arxiv.org/pdf/2506.07633)  

**Abstract**: Current robotic agents, such as autonomous vehicles (AVs) and drones, need to deal with uncertain real-world environments with appropriate situational awareness (SA), risk awareness, coordination, and decision-making. The SymAware project strives to address this issue by designing an architecture for artificial awareness in multi-agent systems, enabling safe collaboration of autonomous vehicles and drones. However, these agents will also need to interact with human users (drivers, pedestrians, drone operators), which in turn requires an understanding of how to model the human in the interaction scenario, and how to foster trust and transparency between the agent and the human.
In this work, we aim to create a data-driven model of a human driver to be integrated into our SA architecture, grounding our research in the principles of trustworthy human-agent interaction. To collect the data necessary for creating the model, we conducted a large-scale user-centered study on human-AV interaction, in which we investigate the interaction between the AV's transparency and the users' behavior.
The contributions of this paper are twofold: First, we illustrate in detail our human-AV study and its findings, and second we present the resulting Markov chain models of the human driver computed from the study's data. Our results show that depending on the AV's transparency, the scenario's environment, and the users' demographics, we can obtain significant differences in the model's transitions. 

**Abstract (ZH)**: 当前的机器人代理，如自动驾驶车辆（AVs）和无人机，需要在不确定的现实环境中具备适当的情境意识（SA）、风险意识、协调和决策能力。SymAware项目致力于通过为多智能体系统设计人工意识架构，促进自动驾驶车辆和无人机的安全协作。然而，这些代理还将需要与人类用户（驾驶员、行人、无人机操作员）互动，这就要求我们理解如何在交互场景中建模人类行为，并促进代理与人类之间的信任和透明度。

在本文中，我们旨在创建一个基于数据的人类驾驶员模型，将其集成到我们的SA架构中，以实现可信赖的人机交互。为了收集创建模型所必需的数据，我们进行了大规模的以用户为中心的研究，探讨了AV透明度与用户行为之间的交互。

本文的贡献主要有两点：首先，我们详细介绍了人类-AV研究及其发现；其次，我们展示了从研究数据中计算得到的人类驾驶员马尔可夫链模型。我们的结果显示，根据不同AV的透明度、场景的环境以及用户的人口统计特征，模型的状态转换会存在显著差异。 

---
# BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation 

**Title (ZH)**: BitVLA: 1-bit Vision-Language-Action模型用于机器人 manipulation 

**Authors**: Hongyu Wang, Chuyan Xiong, Ruiping Wang, Xilin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07530)  

**Abstract**: Vision-Language-Action (VLA) models have shown impressive capabilities across a wide range of robotics manipulation tasks. However, their growing model size poses significant challenges for deployment on resource-constrained robotic systems. While 1-bit pretraining has proven effective for enhancing the inference efficiency of large language models with minimal performance loss, its application to VLA models remains underexplored. In this work, we present BitVLA, the first 1-bit VLA model for robotics manipulation, in which every parameter is ternary, i.e., {-1, 0, 1}. To further reduce the memory footprint of the vision encoder, we propose the distillation-aware training strategy that compresses the full-precision encoder to 1.58-bit weights. During this process, a full-precision encoder serves as a teacher model to better align latent representations. Despite the lack of large-scale robotics pretraining, BitVLA achieves performance comparable to the state-of-the-art model OpenVLA-OFT with 4-bit post-training quantization on the LIBERO benchmark, while consuming only 29.8% of the memory. These results highlight BitVLA's promise for deployment on memory-constrained edge devices. We release the code and model weights in this https URL. 

**Abstract (ZH)**: 面向机器人操作的1比特Vision-Language-Action (VLA) 模型 

---
# Taking Flight with Dialogue: Enabling Natural Language Control for PX4-based Drone Agent 

**Title (ZH)**: 凭借对话起飞：为基于PX4的无人机代理启用自然语言控制 

**Authors**: Shoon Kit Lim, Melissa Jia Ying Chong, Jing Huey Khor, Ting Yang Ling  

**Link**: [PDF](https://arxiv.org/pdf/2506.07509)  

**Abstract**: Recent advances in agentic and physical artificial intelligence (AI) have largely focused on ground-based platforms such as humanoid and wheeled robots, leaving aerial robots relatively underexplored. Meanwhile, state-of-the-art unmanned aerial vehicle (UAV) multimodal vision-language systems typically rely on closed-source models accessible only to well-resourced organizations. To democratize natural language control of autonomous drones, we present an open-source agentic framework that integrates PX4-based flight control, Robot Operating System 2 (ROS 2) middleware, and locally hosted models using Ollama. We evaluate performance both in simulation and on a custom quadcopter platform, benchmarking four large language model (LLM) families for command generation and three vision-language model (VLM) families for scene understanding. 

**Abstract (ZH)**: 近期在代理人和物理人工智能领域的进展主要集中在地面平台如人形机器人和轮式机器人上，而对飞行机器人则相对探索较少。与此同时，最先进的无人驾驶飞行器（UAV）多模态视觉-语言系统通常依赖于仅对资源充足的组织开放的闭源模型。为了使自然语言控制自主无人机的应用更加普及，我们提出了一种开源代理人框架，该框架结合了基于PX4的飞行控制、Robot Operating System 2（ROS 2）中间件以及使用Ollama托管的本地模型。我们在模拟和自定义四旋翼飞行器平台上评估了性能，并对四种大型语言模型（LLM）家族的命令生成和三种视觉-语言模型（VLM）家族的场景理解进行了基准测试。 

---
# MapBERT: Bitwise Masked Modeling for Real-Time Semantic Mapping Generation 

**Title (ZH)**: MapBERT：位级掩码建模以实现实时语义地图生成 

**Authors**: Yijie Deng, Shuaihang Yuan, Congcong Wen, Hao Huang, Anthony Tzes, Geeta Chandra Raju Bethala, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07350)  

**Abstract**: Spatial awareness is a critical capability for embodied agents, as it enables them to anticipate and reason about unobserved regions. The primary challenge arises from learning the distribution of indoor semantics, complicated by sparse, imbalanced object categories and diverse spatial scales. Existing methods struggle to robustly generate unobserved areas in real time and do not generalize well to new environments. To this end, we propose \textbf{MapBERT}, a novel framework designed to effectively model the distribution of unseen spaces. Motivated by the observation that the one-hot encoding of semantic maps aligns naturally with the binary structure of bit encoding, we, for the first time, leverage a lookup-free BitVAE to encode semantic maps into compact bitwise tokens. Building on this, a masked transformer is employed to infer missing regions and generate complete semantic maps from limited observations. To enhance object-centric reasoning, we propose an object-aware masking strategy that masks entire object categories concurrently and pairs them with learnable embeddings, capturing implicit relationships between object embeddings and spatial tokens. By learning these relationships, the model more effectively captures indoor semantic distributions crucial for practical robotic tasks. Experiments on Gibson benchmarks show that MapBERT achieves state-of-the-art semantic map generation, balancing computational efficiency with accurate reconstruction of unobserved regions. 

**Abstract (ZH)**: 基于空间意识的MapBERT：一种用于有效建模未见空间分布的新框架 

---
# Real-Time Execution of Action Chunking Flow Policies 

**Title (ZH)**: 实时执行行动切片区块流策略 

**Authors**: Kevin Black, Manuel Y. Galliker, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2506.07339)  

**Abstract**: Modern AI systems, especially those interacting with the physical world, increasingly require real-time performance. However, the high latency of state-of-the-art generalist models, including recent vision-language action models (VLAs), poses a significant challenge. While action chunking has enabled temporal consistency in high-frequency control tasks, it does not fully address the latency problem, leading to pauses or out-of-distribution jerky movements at chunk boundaries. This paper presents a novel inference-time algorithm that enables smooth asynchronous execution of action chunking policies. Our method, real-time chunking (RTC), is applicable to any diffusion- or flow-based VLA out of the box with no re-training. It generates the next action chunk while executing the current one, "freezing" actions guaranteed to execute and "inpainting" the rest. To test RTC, we introduce a new benchmark of 12 highly dynamic tasks in the Kinetix simulator, as well as evaluate 6 challenging real-world bimanual manipulation tasks. Results demonstrate that RTC is fast, performant, and uniquely robust to inference delay, significantly improving task throughput and enabling high success rates in precise tasks $\unicode{x2013}$ such as lighting a match $\unicode{x2013}$ even in the presence of significant latency. See this https URL for videos. 

**Abstract (ZH)**: 现代AI系统，尤其是那些与物理世界交互的系统，越来越需要实时性能。然而，最先进的通用模型的高度延迟，包括最近的视觉-语言动作模型（VLAs），提出了一个重大的挑战。尽管动作切分使高频率控制任务具有时间一致性，但它并未完全解决延迟问题，导致在切分边界处出现暂停或不自然的运动。本文提出了一种新颖的推理时算法，可以实现动作切分策略的平滑异步执行。我们的方法实时切分（RTC）适用于任何基于扩散或流动的VLA，无需重新训练即可直接应用。它在执行当前动作的同时生成下一个动作切片，“冻结”已确保执行的动作，并“修复”其余部分。为了测试RTC，我们引入了Kinetix模拟器中的12个高度动态任务的新基准，并评估了6个具有挑战性的实际双臂操作任务。结果表明，RTC不仅快速且性能优越，而且对推理延迟具有独特 robust性，显著提高了任务吞吐量，并即使在存在显著延迟的情况下也能在精确任务中实现高成功率，例如点火。请见此链接获取视频。 

---
# Model Analysis And Design Of Ellipse Based Segmented Varying Curved Foot For Biped Robot Walking 

**Title (ZH)**: 基于椭圆分段变曲率足的 biped 机器人行走模型分析与设计 

**Authors**: Boyang Chen, Xizhe Zang, Chao Song, Yue Zhang, Jie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07283)  

**Abstract**: This paper presents the modeling, design, and experimental validation of an Ellipse-based Segmented Varying Curvature (ESVC) foot for bipedal robots. Inspired by the segmented curvature rollover shape of human feet, the ESVC foot aims to enhance gait energy efficiency while maintaining analytical tractability for foot location based controller. First, we derive a complete analytical contact model for the ESVC foot by formulating spatial transformations of elliptical segments only using elementary functions. Then a nonlinear programming approach is engaged to determine optimal elliptical parameters of hind foot and fore foot based on a known mid-foot. An error compensation method is introduced to address approximation inaccuracies in rollover length calculation. The proposed ESVC foot is then integrated with a Hybrid Linear Inverted Pendulum model-based walking controller and validated through both simulation and physical experiments on the TT II biped robot. Experimental results across marking time, sagittal, and lateral walking tasks show that the ESVC foot consistently reduces energy consumption compared to line, and flat feet, with up to 18.52\% improvement in lateral walking. These findings demonstrate that the ESVC foot provides a practical and energy-efficient alternative for real-world bipedal locomotion. The proposed design methodology also lays a foundation for data-driven foot shape optimization in future research. 

**Abstract (ZH)**: 基于椭圆分段变曲率foot的设计与实验验证：提升双足机器人步态能量效率的新方法 

---
# Robotic Policy Learning via Human-assisted Action Preference Optimization 

**Title (ZH)**: 通过人类辅助动作偏好优化的机器人策略学习 

**Authors**: Wenke xia, Yichu Yang, Hongtao Wu, Xiao Ma, Tao Kong, Di Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07127)  

**Abstract**: Establishing a reliable and iteratively refined robotic system is essential for deploying real-world applications. While Vision-Language-Action (VLA) models are widely recognized as the foundation model for such robotic deployment, their dependence on expert demonstrations hinders the crucial capabilities of correction and learning from failures. To mitigate this limitation, we introduce a Human-assisted Action Preference Optimization method named HAPO, designed to correct deployment failures and foster effective adaptation through preference alignment for VLA models. This method begins with a human-robot collaboration framework for reliable failure correction and interaction trajectory collection through human intervention. These human-intervention trajectories are further employed within the action preference optimization process, facilitating VLA models to mitigate failure action occurrences while enhancing corrective action adaptation. Specifically, we propose an adaptive reweighting algorithm to address the issues of irreversible interactions and token probability mismatch when introducing preference optimization into VLA models, facilitating model learning from binary desirability signals derived from interactions. Through combining these modules, our human-assisted action preference optimization method ensures reliable deployment and effective learning from failure for VLA models. The experiments conducted in simulation and real-world scenarios prove superior generalization and robustness of our framework across a variety of manipulation tasks. 

**Abstract (ZH)**: 基于人类辅助动作偏好的优化方法：实现VLA模型的可靠部署与从失败中有效学习 

---
# CARoL: Context-aware Adaptation for Robot Learning 

**Title (ZH)**: CARoL: 基于上下文的机器人学习适应机制 

**Authors**: Zechen Hu, Tong Xu, Xuesu Xiao, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07006)  

**Abstract**: Using Reinforcement Learning (RL) to learn new robotic tasks from scratch is often inefficient. Leveraging prior knowledge has the potential to significantly enhance learning efficiency, which, however, raises two critical challenges: how to determine the relevancy of existing knowledge and how to adaptively integrate them into learning a new task. In this paper, we propose Context-aware Adaptation for Robot Learning (CARoL), a novel framework to efficiently learn a similar but distinct new task from prior knowledge. CARoL incorporates context awareness by analyzing state transitions in system dynamics to identify similarities between the new task and prior knowledge. It then utilizes these identified similarities to prioritize and adapt specific knowledge pieces for the new task. Additionally, CARoL has a broad applicability spanning policy-based, value-based, and actor-critic RL algorithms. We validate the efficiency and generalizability of CARoL on both simulated robotic platforms and physical ground vehicles. The simulations include CarRacing and LunarLander environments, where CARoL demonstrates faster convergence and higher rewards when learning policies for new tasks. In real-world experiments, we show that CARoL enables a ground vehicle to quickly and efficiently adapt policies learned in simulation to smoothly traverse real-world off-road terrain. 

**Abstract (ZH)**: 基于上下文的适应性机器人学习（CARoL）：从先验知识高效学习新任务 

---
# Hierarchical Intention Tracking with Switching Trees for Real-Time Adaptation to Dynamic Human Intentions during Collaboration 

**Title (ZH)**: 基于切换树的分层意图跟踪：实现协作中动态人类意图的实时适应 

**Authors**: Zhe Huang, Ye-Ji Mun, Fatemeh Cheraghi Pouria, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2506.07004)  

**Abstract**: During collaborative tasks, human behavior is guided by multiple levels of intentions that evolve over time, such as task sequence preferences and interaction strategies. To adapt to these changing preferences and promptly correct any inaccurate estimations, collaborative robots must accurately track these dynamic human intentions in real time. We propose a Hierarchical Intention Tracking (HIT) algorithm for collaborative robots to track dynamic and hierarchical human intentions effectively in real time. HIT represents human intentions as intention trees with arbitrary depth, and probabilistically tracks human intentions by Bayesian filtering, upward measurement propagation, and downward posterior propagation across all levels. We develop a HIT-based robotic system that dynamically switches between Interaction-Task and Verification-Task trees for a collaborative assembly task, allowing the robot to effectively coordinate human intentions at three levels: task-level (subtask goal locations), interaction-level (mode of engagement with the robot), and verification-level (confirming or correcting intention recognition). Our user study shows that our HIT-based collaborative robot system surpasses existing collaborative robot solutions by achieving a balance between efficiency, physical workload, and user comfort while ensuring safety and task completion. Post-experiment surveys further reveal that the HIT-based system enhances the user trust and minimizes interruptions to user's task flow through its effective understanding of human intentions across multiple levels. 

**Abstract (ZH)**: 基于层次意图跟踪的协作机器人系统 

---
# Multimodal Spatial Language Maps for Robot Navigation and Manipulation 

**Title (ZH)**: 多模态空间语言地图在机器人导航与操作中的应用 

**Authors**: Chenguang Huang, Oier Mees, Andy Zeng, Wolfram Burgard  

**Link**: [PDF](https://arxiv.org/pdf/2506.06862)  

**Abstract**: Grounding language to a navigating agent's observations can leverage pretrained multimodal foundation models to match perceptions to object or event descriptions. However, previous approaches remain disconnected from environment mapping, lack the spatial precision of geometric maps, or neglect additional modality information beyond vision. To address this, we propose multimodal spatial language maps as a spatial map representation that fuses pretrained multimodal features with a 3D reconstruction of the environment. We build these maps autonomously using standard exploration. We present two instances of our maps, which are visual-language maps (VLMaps) and their extension to audio-visual-language maps (AVLMaps) obtained by adding audio information. When combined with large language models (LLMs), VLMaps can (i) translate natural language commands into open-vocabulary spatial goals (e.g., "in between the sofa and TV") directly localized in the map, and (ii) be shared across different robot embodiments to generate tailored obstacle maps on demand. Building upon the capabilities above, AVLMaps extend VLMaps by introducing a unified 3D spatial representation integrating audio, visual, and language cues through the fusion of features from pretrained multimodal foundation models. This enables robots to ground multimodal goal queries (e.g., text, images, or audio snippets) to spatial locations for navigation. Additionally, the incorporation of diverse sensory inputs significantly enhances goal disambiguation in ambiguous environments. Experiments in simulation and real-world settings demonstrate that our multimodal spatial language maps enable zero-shot spatial and multimodal goal navigation and improve recall by 50% in ambiguous scenarios. These capabilities extend to mobile robots and tabletop manipulators, supporting navigation and interaction guided by visual, audio, and spatial cues. 

**Abstract (ZH)**: 将语言与导航代理的观察相结合可以利用预训练的多模态基础模型将感知与物体或事件描述匹配起来。然而，以前的方法仍然与环境建图脱离，缺乏几何地图的 spatial 精度，或者忽略了超越视觉的其他模态信息。为了解决这个问题，我们提出多模态空间语言地图作为一种融合预训练多模态特征与环境 3D 重建的空间地图表示。我们通过标准探索方式自动构建这些地图。我们展示了两种我们的地图实例，即视觉-语言地图（VLMaps）及其通过增加音频信息扩展为音频-视觉-语言地图（AVLMaps）。当与大型语言模型（LLMs）结合使用时，VLMaps 可以（i）直接将自然语言命令翻译成开放词汇的空间目标（例如，“在沙发和电视之间”），并将这些目标定位到地图上，并且可以（ii）在不同机器人载体之间共享以按需生成定制的障碍地图。基于上述能力，AVLMaps 通过引入综合音频、视觉和语言线索的统一 3D 空间表示来扩展 VLMaps，这通过预训练多模态基础模型的特征融合实现。这使机器人能够将多模态目标查询（例如，文本、图像或音频片段）定位到空间位置用于导航。此外，多样化感官输入的整合显著提高了在模棱两可环境中目标的消歧。在模拟和真实环境中的实验表明，我们的多模态空间语言地图能够实现零样本的空间和多模态目标导航，并在模棱两可场景中将召回率提高 50%。这些能力扩展到移动机器人和台式操作器，支持由视觉、音频和空间线索引导的导航和交互。 

---
# SpikePingpong: High-Frequency Spike Vision-based Robot Learning for Precise Striking in Table Tennis Game 

**Title (ZH)**: SpikePingpong：基于尖峰视觉的高频率击球机器人学习方法以实现乒乓球精确打击 

**Authors**: Hao Wang, Chengkai Hou, Xianglong Li, Yankai Fu, Chenxuan Li, Ning Chen, Gaole Dai, Jiaming Liu, Tiejun Huang, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06690)  

**Abstract**: Learning to control high-speed objects in the real world remains a challenging frontier in robotics. Table tennis serves as an ideal testbed for this problem, demanding both rapid interception of fast-moving balls and precise adjustment of their trajectories. This task presents two fundamental challenges: it requires a high-precision vision system capable of accurately predicting ball trajectories, and it necessitates intelligent strategic planning to ensure precise ball placement to target regions. The dynamic nature of table tennis, coupled with its real-time response requirements, makes it particularly well-suited for advancing robotic control capabilities in fast-paced, precision-critical domains. In this paper, we present SpikePingpong, a novel system that integrates spike-based vision with imitation learning for high-precision robotic table tennis. Our approach introduces two key attempts that directly address the aforementioned challenges: SONIC, a spike camera-based module that achieves millimeter-level precision in ball-racket contact prediction by compensating for real-world uncertainties such as air resistance and friction; and IMPACT, a strategic planning module that enables accurate ball placement to targeted table regions. The system harnesses a 20 kHz spike camera for high-temporal resolution ball tracking, combined with efficient neural network models for real-time trajectory correction and stroke planning. Experimental results demonstrate that SpikePingpong achieves a remarkable 91% success rate for 30 cm accuracy target area and 71% in the more challenging 20 cm accuracy task, surpassing previous state-of-the-art approaches by 38% and 37% respectively. These significant performance improvements enable the robust implementation of sophisticated tactical gameplay strategies, providing a new research perspective for robotic control in high-speed dynamic tasks. 

**Abstract (ZH)**: 在现实世界中控制高速物体仍然是机器人技术的一个挑战性前沿问题。乒乓球为这一问题提供了一个理想的测试平台，既要求快速拦截快速移动的球，又要求精确调整球的轨迹。该任务提出了两个基本挑战：它需要一个高精度的视觉系统来准确预测球的轨迹，并需要智能化的战略规划以确保将球精确放置到目标区域。由于乒乓球的动态特性和实时响应要求，它特别适合于推进机器人在快节奏、高精度关键领域的控制能力。在本文中，我们提出了一种名为SpikePingpong的新系统，该系统结合了基于尖峰的视觉与模仿学习，以实现高精度的机器人乒乓球。我们的方法引入了两个关键尝试，直接应对上述挑战：SONIC，一种基于尖峰摄像头的模块，通过补偿诸如空气阻力和摩擦等现实世界不确定性，实现了毫米级精度的球拍接触预测；和IMPACT，一种战略规划模块，能够实现将球精确放置到指定球台区域的准确性。该系统利用20 kHz的尖峰摄像头进行高时间分辨率的球跟踪，结合高效的神经网络模型进行实时轨迹校正和击球规划。实验结果表明，SpikePingpong在30 cm精度目标区域的击球成功率达到了91%，在更具挑战性的20 cm精度任务中为71%，分别超越了之前最先进的方法38%和37%。这些显著的性能提升使得复杂的战术游戏策略得以稳健实施，为在高速动态任务中的机器人控制提供了新的研究视角。 

---
# RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation 

**Title (ZH)**: RoboCerebra：大规模长期 horizon 机器人操控评估基准 

**Authors**: Songhao Han, Boxiang Qiu, Yue Liao, Siyuan Huang, Chen Gao, Shuicheng Yan, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06677)  

**Abstract**: Recent advances in vision-language models (VLMs) have enabled instruction-conditioned robotic systems with improved generalization. However, most existing work focuses on reactive System 1 policies, underutilizing VLMs' strengths in semantic reasoning and long-horizon planning. These System 2 capabilities-characterized by deliberative, goal-directed thinking-remain under explored due to the limited temporal scale and structural complexity of current benchmarks. To address this gap, we introduce RoboCerebra, a benchmark for evaluating high-level reasoning in long-horizon robotic manipulation. RoboCerebra includes: (1) a large-scale simulation dataset with extended task horizons and diverse subtask sequences in household environments; (2) a hierarchical framework combining a high-level VLM planner with a low-level vision-language-action (VLA) controller; and (3) an evaluation protocol targeting planning, reflection, and memory through structured System 1-System 2 interaction. The dataset is constructed via a top-down pipeline, where GPT generates task instructions and decomposes them into subtask sequences. Human operators execute the subtasks in simulation, yielding high-quality trajectories with dynamic object variations. Compared to prior benchmarks, RoboCerebra features significantly longer action sequences and denser annotations. We further benchmark state-of-the-art VLMs as System 2 modules and analyze their performance across key cognitive dimensions, advancing the development of more capable and generalizable robotic planners. 

**Abstract (ZH)**: 近期在视觉-语言模型（VLMs）方面的进展使得基于指令的机器人系统具备了更强的泛化能力。然而，大多数现有工作集中在反应性System 1策略上，未能充分利用VLMs在语义推理和长远规划方面的优势。这些体现为深思熟虑、目标导向思考的System 2能力由于当前基准在时间尺度和结构复杂性上的限制而未得到充分探索。为解决这一问题，我们引入了RoboCerebra，一个评估长时机器人操作高级推理能力的标准。RoboCerebra包括：（1）一个大规模模拟数据集，包含扩展的任务时间轴和多样的子任务序列在家用环境中；（2）一个层次框架，结合高层VLM规划器和低层视觉-语言-行动（VLA）控制器；（3）一种评估计划、反思和记忆的评价协议，通过结构化的System 1-System 2交互实现。该数据集通过自上而下的管道构建，其中GPT生成任务指令并分解为子任务序列。人类操作员在模拟中执行子任务，生成具有动态物体变化的高质量轨迹。与以前的基准相比，RoboCerebra的特点是显著较长的动作序列和更密集的标注。我们进一步以System 2模块形式评估最先进的VLMs，并在关键认知维度上分析其性能，促进更强大和通用的机器人规划器的发展。 

---
# Generalized Trajectory Scoring for End-to-end Multimodal Planning 

**Title (ZH)**: 端到端多模态规划的广义轨迹评分 

**Authors**: Zhenxin Li, Wenhao Yao, Zi Wang, Xinglong Sun, Joshua Chen, Nadine Chang, Maying Shen, Zuxuan Wu, Shiyi Lan, Jose M. Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2506.06664)  

**Abstract**: End-to-end multi-modal planning is a promising paradigm in autonomous driving, enabling decision-making with diverse trajectory candidates. A key component is a robust trajectory scorer capable of selecting the optimal trajectory from these candidates. While recent trajectory scorers focus on scoring either large sets of static trajectories or small sets of dynamically generated ones, both approaches face significant limitations in generalization. Static vocabularies provide effective coarse discretization but struggle to make fine-grained adaptation, while dynamic proposals offer detailed precision but fail to capture broader trajectory distributions. To overcome these challenges, we propose GTRS (Generalized Trajectory Scoring), a unified framework for end-to-end multi-modal planning that combines coarse and fine-grained trajectory evaluation. GTRS consists of three complementary innovations: (1) a diffusion-based trajectory generator that produces diverse fine-grained proposals; (2) a vocabulary generalization technique that trains a scorer on super-dense trajectory sets with dropout regularization, enabling its robust inference on smaller subsets; and (3) a sensor augmentation strategy that enhances out-of-domain generalization while incorporating refinement training for critical trajectory discrimination. As the winning solution of the Navsim v2 Challenge, GTRS demonstrates superior performance even with sub-optimal sensor inputs, approaching privileged methods that rely on ground-truth perception. Code will be available at this https URL. 

**Abstract (ZH)**: 端到端多模态规划是自动驾驶的一个有前途的范式，能够实现具有多样化轨迹候选者的决策制定。关键组件是一个 robust 轨迹评分器，能够从中选择最优轨迹。虽然最近的轨迹评分器专注于评分大量静态轨迹或少量动态生成的轨迹，但两种方法都面临着泛化能力的显著限制。静态词汇表提供有效的粗粒度离散化，但在细粒度适应方面存在困难，而动态提案虽然提供了详细精度，但无法捕捉更广泛的轨迹分布。为克服这些挑战，我们提出了 GTRS（通用轨迹评分），这是一种结合粗粒度和细粒度轨迹评估的端到端多模态规划统一框架。GTRS 包含三个互补创新：（1）基于扩散的轨迹生成器，产生多样化的细粒度提案；（2）词汇表泛化技术，通过 Dropout 正则化在超密轨迹集中训练评分器，使其能够在较小的子集上进行稳健推理；以及（3）传感器增强策略，在增强域外泛化的同时结合关键轨迹区分的细化训练。作为 Navsim v2 挑战赛的获胜解决方案，GTRS 即使在传感器输入不理想的条件下也能表现出优越的性能，接近依赖地面真实感知的特权方法。代码将在此链接获得。 

---
# Self-Adapting Improvement Loops for Robotic Learning 

**Title (ZH)**: 自适应改进循环在机器人学习中的应用 

**Authors**: Calvin Luo, Zilai Zeng, Mingxi Jia, Yilun Du, Chen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.06658)  

**Abstract**: Video generative models trained on expert demonstrations have been utilized as performant text-conditioned visual planners for solving robotic tasks. However, generalization to unseen tasks remains a challenge. Whereas improved generalization may be facilitated by leveraging learned prior knowledge from additional pre-collected offline data sources, such as web-scale video datasets, in the era of experience we aim to design agents that can continuously improve in an online manner from self-collected behaviors. In this work we thus propose the Self-Adapting Improvement Loop (SAIL), where an in-domain video model iteratively updates itself on self-produced trajectories, collected through adaptation with an internet-scale pretrained video model, and steadily improves its performance for a specified task of interest. We apply SAIL to a diverse suite of MetaWorld tasks, as well as two manipulation tasks on a real robot arm, and find that performance improvements continuously emerge over multiple iterations for novel tasks initially unseen during original in-domain video model training. Furthermore, we discover that SAIL is surprisingly robust regarding if and how the self-collected experience is filtered, and the quality of the initial in-domain demonstrations. Through adaptation with summarized internet-scale data, and learning through online experience, we thus demonstrate a way to iteratively bootstrap a high-performance video model for solving novel robotic tasks through self-improvement. 

**Abstract (ZH)**: 自我适应改进循环（SAIL）：通过自我收集的行为持续改进视觉规划模型以解决新型机器人任务 

---
# Active Test-time Vision-Language Navigation 

**Title (ZH)**: 主动测试时视觉-语言导航 

**Authors**: Heeju Ko, Sungjune Kim, Gyeongrok Oh, Jeongyoon Yoon, Honglak Lee, Sujin Jang, Seungryong Kim, Sangpil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.06630)  

**Abstract**: Vision-Language Navigation (VLN) policies trained on offline datasets often exhibit degraded task performance when deployed in unfamiliar navigation environments at test time, where agents are typically evaluated without access to external interaction or feedback. Entropy minimization has emerged as a practical solution for reducing prediction uncertainty at test time; however, it can suffer from accumulated errors, as agents may become overconfident in incorrect actions without sufficient contextual grounding. To tackle these challenges, we introduce ATENA (Active TEst-time Navigation Agent), a test-time active learning framework that enables a practical human-robot interaction via episodic feedback on uncertain navigation outcomes. In particular, ATENA learns to increase certainty in successful episodes and decrease it in failed ones, improving uncertainty calibration. Here, we propose mixture entropy optimization, where entropy is obtained from a combination of the action and pseudo-expert distributions-a hypothetical action distribution assuming the agent's selected action to be optimal-controlling both prediction confidence and action preference. In addition, we propose a self-active learning strategy that enables an agent to evaluate its navigation outcomes based on confident predictions. As a result, the agent stays actively engaged throughout all iterations, leading to well-grounded and adaptive decision-making. Extensive evaluations on challenging VLN benchmarks-REVERIE, R2R, and R2R-CE-demonstrate that ATENA successfully overcomes distributional shifts at test time, outperforming the compared baseline methods across various settings. 

**Abstract (ZH)**: ATENA：测试时主动学习导航代理 

---
# Underwater Multi-Robot Simulation and Motion Planning in Angler 

**Title (ZH)**: Angler中的水下多机器人仿真与运动规划 

**Authors**: Akshaya Agrawal, Evan Palmer, Zachary Kingston, Geoffrey A. Hollinger  

**Link**: [PDF](https://arxiv.org/pdf/2506.06612)  

**Abstract**: Deploying multi-robot systems in underwater environments is expensive and lengthy; testing algorithms and software in simulation improves development by decoupling software and hardware. However, this requires a simulation framework that closely resembles the real-world. Angler is an open-source framework that simulates low-level communication protocols for an onboard autopilot, such as ArduSub, providing a framework that is close to reality, but unfortunately lacking support for simulating multiple robots. We present an extension to Angler that supports multi-robot simulation and motion planning. Our extension has a modular architecture that creates non-conflicting communication channels between Gazebo, ArduSub Software-in-the-Loop (SITL), and MAVROS to operate multiple robots simultaneously in the same environment. Our multi-robot motion planning module interfaces with cascaded controllers via a JointTrajectory controller in ROS~2. We also provide an integration with the Open Motion Planning Library (OMPL), a collision avoidance module, and tools for procedural environment generation. Our work enables the development and benchmarking of underwater multi-robot motion planning in dynamic environments. 

**Abstract (ZH)**: 多机器人系统在水下环境中的部署成本高且耗时；通过在仿真中测试算法和软件可以解耦软件和硬件从而提高开发效率。然而，这需要一个与真实世界高度相似的仿真框架。Angler是一个开源框架，用于模拟岸上自主导航系统的低层通信协议，如ArduSub，提供了一个接近现实的框架，但不幸的是缺乏多个机器人仿真的支持。我们提出了一种扩展Angler，以支持多机器人仿真和运动规划。我们的扩展具有模块化架构，通过ROS~2中的JointTrajectory控制器接口与嵌套控制器进行交互，在Gazebo、ArduSub Software-in-the-Loop (SITL)和MAVROS之间创建非冲突的通信通道，以便在同一环境中同时操作多个机器人。我们的多机器人运动规划模块与OMPL、碰撞避免模块以及用于生成程序化环境的工具进行集成。我们的工作使在动态环境中开发和基准测试水下多机器人运动规划成为可能。 

---
# Enhancing Robot Safety via MLLM-Based Semantic Interpretation of Failure Data 

**Title (ZH)**: 基于MLLM的故障数据语义解释增强机器人安全性 

**Authors**: Aryaman Gupta, Yusuf Umut Ciftci, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2506.06570)  

**Abstract**: As robotic systems become increasingly integrated into real-world environments, ranging from autonomous vehicles to household assistants, they inevitably encounter diverse and unstructured scenarios that lead to failures. While such failures pose safety and reliability challenges, they also provide rich perceptual data for improving future performance. However, manually analyzing large-scale failure datasets is impractical. In this work, we present a method for automatically organizing large-scale robotic failure data into semantically meaningful clusters, enabling scalable learning from failure without human supervision. Our approach leverages the reasoning capabilities of Multimodal Large Language Models (MLLMs), trained on internet-scale data, to infer high-level failure causes from raw perceptual trajectories and discover interpretable structure within uncurated failure logs. These semantic clusters reveal latent patterns and hypothesized causes of failure, enabling scalable learning from experience. We demonstrate that the discovered failure modes can guide targeted data collection for policy refinement, accelerating iterative improvement in agent policies and overall safety. Additionally, we show that these semantic clusters can be employed for online failure detection, offering a lightweight yet powerful safeguard for real-time adaptation. We demonstrate that this framework enhances robot learning and robustness by transforming real-world failures into actionable and interpretable signals for adaptation. 

**Abstract (ZH)**: 随着机器人系统越来越广泛地集成到现实环境中，从自动驾驶车辆到家庭助手，它们不可避免地会遇到各种未结构化的场景，导致失败。虽然这些失败带来了安全性和可靠性的挑战，但也提供了丰富的感知数据，有助于改善未来的性能。然而，手动分析大规模失败数据集是不实际的。本文提出了一种方法，可以在无需人工监督的情况下，自动将大规模机器人失败数据组织成语义上有意义的簇，从而使能够在失败中进行可扩展的学习。我们的方法利用了预训练于互联网规模数据的多模态大型语言模型（MLLMs）的推理能力，从原始感知轨迹中推断出高层级的失败原因，并在未整理的失败日志中发现可解释的结构。这些语义簇揭示了隐藏的模式和失败的假设原因，有助于在失败中进行可扩展的学习。我们证明，发现的失败模式可以引导针对策略细化的目标数据采集，加速智能体策略的迭代改进和整体安全性。此外，我们展示了这些语义簇可以用于在线故障检测，提供了轻量级但强大的实时适应保护措施。我们证明，该框架通过将现实世界的故障转化为可操作和可解释的适应信号，增强了机器人的学习和鲁棒性。 

---
# Towards Terrain-Aware Task-Driven 3D Scene Graph Generation in Outdoor Environments 

**Title (ZH)**: 面向地形感知的任务驱动户外环境3D场景图生成 

**Authors**: Chad R Samuelson, Timothy W McLain, Joshua G Mangelson  

**Link**: [PDF](https://arxiv.org/pdf/2506.06562)  

**Abstract**: High-level autonomous operations depend on a robot's ability to construct a sufficiently expressive model of its environment. Traditional three-dimensional (3D) scene representations, such as point clouds and occupancy grids, provide detailed geometric information but lack the structured, semantic organization needed for high-level reasoning. 3D scene graphs (3DSGs) address this limitation by integrating geometric, topological, and semantic relationships into a multi-level graph-based representation. By capturing hierarchical abstractions of objects and spatial layouts, 3DSGs enable robots to reason about environments in a structured manner, improving context-aware decision-making and adaptive planning. Although most recent work has focused on indoor 3DSGs, this paper investigates their construction and utility in outdoor environments. We present a method for generating a task-agnostic metric-semantic point cloud for large outdoor settings and propose modifications to existing indoor 3DSG generation techniques for outdoor applicability. Our preliminary qualitative results demonstrate the feasibility of outdoor 3DSGs and highlight their potential for future deployment in real-world field robotic applications. 

**Abstract (ZH)**: 高阶自主操作依赖于机器人构建其环境的充分表达模型的能力。传统的三维（3D）场景表示，如点云和占用网格，提供了详细的几何信息，但缺乏进行高阶推理所需的结构化、语义组织。3D场景图（3DSGs）通过将几何、拓扑和语义关系整合到多级图表示中，解决了这一限制。通过捕捉对象和空间布局的层级抽象，3DSGs使机器人能够以结构化的方式推理环境，从而改善上下文相关的决策制定和适应性规划。尽管最近大多数工作都集中在室内3D场景图上，本文则探讨了它们在户外环境中的构造和实用性。我们提出了一种方法，用于生成适用于大型户外环境的无任务特定的度量语义点云，并对现有的室内3D场景图生成技术进行了修改，以使其适用于户外应用。初步的定性结果表明，户外3D场景图的可行性，并强调了其在未来在实际野外机器人应用中的潜在作用。 

---
# MapleGrasp: Mask-guided Feature Pooling for Language-driven Efficient Robotic Grasping 

**Title (ZH)**: MapleGrasp: 基于掩码引导的特征池化用于语言驱动的高效机器人抓取 

**Authors**: Vineet Bhat, Naman Patel, Prashanth Krishnamurthy, Ramesh Karri, Farshad Khorrami  

**Link**: [PDF](https://arxiv.org/pdf/2506.06535)  

**Abstract**: Robotic manipulation of unseen objects via natural language commands remains challenging. Language driven robotic grasping (LDRG) predicts stable grasp poses from natural language queries and RGB-D images. Here we introduce Mask-guided feature pooling, a lightweight enhancement to existing LDRG methods. Our approach employs a two-stage training strategy: first, a vision-language model generates feature maps from CLIP-fused embeddings, which are upsampled and weighted by text embeddings to produce segmentation masks. Next, the decoder generates separate feature maps for grasp prediction, pooling only token features within these masked regions to efficiently predict grasp poses. This targeted pooling approach reduces computational complexity, accelerating both training and inference. Incorporating mask pooling results in a 12% improvement over prior approaches on the OCID-VLG benchmark. Furthermore, we introduce RefGraspNet, an open-source dataset eight times larger than existing alternatives, significantly enhancing model generalization for open-vocabulary grasping. By extending 2D grasp predictions to 3D via depth mapping and inverse kinematics, our modular method achieves performance comparable to recent Vision-Language-Action (VLA) models on the LIBERO simulation benchmark, with improved generalization across different task suites. Real-world experiments on a 7 DoF Franka robotic arm demonstrate a 57% success rate with unseen objects, surpassing competitive baselines by 7%. Code will be released post publication. 

**Abstract (ZH)**: 通过自然语言指令操纵未见物体的机器人操作仍具有挑战性。基于语言的机器人抓取（LDRG）方法从自然语言查询和RGB-D图像中预测稳定的手 grasp 姿态。我们介绍了掩码引导特征池化，这是一种对现有 LDRG 方法的轻量级增强。我们的方法采用两阶段训练策略：首先，视觉-语言模型从 CLIP 融合嵌入中生成特征图，并通过文本嵌入上采样和加权生成分割掩码。然后，解码器为抓取预测生成单独的特征图，仅在这些掩码区域内池化标记特征，从而高效地预测抓取姿态。这种目标导向的池化方法减少了计算复杂性，加速了训练和推理过程。引入掩码池化后，在 OCID-VLG 基准上的性能提高了 12%。此外，我们引入了 RefGraspNet 数据集，其大小是现有替代数据集的八倍，显著增强了开放词汇抓取的模型泛化能力。通过深度映射和逆运动学将 2D 抓取预测扩展到 3D，我们的模块化方法在 LIBERO 模拟基准上的性能与最近的视觉-语言-动作（VLA）模型相当，并且在不同任务套件上的泛化能力得到提高。在具有 7 自由度的 Franka 机器人手臂上的真实世界实验中，对于未见物体的成功率为 57%，超过了竞争性基线 7%。代码将在发表后发布。 

---
# Active Illumination Control in Low-Light Environments using NightHawk 

**Title (ZH)**: 低光环境下使用NightHawk的主动照明控制 

**Authors**: Yash Turkar, Youngjin Kim, Karthik Dantu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06394)  

**Abstract**: Subterranean environments such as culverts present significant challenges to robot vision due to dim lighting and lack of distinctive features. Although onboard illumination can help, it introduces issues such as specular reflections, overexposure, and increased power consumption. We propose NightHawk, a framework that combines active illumination with exposure control to optimize image quality in these settings. NightHawk formulates an online Bayesian optimization problem to determine the best light intensity and exposure-time for a given scene. We propose a novel feature detector-based metric to quantify image utility and use it as the cost function for the optimizer. We built NightHawk as an event-triggered recursive optimization pipeline and deployed it on a legged robot navigating a culvert beneath the Erie Canal. Results from field experiments demonstrate improvements in feature detection and matching by 47-197% enabling more reliable visual estimation in challenging lighting conditions. 

**Abstract (ZH)**: 地下环境如涵洞对机器人视觉构成了显著挑战，由于光线昏暗和缺乏 distinctive 特征。尽管可以采用机载照明，但这种方法会导致镜面反射、过度曝光和增加能耗等问题。我们提出 NightHawk，一种结合主动照明与曝光控制的框架，以优化这些环境中的图像质量。NightHawk 构建了一个在线贝叶斯优化问题，以确定给定场景的最佳光照强度和曝光时间。我们提出了一种基于特征检测的新颖度量标准来量化图像的实用性，并将其用作优化器的成本函数。我们构建了 NightHawk 作为一个事件触发的递归优化管道，并将其部署在伊利运河下方的腿式机器人上。实地实验结果表明，特征检测和匹配性能提高了 47-197%，从而在恶劣光照条件下实现了更可靠的视觉估计。 

---
# Deep Equivariant Multi-Agent Control Barrier Functions 

**Title (ZH)**: 深度同变多代理控制 barrier 函数 

**Authors**: Nikolaos Bousias, Lars Lindemann, George Pappas  

**Link**: [PDF](https://arxiv.org/pdf/2506.07755)  

**Abstract**: With multi-agent systems increasingly deployed autonomously at scale in complex environments, ensuring safety of the data-driven policies is critical. Control Barrier Functions have emerged as an effective tool for enforcing safety constraints, yet existing learning-based methods often lack in scalability, generalization and sampling efficiency as they overlook inherent geometric structures of the system. To address this gap, we introduce symmetries-infused distributed Control Barrier Functions, enforcing the satisfaction of intrinsic symmetries on learnable graph-based safety certificates. We theoretically motivate the need for equivariant parametrization of CBFs and policies, and propose a simple, yet efficient and adaptable methodology for constructing such equivariant group-modular networks via the compatible group actions. This approach encodes safety constraints in a distributed data-efficient manner, enabling zero-shot generalization to larger and denser swarms. Through extensive simulations on multi-robot navigation tasks, we demonstrate that our method outperforms state-of-the-art baselines in terms of safety, scalability, and task success rates, highlighting the importance of embedding symmetries in safe distributed neural policies. 

**Abstract (ZH)**: 多智能体系统在复杂环境中大规模自治部署时，确保数据驱动策略的安全性至关重要。为此，我们引入了融合对称性的分布式控制屏障函数，通过对可学习的图基安全证书施加内在对称性来确保其满足条件。我们从理论上阐述了对称性在控制屏障函数和策略参数化中的必要性，并提出了一种简单但高效且可适应的方法，通过兼容的群作用构建此类对称性群模网络。该方法以分布式数据高效的方式编码安全约束，使算法能够零样本泛化到更大、更密集的群体中。通过在多机器人导航任务中的广泛仿真实验，我们证明了该方法在安全性、可扩展性和任务成功率方面优于现有先进基线，强调了在安全分布式神经策略中嵌入对称性的的重要性。 

---
# Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning 

**Title (ZH)**: 基于图辅助缝合的离线分层强化学习 

**Authors**: Seungho Baek, Taegeon Park, Jongchan Park, Seungjun Oh, Yusung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07744)  

**Abstract**: Existing offline hierarchical reinforcement learning methods rely on high-level policy learning to generate subgoal sequences. However, their efficiency degrades as task horizons increase, and they lack effective strategies for stitching useful state transitions across different trajectories. We propose Graph-Assisted Stitching (GAS), a novel framework that formulates subgoal selection as a graph search problem rather than learning an explicit high-level policy. By embedding states into a Temporal Distance Representation (TDR) space, GAS clusters semantically similar states from different trajectories into unified graph nodes, enabling efficient transition stitching. A shortest-path algorithm is then applied to select subgoal sequences within the graph, while a low-level policy learns to reach the subgoals. To improve graph quality, we introduce the Temporal Efficiency (TE) metric, which filters out noisy or inefficient transition states, significantly enhancing task performance. GAS outperforms prior offline HRL methods across locomotion, navigation, and manipulation tasks. Notably, in the most stitching-critical task, it achieves a score of 88.3, dramatically surpassing the previous state-of-the-art score of 1.0. Our source code is available at: this https URL. 

**Abstract (ZH)**: 基于图辅助拼接的离线层次 reinforcement 学习方法 

---
# Multi-Step Guided Diffusion for Image Restoration on Edge Devices: Toward Lightweight Perception in Embodied AI 

**Title (ZH)**: 基于边缘设备的多步引导扩散影像恢复：朝向具身AI的轻量级感知 

**Authors**: Aditya Chakravarty  

**Link**: [PDF](https://arxiv.org/pdf/2506.07286)  

**Abstract**: Diffusion models have shown remarkable flexibility for solving inverse problems without task-specific retraining. However, existing approaches such as Manifold Preserving Guided Diffusion (MPGD) apply only a single gradient update per denoising step, limiting restoration fidelity and robustness, especially in embedded or out-of-distribution settings. In this work, we introduce a multistep optimization strategy within each denoising timestep, significantly enhancing image quality, perceptual accuracy, and generalization. Our experiments on super-resolution and Gaussian deblurring demonstrate that increasing the number of gradient updates per step improves LPIPS and PSNR with minimal latency overhead. Notably, we validate this approach on a Jetson Orin Nano using degraded ImageNet and a UAV dataset, showing that MPGD, originally trained on face datasets, generalizes effectively to natural and aerial scenes. Our findings highlight MPGD's potential as a lightweight, plug-and-play restoration module for real-time visual perception in embodied AI agents such as drones and mobile robots. 

**Abstract (ZH)**: 多步优化策略在去噪每个时间步的应用：显著提高图像质量、感知准确性和通用性 

---
# Safety-Aware Reinforcement Learning for Control via Risk-Sensitive Action-Value Iteration and Quantile Regression 

**Title (ZH)**: 基于风险敏感动作价值迭代和分位数回归的 Awareness 安全强化学习控制 

**Authors**: Clinton Enwerem, Aniruddh G. Puranic, John S. Baras, Calin Belta  

**Link**: [PDF](https://arxiv.org/pdf/2506.06954)  

**Abstract**: Mainstream approximate action-value iteration reinforcement learning (RL) algorithms suffer from overestimation bias, leading to suboptimal policies in high-variance stochastic environments. Quantile-based action-value iteration methods reduce this bias by learning a distribution of the expected cost-to-go using quantile regression. However, ensuring that the learned policy satisfies safety constraints remains a challenge when these constraints are not explicitly integrated into the RL framework. Existing methods often require complex neural architectures or manual tradeoffs due to combined cost functions. To address this, we propose a risk-regularized quantile-based algorithm integrating Conditional Value-at-Risk (CVaR) to enforce safety without complex architectures. We also provide theoretical guarantees on the contraction properties of the risk-sensitive distributional Bellman operator in Wasserstein space, ensuring convergence to a unique cost distribution. Simulations of a mobile robot in a dynamic reach-avoid task show that our approach leads to more goal successes, fewer collisions, and better safety-performance trade-offs compared to risk-neutral methods. 

**Abstract (ZH)**: 基于分位数的风险正则化行动价值迭代强化学习算法：通过条件值风险（CVaR）确保安全而不依赖复杂架构 

---
# Reading in the Dark with Foveated Event Vision 

**Title (ZH)**: 在暗光环境下使用聚焦事件视觉阅读 

**Authors**: Carl Brander, Giovanni Cioffi, Nico Messikommer, Davide Scaramuzza  

**Link**: [PDF](https://arxiv.org/pdf/2506.06918)  

**Abstract**: Current smart glasses equipped with RGB cameras struggle to perceive the environment in low-light and high-speed motion scenarios due to motion blur and the limited dynamic range of frame cameras. Additionally, capturing dense images with a frame camera requires large bandwidth and power consumption, consequently draining the battery faster. These challenges are especially relevant for developing algorithms that can read text from images. In this work, we propose a novel event-based Optical Character Recognition (OCR) approach for smart glasses. By using the eye gaze of the user, we foveate the event stream to significantly reduce bandwidth by around 98% while exploiting the benefits of event cameras in high-dynamic and fast scenes. Our proposed method performs deep binary reconstruction trained on synthetic data and leverages multimodal LLMs for OCR, outperforming traditional OCR solutions. Our results demonstrate the ability to read text in low light environments where RGB cameras struggle while using up to 2400 times less bandwidth than a wearable RGB camera. 

**Abstract (ZH)**: 基于事件的双眼佩戴式光学字符识别方法 

---
# SAFEFLOW: A Principled Protocol for Trustworthy and Transactional Autonomous Agent Systems 

**Title (ZH)**: SAFEFLOW：可信赖且事务性的自主代理系统原理协议 

**Authors**: Peiran Li, Xinkai Zou, Zhuohang Wu, Ruifeng Li, Shuo Xing, Hanwen Zheng, Zhikai Hu, Yuping Wang, Haoxi Li, Qin Yuan, Yingmo Zhang, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07564)  

**Abstract**: Recent advances in large language models (LLMs) and vision-language models (VLMs) have enabled powerful autonomous agents capable of complex reasoning and multi-modal tool use. Despite their growing capabilities, today's agent frameworks remain fragile, lacking principled mechanisms for secure information flow, reliability, and multi-agent coordination. In this work, we introduce SAFEFLOW, a new protocol-level framework for building trustworthy LLM/VLM-based agents. SAFEFLOW enforces fine-grained information flow control (IFC), precisely tracking provenance, integrity, and confidentiality of all the data exchanged between agents, tools, users, and environments. By constraining LLM reasoning to respect these security labels, SAFEFLOW prevents untrusted or adversarial inputs from contaminating high-integrity decisions. To ensure robustness in concurrent multi-agent settings, SAFEFLOW introduces transactional execution, conflict resolution, and secure scheduling over shared state, preserving global consistency across agents. We further introduce mechanisms, including write-ahead logging, rollback, and secure caches, that further enhance resilience against runtime errors and policy violations. To validate the performances, we built SAFEFLOWBENCH, a comprehensive benchmark suite designed to evaluate agent reliability under adversarial, noisy, and concurrent operational conditions. Extensive experiments demonstrate that agents built with SAFEFLOW maintain impressive task performance and security guarantees even in hostile environments, substantially outperforming state-of-the-art. Together, SAFEFLOW and SAFEFLOWBENCH lay the groundwork for principled, robust, and secure agent ecosystems, advancing the frontier of reliable autonomy. 

**Abstract (ZH)**: 最近大型语言模型（LLMs）和视觉-语言模型（VLMs）的发展使得强大的自主代理能够进行复杂的推理和多模态工具使用。尽管这些代理的能力在不断增强，但当前的代理框架仍然脆弱，缺乏通过安全信息流、可靠性和多代理协调的原则性机制。在本工作中，我们引入了SAFEFLOW，这是一种新的协议级框架，用于构建可信赖的LLM/VLM基自主代理。SAFEFLOW实施了细粒度的信息流控制（IFC），精确追踪代理、工具、用户和环境之间交换的所有数据的来源、完整性和保密性。通过限制LLM推理尊重这些安全标签，SAFEFLOW防止未信任或恶意输入污染高完整性决策。为确保并发多代理设置中的鲁棒性，SAFEFLOW引入了事务执行、冲突解决和共享状态上的安全调度，从而在代理之间保持全局一致性。我们还引入了包括预先写日志、回滚和安全缓存在内的机制，进一步增强对运行时错误和策略违规的抗御能力。为了验证性能，我们构建了SAFEFLOWBENCH，这是一个全面的基准套件，旨在评估代理在对抗性、嘈杂和并发操作条件下的可靠性。广泛的实验表明，使用SAFEFLOW构建的代理即使在敌对环境中也能保持出色的任务性能和安全保证，并显著优于现有最佳方案。SAFEFLOW和SAFEFLOWBENCH为原则性、稳健和安全的代理生态系统奠定了基础，推动可靠自主性的前沿。 

---
# Efficient Generation of Diverse Cooperative Agents with World Models 

**Title (ZH)**: 高效的生成多样化协同代理模型 

**Authors**: Yi Loo, Akshunn Trivedi, Malika Meghjani  

**Link**: [PDF](https://arxiv.org/pdf/2506.07450)  

**Abstract**: A major bottleneck in the training process for Zero-Shot Coordination (ZSC) agents is the generation of partner agents that are diverse in collaborative conventions. Current Cross-play Minimization (XPM) methods for population generation can be very computationally expensive and sample inefficient as the training objective requires sampling multiple types of trajectories. Each partner agent in the population is also trained from scratch, despite all of the partners in the population learning policies of the same coordination task. In this work, we propose that simulated trajectories from the dynamics model of an environment can drastically speed up the training process for XPM methods. We introduce XPM-WM, a framework for generating simulated trajectories for XPM via a learned World Model (WM). We show XPM with simulated trajectories removes the need to sample multiple trajectories. In addition, we show our proposed method can effectively generate partners with diverse conventions that match the performance of previous methods in terms of SP population training reward as well as training partners for ZSC agents. Our method is thus, significantly more sample efficient and scalable to a larger number of partners. 

**Abstract (ZH)**: Zero-Shot Coordination代理训练过程中的主要瓶颈是在协作惯例多样性方面生成伙伴代理。当前用于群体生成的跨游戏最小化(XPM)方法可能非常计算成本高且采样效率低，因为训练目标需要采样多种类型的轨迹。尽管环境中的所有伙伴代理都在同一协调任务中学习策略，但群体中的每个伙伴代理都从头开始训练。在此工作中，我们提出使用环境动力学模型生成的模拟轨迹可以大幅加快XPM方法的训练过程。我们提出了XPM-WM框架，通过学习的世界模型（WM）生成XPM的模拟轨迹。我们展示了使用模拟轨迹的XPM消除了需要采样多种轨迹的需求。此外，我们展示了我们提出的方法能够有效地生成具有多样性协作惯例的伙伴，这些伙伴在SP群体训练奖励以及零-shot协调代理训练伙伴方面与先前方法具有竞争力。因此，我们的方法在采样效率方面显著提高，并且适用于更大数量的伙伴。 

---
# LLM-Enhanced Rapid-Reflex Async-Reflect Embodied Agent for Real-Time Decision-Making in Dynamically Changing Environments 

**Title (ZH)**: 增强大型语言模型的快速反应异步反射体现代理在动态变化环境中的实时决策-making 

**Authors**: Yangqing Zheng, Shunqi Mao, Dingxin Zhang, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2506.07223)  

**Abstract**: In the realm of embodied intelligence, the evolution of large language models (LLMs) has markedly enhanced agent decision making. Consequently, researchers have begun exploring agent performance in dynamically changing high-risk scenarios, i.e., fire, flood, and wind scenarios in the HAZARD benchmark. Under these extreme conditions, the delay in decision making emerges as a crucial yet insufficiently studied issue. We propose a Time Conversion Mechanism (TCM) that translates inference delays in decision-making into equivalent simulation frames, thus aligning cognitive and physical costs under a single FPS-based metric. By extending HAZARD with Respond Latency (RL) and Latency-to-Action Ratio (LAR), we deliver a fully latency-aware evaluation protocol. Moreover, we present the Rapid-Reflex Async-Reflect Agent (RRARA), which couples a lightweight LLM-guided feedback module with a rule-based agent to enable immediate reactive behaviors and asynchronous reflective refinements in situ. Experiments on HAZARD show that RRARA substantially outperforms existing baselines in latency-sensitive scenarios. 

**Abstract (ZH)**: 在本体智能领域，大型语言模型（LLMs）的发展显著增强了代理决策能力。因此，研究人员开始探索代理在动态变化的高风险场景中的表现，例如HAZARD基准中的火灾、洪水和风灾场景。在这些极端条件下，决策延迟成为了一个重要但研究不足的问题。我们提出了一种时间转换机制（TCM），将决策推理延迟转换为等效的模拟帧，从而在基于FPS的单一度量下对认知和物理成本进行对齐。通过将HAZARD扩展为响应延迟（RL）和动作延迟比（LAR），我们提供了一个完全关注延迟的评估协议。此外，我们提出了快速反应异步反思代理（RRARA），该代理结合了一个轻量级的LLM指导反馈模块和基于规则的代理，以实现即时的反应行为和现场异步的反思性改进。在HAZARD上的实验表明，RRARA在延迟敏感场景中显著优于现有基线。 

---
# Incorporating Failure of Machine Learning in Dynamic Probabilistic Safety Assurance 

**Title (ZH)**: 在动态概率安全保证中纳入机器学习失效因素 

**Authors**: Razieh Arshadizadeh, Mahmoud Asgari, Zeinab Khosravi, Yiannis Papadopoulos, Koorosh Aslansefat  

**Link**: [PDF](https://arxiv.org/pdf/2506.06868)  

**Abstract**: Machine Learning (ML) models are increasingly integrated into safety-critical systems, such as autonomous vehicle platooning, to enable real-time decision-making. However, their inherent imperfection introduces a new class of failure: reasoning failures often triggered by distributional shifts between operational and training data. Traditional safety assessment methods, which rely on design artefacts or code, are ill-suited for ML components that learn behaviour from data. SafeML was recently proposed to dynamically detect such shifts and assign confidence levels to the reasoning of ML-based components. Building on this, we introduce a probabilistic safety assurance framework that integrates SafeML with Bayesian Networks (BNs) to model ML failures as part of a broader causal safety analysis. This allows for dynamic safety evaluation and system adaptation under uncertainty. We demonstrate the approach on an simulated automotive platooning system with traffic sign recognition. The findings highlight the potential broader benefits of explicitly modelling ML failures in safety assessment. 

**Abstract (ZH)**: 机器学习模型正越来越多地集成到自动驾驶车辆编队等安全关键系统中，以实现实时决策。然而，其固有的不完美性引入了一类新的故障：由于运行数据和训练数据分布变化引发的推理故障。传统的安全评估方法依赖于设计 artefacts 或代码，不适合学习行为的 ML 组件。近期提出了 SafeML 来动态检测这类变化并为基于 ML 的组件的推理赋予置信水平。在此基础上，我们提出了一种结合 SafeML 和贝叶斯网络 (BNs) 的概率安全保证框架，以将 ML 失败建模为更广泛的因果安全分析的一部分。这允许在不确定性下进行动态安全评估和系统适应。我们通过一个带有交通标志识别的模拟自动驾驶车辆编队系统展示了该方法。研究结果突显了明确建模 ML 失败在安全性评估中的潜在更广泛益处。 

---
# Learning What Matters Now: A Dual-Critic Context-Aware RL Framework for Priority-Driven Information Gain 

**Title (ZH)**: 学习当前重要的内容：一种基于优先级驱动信息增益的双重评论家上下文感知 reinforcement 学习框架 

**Authors**: Dimitris Panagopoulos, Adolfo Perrusquia, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.06786)  

**Abstract**: Autonomous systems operating in high-stakes search-and-rescue (SAR) missions must continuously gather mission-critical information while flexibly adapting to shifting operational priorities. We propose CA-MIQ (Context-Aware Max-Information Q-learning), a lightweight dual-critic reinforcement learning (RL) framework that dynamically adjusts its exploration strategy whenever mission priorities change. CA-MIQ pairs a standard extrinsic critic for task reward with an intrinsic critic that fuses state-novelty, information-location awareness, and real-time priority alignment. A built-in shift detector triggers transient exploration boosts and selective critic resets, allowing the agent to re-focus after a priority revision. In a simulated SAR grid-world, where experiments specifically test adaptation to changes in the priority order of information types the agent is expected to focus on, CA-MIQ achieves nearly four times higher mission-success rates than baselines after a single priority shift and more than three times better performance in multiple-shift scenarios, achieving 100% recovery while baseline methods fail to adapt. These results highlight CA-MIQ's effectiveness in any discrete environment with piecewise-stationary information-value distributions. 

**Abstract (ZH)**: 自主系统在高风险搜救（SAR）任务中的运行必须在不断收集关键任务信息的同时，灵活适应不断变化的操作优先级。我们提出了一种轻量级的双 Critic 强化学习（RL）框架 CA-MIQ（情境感知最大化信息 Q 学习），该框架在任务优先级变化时动态调整其探索策略。CA-MIQ 结合了一个标准的外在 Critic 用于任务奖励，以及一个内在 Critic，该 Critic 融合了状态新颖性、信息位置意识和实时优先级对齐。内置的偏移检测器触发临时的探索增强和选择性的 Critic 重置，使代理在优先级修订后能够重新聚焦。在模拟的 SAR 网格世界中，实验特别测试了代理需要重点关注的信息类型优先级变化的适应能力，CA-MIQ 在单次优先级变化后将任务成功率提高了近四倍，在多次变化场景中的表现提高了三倍以上，并实现了 100% 的恢复，而基线方法则无法适应。这些结果突显了 CA-MIQ 在任何离散环境中信息价值分布分段稳定的效果。 

---
# AI Simulation by Digital Twins: Systematic Survey, Reference Framework, and Mapping to a Standardized Architecture 

**Title (ZH)**: 数字孪生驱动的AI仿真：系统性综述、参考框架及其与标准化架构的映射 

**Authors**: Xiaoran Liu, Istvan David  

**Link**: [PDF](https://arxiv.org/pdf/2506.06580)  

**Abstract**: Insufficient data volume and quality are particularly pressing challenges in the adoption of modern subsymbolic AI. To alleviate these challenges, AI simulation uses virtual training environments in which AI agents can be safely and efficiently developed with simulated, synthetic data. Digital twins open new avenues in AI simulation, as these high-fidelity virtual replicas of physical systems are equipped with state-of-the-art simulators and the ability to further interact with the physical system for additional data collection. In this article, we report on our systematic survey of digital twin-enabled AI simulation. By analyzing 22 primary studies, we identify technological trends and derive a reference framework to situate digital twins and AI components. Based on our findings, we derive a reference framework and provide architectural guidelines by mapping it onto the ISO 23247 reference architecture for digital twins. Finally, we identify challenges and research opportunities for prospective researchers. 

**Abstract (ZH)**: 现代亚符号人工智能采用中数据量和质量不足的挑战尤为迫切。为缓解这些挑战，AI模拟使用虚拟训练环境，在其中可以安全高效地使用模拟合成数据开发AI代理。数字孪生为AI模拟开辟了新途径，这些高度逼真的物理系统虚拟复制品配备了最先进的模拟器，并能够进一步与物理系统交互以收集额外数据。本文报告了我们对数字孪生赋能的AI模拟的系统性综述。通过分析22篇主要研究，我们确定了技术趋势并推导出一个参考框架，以确定数字孪生和AI组件的位置。基于我们的研究发现，我们推导出一个参考框架并提供架构指南，将其映射到ISO 23247数字孪生参考架构之上。最后，我们识别出未来研究人员面临的挑战和研究机会。 

---
# Thinking vs. Doing: Agents that Reason by Scaling Test-Time Interaction 

**Title (ZH)**: 思考 vs. 做事：通过扩展测试时交互进行推理的代理 

**Authors**: Junhong Shen, Hao Bai, Lunjun Zhang, Yifei Zhou, Amrith Setlur, Shengbang Tong, Diego Caples, Nan Jiang, Tong Zhang, Ameet Talwalkar, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.07976)  

**Abstract**: The current paradigm of test-time scaling relies on generating long reasoning traces ("thinking" more) before producing a response. In agent problems that require interaction, this can be done by generating thinking traces before acting in the world. However, this process does not allow agents to acquire new information from the environment or adapt their behavior over time. In this work, we propose to scale test-time interaction, an untapped dimension of test-time scaling that increases the agent's interaction horizon to enable running rich behaviors such as exploration, backtracking, and dynamic re-planning within a single rollout. To demonstrate the promise of this scaling dimension, we study the domain of web agents. We first show that even prompting-based interaction scaling without any training can improve task success on web benchmarks non-trivially. Building on this, we introduce TTI (Test-Time Interaction), a curriculum-based online reinforcement learning (RL) approach that trains agents by adaptively adjusting their rollout lengths. Using a Gemma 3 12B model, TTI produces state-of-the-art open-source, open-data web agents on WebVoyager and WebArena benchmarks. We further show that TTI enables agents to balance exploration and exploitation adaptively. Our results establish interaction scaling as a powerful, complementary axis to scaling per-step compute, offering new avenues for training adaptive agents. 

**Abstract (ZH)**: 当前的测试时放大规模的范式依赖于生成长的推理轨迹（“思考”更多）后再生成响应。在需要交互的代理问题中，可以在实际操作前生成思考轨迹。然而，这一过程不允许代理从环境中学到新的信息或随时间调整其行为。在本工作中，我们提出了一种测试时交互放大规模的方法，这是一种未充分利用的测试时放大规模维度，能够扩展代理的交互范围，使代理能够在单个展开过程中运行丰富的行为，如探索、回溯和动态重规划。为了展示这一放大规模维度的潜力，我们研究了网络代理领域。我们首先表明，即使是基于提示的交互放大规模，无需任何训练也能在网页基准测试中显著提高任务成功率。在此基础上，我们引入了TTI（测试时交互）方法，这是一种基于课程的在线强化学习方法，通过自适应调整展开长度来训练代理。使用Gemma 3 12B模型，TTI在WebVoyager和WebArena基准测试中生成了最先进的开源、开源数据网页代理。我们进一步证明，TTI使代理能够自适应地平衡探索和利用。我们的结果确立了交互放大规模作为扩展每步计算量的强大补充维度，并为训练自适应代理开辟了新的途径。 

---
# BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models 

**Title (ZH)**: BridgeVLA: 输入-输出对齐以实现高效基于视觉-语言模型的3D manipulation学习 

**Authors**: Peiyan Li, Yixiang Chen, Hongtao Wu, Xiao Ma, Xiangnan Wu, Yan Huang, Liang Wang, Tao Kong, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07961)  

**Abstract**: Recently, leveraging pre-trained vision-language models (VLMs) for building vision-language-action (VLA) models has emerged as a promising approach to effective robot manipulation learning. However, only few methods incorporate 3D signals into VLMs for action prediction, and they do not fully leverage the spatial structure inherent in 3D data, leading to low sample efficiency. In this paper, we introduce BridgeVLA, a novel 3D VLA model that (1) projects 3D inputs to multiple 2D images, ensuring input alignment with the VLM backbone, and (2) utilizes 2D heatmaps for action prediction, unifying the input and output spaces within a consistent 2D image space. In addition, we propose a scalable pre-training method that equips the VLM backbone with the capability to predict 2D heatmaps before downstream policy learning. Extensive experiments show the proposed method is able to learn 3D manipulation efficiently and effectively. BridgeVLA outperforms state-of-the-art baseline methods across three simulation benchmarks. In RLBench, it improves the average success rate from 81.4% to 88.2%. In COLOSSEUM, it demonstrates significantly better performance in challenging generalization settings, boosting the average success rate from 56.7% to 64.0%. In GemBench, it surpasses all the comparing baseline methods in terms of average success rate. In real-robot experiments, BridgeVLA outperforms a state-of-the-art baseline method by 32% on average. It generalizes robustly in multiple out-of-distribution settings, including visual disturbances and unseen instructions. Remarkably, it is able to achieve a success rate of 96.8% on 10+ tasks with only 3 trajectories per task, highlighting its extraordinary sample efficiency. Project Website:this https URL 

**Abstract (ZH)**: Recently, 利用预训练的多模态模型构建视觉-语言-动作模型以有效学习机器人操作正逐渐成为一种有前途的方法。然而，目前仅有少数方法将3D信号整合进多模态模型进行动作预测，且这些方法未能充分利用3D数据中固有的空间结构，导致样本效率较低。本文介绍了一种新颖的3D VLA模型BridgeVLA，该模型通过（1）将3D输入投影为多个2D图像，确保输入与多模态模型主干对齐，以及（2）利用2D热图进行动作预测，将输入和输出空间统一在一致的2D图像空间中。此外，我们提出了一种可扩展的预训练方法，使多模态模型主干能够预测2D热图，从而为下游策略学习做好准备。大量实验证明，提出的方法能够高效有效地学习3D操作。BridgeVLA在三个仿真基准上均优于最先进的基线方法。在RLBench中，它将平均成功率从81.4%提升到88.2%。在COLOSSEUM中，它在具有挑战性的泛化设置中表现出显著更好的性能，将平均成功率从56.7%提升到64.0%。在GemBench中，它在平均成功率上超越了所有比较的基线方法。在真实机器人实验中，BridgeVLA在平均上优于最先进的基线方法32%，并且在包括视觉干扰和未见指令在内的多种非分布外设置中表现出鲁棒泛化能力。令人惊讶的是，它仅使用每个任务3条轨迹就能实现96.8%的成功率，突显了其卓越的样本效率。项目网站：this https URL 

---
# Decoupling the Image Perception and Multimodal Reasoning for Reasoning Segmentation with Digital Twin Representations 

**Title (ZH)**: 解耦图像感知与多模态推理以实现数字孪生表示的语义分割 

**Authors**: Yizhen Li, Dell Zhang, Xuelong Li, Yiqing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07943)  

**Abstract**: Reasoning Segmentation (RS) is a multimodal vision-text task that requires segmenting objects based on implicit text queries, demanding both precise visual perception and vision-text reasoning capabilities. Current RS approaches rely on fine-tuning vision-language models (VLMs) for both perception and reasoning, but their tokenization of images fundamentally disrupts continuous spatial relationships between objects. We introduce DTwinSeger, a novel RS approach that leverages Digital Twin (DT) representation as an intermediate layer to decouple perception from reasoning. Innovatively, DTwinSeger reformulates RS as a two-stage process, where the first transforms the image into a structured DT representation that preserves spatial relationships and semantic properties and then employs a Large Language Model (LLM) to perform explicit reasoning over this representation to identify target objects. We propose a supervised fine-tuning method specifically for LLM with DT representation, together with a corresponding fine-tuning dataset Seg-DT, to enhance the LLM's reasoning capabilities with DT representations. Experiments show that our method can achieve state-of-the-art performance on two image RS benchmarks and three image referring segmentation benchmarks. It yields that DT representation functions as an effective bridge between vision and text, enabling complex multimodal reasoning tasks to be accomplished solely with an LLM. 

**Abstract (ZH)**: 基于数字孪生的推理分割（DTwinSeger：一种结合数字孪生表示的多模态视图-文本任务方法） 

---
# Reinforcement Learning via Implicit Imitation Guidance 

**Title (ZH)**: 强化学习通过隐式模仿引导 

**Authors**: Perry Dong, Alec M. Lessing, Annie S. Chen, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2506.07505)  

**Abstract**: We study the problem of sample efficient reinforcement learning, where prior data such as demonstrations are provided for initialization in lieu of a dense reward signal. A natural approach is to incorporate an imitation learning objective, either as regularization during training or to acquire a reference policy. However, imitation learning objectives can ultimately degrade long-term performance, as it does not directly align with reward maximization. In this work, we propose to use prior data solely for guiding exploration via noise added to the policy, sidestepping the need for explicit behavior cloning constraints. The key insight in our framework, Data-Guided Noise (DGN), is that demonstrations are most useful for identifying which actions should be explored, rather than forcing the policy to take certain actions. Our approach achieves up to 2-3x improvement over prior reinforcement learning from offline data methods across seven simulated continuous control tasks. 

**Abstract (ZH)**: 基于数据指导噪声的样本高效强化学习 

---
# DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO 

**Title (ZH)**: DeepVideo-R1: 视频强化微调通过难度感知递归GRPO 

**Authors**: Jinyoung Park, Jeehye Na, Jinyoung Kim, Hyunwoo J. Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07464)  

**Abstract**: Recent works have demonstrated the effectiveness of reinforcement learning (RL)-based post-training in enhancing the reasoning capabilities of large language models (LLMs). In particular, Group Relative Policy Optimization (GRPO) has shown impressive success by employing a PPO-style reinforcement algorithm with group-based normalized rewards. However, the application of GRPO to Video Large Language Models (Video LLMs) has been less studied. In this paper, we explore GRPO for video LLMs and identify two primary issues that impede its effective learning: (1) reliance on safeguards, and (2) the vanishing advantage problem. To mitigate these challenges, we propose DeepVideo-R1, a video large language model trained with our proposed Reg-GRPO (Regressive GRPO) and difficulty-aware data augmentation strategy. Reg-GRPO reformulates the GRPO objective as a regression task, directly predicting the advantage in GRPO. This design eliminates the need for safeguards like clipping and min functions, thereby facilitating more direct policy guidance by aligning the model with the advantage values. We also design the difficulty-aware data augmentation strategy that dynamically augments training samples at solvable difficulty levels, fostering diverse and informative reward signals. Our comprehensive experiments show that DeepVideo-R1 significantly improves video reasoning performance across multiple video reasoning benchmarks. 

**Abstract (ZH)**: Recent Works on Using Reinforcement Learning for Enhancing Reasoning Capabilities of Video Large Language Models: Addressing Challenges with DeepVideo-R1 

---
# Language-Grounded Hierarchical Planning and Execution with Multi-Robot 3D Scene Graphs 

**Title (ZH)**: 基于语言的多层次规划与执行：多机器人3D场景图 

**Authors**: Jared Strader, Aaron Ray, Jacob Arkin, Mason B. Peterson, Yun Chang, Nathan Hughes, Christopher Bradley, Yi Xuan Jia, Carlos Nieto-Granda, Rajat Talak, Chuchu Fan, Luca Carlone, Jonathan P. How, Nicholas Roy  

**Link**: [PDF](https://arxiv.org/pdf/2506.07454)  

**Abstract**: In this paper, we introduce a multi-robot system that integrates mapping, localization, and task and motion planning (TAMP) enabled by 3D scene graphs to execute complex instructions expressed in natural language. Our system builds a shared 3D scene graph incorporating an open-set object-based map, which is leveraged for multi-robot 3D scene graph fusion. This representation supports real-time, view-invariant relocalization (via the object-based map) and planning (via the 3D scene graph), allowing a team of robots to reason about their surroundings and execute complex tasks. Additionally, we introduce a planning approach that translates operator intent into Planning Domain Definition Language (PDDL) goals using a Large Language Model (LLM) by leveraging context from the shared 3D scene graph and robot capabilities. We provide an experimental assessment of the performance of our system on real-world tasks in large-scale, outdoor environments. 

**Abstract (ZH)**: 基于3D场景图的多机器人系统：结合映射、定位和自然语言表达复杂指令的Task and Motion Planning (TAMP) 

---
# LiteVLM: A Low-Latency Vision-Language Model Inference Pipeline for Resource-Constrained Environments 

**Title (ZH)**: LiteVLM：一种面向资源受限环境的低延迟视觉-语言模型推理管道 

**Authors**: Jin Huang, Yuchao Jin, Le An, Josh Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.07416)  

**Abstract**: This paper introduces an efficient Vision-Language Model (VLM) pipeline specifically optimized for deployment on embedded devices, such as those used in robotics and autonomous driving. The pipeline significantly reduces the computational overhead by jointly leveraging patch selection to filter irrelevant camera views, a token selection module to reduce input sequence length for the LLM, and speculative decoding to accelerate token generation. Evaluation on the NVIDIA DRIVE Thor platform for automonous driving application, our pipeline achieves $2.5\times$ end-to-end latency reduction without compromising task accuracy. The speed-up further increases to $3.2\times$ when applying FP8 post-training quantization. These results demonstrate our pipeline as a viable solution for enabling real-time VLM deployment in resource-constrained environments. 

**Abstract (ZH)**: 一种针对嵌入式设备优化的高效视觉-语言模型管道：在机器人和自动驾驶应用中的部署与加速 

---
# From Static to Adaptive Defense: Federated Multi-Agent Deep Reinforcement Learning-Driven Moving Target Defense Against DoS Attacks in UAV Swarm Networks 

**Title (ZH)**: 从静态防御到适应性防御：无人机 swarm 网络中面向 DoS 攻击的联邦多代理深度强化学习驱动的动目标防御 

**Authors**: Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen, Tian Qin, Yuyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07392)  

**Abstract**: The proliferation of unmanned aerial vehicle (UAV) swarms has enabled a wide range of mission-critical applications, but also exposes UAV networks to severe Denial-of-Service (DoS) threats due to their open wireless environment, dynamic topology, and resource constraints. Traditional static or centralized defense mechanisms are often inadequate for such dynamic and distributed scenarios. To address these challenges, we propose a novel federated multi-agent deep reinforcement learning (FMADRL)-driven moving target defense (MTD) framework for proactive and adaptive DoS mitigation in UAV swarm networks. Specifically, we design three lightweight and coordinated MTD mechanisms, including leader switching, route mutation, and frequency hopping, that leverage the inherent flexibility of UAV swarms to disrupt attacker efforts and enhance network resilience. The defense problem is formulated as a multi-agent partially observable Markov decision process (POMDP), capturing the distributed, resource-constrained, and uncertain nature of UAV swarms under attack. Each UAV is equipped with a local policy agent that autonomously selects MTD actions based on partial observations and local experiences. By employing a policy gradient-based FMADRL algorithm, UAVs collaboratively optimize their defense policies via reward-weighted aggregation, enabling distributed learning without sharing raw data and thus reducing communication overhead. Extensive simulations demonstrate that our approach significantly outperforms state-of-the-art baselines, achieving up to a 34.6% improvement in attack mitigation rate, a reduction in average recovery time of up to 94.6%, and decreases in energy consumption and defense cost by as much as 29.3% and 98.3%, respectively, while maintaining robust mission continuity under various DoS attack strategies. 

**Abstract (ZH)**: 基于联邦多智能体深度强化学习的无人机 swarm 网络动态目标防御框架 

---
# Learn as Individuals, Evolve as a Team: Multi-agent LLMs Adaptation in Embodied Environments 

**Title (ZH)**: 个体学习，团队进化：多智能体LLMs在体言环境中的适应性发展 

**Authors**: Xinran Li, Chenjia Bai, Zijian Li, Jiakun Zheng, Ting Xiao, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07232)  

**Abstract**: Large language models (LLMs) possess extensive knowledge bases and strong reasoning capabilities, making them promising tools for complex, multi-agent planning in embodied environments. However, despite LLMs' advanced abilities and the sophisticated modular design of agentic methods, existing LLM-based planning algorithms remain limited by weak adaptation capabilities to multi-agent embodied scenarios. We address this limitation by introducing a framework that enables LLM agents to learn and evolve both before and during test time, equipping them with environment-relevant knowledge for better planning and enhanced communication for improved cooperation. Inspired by centralized training with decentralized execution in multi-agent reinforcement learning, we propose a \textit{Learn as Individuals, Evolve as a Team (LIET)} paradigm for multi-agent LLMs adaptation. At the individual level, LLM agents learn a local utility function from exploratory datasets to better comprehend the embodied environment, which is then queried during test time to support informed decision-making. At the team level, LLM agents collaboratively and iteratively maintain and update a shared cooperation knowledge list based on new experiences, using it to guide more effective communication. By combining individual learning with team evolution, LIET enables comprehensive and flexible adaptation for LLM agents. Our experiments on Communicative Watch-And-Help and ThreeD-World Multi-Agent Transport benchmarks demonstrate that LIET, instantiated with both LLaMA and GPT-4o, outperforms existing baselines and exhibits strong cooperative planning abilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）具备广泛的知识库和强大的推理能力，使其成为复杂、多智能体环境感知规划的有希望工具。然而，尽管LLMs具有高级能力且智能体方法具有精妙的模块化设计，现有的基于LLM的规划算法在多智能体感知场景的适应能力上仍然有限。我们通过引入一个框架来解决这一限制，该框架使LLM智能体能够在测试前后学习和进化，从而它们能够获得与环境相关的关键知识，进行更好的规划并增强交流以提高合作效果。受多智能体强化学习中集中训练分散执行的启发，我们提出了一种多智能体大型语言模型适应的“个体学习，团队进化（LIET）”范式。在个体层面，LLM智能体从探索性数据集中学习局部效用函数，以便更好地理解感知环境，测试时查询该效用函数以支持明智的决策。在团队层面，智能体协作并迭代地维护和更新基于新体验的共享合作知识列表，使用该列表来指导更有效的交流。通过结合个体学习与团队进化，LIET能够实现全面且灵活的适应。我们在Communicative Watch-And-Help和ThreeD-World多智能体运输基准测试上的实验表明，无论使用LLaMA还是GPT-4o实现，LIET都能超越现有基线，并展示出强大的协同规划能力。 

---
# Reliable Critics: Monotonic Improvement and Convergence Guarantees for Reinforcement Learning 

**Title (ZH)**: 可靠的批评者：强化学习中的单调改进与收敛保证 

**Authors**: Eshwar S. R., Gugan Thoppe, Aditya Gopalan, Gal Dalal  

**Link**: [PDF](https://arxiv.org/pdf/2506.07134)  

**Abstract**: Despite decades of research, it remains challenging to correctly use Reinforcement Learning (RL) algorithms with function approximation. A prime example is policy iteration, whose fundamental guarantee of monotonic improvement collapses even under linear function approximation. To address this issue, we introduce Reliable Policy Iteration (RPI). It replaces the common projection or Bellman-error minimization during policy evaluation with a Bellman-based constrained optimization. We prove that not only does RPI confer textbook monotonicity on its value estimates but these estimates also lower bound the true return. Also, their limit partially satisfies the unprojected Bellman equation, emphasizing RPI's natural fit within RL. RPI is the first algorithm with such monotonicity and convergence guarantees under function approximation. For practical use, we provide a model-free variant of RPI that amounts to a novel critic. It can be readily integrated into primary model-free PI implementations such as DQN and DDPG. In classical control tasks, such RPI-enhanced variants consistently maintain their lower-bound guarantee while matching or surpassing the performance of all baseline methods. 

**Abstract (ZH)**: 尽管已有数十年的研究，但在功能逼近的情况下正确使用强化学习（RL）算法仍具挑战性。可靠政策迭代（RPI）通过在策略评估中使用基于贝尔曼的约束优化，替代常见的投影或贝尔曼误差最小化，解决了这一问题。我们证明RPI不仅赋予其价值估计教科书级别的单调性，同时也提供了真实回报的下界。此外，其极限部分满足未投影的贝尔曼方程，强调了RPI在强化学习中的自然契合度。RPI是首个在功能逼近情况下具备单调性和收敛性保证的算法。为实际应用，我们提供了一种基于模型的RPI变体，它相当于一种新的critic。它可以无缝集成到如DQN和DDPG等主要的基于模型的PI实现中。在经典控制任务中，这些RPI增强的变体能够持续保持其下界保证，并且匹配或超越所有基线方法的表现。 

---
# Interpretable and Reliable Detection of AI-Generated Images via Grounded Reasoning in MLLMs 

**Title (ZH)**: 基于MLLMs的 grounded reasoning 的可解释和可靠检测方法：识别AI生成的图像 

**Authors**: Yikun Ji, Hong Yan, Jun Lan, Huijia Zhu, Weiqiang Wang, Qi Fan, Liqing Zhang, Jianfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07045)  

**Abstract**: The rapid advancement of image generation technologies intensifies the demand for interpretable and robust detection methods. Although existing approaches often attain high accuracy, they typically operate as black boxes without providing human-understandable justifications. Multi-modal Large Language Models (MLLMs), while not originally intended for forgery detection, exhibit strong analytical and reasoning capabilities. When properly fine-tuned, they can effectively identify AI-generated images and offer meaningful explanations. However, existing MLLMs still struggle with hallucination and often fail to align their visual interpretations with actual image content and human reasoning. To bridge this gap, we construct a dataset of AI-generated images annotated with bounding boxes and descriptive captions that highlight synthesis artifacts, establishing a foundation for human-aligned visual-textual grounded reasoning. We then finetune MLLMs through a multi-stage optimization strategy that progressively balances the objectives of accurate detection, visual localization, and coherent textual explanation. The resulting model achieves superior performance in both detecting AI-generated images and localizing visual flaws, significantly outperforming baseline methods. 

**Abstract (ZH)**: 图像生成技术的迅速发展加剧了对可解释和稳健检测方法的需求。尽管现有方法通常能获得高准确率，但它们通常作为黑盒操作，无法提供人类可理解的解释。多模态大型语言模型（MLLMs）虽然最初并非设计用于检测伪造，但展现出强大的分析与推理能力。通过适当微调，它们可以有效识别AI生成的图像并提供有意义的解释。然而，现有的MLLMs仍然难以避免生成幻觉，并常无法与其视觉解释和实际图像内容及人类推理保持一致。为此，我们构建了一个标注有边界框和描述性注释的AI生成图像数据集，突显合成痕迹，为人类对齐的视觉-文本推理奠定了基础。随后，我们通过多阶段优化策略微调MLLMs，逐步平衡准确检测、视觉定位和一致文本解释的目标。最终模型在检测AI生成图像和定位视觉缺陷方面表现出色，显著优于基线方法。 

---
# Towards Physics-informed Diffusion for Anomaly Detection in Trajectories 

**Title (ZH)**: 面向物理约束扩散的轨迹异常检测 

**Authors**: Arun Sharma, Mingzhou Yang, Majid Farhadloo, Subhankar Ghosh, Bharat Jayaprakash, Shashi Shekhar  

**Link**: [PDF](https://arxiv.org/pdf/2506.06999)  

**Abstract**: Given trajectory data, a domain-specific study area, and a user-defined threshold, we aim to find anomalous trajectories indicative of possible GPS spoofing (e.g., fake trajectory). The problem is societally important to curb illegal activities in international waters, such as unauthorized fishing and illicit oil transfers. The problem is challenging due to advances in AI generated in deep fakes generation (e.g., additive noise, fake trajectories) and lack of adequate amount of labeled samples for ground-truth verification. Recent literature shows promising results for anomalous trajectory detection using generative models despite data sparsity. However, they do not consider fine-scale spatiotemporal dependencies and prior physical knowledge, resulting in higher false-positive rates. To address these limitations, we propose a physics-informed diffusion model that integrates kinematic constraints to identify trajectories that do not adhere to physical laws. Experimental results on real-world datasets in the maritime and urban domains show that the proposed framework results in higher prediction accuracy and lower estimation error rate for anomaly detection and trajectory generation methods, respectively. Our implementation is available at this https URL. 

**Abstract (ZH)**: 给定轨迹数据、特定研究区域和用户定义的阈值，我们旨在找到指示可能的GPS欺骗（例如，虚假轨迹）的异常轨迹。该问题对于遏制国际海域中的非法活动（如未授权捕鱼和非法油品转运）具有社会重要性。由于在深度假信息生成（例如，添加噪声、虚假轨迹）方面AI的进展以及缺乏足够的标注样本进行真实情况验证，这一问题具有挑战性。近期文献表明，尽管存在数据稀疏问题，生成模型在异常轨迹检测方面仍显示出有希望的结果。然而，它们并未考虑细粒度的空间-时间依赖性和先前的物理知识，导致较高的误报率。为解决这些局限性，我们提出了一种基于物理的扩散模型，该模型整合了运动约束以识别不符合物理定律的轨迹。在海洋和城市领域的实际数据集上的实验结果显示，提出的框架分别在异常检测和轨迹生成方法中提高了预测准确性和降低了估计误差率。我们的实现可通过以下网址访问：this https URL。 

---
# Position: Simulating Society Requires Simulating Thought 

**Title (ZH)**: 位置：模拟社会需要模拟思想 

**Authors**: Chance Jiajie Li, Jiayi Wu, Zhenze Mo, Ao Qu, Yuhan Tang, Kaiya Ivy Zhao, Yulu Gan, Jie Fan, Jiangbo Yu, Jinhua Zhao, Paul Liang, Luis Alonso, Kent Larson  

**Link**: [PDF](https://arxiv.org/pdf/2506.06958)  

**Abstract**: Simulating society with large language models (LLMs), we argue, requires more than generating plausible behavior -- it demands cognitively grounded reasoning that is structured, revisable, and traceable. LLM-based agents are increasingly used to emulate individual and group behavior -- primarily through prompting and supervised fine-tuning. Yet they often lack internal coherence, causal reasoning, and belief traceability -- making them unreliable for analyzing how people reason, deliberate, or respond to interventions.
To address this, we present a conceptual modeling paradigm, Generative Minds (GenMinds), which draws from cognitive science to support structured belief representations in generative agents. To evaluate such agents, we introduce the RECAP (REconstructing CAusal Paths) framework, a benchmark designed to assess reasoning fidelity via causal traceability, demographic grounding, and intervention consistency. These contributions advance a broader shift: from surface-level mimicry to generative agents that simulate thought -- not just language -- for social simulations. 

**Abstract (ZH)**: 使用大型语言模型（LLMs）模拟社会，我们argue需要的不仅仅是生成可信的行为，还需要基于认知的基础进行结构化、可修改和可追溯的推理。基于LLM的代理越来越多地用于模仿个体和群体的行为——主要是通过提示和监督微调实现的。然而，它们往往缺乏内部一致性、因果推理和信念可追溯性——这使它们在分析人们如何推理、审议或回应干预方面不可靠。

为此，我们提出了一个基于认知科学的概念建模范式——生成性心智（GenMinds），旨在支持生成代理的结构化信念表示。为评估此类代理，我们引入了RECAP（重构因果路径）框架，该框架旨在通过因果可追溯性、人口统计学基础和干预一致性来评估推理的准确性。这些贡献推动了更广泛的转变：从表面模仿转向能够模拟思维（而不仅仅是语言）的生成代理，以用于社会模拟。 

---
# PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation 

**Title (ZH)**: PCoT: 说服增强的思维链用于检测假新闻和社会媒体误导信息 

**Authors**: Arkadiusz Modzelewski, Witold Sosnowski, Tiziano Labruna, Adam Wierzbicki, Giovanni Da San Martino  

**Link**: [PDF](https://arxiv.org/pdf/2506.06842)  

**Abstract**: Disinformation detection is a key aspect of media literacy. Psychological studies have shown that knowledge of persuasive fallacies helps individuals detect disinformation. Inspired by these findings, we experimented with large language models (LLMs) to test whether infusing persuasion knowledge enhances disinformation detection. As a result, we introduce the Persuasion-Augmented Chain of Thought (PCoT), a novel approach that leverages persuasion to improve disinformation detection in zero-shot classification. We extensively evaluate PCoT on online news and social media posts. Moreover, we publish two novel, up-to-date disinformation datasets: EUDisinfo and MultiDis. These datasets enable the evaluation of PCoT on content entirely unseen by the LLMs used in our experiments, as the content was published after the models' knowledge cutoffs. We show that, on average, PCoT outperforms competitive methods by 15% across five LLMs and five datasets. These findings highlight the value of persuasion in strengthening zero-shot disinformation detection. 

**Abstract (ZH)**: 媒体素养中的虚假信息检测是一个关键方面。心理学研究表明，了解有说服力的谬误知识有助于个体检测虚假信息。受这些发现的启发，我们通过实验测试大型语言模型（LLMs）中灌输说服知识是否能增强虚假信息检测。因此，我们提出了说服增强思维链（PCoT）这一新颖方法，利用说服力来改进零样本分类中的虚假信息检测。我们对在线新闻和社交媒体帖子进行了广泛的评估。此外，我们发布了两个最新的虚假信息数据集：EUDisinfo和MultiDis。这些数据集使PCoT能够在实验中使用的LLM从未见过的内容上进行评估，因为内容是在模型知识截止点之后发布的。结果显示，平均而言，PCoT在五种LLM和五种数据集上的表现比竞争方法高出15%。这些发现突显了说服力在增强零样本虚假信息检测方面的价值。 

---
# RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks 

**Title (ZH)**: RoboPARA：跨任务的并行分配与重组的双臂机器人规划 

**Authors**: Shiying Duan, Pei Ren, Nanxiang Jiang, Zhengping Che, Jian Tang, Yifan Sun, Zhaoxin Fan, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06683)  

**Abstract**: Dual-arm robots play a crucial role in improving efficiency and flexibility in complex multitasking scenarios. While existing methods have achieved promising results in task planning, they often fail to fully optimize task parallelism, limiting the potential of dual-arm collaboration. To address this issue, we propose RoboPARA, a novel large language model (LLM)-driven framework for dual-arm task parallelism planning. RoboPARA employs a two-stage process: (1) Dependency Graph-based Planning Candidates Generation, which constructs directed acyclic graphs (DAGs) to model task dependencies and eliminate redundancy, and (2) Graph Re-Traversal-based Dual-Arm Parallel Planning, which optimizes DAG traversal to maximize parallelism while maintaining task coherence. In addition, we introduce the Cross-Scenario Dual-Arm Parallel Task dataset (X-DAPT dataset), the first dataset specifically designed to evaluate dual-arm task parallelism across diverse scenarios and difficulty levels. Extensive experiments on the X-DAPT dataset demonstrate that RoboPARA significantly outperforms existing methods, achieving higher efficiency and reliability, particularly in complex task combinations. The code and dataset will be released upon acceptance. 

**Abstract (ZH)**: 双臂机器人在复杂多任务场景中对于提高效率和灵活性起着关键作用。尽管现有方法在任务规划方面取得了令人鼓舞的结果，但它们往往未能充分优化任务并行性，限制了双臂协作的潜力。为了解决这一问题，我们提出了一种新的基于大型语言模型（LLM）的双臂任务并行规划框架RoboPARA。RoboPARA采用两阶段过程：(1) 基于依赖图的规划候选生成，该过程构建有向无环图（DAG）以建模任务依赖关系并消除冗余；(2) 基于图重新遍历的双臂并行规划，该过程优化DAG遍历以最大化并行性同时保持任务连贯性。此外，我们引入了跨场景双臂并行任务数据集（X-DAPT数据集），这是第一个专门设计用于评估不同场景和难度级别下双臂任务并行性的数据集。在X-DAPT数据集上进行的大量实验表明，RoboPARA显著优于现有方法，特别是在复杂任务组合中实现更高的效率和可靠性。该代码和数据集将在接受后发布。 

---
# From Model-Based and Adaptive Control to Evolving Fuzzy Control 

**Title (ZH)**: 从模型基础和自适应控制到演化模糊控制 

**Authors**: Daniel Leite, Igor Škrjanc, Fernando Gomide  

**Link**: [PDF](https://arxiv.org/pdf/2506.06594)  

**Abstract**: Evolving fuzzy systems build and adapt fuzzy models - such as predictors and controllers - by incrementally updating their rule-base structure from data streams. On the occasion of the 60-year anniversary of fuzzy set theory, commemorated during the Fuzz-IEEE 2025 event, this brief paper revisits the historical development and core contributions of classical fuzzy and adaptive modeling and control frameworks. It then highlights the emergence and significance of evolving intelligent systems in fuzzy modeling and control, emphasizing their advantages in handling nonstationary environments. Key challenges and future directions are discussed, including safety, interpretability, and principled structural evolution. 

**Abstract (ZH)**: 演化模糊系统通过增量更新规则基结构从数据流中构建和适应模糊模型——如预测器和控制器。在模糊集理论诞辰60周年之际，Fuzz-IEEE 2025会议期间，本文简要回顾了经典模糊和自适应建模与控制框架的历史发展和核心贡献，强调了在模糊建模与控制中演化智能系统的发展及其在处理非稳定环境方面的优势，讨论了安全性、可解释性和原理性的结构演化等关键挑战和未来方向。 

---
# Tactile MNIST: Benchmarking Active Tactile Perception 

**Title (ZH)**: 触觉MNIST：评估主动触觉感知 

**Authors**: Tim Schneider, Guillaume Duret, Cristiana de Farias, Roberto Calandra, Liming Chen, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2506.06361)  

**Abstract**: Tactile perception has the potential to significantly enhance dexterous robotic manipulation by providing rich local information that can complement or substitute for other sensory modalities such as vision. However, because tactile sensing is inherently local, it is not well-suited for tasks that require broad spatial awareness or global scene understanding on its own. A human-inspired strategy to address this issue is to consider active perception techniques instead. That is, to actively guide sensors toward regions with more informative or significant features and integrate such information over time in order to understand a scene or complete a task. Both active perception and different methods for tactile sensing have received significant attention recently. Yet, despite advancements, both fields lack standardized benchmarks. To bridge this gap, we introduce the Tactile MNIST Benchmark Suite, an open-source, Gymnasium-compatible benchmark specifically designed for active tactile perception tasks, including localization, classification, and volume estimation. Our benchmark suite offers diverse simulation scenarios, from simple toy environments all the way to complex tactile perception tasks using vision-based tactile sensors. Furthermore, we also offer a comprehensive dataset comprising 13,500 synthetic 3D MNIST digit models and 153,600 real-world tactile samples collected from 600 3D printed digits. Using this dataset, we train a CycleGAN for realistic tactile simulation rendering. By providing standardized protocols and reproducible evaluation frameworks, our benchmark suite facilitates systematic progress in the fields of tactile sensing and active perception. 

**Abstract (ZH)**: 触觉感知有潜力通过提供丰富的局部信息来显著增强灵巧的机器人操作，这些信息可以补充或替代其他传感模态，如视觉。然而，由于触觉感知本质上是局部的，它不适用于仅依靠自身进行广域意识或全局场景理解的任务。基于人类的策略是考虑主动感知技术，即引导传感器朝着具有更多信息或显著特征的区域，并通过时间上的信息整合来理解场景或完成任务。主动感知和不同类型的触觉传感方法近年来都得到了广泛关注。尽管取得了进展，这两个领域仍然缺乏标准化的基准。为了填补这一空白，我们引入了触觉MNIST基准套件，这是一个开源的、兼容Gymnasium的基准，专门设计用于主动触觉感知任务，包括定位、分类和体积估计。我们的基准套件提供了从简单玩具环境到使用视觉触觉传感器进行复杂触觉感知任务的多样化的模拟场景。此外，我们还提供了一个全面的数据集，包含13,500个合成的3D MNIST数字模型和来自600个3D打印数字的153,600个真实世界的触觉样本。利用这一数据集，我们训练了一个CycleGAN进行逼真的触觉仿真渲染。通过提供标准化协议和可重现的评估框架，我们的基准套件促进了触觉传感和主动感知领域的系统性进步。 

---
# A Reinforcement Learning Approach for RIS-aided Fair Communications 

**Title (ZH)**: 基于RIS辅助的公平通信 reinforcement learning方法 

**Authors**: Alex Pierron, Michel Barbeau, Luca De Cicco, Jose Rubio-Hernan, Joaquin Garcia-Alfaro  

**Link**: [PDF](https://arxiv.org/pdf/2506.06344)  

**Abstract**: Reconfigurable Intelligent Surfaces (RISs) are composed of physical elements that can dynamically alter electromagnetic wave properties to enhance beamforming and leading to improvements in areas with low coverage properties. They have the potential to be combined with Reinforcement Learning (RL) techniques to achieve network performance and energy efficiency via optimization techniques. In addition to performance and energy improvements, it is also crucial to consider the concept of fair communications. RISs must ensure that User Equipment (UE) units receive their signals with adequate strength, without other UE being deprived of service due to insufficient power. In this paper, we address such a problem. We explore the fairness properties of previous work and propose a novel method that aims at obtaining an efficient and fair duplex RIS-RL system for multiple legitimate UE units. We report and discuss our experimental work and simulation results. We also release our code and datasets to foster further research in the topic. 

**Abstract (ZH)**: 可重构智能表面(RIS)由能够动态改变电磁波性质以增强波束成形并改善覆盖不足区域性能的物理元件组成。它们有潜力与强化学习(RL)技术相结合，通过优化技术实现网络性能和能效提升。除了性能和能效的提升，公平通信的概念也同样重要。RIS必须确保用户设备(UE)接收到足够的信号强度，而不使其他UE因功率不足而无法服务。在本文中，我们探讨了先前工作的公平性特性，并提出了一种新型方法，旨在为多个合法UE单位获得高效且公平的单工RIS-RL系统。我们报告并讨论了我们的实验工作和仿真结果，同时发布我们的代码和数据集以促进该领域进一步研究。 

---
# A Reinforcement-Learning-Enhanced LLM Framework for Automated A/B Testing in Personalized Marketing 

**Title (ZH)**: 基于强化学习增强的LLM框架在个性化营销中的自动化A/B测试 

**Authors**: Haoyang Feng, Yanjun Dai, Yuan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06316)  

**Abstract**: For personalized marketing, a new challenge of how to effectively algorithm the A/B testing to maximize user response is urgently to be overcome. In this paper, we present a new approach, the RL-LLM-AB test framework, for using reinforcement learning strategy optimization combined with LLM to automate and personalize A/B tests. The RL-LLM-AB test is built upon the pre-trained instruction-tuned language model. It first generates A/B versions of candidate content variants using a Prompt-Conditioned Generator, and then dynamically embeds and fuses the user portrait and the context of the current query with the multi-modal perception module to constitute the current interaction state. The content version is then selected in real-time through the policy optimization module with an Actor-Critic structure, and long-term revenue is estimated according to real-time feedback (such as click-through rate and conversion rate). Furthermore, a Memory-Augmented Reward Estimator is embedded into the framework to capture long-term user preference drift, which helps to generalize policy across multiple users and content contexts. Numerical results demonstrate the superiority of our proposed RL-LLM-ABTest over existing A/B testing methods, including classical A/B testing, Contextual Bandits, and benchmark reinforcement learning approaches on real-world marketing data. 

**Abstract (ZH)**: 针对个性化营销的新挑战，如何有效算法化A/B测试以最大化用户响应亟待解决。本文提出了一种新的方法——RL-LLM-AB测试框架，结合强化学习策略优化和 Large Language Model 自动化并个性化地进行A/B测试。RL-LLM-AB测试基于预训练的指令调整语言模型构建。它首先利用提示条件生成器生成候选内容变体的A/B版本，然后动态嵌入和融合用户画像和当前查询的上下文至多模态感知模块，构成当前交互状态。通过具有Actor-Critic结构的策略优化模块实时选择内容版本，并根据实时反馈（如点击率和转化率）估计长期收益。此外，框架中嵌入了记忆增强的奖励估计算法，以捕捉长期用户偏好漂移，有助于跨多个用户和内容上下文推广策略。实证结果表明，我们的RL-LLM-ABTest在实际营销数据上的表现优于现有的A/B测试方法，包括经典的A/B测试、上下文多臂老虎机以及基准强化学习方法。 

---
# DISRetrieval: Harnessing Discourse Structure for Long Document Retrieval 

**Title (ZH)**: DISRetrieval：利用话语结构进行长文档检索 

**Authors**: Huiyao Chen, Yi Yang, Yinghui Li, Meishan Zhang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06313)  

**Abstract**: Long document understanding has become increasingly crucial in natural language processing, with retrieval-based methods emerging as a promising solution to address the context length limitations of large language models (LLMs). However, existing approaches either treat documents as flat sequences or employ arbitrary chunking strategies, failing to capture the inherent discourse structure that guides human comprehension. We present DISRetrieval, a novel hierarchical retrieval framework that leverages linguistic discourse structure to enhance long document understanding. Our approach introduces three key innovations: (1) a discourse-aware document organization framework that utilizes rhetorical structure theory (RST) to create sentence-level hierarchical representations, preserving both semantic relationships and natural document flow; (2) an LLM-enhanced node representation technique that combines discourse structure with adaptive summarization to enrich tree nodes with contextual information; and (3) a hierarchical evidence retrieval mechanism that effectively selects relevant content while maintaining discourse coherence. Through comprehensive experiments on QASPER and QuALITY datasets, DISRetrieval demonstrates substantial improvements over existing methods in both token-level retrieval metrics and downstream question answering tasks. Our ablation studies confirm that incorporating discourse structure significantly enhances retrieval effectiveness across different document lengths and query types, validating the importance of linguistically-informed document representation in long-text understanding. Our code and datasets are publicly available at github/DreamH1gh/DISRetrieval to facilitate future research. 

**Abstract (ZH)**: 长文档理解在自然语言处理中变得愈发关键，基于检索的方法因其能解决大规模语言模型上下文长度限制而展现出潜力。然而，现有方法要么将文档视为扁平序列，要么采用任意的分块策略，无法捕捉引导人类理解的固有语篇结构。我们提出DISRetrieval，一种新颖的层次检索框架，利用语篇结构提升长文档理解能力。该方法引入了三个方面的主要创新：(1) 一种语篇意识的文档组织框架，利用论理性结构理论 (RST) 创建句级层次表示，同时保持语义关系和自然文档流程；(2) 结合语篇结构与自适应总结的LLM增强节点表示技术，丰富树节点的上下文信息；(3) 一种有效的层次证据检索机制，能够在保持语篇连贯性的同时选择相关内容。通过在QASPER和QuALITY数据集上的全面实验，DISRetrieval在token级检索指标和下游问答任务中均显著优于现有方法。我们的消融研究证实，结合语篇结构在不同文档长度和查询类型下的检索效果均有显著提升，验证了基于语言信息的文档表示在长文本理解中的重要性。我们的代码和数据集可在github/DreamH1gh/DISRetrieval公开获取，以促进未来研究。 

---
