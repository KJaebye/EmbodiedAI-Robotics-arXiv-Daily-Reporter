# Next-Future: Sample-Efficient Policy Learning for Robotic-Arm Tasks 

**Title (ZH)**: Next-Future: 样本高效机器人臂任务策略学习 

**Authors**: Fikrican Özgür, René Zurbrügg, Suryansh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2504.11247)  

**Abstract**: Hindsight Experience Replay (HER) is widely regarded as the state-of-the-art algorithm for achieving sample-efficient multi-goal reinforcement learning (RL) in robotic manipulation tasks with binary rewards. HER facilitates learning from failed attempts by replaying trajectories with redefined goals. However, it relies on a heuristic-based replay method that lacks a principled framework. To address this limitation, we introduce a novel replay strategy, "Next-Future", which focuses on rewarding single-step transitions. This approach significantly enhances sample efficiency and accuracy in learning multi-goal Markov decision processes (MDPs), particularly under stringent accuracy requirements -- a critical aspect for performing complex and precise robotic-arm tasks. We demonstrate the efficacy of our method by highlighting how single-step learning enables improved value approximation within the multi-goal RL framework. The performance of the proposed replay strategy is evaluated across eight challenging robotic manipulation tasks, using ten random seeds for training. Our results indicate substantial improvements in sample efficiency for seven out of eight tasks and higher success rates in six tasks. Furthermore, real-world experiments validate the practical feasibility of the learned policies, demonstrating the potential of "Next-Future" in solving complex robotic-arm tasks. 

**Abstract (ZH)**: 基于“ hindsight经验重放（HER）的“Next-Future”重放策略在机器人 manipulation 任务中的高效多目标强化学习 

---
# A Real-time Anomaly Detection Method for Robots based on a Flexible and Sparse Latent Space 

**Title (ZH)**: 基于柔性稀疏潜在空间的实时机器人异常检测方法 

**Authors**: Taewook Kang, Bum-Jae You, Juyoun Park, Yisoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.11170)  

**Abstract**: The growing demand for robots to operate effectively in diverse environments necessitates the need for robust real-time anomaly detection techniques during robotic operations. However, deep learning-based models in robotics face significant challenges due to limited training data and highly noisy signal features. In this paper, we present Sparse Masked Autoregressive Flow-based Adversarial AutoEncoders model to address these problems. This approach integrates Masked Autoregressive Flow model into Adversarial AutoEncoders to construct a flexible latent space and utilize Sparse autoencoder to efficiently focus on important features, even in scenarios with limited feature space. Our experiments demonstrate that the proposed model achieves a 4.96% to 9.75% higher area under the receiver operating characteristic curve for pick-and-place robotic operations with randomly placed cans, compared to existing state-of-the-art methods. Notably, it showed up to 19.67% better performance in scenarios involving collisions with lightweight objects. Additionally, unlike the existing state-of-the-art model, our model performs inferences within 1 millisecond, ensuring real-time anomaly detection. These capabilities make our model highly applicable to machine learning-based robotic safety systems in dynamic environments. The code will be made publicly available after acceptance. 

**Abstract (ZH)**: 基于稀疏掩码自回归流的对抗自编码器模型：应对机器人操作中多样环境下的实时异常检测需求 

---
# The Robotability Score: Enabling Harmonious Robot Navigation on Urban Streets 

**Title (ZH)**: 机器人化得分：实现城市街道和谐机器人导航 

**Authors**: Matt Franchi, Maria Teresa Parreira, Fanjun Bu, Wendy Ju  

**Link**: [PDF](https://arxiv.org/pdf/2504.11163)  

**Abstract**: This paper introduces the Robotability Score ($R$), a novel metric that quantifies the suitability of urban environments for autonomous robot navigation. Through expert interviews and surveys, we identify and weigh key features contributing to R for wheeled robots on urban streets. Our findings reveal that pedestrian density, crowd dynamics and pedestrian flow are the most critical factors, collectively accounting for 28% of the total score. Computing robotability across New York City yields significant variation; the area of highest R is 3.0 times more "robotable" than the area of lowest R. Deployments of a physical robot on high and low robotability areas show the adequacy of the score in anticipating the ease of robot navigation. This new framework for evaluating urban landscapes aims to reduce uncertainty in robot deployment while respecting established mobility patterns and urban planning principles, contributing to the discourse on harmonious human-robot environments. 

**Abstract (ZH)**: 这篇论文介绍了Robotability得分（$R$），这是一个新颖的指标，用于量化城市环境对自主机器人导航的适宜性。通过专家访谈和调查，我们识别并加权了对城市街道上轮式机器人导航至关重要的关键特征。我们的研究发现，行人密度、人群动态和行人流量是最重要的因素，共同占总分的28%。在整个纽约市计算机器人适宜性揭示了显著的变化；最高机器人适宜性区域的得分是最低区域的3.0倍。在高机器人适宜性和低机器人适宜性区域部署物理机器人显示了该得分在预测机器人导航难度方面的有效性。这一新的评估城市景观的框架旨在减少机器人部署的不确定性，同时尊重已有的移动模式和城市规划原则，为和谐的人机共存环境的讨论做出贡献。 

---
# ZeroGrasp: Zero-Shot Shape Reconstruction Enabled Robotic Grasping 

**Title (ZH)**: ZeroGrasp: 零样本形状重建赋能机器人抓取 

**Authors**: Shun Iwase, Zubair Irshad, Katherine Liu, Vitor Guizilini, Robert Lee, Takuya Ikeda, Ayako Amma, Koichi Nishiwaki, Kris Kitani, Rares Ambrus, Sergey Zakharov  

**Link**: [PDF](https://arxiv.org/pdf/2504.10857)  

**Abstract**: Robotic grasping is a cornerstone capability of embodied systems. Many methods directly output grasps from partial information without modeling the geometry of the scene, leading to suboptimal motion and even collisions. To address these issues, we introduce ZeroGrasp, a novel framework that simultaneously performs 3D reconstruction and grasp pose prediction in near real-time. A key insight of our method is that occlusion reasoning and modeling the spatial relationships between objects is beneficial for both accurate reconstruction and grasping. We couple our method with a novel large-scale synthetic dataset, which comprises 1M photo-realistic images, high-resolution 3D reconstructions and 11.3B physically-valid grasp pose annotations for 12K objects from the Objaverse-LVIS dataset. We evaluate ZeroGrasp on the GraspNet-1B benchmark as well as through real-world robot experiments. ZeroGrasp achieves state-of-the-art performance and generalizes to novel real-world objects by leveraging synthetic data. 

**Abstract (ZH)**: 机器人抓取是具身系统的一项基石能力。许多方法直接从部分信息输出抓取，而不建模场景几何，导致运动次优甚至发生碰撞。为解决这些问题，我们提出了ZeroGrasp，这是一种新颖的框架，能够同时在近实时下进行3D重建和抓取姿态预测。我们方法的一个关键洞察是，遮挡推理和建模物体之间的空间关系对于准确的重建和抓取都有益处。我们将该方法与一个新的大规模合成数据集相结合，该数据集包含100万张逼真图像、高分辨率3D重建以及对Objaverse-LVIS数据集中12000个对象的1130亿个物理上有效的抓取姿态注释。我们在GraspNet-1B基准上评估了ZeroGrasp，并通过真实世界机器人实验进行了评估。ZeroGrasp实现了最先进的性能，并通过利用合成数据对新型真实世界对象进行了泛化。 

---
# Following Is All You Need: Robot Crowd Navigation Using People As Planners 

**Title (ZH)**: 跟随即一切：利用人群作为规划者的机器人群体导航 

**Authors**: Yuwen Liao, Xinhang Xu, Ruofei Bai, Yizhuo Yang, Muqing Cao, Shenghai Yuan, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2504.10828)  

**Abstract**: Navigating in crowded environments requires the robot to be equipped with high-level reasoning and planning techniques. Existing works focus on developing complex and heavyweight planners while ignoring the role of human intelligence. Since humans are highly capable agents who are also widely available in a crowd navigation setting, we propose an alternative scheme where the robot utilises people as planners to benefit from their effective planning decisions and social behaviours. Through a set of rule-based evaluations, we identify suitable human leaders who exhibit the potential to guide the robot towards its goal. Using a simple base planner, the robot follows the selected leader through shorthorizon subgoals that are designed to be straightforward to achieve. We demonstrate through both simulated and real-world experiments that our novel framework generates safe and efficient robot plans compared to existing planners, even without predictive or data-driven modules. Our method also brings human-like robot behaviours without explicitly defining traffic rules and social norms. Code will be available at this https URL. 

**Abstract (ZH)**: 在拥挤环境中导航需要机器人配备高级推理和规划技术。现有工作集中在开发复杂和沉重的规划器上，而忽视了人类智能的作用。由于人类是高度有能力且在拥挤环境下广泛可用的代理，我们提出了一种替代方案，其中机器人利用人类作为规划者，以受益于他们有效的规划决策和社会行为。通过一套基于规则的评估，我们识别出合适的 human leaders，他们具有引导机器人实现其目标的潜力。使用一个简单的基础规划器，机器人通过设计易于实现的短期子目标跟随选定的领导者。通过模拟和真实世界的实验，我们展示我们的新颖框架生成的安全且高效的机器人规划，即使没有预测或数据驱动的模块。我们的方法也带来了类似人类的机器人行为，而无需明确定义交通规则和社会规范。代码将在此处提供。 

---
# E2E Parking Dataset: An Open Benchmark for End-to-End Autonomous Parking 

**Title (ZH)**: E2E停车数据集：端到端自主停车的开放基准 

**Authors**: Kejia Gao, Liguo Zhou, Mingjun Liu, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2504.10812)  

**Abstract**: End-to-end learning has shown great potential in autonomous parking, yet the lack of publicly available datasets limits reproducibility and benchmarking. While prior work introduced a visual-based parking model and a pipeline for data generation, training, and close-loop test, the dataset itself was not released. To bridge this gap, we create and open-source a high-quality dataset for end-to-end autonomous parking. Using the original model, we achieve an overall success rate of 85.16% with lower average position and orientation errors (0.24 meters and 0.34 degrees). 

**Abstract (ZH)**: 端到端学习在自主停车中的应用展现出巨大潜力，然而缺乏公开的数据集限制了其实现的可重复性和基准测试。尽管先前的工作引入了基于视觉的停车模型及数据生成、训练和闭环测试的管道，但数据集本身并未公开。为了弥合这一差距，我们创建并开源了一个高质量的端到端自主停车数据集。使用原始模型，我们实现了85.16%的整体成功率，并且平均位置和方向误差分别为0.24米和0.34度。 

---
# ATLASv2: LLM-Guided Adaptive Landmark Acquisition and Navigation on the Edge 

**Title (ZH)**: ATLASv2: LLM引导的边缘端自适应地标获取与导航 

**Authors**: Mikolaj Walczak, Uttej Kallakuri, Tinoosh Mohsenin  

**Link**: [PDF](https://arxiv.org/pdf/2504.10784)  

**Abstract**: Autonomous systems deployed on edge devices face significant challenges, including resource constraints, real-time processing demands, and adapting to dynamic environments. This work introduces ATLASv2, a novel system that integrates a fine-tuned TinyLLM, real-time object detection, and efficient path planning to enable hierarchical, multi-task navigation and manipulation all on the edge device, Jetson Nano. ATLASv2 dynamically expands its navigable landmarks by detecting and localizing objects in the environment which are saved to its internal knowledge base to be used for future task execution. We evaluate ATLASv2 in real-world environments, including a handcrafted home and office setting constructed with diverse objects and landmarks. Results show that ATLASv2 effectively interprets natural language instructions, decomposes them into low-level actions, and executes tasks with high success rates. By leveraging generative AI in a fully on-board framework, ATLASv2 achieves optimized resource utilization with minimal prompting latency and power consumption, bridging the gap between simulated environments and real-world applications. 

**Abstract (ZH)**: 边缘设备上部署的自主系统面临显著挑战，包括资源限制、实时处理需求以及适应动态环境的能力。本文介绍了ATLASv2新型系统，该系统整合了精细化调优的TinyLLM、实时物体检测和高效路径规划，以在Jetson Nano边缘设备上实现分层次的多任务导航和操作。ATLASv2通过检测和定位环境中的物体来动态扩展可导航地标，并将这些信息保存到其内部知识库中，以供未来任务执行使用。我们在包括手工构建的多样化家庭和办公室环境在内的真实环境中评估了ATLASv2，结果表明ATLASv2能够有效解释自然语言指令，将其分解为低级操作，并以高成功率执行任务。通过在一个完整的车载框架中利用生成式AI，ATLASv2实现了资源的高效利用，具有最小的提示延迟和能耗，从而在模拟环境与实际应用之间架起了桥梁。 

---
# Communication-aware Hierarchical Map Compression of Time-Varying Environments for Mobile Robots 

**Title (ZH)**: 通信感知分层时间变化环境地图压缩方法及其在移动机器人中的应用 

**Authors**: Daniel T. Larsson, Dipankar Maity  

**Link**: [PDF](https://arxiv.org/pdf/2504.10751)  

**Abstract**: In this paper, we develop a systematic framework for the time-sequential compression of dynamic probabilistic occupancy grids. Our approach leverages ideas from signal compression theory to formulate an optimization problem that searches for a multi-resolution hierarchical encoder that balances the quality of the compressed map (distortion) with its description size, the latter of which relates to the bandwidth required to reliably transmit the map to other agents or to store map estimates in on-board memory. The resulting optimization problem allows for multi-resolution map compressions to be obtained that satisfy available communication or memory resources, and does not require knowledge of the occupancy map dynamics. We develop an algorithm to solve our problem, and demonstrate the utility of the proposed framework in simulation on both static (i.e., non-time varying) and dynamic (time-varying) occupancy maps. 

**Abstract (ZH)**: 本文开发了一种系统框架，用于动态概率占用网格的时间序列压缩。我们的方法借鉴信号压缩理论，通过建立优化问题来寻找一个多分辨率层次编码器，该编码器在压缩图的质量（失真）与描述大小之间寻求平衡，后者与可靠传输地图所需带宽相关。该优化问题允许获得满足可用通信或存储资源的多分辨率地图压缩，并且不需要了解占用地图的动力学。我们开发了一种算法来解决该问题，并在模拟中展示了所提框架在静态和动态占用地图上的应用价值。 

---
# A Clean Slate for Offline Reinforcement Learning 

**Title (ZH)**: 离线强化学习的全新起点 

**Authors**: Matthew Thomas Jackson, Uljad Berdica, Jarek Liesen, Shimon Whiteson, Jakob Nicolaus Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2504.11453)  

**Abstract**: Progress in offline reinforcement learning (RL) has been impeded by ambiguous problem definitions and entangled algorithmic designs, resulting in inconsistent implementations, insufficient ablations, and unfair evaluations. Although offline RL explicitly avoids environment interaction, prior methods frequently employ extensive, undocumented online evaluation for hyperparameter tuning, complicating method comparisons. Moreover, existing reference implementations differ significantly in boilerplate code, obscuring their core algorithmic contributions. We address these challenges by first introducing a rigorous taxonomy and a transparent evaluation protocol that explicitly quantifies online tuning budgets. To resolve opaque algorithmic design, we provide clean, minimalistic, single-file implementations of various model-free and model-based offline RL methods, significantly enhancing clarity and achieving substantial speed-ups. Leveraging these streamlined implementations, we propose Unifloral, a unified algorithm that encapsulates diverse prior approaches within a single, comprehensive hyperparameter space, enabling algorithm development in a shared hyperparameter space. Using Unifloral with our rigorous evaluation protocol, we develop two novel algorithms - TD3-AWR (model-free) and MoBRAC (model-based) - which substantially outperform established baselines. Our implementation is publicly available at this https URL. 

**Abstract (ZH)**: Offline Reinforcement Learning 进展受困于模糊的问题定义和纠缠的算法设计，导致实现不一致、消融实验不足和评估不公平。尽管 Offline RL 明确避免环境交互，先前的方法通常频繁使用广泛的、未文档化的在线调优评估，使得方法比较复杂。此外，现有的参考实现之间在样板代码方面差异显著，掩盖了其核心算法贡献。我们通过首先引入严格的分类学和透明的评估协议来解决这些挑战，该协议明确量化了在线调优预算。为了解决不透明的算法设计，我们提供了各种模型自由和模型依赖的 Offline RL 方法的简洁、简约、单文件实现，显著增强了清晰度并实现了显著的加速。利用这些精简的实现，我们提出了 Unifloral，一种统一算法，将多种先前的方法封装在一个全面的超参数空间内，使算法开发可以在共享的超参数空间中进行。使用我们的严格评估协议以及 Unifloral，我们开发了两个新的算法——TD3-AWR（模型自由）和 MoBRAC（模型依赖），它们显著优于现有的基准算法。我们的实现已公开，网址为：这个 https URL。 

---
# Hallucination-Aware Generative Pretrained Transformer for Cooperative Aerial Mobility Control 

**Title (ZH)**: 面向幻觉的生成预训练变换器在协同空中移动控制中应用 

**Authors**: Hyojun Ahn, Seungcheol Oh, Gyu Seon Kim, Soyi Jung, Soohyun Park, Joongheon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.10831)  

**Abstract**: This paper proposes SafeGPT, a two-tiered framework that integrates generative pretrained transformers (GPTs) with reinforcement learning (RL) for efficient and reliable unmanned aerial vehicle (UAV) last-mile deliveries. In the proposed design, a Global GPT module assigns high-level tasks such as sector allocation, while an On-Device GPT manages real-time local route planning. An RL-based safety filter monitors each GPT decision and overrides unsafe actions that could lead to battery depletion or duplicate visits, effectively mitigating hallucinations. Furthermore, a dual replay buffer mechanism helps both the GPT modules and the RL agent refine their strategies over time. Simulation results demonstrate that SafeGPT achieves higher delivery success rates compared to a GPT-only baseline, while substantially reducing battery consumption and travel distance. These findings validate the efficacy of combining GPT-based semantic reasoning with formal safety guarantees, contributing a viable solution for robust and energy-efficient UAV logistics. 

**Abstract (ZH)**: SafeGPT：结合强化学习的两层框架以实现高效可靠的无人驾驶航空车辆最后一公里交付 

---
# Embodied World Models Emerge from Navigational Task in Open-Ended Environments 

**Title (ZH)**: 具身世界模型在开放环境中的导航任务中 Emerge 从导航任务中在开放-ended 环境中 

**Authors**: Li Jin, Liu Jia  

**Link**: [PDF](https://arxiv.org/pdf/2504.11419)  

**Abstract**: Understanding how artificial systems can develop spatial awareness and reasoning has long been a challenge in AI research. Traditional models often rely on passive observation, but embodied cognition theory suggests that deeper understanding emerges from active interaction with the environment. This study investigates whether neural networks can autonomously internalize spatial concepts through interaction, focusing on planar navigation tasks. Using Gated Recurrent Units (GRUs) combined with Meta-Reinforcement Learning (Meta-RL), we show that agents can learn to encode spatial properties like direction, distance, and obstacle avoidance. We introduce Hybrid Dynamical Systems (HDS) to model the agent-environment interaction as a closed dynamical system, revealing stable limit cycles that correspond to optimal navigation strategies. Ridge Representation allows us to map navigation paths into a fixed-dimensional behavioral space, enabling comparison with neural states. Canonical Correlation Analysis (CCA) confirms strong alignment between these representations, suggesting that the agent's neural states actively encode spatial knowledge. Intervention experiments further show that specific neural dimensions are causally linked to navigation performance. This work provides an approach to bridging the gap between action and perception in AI, offering new insights into building adaptive, interpretable models that can generalize across complex environments. The causal validation of neural representations also opens new avenues for understanding and controlling the internal mechanisms of AI systems, pushing the boundaries of how machines learn and reason in dynamic, real-world scenarios. 

**Abstract (ZH)**: 理解人工系统如何发展空间意识和推理一直是人工智能研究中的挑战。传统模型往往依赖于被动观察，但本体认知理论表明，更深层次的理解来自于与环境的积极互动。本研究 investigate 是否可以通过互动使神经网络自主内化空间概念，重点是平面导航任务。通过将门控循环单元（GRUs）与元强化学习（Meta-RL）结合使用，我们表明智能体可以学会编码方向、距离和障碍避让等空间属性。我们引入混合动力系统（HDS）来模型化智能体-环境交互，揭示出与最优导航策略对应的稳定极限环。岭表示法使我们能够将导航路径映射到固定维度的行为空间，从而便于与神经状态进行比较。通过对角相关分析（CCA）证实了这些表示之间的强烈对齐，表明智能体的神经状态主动编码了空间知识。干预实验进一步表明特定的神经维度与导航性能之间存在因果关系。这项工作提供了一种弥合人工智能中动作与感知之间差距的方法，为构建能够跨复杂环境泛化的适应性和可解释性模型提供了新见解。神经表示的因果验证也为理解并控制人工智能系统的内部机制开辟了新途径，推动了机器在动态、实际环境中的学习和推理能力的提升。 

---
# Kimina-Prover Preview: Towards Large Formal Reasoning Models with Reinforcement Learning 

**Title (ZH)**: Kimina-Prover 预览：基于 reinforcement learning 的大规模形式化推理模型探索 

**Authors**: Haiming Wang, Mert Unsal, Xiaohan Lin, Mantas Baksys, Junqi Liu, Marco Dos Santos, Flood Sung, Marina Vinyes, Zhenzhe Ying, Zekai Zhu, Jianqiao Lu, Hugues de Saxcé, Bolton Bailey, Chendong Song, Chenjun Xiao, Dehao Zhang, Ebony Zhang, Frederick Pu, Han Zhu, Jiawei Liu, Jonas Bayer, Julien Michel, Longhui Yu, Léo Dreyfus-Schmidt, Lewis Tunstall, Luigi Pagani, Moreira Machado, Pauline Bourigault, Ran Wang, Stanislas Polu, Thibaut Barroyer, Wen-Ding Li, Yazhe Niu, Yann Fleureau, Yangyang Hu, Zhouliang Yu, Zihan Wang, Zhilin Yang, Zhengying Liu, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.11354)  

**Abstract**: We introduce Kimina-Prover Preview, a large language model that pioneers a novel reasoning-driven exploration paradigm for formal theorem proving, as showcased in this preview release. Trained with a large-scale reinforcement learning pipeline from Qwen2.5-72B, Kimina-Prover demonstrates strong performance in Lean 4 proof generation by employing a structured reasoning pattern we term \textit{formal reasoning pattern}. This approach allows the model to emulate human problem-solving strategies in Lean, iteratively generating and refining proof steps. Kimina-Prover sets a new state-of-the-art on the miniF2F benchmark, reaching 80.7% with pass@8192. Beyond improved benchmark performance, our work yields several key insights: (1) Kimina-Prover exhibits high sample efficiency, delivering strong results even with minimal sampling (pass@1) and scaling effectively with computational budget, stemming from its unique reasoning pattern and RL training; (2) we demonstrate clear performance scaling with model size, a trend previously unobserved for neural theorem provers in formal mathematics; (3) the learned reasoning style, distinct from traditional search algorithms, shows potential to bridge the gap between formal verification and informal mathematical intuition. We open source distilled versions with 1.5B and 7B parameters of Kimina-Prover 

**Abstract (ZH)**: Kimina-Prover Preview:一种基于新颖推理驱动探索范式的大型语言模型及其在形式定理证明中的应用 

---
# Mutual Understanding between People and Systems via Neurosymbolic AI and Knowledge Graphs 

**Title (ZH)**: 人与系统之间的神经符号AI与知识图谱相互理解 

**Authors**: Irene Celino, Mario Scrocca, Agnese Chiatti  

**Link**: [PDF](https://arxiv.org/pdf/2504.11200)  

**Abstract**: This chapter investigates the concept of mutual understanding between humans and systems, positing that Neuro-symbolic Artificial Intelligence (NeSy AI) methods can significantly enhance this mutual understanding by leveraging explicit symbolic knowledge representations with data-driven learning models. We start by introducing three critical dimensions to characterize mutual understanding: sharing knowledge, exchanging knowledge, and governing knowledge. Sharing knowledge involves aligning the conceptual models of different agents to enable a shared understanding of the domain of interest. Exchanging knowledge relates to ensuring the effective and accurate communication between agents. Governing knowledge concerns establishing rules and processes to regulate the interaction between agents. Then, we present several different use case scenarios that demonstrate the application of NeSy AI and Knowledge Graphs to aid meaningful exchanges between human, artificial, and robotic agents. These scenarios highlight both the potential and the challenges of combining top-down symbolic reasoning with bottom-up neural learning, guiding the discussion of the coverage provided by current solutions along the dimensions of sharing, exchanging, and governing knowledge. Concurrently, this analysis facilitates the identification of gaps and less developed aspects in mutual understanding to address in future research. 

**Abstract (ZH)**: 本章探讨了人类与系统之间相互理解的概念，并提出神经符号人工智能（NeSy AI）方法可以通过结合显式符号知识表示与数据驱动的学习模型，显著增强这种相互理解。我们首先介绍三个关键维度来刻画相互理解：共享知识、交换知识和治理知识。共享知识涉及对齐不同代理的conceptual模型，以实现对感兴趣领域的共同理解。交换知识涉及确保代理之间有效的准确通信。治理知识涉及建立规则和流程，以调节代理之间的交互。然后，我们展示了NeSy AI和知识图谱在促进人类、人工和机器人代理之间有意义的交流中的应用实例。这些场景突显了自上而下的符号推理与自下而上的神经学习结合的潜力和挑战，并指导了沿共享、交换和治理知识维度的当前解决方案的讨论。同时，这种分析也有助于识别相互理解中的空白和未充分开发的方面，为未来研究提供方向。 

---
# Emergence of Goal-Directed Behaviors via Active Inference with Self-Prior 

**Title (ZH)**: 基于自我先验的主动推断中目标导向行为的涌现 

**Authors**: Dongmin Kim, Hoshinori Kanazawa, Naoto Yoshida, Yasuo Kuniyoshi  

**Link**: [PDF](https://arxiv.org/pdf/2504.11075)  

**Abstract**: Infants often exhibit goal-directed behaviors, such as reaching for a sensory stimulus, even when no external reward criterion is provided. These intrinsically motivated behaviors facilitate spontaneous exploration and learning of the body and environment during early developmental stages. Although computational modeling can offer insight into the mechanisms underlying such behaviors, many existing studies on intrinsic motivation focus primarily on how exploration contributes to acquiring external rewards. In this paper, we propose a novel density model for an agent's own multimodal sensory experiences, called the "self-prior," and investigate whether it can autonomously induce goal-directed behavior. Integrated within an active inference framework based on the free energy principle, the self-prior generates behavioral references purely from an intrinsic process that minimizes mismatches between average past sensory experiences and current observations. This mechanism is also analogous to the acquisition and utilization of a body schema through continuous interaction with the environment. We examine this approach in a simulated environment and confirm that the agent spontaneously reaches toward a tactile stimulus. Our study implements intrinsically motivated behavior shaped by the agent's own sensory experiences, demonstrating the spontaneous emergence of intentional behavior during early development. 

**Abstract (ZH)**: 婴儿经常表现出目标导向的行为，即使没有提供外部奖励标准，也会伸手去抓感觉刺激。这些内驱动行為促进了婴儿在早期发展阶段对身体和环境的自发探索和学习。尽管计算建模可以揭示这些行为背后的机制，但许多关于内驱动的研究主要关注探索如何有助于获得外部奖励。在本文中，我们提出了一种基于代理自身多种感官体验的新密度模型，称为“自我先验”，并探讨它是否可以自主诱导目标导向行为。该模型整合在基于自由能原则的主动推理框架之内，从一个内在过程中生成行为参考，该过程旨在最小化平均过去感官体验与当前观察之间的差异匹配。这种机制类似于通过与环境的持续互动获得和利用身体图示的过程。我们在模拟环境中对这种方法进行了测试，并确认代理自发地向触觉刺激伸展。我们的研究实现了由代理自身感官经验塑造的内驱动行为，展示了在早期发展阶段自发产生有意图行为的出现。 

---
# Toward Super Agent System with Hybrid AI Routers 

**Title (ZH)**: 具有混合AI交换机的超 agent 系统研究 

**Authors**: Yuhang Yao, Haixin Wang, Yibo Chen, Jiawen Wang, Min Chang Jordan Ren, Bosheng Ding, Salman Avestimehr, Chaoyang He  

**Link**: [PDF](https://arxiv.org/pdf/2504.10519)  

**Abstract**: AI Agents powered by Large Language Models are transforming the world through enormous applications. A super agent has the potential to fulfill diverse user needs, such as summarization, coding, and research, by accurately understanding user intent and leveraging the appropriate tools to solve tasks. However, to make such an agent viable for real-world deployment and accessible at scale, significant optimizations are required to ensure high efficiency and low cost. This paper presents a design of the Super Agent System. Upon receiving a user prompt, the system first detects the intent of the user, then routes the request to specialized task agents with the necessary tools or automatically generates agentic workflows. In practice, most applications directly serve as AI assistants on edge devices such as phones and robots. As different language models vary in capability and cloud-based models often entail high computational costs, latency, and privacy concerns, we then explore the hybrid mode where the router dynamically selects between local and cloud models based on task complexity. Finally, we introduce the blueprint of an on-device super agent enhanced with cloud. With advances in multi-modality models and edge hardware, we envision that most computations can be handled locally, with cloud collaboration only as needed. Such architecture paves the way for super agents to be seamlessly integrated into everyday life in the near future. 

**Abstract (ZH)**: 基于大型语言模型的AI代理正在通过众多应用改变世界。超级代理有能力通过准确理解用户意图并利用适当的工具来满足多样化的用户需求，如总结、编程和研究。然而，为了使这样的代理能够在现实世界中部署并大规模使用，需要进行重大优化以确保高效率和低成本。本文介绍了超级代理系统的架构设计。系统接收到用户提示后，首先检测用户的意图，然后将请求路由到具有必要工具的专业任务代理，或者自动生成代理工作流。实践中，大多数应用直接作为边缘设备上的AI助手运行，如手机和机器人。鉴于不同语言模型的能力差异以及基于云的模型通常带来的高计算成本、延迟和隐私问题，我们探索了混合模式，即路由器根据任务复杂度动态选择本地或云模型。最后，我们介绍了兼具云端功能的边缘超级代理蓝图。随着多模态模型和边缘硬件的进步，我们设想大多数计算可以本地处理，必要时才进行云协作。这样的架构为超级代理在未来无缝融入日常生活铺平了道路。 

---
# Neural Control Barrier Functions from Physics Informed Neural Networks 

**Title (ZH)**: 基于物理知情神经网络的神经控制障碍函数 

**Authors**: Shreenabh Agrawal, Manan Tayal, Aditya Singh, Shishir Kolathaya  

**Link**: [PDF](https://arxiv.org/pdf/2504.11045)  

**Abstract**: As autonomous systems become increasingly prevalent in daily life, ensuring their safety is paramount. Control Barrier Functions (CBFs) have emerged as an effective tool for guaranteeing safety; however, manually designing them for specific applications remains a significant challenge. With the advent of deep learning techniques, recent research has explored synthesizing CBFs using neural networks-commonly referred to as neural CBFs. This paper introduces a novel class of neural CBFs that leverages a physics-inspired neural network framework by incorporating Zubov's Partial Differential Equation (PDE) within the context of safety. This approach provides a scalable methodology for synthesizing neural CBFs applicable to high-dimensional systems. Furthermore, by utilizing reciprocal CBFs instead of zeroing CBFs, the proposed framework allows for the specification of flexible, user-defined safe regions. To validate the effectiveness of the approach, we present case studies on three different systems: an inverted pendulum, autonomous ground navigation, and aerial navigation in obstacle-laden environments. 

**Abstract (ZH)**: 随着自主系统在日常生活中越来越普遍，确保其安全性变得至关重要。控制屏障函数（CBFs）已成为确保安全的有效工具；然而，为特定应用手动设计它们仍然是一个重大挑战。随着深度学习技术的发展，近期研究探索了使用神经网络合成CBFs的方法——通常称为神经CBFs。本文介绍了一类新颖的神经CBFs，通过在安全性框架中引入Zubov的部分微分方程（PDE），利用物理启发式的神经网络框架。该方法为高维系统提供了可扩展的CBFs合成方法。此外，通过使用互惠CBFs而非零值CBFs，所提出的框架允许用户定义灵活的安全区域。为了验证该方法的有效性，我们在三种不同的系统上进行了案例研究：倒立摆、自主地面导航和障碍环境中的航路规划。 

---
