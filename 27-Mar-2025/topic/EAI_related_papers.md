# MoLe-VLA: Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers for Efficient Robot Manipulation 

**Title (ZH)**: MoLe-VLA：基于混合层的动态层跳过视觉语言动作模型及其在高效机器人操作中的应用 

**Authors**: Rongyu Zhang, Menghang Dong, Yuan Zhang, Liang Heng, Xiaowei Chi, Gaole Dai, Li Du, Dan Wang, Yuan Du, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20384)  

**Abstract**: Multimodal Large Language Models (MLLMs) excel in understanding complex language and visual data, enabling generalist robotic systems to interpret instructions and perform embodied tasks. Nevertheless, their real-world deployment is hindered by substantial computational and storage demands. Recent insights into the homogeneous patterns in the LLM layer have inspired sparsification techniques to address these challenges, such as early exit and token pruning. However, these methods often neglect the critical role of the final layers that encode the semantic information most relevant to downstream robotic tasks. Aligning with the recent breakthrough of the Shallow Brain Hypothesis (SBH) in neuroscience and the mixture of experts in model sparsification, we conceptualize each LLM layer as an expert and propose a Mixture-of-Layers Vision-Language-Action model (MoLe-VLA, or simply MoLe) architecture for dynamic LLM layer activation. We introduce a Spatial-Temporal Aware Router (STAR) for MoLe to selectively activate only parts of the layers based on the robot's current state, mimicking the brain's distinct signal pathways specialized for cognition and causal reasoning. Additionally, to compensate for the cognitive ability of LLMs lost in MoLe, we devise a Cognition Self-Knowledge Distillation (CogKD) framework. CogKD enhances the understanding of task demands and improves the generation of task-relevant action sequences by leveraging cognitive features. Extensive experiments conducted in both RLBench simulation and real-world environments demonstrate the superiority of MoLe-VLA in both efficiency and performance. Specifically, MoLe-VLA achieves an 8% improvement in the mean success rate across ten tasks while reducing computational costs by up to x5.6 compared to standard LLMs. 

**Abstract (ZH)**: 多模态大规模语言模型（MLLMs）在理解和处理复杂语言与视觉数据方面表现出色，使通用机器人系统能够解释指令并执行实体任务。然而，它们的实际部署受到计算和存储需求的限制。近期对LLM层中同质模式的理解启发了通过早期退出和令牌修剪等稀疏化技术来解决这些挑战的方法。然而，这些方法往往忽略了对下游机器人任务最为相关的语义信息进行编码的最终层的关键作用。顺应神经科学中浅脑假说（SBH）的最新突破以及模型稀疏化中的专家混合，我们将每个LLM层视为专家，并提出了一种动态LLM层激活的多层视觉语言行动模型（MoLe-VLA，或MoLe）架构。我们引入了一种时空感知路由器（STAR）以根据机器人当前状态仅选择性地激活部分层，模拟大脑专门化的认知和因果推理信号路径。此外，为了补偿MoLe中LLM认知能力的损失，我们设计了一种认知自知知识蒸馏（CogKD）框架。CogKD通过利用认知特征增强了对任务需求的理解，并改善了相关行动序列的生成。在RLBench模拟和真实环境中的广泛实验表明，MoLe-VLA在效率和性能上均优于标准的大规模语言模型。具体而言，MoLe-VLA在十个任务上的平均成功率提高了8%，计算成本降低了最多5.6倍。 

---
# LGR: LLM-Guided Ranking of Frontiers for Object Goal Navigation 

**Title (ZH)**: LLM引导的物体目标导航前沿排名：LGR 

**Authors**: Mitsuaki Uno, Kanji Tanaka, Daiki Iwata, Yudai Noda, Shoya Miyazaki, Kouki Terashima  

**Link**: [PDF](https://arxiv.org/pdf/2503.20241)  

**Abstract**: Object Goal Navigation (OGN) is a fundamental task for robots and AI, with key applications such as mobile robot image databases (MRID). In particular, mapless OGN is essential in scenarios involving unknown or dynamic environments. This study aims to enhance recent modular mapless OGN systems by leveraging the commonsense reasoning capabilities of large language models (LLMs). Specifically, we address the challenge of determining the visiting order in frontier-based exploration by framing it as a frontier ranking problem. Our approach is grounded in recent findings that, while LLMs cannot determine the absolute value of a frontier, they excel at evaluating the relative value between multiple frontiers viewed within a single image using the view image as context. We dynamically manage the frontier list by adding and removing elements, using an LLM as a ranking model. The ranking results are represented as reciprocal rank vectors, which are ideal for multi-view, multi-query information fusion. We validate the effectiveness of our method through evaluations in Habitat-Sim. 

**Abstract (ZH)**: 无地图物体目标导航（OGN）是机器人和AI领域的一项基础任务，具有移动机器人图像数据库（MRID）等关键应用。特别是，无地图OGN在涉及未知或动态环境的场景中尤为重要。本研究旨在通过利用大规模语言模型（LLM）的常识推理能力，提升最近的模块化无地图OGN系统。具体而言，我们通过将前沿探索中的访问顺序问题重新定义为前沿排名问题来应对基于前沿探索的挑战。我们的方法基于近期发现，虽然LLM无法确定前沿的绝对值，但在单张图像作为上下文的情况下，它们能够出色地评估多个前沿的相对价值。我们通过动态管理前沿列表——添加和移除元素——使用LLM作为排名模型。排名结果以互逆排名向量表示，非常适合多视角、多查询信息融合。我们通过在Habitat-Sim中的评估验证了该方法的有效性。 

---
# Learning Adaptive Dexterous Grasping from Single Demonstrations 

**Title (ZH)**: 基于单次演示学习适应性灵巧抓取 

**Authors**: Liangzhi Shi, Yulin Liu, Lingqi Zeng, Bo Ai, Zhengdong Hong, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2503.20208)  

**Abstract**: How can robots learn dexterous grasping skills efficiently and apply them adaptively based on user instructions? This work tackles two key challenges: efficient skill acquisition from limited human demonstrations and context-driven skill selection. We introduce AdaDexGrasp, a framework that learns a library of grasping skills from a single human demonstration per skill and selects the most suitable one using a vision-language model (VLM). To improve sample efficiency, we propose a trajectory following reward that guides reinforcement learning (RL) toward states close to a human demonstration while allowing flexibility in exploration. To learn beyond the single demonstration, we employ curriculum learning, progressively increasing object pose variations to enhance robustness. At deployment, a VLM retrieves the appropriate skill based on user instructions, bridging low-level learned skills with high-level intent. We evaluate AdaDexGrasp in both simulation and real-world settings, showing that our approach significantly improves RL efficiency and enables learning human-like grasp strategies across varied object configurations. Finally, we demonstrate zero-shot transfer of our learned policies to a real-world PSYONIC Ability Hand, with a 90% success rate across objects, significantly outperforming the baseline. 

**Abstract (ZH)**: 如何高效地让机器人从少量的人类示范中学习灵巧抓握技能，并根据用户指令进行适配性应用？本研究解决了两个关键挑战：从有限的人类示范中高效学习技能以及基于上下文选择技能。我们介绍了AdaDexGrasp框架，该框架从每种技能单个人类示范中学习抓握技能库，并使用视觉语言模型进行最合适的技能选择。为提高样本效率，我们提出了轨迹跟随奖励，引导强化学习（RL）向接近人类示范的状态发展，同时允许在探索中保持灵活性。为超越单示范学习，我们采用了逐步学习策略，逐步增加物体姿态变化以增强鲁棒性。在部署时，视觉语言模型根据用户指令检索合适的技能，将底层学习的技能与高层意图连接起来。我们在仿真和现实世界环境中评估了AdaDexGrasp，结果显示我们的方法显著提高了RL效率，并能在各种物体配置中学习类似人类的抓握策略。最后，我们在现实世界的PSYONIC Ability Hand上展示了我们学习策咯的零样本转移，并在各类物体上实现了90%的成功率，显著优于基线方法。 

---
# DRPA-MPPI: Dynamic Repulsive Potential Augmented MPPI for Reactive Navigation in Unstructured Environments 

**Title (ZH)**: DRPA-MPPI: 动态排斥势增强的MPPI在不规则环境中的反应 navigation 

**Authors**: Takahiro Fuke, Masafumi Endo, Kohei Honda, Genya Ishigami  

**Link**: [PDF](https://arxiv.org/pdf/2503.20134)  

**Abstract**: Reactive mobile robot navigation in unstructured environments is challenging when robots encounter unexpected obstacles that invalidate previously planned trajectories. Model predictive path integral control (MPPI) enables reactive planning, but still suffers from limited prediction horizons that lead to local minima traps near obstacles. Current solutions rely on heuristic cost design or scenario-specific pre-training, which often limits their adaptability to new environments. We introduce dynamic repulsive potential augmented MPPI (DRPA-MPPI), which dynamically detects potential entrapments on the predicted trajectories. Upon detecting local minima, DRPA-MPPI automatically switches between standard goal-oriented optimization and a modified cost function that generates repulsive forces away from local minima. Comprehensive testing in simulated obstacle-rich environments confirms DRPA-MPPI's superior navigation performance and safety compared to conventional methods with less computational burden. 

**Abstract (ZH)**: 动态排斥势增强的MPPI控制（DRPA-MPPI）及其在未结构化环境中的反应式导航 

---
# Gemini Robotics: Bringing AI into the Physical World 

**Title (ZH)**: Gemini机器人：将AI带入物理世界 

**Authors**: Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, Steven Bohez, Konstantinos Bousmalis, Anthony Brohan, Thomas Buschmann, Arunkumar Byravan, Serkan Cabi, Ken Caluwaerts, Federico Casarini, Oscar Chang, Jose Enrique Chen, Xi Chen, Hao-Tien Lewis Chiang, Krzysztof Choromanski, David D'Ambrosio, Sudeep Dasari, Todor Davchev, Coline Devin, Norman Di Palo, Tianli Ding, Adil Dostmohamed, Danny Driess, Yilun Du, Debidatta Dwibedi, Michael Elabd, Claudio Fantacci, Cody Fong, Erik Frey, Chuyuan Fu, Marissa Giustina, Keerthana Gopalakrishnan, Laura Graesser, Leonard Hasenclever, Nicolas Heess, Brandon Hernaez, Alexander Herzog, R. Alex Hofer, Jan Humplik, Atil Iscen, Mithun George Jacob, Deepali Jain, Ryan Julian, Dmitry Kalashnikov, M. Emre Karagozler, Stefani Karp, Chase Kew, Jerad Kirkland, Sean Kirmani, Yuheng Kuang, Thomas Lampe, Antoine Laurens, Isabel Leal, Alex X. Lee, Tsang-Wei Edward Lee, Jacky Liang, Yixin Lin, Sharath Maddineni, Anirudha Majumdar, Assaf Hurwitz Michaely, Robert Moreno, Michael Neunert, Francesco Nori, Carolina Parada, Emilio Parisotto, Peter Pastor, Acorn Pooley, Kanishka Rao, Krista Reymann, Dorsa Sadigh, Stefano Saliceti, Pannag Sanketi, Pierre Sermanet, Dhruv Shah, Mohit Sharma, Kathryn Shea, Charles Shu, Vikas Sindhwani, Sumeet Singh, Radu Soricut, Jost Tobias Springenberg, Rachel Sterneck, Razvan Surdulescu, Jie Tan, Jonathan Tompson, Vincent Vanhoucke, Jake Varley, Grace Vesom, Giulia Vezzani, Oriol Vinyals, Ayzaan Wahid, Stefan Welker  

**Link**: [PDF](https://arxiv.org/pdf/2503.20020)  

**Abstract**: Recent advancements in large multimodal models have led to the emergence of remarkable generalist capabilities in digital domains, yet their translation to physical agents such as robots remains a significant challenge. This report introduces a new family of AI models purposefully designed for robotics and built upon the foundation of Gemini 2.0. We present Gemini Robotics, an advanced Vision-Language-Action (VLA) generalist model capable of directly controlling robots. Gemini Robotics executes smooth and reactive movements to tackle a wide range of complex manipulation tasks while also being robust to variations in object types and positions, handling unseen environments as well as following diverse, open vocabulary instructions. We show that with additional fine-tuning, Gemini Robotics can be specialized to new capabilities including solving long-horizon, highly dexterous tasks, learning new short-horizon tasks from as few as 100 demonstrations and adapting to completely novel robot embodiments. This is made possible because Gemini Robotics builds on top of the Gemini Robotics-ER model, the second model we introduce in this work. Gemini Robotics-ER (Embodied Reasoning) extends Gemini's multimodal reasoning capabilities into the physical world, with enhanced spatial and temporal understanding. This enables capabilities relevant to robotics including object detection, pointing, trajectory and grasp prediction, as well as multi-view correspondence and 3D bounding box predictions. We show how this novel combination can support a variety of robotics applications. We also discuss and address important safety considerations related to this new class of robotics foundation models. The Gemini Robotics family marks a substantial step towards developing general-purpose robots that realizes AI's potential in the physical world. 

**Abstract (ZH)**: Recent advancements in大型多模态模型已在数字领域引发了一种令人瞩目的通用能力，但将其翻译到机器人等物理代理中仍然是一个重大挑战。本报告介绍了专门为机器人设计的一系列新型AI模型，建立在Gemini 2.0的基础上。我们介绍了Gemini Robotics，这是一个先进的Vision-Language-Action（VLA）通用模型，能够直接控制机器人。Gemini Robotics执行流畅且反应灵敏的动作，能够应对一系列复杂的操纵任务，同时能够应对物体类型和位置的变化，处理未见过的环境，并遵循多种多样的开放词汇指令。我们展示了通过额外的微调，Gemini Robotics可以专门用于新能力，包括解决长期目标、高度灵巧的任务，从最少100个演示中学习新的短期任务，以及适应全新的机器人实体。这得益于Gemini Robotics建立在Gemini Robotics-ER模型之上，这是我们在此工作中介绍的第二个模型。Gemini Robotics-ER（实体推理）将Gemini的多模态推理能力扩展到物理世界，增强了空间和时间理解能力。这使得与机器人相关的功能成为可能，包括物体检测、指认、轨迹和抓取预测，以及多视图对应和三维边界框预测。我们展示了这种新颖的组合如何支持各种机器人应用。我们还讨论并解决了与这一新类别的机器人基础模型相关的重要安全问题。Gemini Robotics家族标志着朝着开发能够在物理世界实现人工智能潜力的通用机器人迈出了一大步。 

---
# Hybrid Magnetically and Electrically Powered Metallo-Dielectric Janus Microrobots: Enhanced Motion Control and Operation Beyond Planar Limits 

**Title (ZH)**: 磁电混合动力金属-介质阴阳微机器人：超越平面限制的运动控制与操作 Enhancement of Motion Control and Operation Beyond Planar Limits for Hybrid Magnetically and Electrically Powered Metallo-Dielectric Janus Microrobots 

**Authors**: Ido Rachbuch, Sinwook Park, Yuval Katz, Touvia Miloh, Gilad Yossifon  

**Link**: [PDF](https://arxiv.org/pdf/2503.19984)  

**Abstract**: This study introduces the integration of hybrid magnetic and electric actuation mechanisms to achieve advanced motion capabilities for Janus particle (JP) microrobots. We demonstrate enhanced in-plane motion control through versatile control strategies and present the concepts of interplanar transitions and 2.5-dimensional (2.5D) trajectories, enabled by magnetic levitation and electrostatic trapping. These innovations expand the mobility of JPs into 3D space, allowing dynamic operation beyond the limitations of traditional surface-bound motion. Key functionalities include obstacle crossing, transitions to elevated surfaces, and discrete surface patterning enabling highly localized interventions. Using this set of tools, we also showcase the controlled out-of-plane transport of both synthetic and biological cargo. Together, these advancements lay the groundwork for novel microrobot-related applications in microfluidic systems and biomedical research. 

**Abstract (ZH)**: 本研究介绍了将混合磁性和电性驱动机制集成起来，以实现Janus粒子（JP）微机器人先进的运动能力。我们通过多样化的控制策略展示了平面内运动控制的增强，并介绍了通过磁悬浮和静电捕获实现的层间转换和2.5维轨迹的概念。这些创新将JP的移动性扩展到三维空间，使它们能够在传统表面束缚运动的限制之外实现动态操作。关键功能包括障碍物穿越、过渡到高处表面以及离散表面图案化，以实现高度局部化的干预。利用这一套工具，我们还展示了对合成和生物货物进行可控的层间运输。这些进步为微流控系统和生物医学研究中的新型微机器人相关应用奠定了基础。 

---
# Body Discovery of Embodied AI 

**Title (ZH)**: 具身AI的obody发现 

**Authors**: Zhe Sun, Pengfei Tian, Xiaozhu Hu, Xiaoyu Zhao, Huiying Li, Zhenliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.19941)  

**Abstract**: In the pursuit of realizing artificial general intelligence (AGI), the importance of embodied artificial intelligence (AI) becomes increasingly apparent. Following this trend, research integrating robots with AGI has become prominent. As various kinds of embodiments have been designed, adaptability to diverse embodiments will become important to AGI. We introduce a new challenge, termed "Body Discovery of Embodied AI", focusing on tasks of recognizing embodiments and summarizing neural signal functionality. The challenge encompasses the precise definition of an AI body and the intricate task of identifying embodiments in dynamic environments, where conventional approaches often prove inadequate. To address these challenges, we apply causal inference method and evaluate it by developing a simulator tailored for testing algorithms with virtual environments. Finally, we validate the efficacy of our algorithms through empirical testing, demonstrating their robust performance in various scenarios based on virtual environments. 

**Abstract (ZH)**: 追求实现通用人工智能（AGI）的过程中，具备实体的 artificial intelligence（AI）的重要性 increasingly 明显。在此趋势下，将机器人与 AGI 结合的研究逐渐成为热点。随着各种实体形式的 设计，AGI 对不同实体形式的适应性将变得尤为重要。我们引入了一个新的挑战，称为“实体 AI 的实体发现”，专注于识别实体和总结神经信号功能的任务。该挑战涉及对 AI 实体的精确定义以及在动态环境中辨识实体的复杂任务，而传统的办法往往在这种情况下效果不佳。为了应对这些挑战，我们应用因果推断方法，并通过为测试算法开发专门针对虚拟环境的模拟器来评估其效果。最后，我们通过实证测试验证了算法的有效性，展示了其在多种基于虚拟环境的场景中的稳健表现。 

---
# GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving 

**Title (ZH)**: GAIA-2：一种可控的多视图生成世界模型在自动驾驶中的应用 

**Authors**: Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, Gianluca Corrado  

**Link**: [PDF](https://arxiv.org/pdf/2503.20523)  

**Abstract**: Generative models offer a scalable and flexible paradigm for simulating complex environments, yet current approaches fall short in addressing the domain-specific requirements of autonomous driving - such as multi-agent interactions, fine-grained control, and multi-camera consistency. We introduce GAIA-2, Generative AI for Autonomy, a latent diffusion world model that unifies these capabilities within a single generative framework. GAIA-2 supports controllable video generation conditioned on a rich set of structured inputs: ego-vehicle dynamics, agent configurations, environmental factors, and road semantics. It generates high-resolution, spatiotemporally consistent multi-camera videos across geographically diverse driving environments (UK, US, Germany). The model integrates both structured conditioning and external latent embeddings (e.g., from a proprietary driving model) to facilitate flexible and semantically grounded scene synthesis. Through this integration, GAIA-2 enables scalable simulation of both common and rare driving scenarios, advancing the use of generative world models as a core tool in the development of autonomous systems. Videos are available at this https URL. 

**Abstract (ZH)**: 生成模型提供了一种可扩展且灵活的框架来模拟复杂环境，但当前的方法在应对自动驾驶领域的特定需求（如多Agent交互、精细控制和多摄像头一致性）方面尚存在不足。我们引入了GAIA-2，这是一种生成AI自动驾驶世界模型，将其所有这些功能统合在一个生成框架中。GAIA-2支持基于丰富结构化输入的可控视频生成，包括 ego-车辆动态、Agent配置、环境因素和道路语义。它生成了高分辨率、时空一致的多摄像头视频，适用于多种地理驾驶环境（英国、美国、德国）。该模型结合了结构化条件和外部潜在嵌入（如自有的驾驶模型），以实现灵活且语义化的场景合成。通过这种结合，GAIA-2 使我们能够大规模地模拟常见和罕见的驾驶场景，推动生成世界模型在自动驾驶系统开发中的核心应用。视频可访问此链接：this https URL。 

---
# Exploring the Effect of Robotic Embodiment and Empathetic Tone of LLMs on Empathy Elicitation 

**Title (ZH)**: 探索大型语言模型的机器人具身形式和 empathy 訵调对其 empathy 引发效果的影响 

**Authors**: Liza Darwesh, Jaspreet Singh, Marin Marian, Eduard Alexa, Koen Hindriks, Kim Baraka  

**Link**: [PDF](https://arxiv.org/pdf/2503.20518)  

**Abstract**: This study investigates the elicitation of empathy toward a third party through interaction with social agents. Participants engaged with either a physical robot or a voice-enabled chatbot, both driven by a large language model (LLM) programmed to exhibit either an empathetic tone or remain neutral. The interaction is focused on a fictional character, Katie Banks, who is in a challenging situation and in need of financial donations. The willingness to help Katie, measured by the number of hours participants were willing to volunteer, along with their perceptions of the agent, were assessed for 60 participants. Results indicate that neither robotic embodiment nor empathetic tone significantly influenced participants' willingness to volunteer. While the LLM effectively simulated human empathy, fostering genuine empathetic responses in participants proved challenging. 

**Abstract (ZH)**: 本研究探讨了通过与社会代理互动激发对第三方 empathy 的机制。参与者与物理机器人或声音聊天机器人进行了互动，这两种机器人均由大型语言模型（LLM）驱动，分别被编程为展现出 empathetic 的语气或保持中立。互动的对象是一个虚构的人物凯蒂·班克斯，她遇到了一个棘手的情况，并需要获得经济上的捐赠。通过评估参与者愿意为凯蒂志愿服务的小时数以及他们对代理人的看法，研究分析了 60 名参与者的数据。结果表明，机器人实体化形式或 empathetic 的语气并未显著影响参与者的服务意愿。虽然 LLM 成功模拟了人类的 empathy，但在参与者中引发真正的 empathy 反应却颇具挑战。 

---
# Perspective-Shifted Neuro-Symbolic World Models: A Framework for Socially-Aware Robot Navigation 

**Title (ZH)**: 视角变换的神经符号世界模型：一种社会意识增强的机器人导航框架 

**Authors**: Kevin Alcedo, Pedro U. Lima, Rachid Alami  

**Link**: [PDF](https://arxiv.org/pdf/2503.20425)  

**Abstract**: Navigating in environments alongside humans requires agents to reason under uncertainty and account for the beliefs and intentions of those around them. Under a sequential decision-making framework, egocentric navigation can naturally be represented as a Markov Decision Process (MDP). However, social navigation additionally requires reasoning about the hidden beliefs of others, inherently leading to a Partially Observable Markov Decision Process (POMDP), where agents lack direct access to others' mental states. Inspired by Theory of Mind and Epistemic Planning, we propose (1) a neuro-symbolic model-based reinforcement learning architecture for social navigation, addressing the challenge of belief tracking in partially observable environments; and (2) a perspective-shift operator for belief estimation, leveraging recent work on Influence-based Abstractions (IBA) in structured multi-agent settings. 

**Abstract (ZH)**: 在人类环境中导航需要智能体在不确定性下进行推理，并考虑周围他人的信念和意图。在序列决策框架下，以自我为中心的导航可以自然地表示为马尔可夫决策过程（MDP）。然而，社会导航还要求推理他人的隐藏信念，这导致需要使用部分可观测马尔可夫决策过程（POMDP），其中智能体无法直接访问他人的心理状态。受心智理论和知识规划的启发，我们提出了一种用于社会导航的神经符号模型基强化学习架构，以应对部分可观测环境中信念追踪的挑战；并提出了一种视角转换操作符，用于信念估计，利用了结构化多智能体环境中基于影响的抽象（IBA）的最新研究成果。 

---
# SARGes: Semantically Aligned Reliable Gesture Generation via Intent Chain 

**Title (ZH)**: 基于意图链的语义对齐可靠手势生成 

**Authors**: Nan Gao, Yihua Bao, Dongdong Weng, Jiayi Zhao, Jia Li, Yan Zhou, Pengfei Wan, Di Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20202)  

**Abstract**: Co-speech gesture generation enhances human-computer interaction realism through speech-synchronized gesture synthesis. However, generating semantically meaningful gestures remains a challenging problem. We propose SARGes, a novel framework that leverages large language models (LLMs) to parse speech content and generate reliable semantic gesture labels, which subsequently guide the synthesis of meaningful co-speech this http URL, we constructed a comprehensive co-speech gesture ethogram and developed an LLM-based intent chain reasoning mechanism that systematically parses and decomposes gesture semantics into structured inference steps following ethogram criteria, effectively guiding LLMs to generate context-aware gesture labels. Subsequently, we constructed an intent chain-annotated text-to-gesture label dataset and trained a lightweight gesture label generation model, which then guides the generation of credible and semantically coherent co-speech gestures. Experimental results demonstrate that SARGes achieves highly semantically-aligned gesture labeling (50.2% accuracy) with efficient single-pass inference (0.4 seconds). The proposed method provides an interpretable intent reasoning pathway for semantic gesture synthesis. 

**Abstract (ZH)**: 同步语音手势生成通过语音同步手势合成增强人机交互的真实感。然而，生成语义相关的手势依然是一个具有挑战性的问题。我们提出了SARGes，这是一个利用大型语言模型（LLMs）解析语音内容并生成可靠语义手势标签的新型框架，这些标签随后指导有意义同步语音手势的合成。为了实现这一目标，我们构建了一个全面的同步语音手势志数组，并开发了一种基于LLM的意图链推理机制，该机制系统地按照志数组标准解析和分解手势语义为结构化的推理步骤，有效地指导大型语言模型生成上下文感知的手势标签。随后，我们构建了一个带有意图链标注的文字到手势标签数据集，并训练了一个轻量级的手势标签生成模型，该模型随后指导生成具有可信度和语义一致性的同步语音手势。实验结果表明，SARGes实现了高度语义对齐的手势标注（准确率50.2%）和高效的单次推理（0.4秒）。所提出的方法为语义手势合成提供了可解释的意图推理路径。 

---
# Offline Reinforcement Learning with Discrete Diffusion Skills 

**Title (ZH)**: 离线强化学习中的离散扩散技能 

**Authors**: RuiXi Qiao, Jie Cheng, Xingyuan Dai, Yonglin Tian, Yisheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2503.20176)  

**Abstract**: Skills have been introduced to offline reinforcement learning (RL) as temporal abstractions to tackle complex, long-horizon tasks, promoting consistent behavior and enabling meaningful exploration. While skills in offline RL are predominantly modeled within a continuous latent space, the potential of discrete skill spaces remains largely underexplored. In this paper, we propose a compact discrete skill space for offline RL tasks supported by state-of-the-art transformer-based encoder and diffusion-based decoder. Coupled with a high-level policy trained via offline RL techniques, our method establishes a hierarchical RL framework where the trained diffusion decoder plays a pivotal role. Empirical evaluations show that the proposed algorithm, Discrete Diffusion Skill (DDS), is a powerful offline RL method. DDS performs competitively on Locomotion and Kitchen tasks and excels on long-horizon tasks, achieving at least a 12 percent improvement on AntMaze-v2 benchmarks compared to existing offline RL approaches. Furthermore, DDS offers improved interpretability, training stability, and online exploration compared to previous skill-based methods. 

**Abstract (ZH)**: 离线 reinforcement learning任务中基于Transformer编码器和扩散解码器的紧凑离散技能空间 

---
# Direct Post-Training Preference Alignment for Multi-Agent Motion Generation Models Using Implicit Feedback from Pre-training Demonstrations 

**Title (ZH)**: 直接训练后偏好对齐：利用预训练示范的隐式反馈进行多Agent运动生成模型的偏好对齐 

**Authors**: Ran Tian, Kratarth Goel  

**Link**: [PDF](https://arxiv.org/pdf/2503.20105)  

**Abstract**: Recent advancements in LLMs have revolutionized motion generation models in embodied applications. While LLM-type auto-regressive motion generation models benefit from training scalability, there remains a discrepancy between their token prediction objectives and human preferences. As a result, models pre-trained solely with token-prediction objectives often generate behaviors that deviate from what humans would prefer, making post-training preference alignment crucial for producing human-preferred motions. Unfortunately, post-training alignment requires extensive preference rankings of motions generated by the pre-trained model, which are costly to annotate, especially in multi-agent settings. Recently, there has been growing interest in leveraging pre-training demonstrations to scalably generate preference data for post-training alignment. However, these methods often adopt an adversarial assumption, treating all pre-trained model-generated samples as unpreferred examples. This adversarial approach overlooks the valuable signal provided by preference rankings among the model's own generations, ultimately reducing alignment effectiveness and potentially leading to misaligned behaviors. In this work, instead of treating all generated samples as equally bad, we leverage implicit preferences encoded in pre-training demonstrations to construct preference rankings among the pre-trained model's generations, offering more nuanced preference alignment guidance with zero human cost. We apply our approach to large-scale traffic simulation and demonstrate its effectiveness in improving the realism of pre-trained model's generated behaviors, making a lightweight 1M motion generation model comparable to SOTA large imitation-based models by relying solely on implicit feedback from pre-training demonstrations, without additional post-training human preference annotations or high computational costs. 

**Abstract (ZH)**: 近期大型语言模型的进展彻底改变了具身应用中的动作生成模型。虽然基于LLM的自回归动作生成模型受益于训练可扩展性，但它们的标记预测目标与人类偏好之间仍存在差异。因此，仅通过标记预测目标预训练的模型往往生成偏离人类偏好的行为，这使得后训练偏好对齐变得至关重要。不幸的是，后训练对齐需要大量标注预训练模型生成的动作偏好排名，这在多智能体设置中成本高昂。最近，利用预训练示范来大规模生成后训练偏好数据的兴趣日益增长。然而，这些方法通常采用对抗性假设，将预训练模型生成的所有样本视为不良示例。这种对抗性方法忽略了预训练模型自身生成动作之间的偏好排名提供的宝贵信号，最终降低了对齐效果，并可能导致不一致的行为。在本文中，我们不将所有生成样本视为同样糟糕，而是利用预训练示范中编码的隐式偏好来构造预训练模型生成动作之间的偏好排名，提供更为细腻的偏好对齐指导，且无需任何人工成本。我们将在大规模交通模拟中应用该方法，并通过仅依赖预训练示范中的隐式反馈，展示其在提高预训练模型生成行为逼真度方面的有效性，使一个轻量级的1M动作生成模型达到当前最佳大模型模仿基线的效果，无需额外的后训练人类偏好标注或高昂的计算成本。 

---
# Hyperdimensional Uncertainty Quantification for Multimodal Uncertainty Fusion in Autonomous Vehicles Perception 

**Title (ZH)**: 基于自主车辆感知中多模态不确定性融合的超维度不确定性量化 

**Authors**: Luke Chen, Junyao Wang, Trier Mortlock, Pramod Khargonekar, Mohammad Abdullah Al Faruque  

**Link**: [PDF](https://arxiv.org/pdf/2503.20011)  

**Abstract**: Uncertainty Quantification (UQ) is crucial for ensuring the reliability of machine learning models deployed in real-world autonomous systems. However, existing approaches typically quantify task-level output prediction uncertainty without considering epistemic uncertainty at the multimodal feature fusion level, leading to sub-optimal outcomes. Additionally, popular uncertainty quantification methods, e.g., Bayesian approximations, remain challenging to deploy in practice due to high computational costs in training and inference. In this paper, we propose HyperDUM, a novel deterministic uncertainty method (DUM) that efficiently quantifies feature-level epistemic uncertainty by leveraging hyperdimensional computing. Our method captures the channel and spatial uncertainties through channel and patch -wise projection and bundling techniques respectively. Multimodal sensor features are then adaptively weighted to mitigate uncertainty propagation and improve feature fusion. Our evaluations show that HyperDUM on average outperforms the state-of-the-art (SOTA) algorithms by up to 2.01%/1.27% in 3D Object Detection and up to 1.29% improvement over baselines in semantic segmentation tasks under various types of uncertainties. Notably, HyperDUM requires 2.36x less Floating Point Operations and up to 38.30x less parameters than SOTA methods, providing an efficient solution for real-world autonomous systems. 

**Abstract (ZH)**: 多模态特征融合中的不确定性量化：HyperDUM方法 

---
# Synthesizing world models for bilevel planning 

**Title (ZH)**: 合成层级规划的世界模型 

**Authors**: Zergham Ahmed, Joshua B. Tenenbaum, Christopher J. Bates, Samuel J. Gershman  

**Link**: [PDF](https://arxiv.org/pdf/2503.20124)  

**Abstract**: Modern reinforcement learning (RL) systems have demonstrated remarkable capabilities in complex environments, such as video games. However, they still fall short of achieving human-like sample efficiency and adaptability when learning new domains. Theory-based reinforcement learning (TBRL) is an algorithmic framework specifically designed to address this gap. Modeled on cognitive theories, TBRL leverages structured, causal world models - "theories" - as forward simulators for use in planning, generalization and exploration. Although current TBRL systems provide compelling explanations of how humans learn to play video games, they face several technical limitations: their theory languages are restrictive, and their planning algorithms are not scalable. To address these challenges, we introduce TheoryCoder, an instantiation of TBRL that exploits hierarchical representations of theories and efficient program synthesis methods for more powerful learning and planning. TheoryCoder equips agents with general-purpose abstractions (e.g., "move to"), which are then grounded in a particular environment by learning a low-level transition model (a Python program synthesized from observations by a large language model). A bilevel planning algorithm can exploit this hierarchical structure to solve large domains. We demonstrate that this approach can be successfully applied to diverse and challenging grid-world games, where approaches based on directly synthesizing a policy perform poorly. Ablation studies demonstrate the benefits of using hierarchical abstractions. 

**Abstract (ZH)**: 现代基于理论的强化学习（TBRL）系统已经在复杂环境中展示了显著的能力，如视频游戏。然而，当学习新领域时，它们在样本效率和适应性方面仍无法达到人类的水平。基于理论的强化学习（TBRL）是一种专门为此差距设计的算法框架。受认知理论的启发，TBRL利用结构化因果世界模型——“理论”——作为前向模拟器，用于规划、泛化和探索。尽管当前的TBRL系统为人类如何学习玩视频游戏提供了令人信服的解释，但它们面临一些技术局限性：其理论语言是限制性的，其规划算法也不具备可扩展性。为了解决这些挑战，我们引入了TheoryCoder，这是一种TBRL的实现，借助了层次化的理论表示和高效的程序合成方法，实现了更强大的学习和规划能力。TheoryCoder为代理提供了通用的抽象（如“移动到”），并通过学习低级过渡模型（由大语言模型从观察中合成的Python程序）将这些抽象具体化于特定环境之中。嵌套的规划算法可以利用这种层次结构解决大型领域。我们证明了这种方法可以成功应用于多样且具有挑战性的格状世界游戏，其中直接合成策略的方法表现不佳。消融研究表明，使用层次化抽象可以带来优势。 

---
