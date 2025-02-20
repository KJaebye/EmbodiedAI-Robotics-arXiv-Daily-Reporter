# NavigateDiff: Visual Predictors are Zero-Shot Navigation Assistants 

**Title (ZH)**: NavigateDiff: 可视化预测器是零样本导航助手 

**Authors**: Yiran Qin, Ao Sun, Yuze Hong, Benyou Wang, Ruimao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13894)  

**Abstract**: Navigating unfamiliar environments presents significant challenges for household robots, requiring the ability to recognize and reason about novel decoration and layout. Existing reinforcement learning methods cannot be directly transferred to new environments, as they typically rely on extensive mapping and exploration, leading to time-consuming and inefficient. To address these challenges, we try to transfer the logical knowledge and the generalization ability of pre-trained foundation models to zero-shot navigation. By integrating a large vision-language model with a diffusion network, our approach named \mname ~constructs a visual predictor that continuously predicts the agent's potential observations in the next step which can assist robots generate robust actions. Furthermore, to adapt the temporal property of navigation, we introduce temporal historical information to ensure that the predicted image is aligned with the navigation scene. We then carefully designed an information fusion framework that embeds the predicted future frames as guidance into goal-reaching policy to solve downstream image navigation tasks. This approach enhances navigation control and generalization across both simulated and real-world environments. Through extensive experimentation, we demonstrate the robustness and versatility of our method, showcasing its potential to improve the efficiency and effectiveness of robotic navigation in diverse settings. 

**Abstract (ZH)**: 导航陌生环境对家庭机器人提出了显著挑战，需要识别和推理关于新颖装饰和布局的能力。现有强化学习方法无法直接转移到新环境中，因为它们通常依赖于广泛的制图和探索，导致耗时且低效。为了解决这些挑战，我们尝试将预先训练的基础模型的逻辑知识和泛化能力转移到零样本导航中。通过将大型视觉语言模型与扩散网络结合，我们方法\mname构建了一个视觉预测器，该预测器连续预测代理在下一步中可能的观察结果，从而帮助机器人生成稳健的动作。此外，为了适应导航的时间特性，我们引入了时间历史信息以确保预测图像与导航场景对齐。然后，我们精心设计了一个信息融合框架，将预测的未来帧作为指导嵌入目标达成策略，以解决下游图像导航任务。该方法增强了导航控制和在模拟和实际环境中的泛化能力。通过广泛实验，我们展示了该方法的稳健性和通用性，展示了其在不同环境下的机器人导航效率和效果提升的潜力。 

---
# Active Illumination for Visual Ego-Motion Estimation in the Dark 

**Title (ZH)**: 黑暗环境下基于主动照明的视觉自我运动估计 

**Authors**: Francesco Crocetti, Alberto Dionigi, Raffaele Brilli, Gabriele Costante, Paolo Valigi  

**Link**: [PDF](https://arxiv.org/pdf/2502.13708)  

**Abstract**: Visual Odometry (VO) and Visual SLAM (V-SLAM) systems often struggle in low-light and dark environments due to the lack of robust visual features. In this paper, we propose a novel active illumination framework to enhance the performance of VO and V-SLAM algorithms in these challenging conditions. The developed approach dynamically controls a moving light source to illuminate highly textured areas, thereby improving feature extraction and tracking. Specifically, a detector block, which incorporates a deep learning-based enhancing network, identifies regions with relevant features. Then, a pan-tilt controller is responsible for guiding the light beam toward these areas, so that to provide information-rich images to the ego-motion estimation algorithm. Experimental results on a real robotic platform demonstrate the effectiveness of the proposed method, showing a reduction in the pose estimation error up to 75% with respect to a traditional fixed lighting technique. 

**Abstract (ZH)**: 基于主动照明的低光环境视觉里程计与定位性能提升方法 

---
# Human-Like Robot Impedance Regulation Skill Learning from Human-Human Demonstrations 

**Title (ZH)**: 人类般类人机器人阻抗调节技能学习从人-人示范 

**Authors**: Chenzui Li, Xi Wu, Junjia Liu, Tao Teng, Yiming Chen, Sylvain Calinon, Darwin Caldwell, Fei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13707)  

**Abstract**: Humans are experts in collaborating with others physically by regulating compliance behaviors based on the perception of their partner states and the task requirements. Enabling robots to develop proficiency in human collaboration skills can facilitate more efficient human-robot collaboration (HRC). This paper introduces an innovative impedance regulation skill learning framework for achieving HRC in multiple physical collaborative tasks. The framework is designed to adjust the robot compliance to the human partner states while adhering to reference trajectories provided by human-human demonstrations. Specifically, electromyography (EMG) signals from human muscles are collected and analyzed to extract limb impedance, representing compliance behaviors during demonstrations. Human endpoint motions are captured and represented using a probabilistic learning method to create reference trajectories and corresponding impedance profiles. Meanwhile, an LSTMbased module is implemented to develop task-oriented impedance regulation policies by mapping the muscle synergistic contributions between two demonstrators. Finally, we propose a wholebody impedance controller for a human-like robot, coordinating joint outputs to achieve the desired impedance and reference trajectory during task execution. Experimental validation was conducted through a collaborative transportation task and two interactive Tai Chi pushing hands tasks, demonstrating superior performance from the perspective of interactive forces compared to a constant impedance control method. 

**Abstract (ZH)**: 人类通过基于伙伴状态感知和任务需求调节合作行为，成为物理合作的专家。使机器人具备人类合作技能的 proficiency 能够促进更加高效的机器人-人类协作（HRC）。本文介绍了一种创新的阻抗调节技能学习框架，用于在多种物理合作任务中实现 HRC。该框架设计用于调整机器人阻抗以适应人类伙伴状态，同时遵循人类-人类演示提供的参考轨迹。具体而言，收集和分析人类肌肉的肌电图（EMG）信号以提取肢体阻抗，表示演示期间的合规行为。使用概率学习方法捕捉和表示人类末端运动，创建参考轨迹和相应的阻抗配置文件。同时，实现一个基于 LSTM 的模块，通过映射两名示范者之间的肌肉协同贡献开发任务导向的阻抗调节策略。最后，提出了一种全身阻抗控制器，用于类人机器人，协调关节输出以在任务执行期间实现所需的阻抗和参考轨迹。实验验证通过协作运输任务和两个互动太极推手任务进行，从交互力的角度展示了优于恒定阻抗控制方法的性能。 

---
# A Framework for Semantics-based Situational Awareness during Mobile Robot Deployments 

**Title (ZH)**: 基于语义的移动机器人部署情境awareness框架 

**Authors**: Tianshu Ruan, Aniketh Ramesh, Hao Wang, Alix Johnstone-Morfoisse, Gokcenur Altindal, Paul Norman, Grigoris Nikolaou, Rustam Stolkin, Manolis Chiou  

**Link**: [PDF](https://arxiv.org/pdf/2502.13677)  

**Abstract**: Deployment of robots into hazardous environments typically involves a ``Human-Robot Teaming'' (HRT) paradigm, in which a human supervisor interacts with a remotely operating robot inside the hazardous zone. Situational Awareness (SA) is vital for enabling HRT, to support navigation, planning, and decision-making. This paper explores issues of higher-level ``semantic'' information and understanding in SA. In semi-autonomous, or variable-autonomy paradigms, different types of semantic information may be important, in different ways, for both the human operator and an autonomous agent controlling the robot. We propose a generalizable framework for acquiring and combining multiple modalities of semantic-level SA during remote deployments of mobile robots. We demonstrate the framework with an example application of search and rescue (SAR) in disaster response robotics. We propose a set of ``environment semantic indicators" that can reflect a variety of different types of semantic information, e.g. indicators of risk, or signs of human activity, as the robot encounters different scenes. Based on these indicators, we propose a metric to describe the overall situation of the environment called ``Situational Semantic Richness (SSR)". This metric combines multiple semantic indicators to summarise the overall situation. The SSR indicates if an information-rich and complex situation has been encountered, which may require advanced reasoning for robots and humans and hence the attention of the expert human operator. The framework is tested on a Jackal robot in a mock-up disaster response environment. Experimental results demonstrate that the proposed semantic indicators are sensitive to changes in different modalities of semantic information in different scenes, and the SSR metric reflects overall semantic changes in the situations encountered. 

**Abstract (ZH)**: 机器人在危险环境中的部署通常涉及“人机协同作业”（HRT）范式，在该范式中，人类监督员与远程操作的机器人在危险区域内进行交互。态势感知（SA）对于支撑导航、规划和决策至关重要。本文探讨了较高层次的“语义”信息及其理解在SA中的问题。在半自主或可变自主范式中，不同的语义信息可能以不同方式对人类操作员和控制机器人的自主代理体重要。我们提出了一种框架，用于在远程部署移动机器人期间获取和结合多种模态的语义级态势感知信息。我们通过救灾机器人领域的一个示例应用演示了该框架。我们提出了一组“环境语义指标”，这些指标能够反映机器人遇到的不同场景中的多种不同类型的语义信息，例如风险指标或人类活动迹象。基于这些指标，我们提出了一种描述环境整体情况的度量标准，称为“态势语义丰富度（SSR）”。SSR指标结合了多种语义指标，以总结整体情况。SSR指标可以指示是否遇到了信息丰富且复杂的环境情况，这可能需要机器人和人类进行高级推理，因而需要专家人类操作员的关注。该框架在模拟灾难响应环境的实验中在Jackal机器人上进行了测试。实验结果表明，提出的语义指标对不同场景中不同模态的语义信息的变化敏感，而SSR度量标准反映了遇到的环境整体语义变化。 

---
# VLAS: Vision-Language-Action Model With Speech Instructions For Customized Robot Manipulation 

**Title (ZH)**: VLAS：带有语音指令的视觉-语言-动作模型及其在定制化机器人 manipulation 中的应用 

**Authors**: Wei Zhao, Pengxiang Ding, Min Zhang, Zhefei Gong, Shuanghao Bai, Han Zhao, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13508)  

**Abstract**: Vision-language-action models (VLAs) have become increasingly popular in robot manipulation for their end-to-end design and remarkable performance. However, existing VLAs rely heavily on vision-language models (VLMs) that only support text-based instructions, neglecting the more natural speech modality for human-robot interaction. Traditional speech integration methods usually involves a separate speech recognition system, which complicates the model and introduces error propagation. Moreover, the transcription procedure would lose non-semantic information in the raw speech, such as voiceprint, which may be crucial for robots to successfully complete customized tasks. To overcome above challenges, we propose VLAS, a novel end-to-end VLA that integrates speech recognition directly into the robot policy model. VLAS allows the robot to understand spoken commands through inner speech-text alignment and produces corresponding actions to fulfill the task. We also present two new datasets, SQA and CSI, to support a three-stage tuning process for speech instructions, which empowers VLAS with the ability of multimodal interaction across text, image, speech, and robot actions. Taking a step further, a voice retrieval-augmented generation (RAG) paradigm is designed to enable our model to effectively handle tasks that require individual-specific knowledge. Our extensive experiments show that VLAS can effectively accomplish robot manipulation tasks with diverse speech commands, offering a seamless and customized interaction experience. 

**Abstract (ZH)**: 基于视觉-语言-动作模型的端到端语音集成：VLAS及其应用 

---
# Ephemerality meets LiDAR-based Lifelong Mapping 

**Title (ZH)**: ephemeral现象邂逅基于LiDAR的生命长映射 

**Authors**: Hyeonjae Gil, Dongjae Lee, Giseop Kim, Ayoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.13452)  

**Abstract**: Lifelong mapping is crucial for the long-term deployment of robots in dynamic environments. In this paper, we present ELite, an ephemerality-aided LiDAR-based lifelong mapping framework which can seamlessly align multiple session data, remove dynamic objects, and update maps in an end-to-end fashion. Map elements are typically classified as static or dynamic, but cases like parked cars indicate the need for more detailed categories than binary. Central to our approach is the probabilistic modeling of the world into two-stage $\textit{ephemerality}$, which represent the transiency of points in the map within two different time scales. By leveraging the spatiotemporal context encoded in ephemeralities, ELite can accurately infer transient map elements, maintain a reliable up-to-date static map, and improve robustness in aligning the new data in a more fine-grained manner. Extensive real-world experiments on long-term datasets demonstrate the robustness and effectiveness of our system. The source code is publicly available for the robotics community: this https URL. 

**Abstract (ZH)**: lifelong环境下基于LiDAR的终身映射框架ELite：通过ephemerality辅助的端到端地图更新 

---
# MapNav: A Novel Memory Representation via Annotated Semantic Maps for VLM-based Vision-and-Language Navigation 

**Title (ZH)**: MapNav：基于标注语义地图的新型记忆表示方法用于VLVL导航 

**Authors**: Lingfeng Zhang, Xiaoshuai Hao, Qinwen Xu, Qiang Zhang, Xinyao Zhang, Pengwei Wang, Jing Zhang, Zhongyuan Wang, Shanghang Zhang, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13451)  

**Abstract**: Vision-and-language navigation (VLN) is a key task in Embodied AI, requiring agents to navigate diverse and unseen environments while following natural language instructions. Traditional approaches rely heavily on historical observations as spatio-temporal contexts for decision making, leading to significant storage and computational overhead. In this paper, we introduce MapNav, a novel end-to-end VLN model that leverages Annotated Semantic Map (ASM) to replace historical frames. Specifically, our approach constructs a top-down semantic map at the start of each episode and update it at each timestep, allowing for precise object mapping and structured navigation information. Then, we enhance this map with explicit textual labels for key regions, transforming abstract semantics into clear navigation cues and generate our ASM. MapNav agent using the constructed ASM as input, and use the powerful end-to-end capabilities of VLM to empower VLN. Extensive experiments demonstrate that MapNav achieves state-of-the-art (SOTA) performance in both simulated and real-world environments, validating the effectiveness of our method. Moreover, we will release our ASM generation source code and dataset to ensure reproducibility, contributing valuable resources to the field. We believe that our proposed MapNav can be used as a new memory representation method in VLN, paving the way for future research in this field. 

**Abstract (ZH)**: 基于标注语义地图的端到端视觉-语言导航（MapNav） 

---
# Exploring Embodied Emotional Communication: A Human-oriented Review of Mediated Social Touch 

**Title (ZH)**: 探索具身情感交流：以人为本的介导社会触碰综述 

**Authors**: Liwen He, Zichun Guo, Yanru Mo, Yue Wen, Yun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13816)  

**Abstract**: This paper offers a structured understanding of mediated social touch (MST) using a human-oriented approach, through an extensive review of literature spanning tactile interfaces, emotional information, mapping mechanisms, and the dynamics of human-human and human-robot interactions. By investigating the existing and exploratory mapping strategies of the 37 selected MST cases, we established the emotional expression space of MSTs that accommodated a diverse spectrum of emotions by integrating the categorical and Valence-arousal models, showcasing how emotional cues can be translated into tactile signals. Based on the expressive capacity of MSTs, a practical design space was structured encompassing factors such as the body locations, device form, tactile modalities, and parameters. We also proposed various design strategies for MSTs including workflow, evaluation methods, and ethical and cultural considerations, as well as several future research directions. MSTs' potential is reflected not only in conveying emotional information but also in fostering empathy, comfort, and connection in both human-human and human-robot interactions. This paper aims to serve as a comprehensive reference for design researchers and practitioners, which helps expand the scope of emotional communication of MSTs, facilitating the exploration of diverse applications of affective haptics, and enhancing the naturalness and sociability of haptic interaction. 

**Abstract (ZH)**: 本文采用以人为本的方法，通过广泛Review触觉界面、情感信息、映射机制以及人-人和人-机互动的动力学等内容，构建了中介社交触觉（MST）的结构化理解。通过对选定的37例MST案例中现有和探索性映射策略进行调查，我们建立了MST的情感表达空间，该空间通过整合类别和唤醒模型涵盖了广泛的情感谱系，展示了如何将情感提示转换为触觉信号。基于MST的情感表达能力，我们构建了一个实用的设计空间，包含了身体位置、设备形态、触觉模式和参数等因素。我们还提出了MST的各种设计策略，包括工作流程、评估方法、伦理与文化考量，以及若干未来研究方向。MST不仅在传递情感信息方面具有潜力，还能在人-人和人-机互动中促进同理心、舒适感和联系。本文旨在为设计研究人员和实践者提供一个全面的参考，有助于扩展MST的情感交流范围，促进情感化触觉多样应用的探索，并提高触觉互动的自然性和社交性。 

---
# A Survey of Sim-to-Real Methods in RL: Progress, Prospects and Challenges with Foundation Models 

**Title (ZH)**: 基础模型视角下的RL中从模拟到现实方法综述：进展、前景与挑战 

**Authors**: Longchao Da, Justin Turnau, Thirulogasankar Pranav Kutralingam, Alvaro Velasquez, Paulo Shakarian, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.13187)  

**Abstract**: Deep Reinforcement Learning (RL) has been explored and verified to be effective in solving decision-making tasks in various domains, such as robotics, transportation, recommender systems, etc. It learns from the interaction with environments and updates the policy using the collected experience. However, due to the limited real-world data and unbearable consequences of taking detrimental actions, the learning of RL policy is mainly restricted within the simulators. This practice guarantees safety in learning but introduces an inevitable sim-to-real gap in terms of deployment, thus causing degraded performance and risks in execution. There are attempts to solve the sim-to-real problems from different domains with various techniques, especially in the era with emerging techniques such as large foundations or language models that have cast light on the sim-to-real. This survey paper, to the best of our knowledge, is the first taxonomy that formally frames the sim-to-real techniques from key elements of the Markov Decision Process (State, Action, Transition, and Reward). Based on the framework, we cover comprehensive literature from the classic to the most advanced methods including the sim-to-real techniques empowered by foundation models, and we also discuss the specialties that are worth attention in different domains of sim-to-real problems. Then we summarize the formal evaluation process of sim-to-real performance with accessible code or benchmarks. The challenges and opportunities are also presented to encourage future exploration of this direction. We are actively maintaining a to include the most up-to-date sim-to-real research outcomes to help the researchers in their work. 

**Abstract (ZH)**: 深强化学习的仿真到现实问题：马尔可夫决策过程关键元素的分类综述 

---
# Towards Robust and Secure Embodied AI: A Survey on Vulnerabilities and Attacks 

**Title (ZH)**: 面向鲁棒性和安全性的实体AI：脆弱性和攻击综述 

**Authors**: Wenpeng Xing, Minghao Li, Mohan Li, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.13175)  

**Abstract**: Embodied AI systems, including robots and autonomous vehicles, are increasingly integrated into real-world applications, where they encounter a range of vulnerabilities stemming from both environmental and system-level factors. These vulnerabilities manifest through sensor spoofing, adversarial attacks, and failures in task and motion planning, posing significant challenges to robustness and safety. Despite the growing body of research, existing reviews rarely focus specifically on the unique safety and security challenges of embodied AI systems. Most prior work either addresses general AI vulnerabilities or focuses on isolated aspects, lacking a dedicated and unified framework tailored to embodied AI. This survey fills this critical gap by: (1) categorizing vulnerabilities specific to embodied AI into exogenous (e.g., physical attacks, cybersecurity threats) and endogenous (e.g., sensor failures, software flaws) origins; (2) systematically analyzing adversarial attack paradigms unique to embodied AI, with a focus on their impact on perception, decision-making, and embodied interaction; (3) investigating attack vectors targeting large vision-language models (LVLMs) and large language models (LLMs) within embodied systems, such as jailbreak attacks and instruction misinterpretation; (4) evaluating robustness challenges in algorithms for embodied perception, decision-making, and task planning; and (5) proposing targeted strategies to enhance the safety and reliability of embodied AI systems. By integrating these dimensions, we provide a comprehensive framework for understanding the interplay between vulnerabilities and safety in embodied AI. 

**Abstract (ZH)**: 具身AI系统的体载智能系统，包括机器人和自动驾驶车辆，正越来越多地集成到现实世界应用中，它们面临着来自环境和系统层面因素的一系列漏洞。这些漏洞通过传感器欺骗、恶意攻击以及任务和运动规划失效等形式表现出来，对系统的鲁棒性和安全性构成了重大挑战。尽管已经有大量研究，但现有综述较少专门关注具身AI系统的独特安全和安全挑战。大多数先前的工作要么研究通用AI漏洞，要么仅关注孤立方面，缺乏一个专门针对具身AI的统一框架。本文综述填补了这一关键空白，通过以下方式：（1）将具身AI特有的漏洞分类为外生来源（如物理攻击、网络安全威胁）和内生来源（如传感器故障、软件缺陷）；（2）系统分析具身AI特有的攻击范式，重点关注其对感知、决策和具身交互的影响；（3）研究针对具身系统中大型视觉语言模型（LVLM）和大型语言模型（LLM）的攻击向量，如逃逸攻击和指令误解释；（4）评估具身感知、决策和任务规划算法的鲁棒性挑战；（5）提出针对性策略以增强具身AI系统的安全性和可靠性。通过整合这些维度，我们提供了一个综合框架，用于理解具身AI中漏洞与安全之间的相互作用。 

---
# Model Evolution Framework with Genetic Algorithm for Multi-Task Reinforcement Learning 

**Title (ZH)**: 基于遗传算法的多任务 reinforcement learning 模型演化框架 

**Authors**: Yan Yu, Wengang Zhou, Yaodong Yang, Wanxuan Lu, Yingyan Hou, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.13569)  

**Abstract**: Multi-task reinforcement learning employs a single policy to complete various tasks, aiming to develop an agent with generalizability across different scenarios. Given the shared characteristics of tasks, the agent's learning efficiency can be enhanced through parameter sharing. Existing approaches typically use a routing network to generate specific routes for each task and reconstruct a set of modules into diverse models to complete multiple tasks simultaneously. However, due to the inherent difference between tasks, it is crucial to allocate resources based on task difficulty, which is constrained by the model's structure. To this end, we propose a Model Evolution framework with Genetic Algorithm (MEGA), which enables the model to evolve during training according to the difficulty of the tasks. When the current model is insufficient for certain tasks, the framework will automatically incorporate additional modules, enhancing the model's capabilities. Moreover, to adapt to our model evolution framework, we introduce a genotype module-level model, using binary sequences as genotype policies for model reconstruction, while leveraging a non-gradient genetic algorithm to optimize these genotype policies. Unlike routing networks with fixed output dimensions, our approach allows for the dynamic adjustment of the genotype policy length, enabling it to accommodate models with a varying number of modules. We conducted experiments on various robotics manipulation tasks in the Meta-World benchmark. Our state-of-the-art performance demonstrated the effectiveness of the MEGA framework. We will release our source code to the public. 

**Abstract (ZH)**: 基于遗传算法的模型进化框架（MEGA）在多任务强化学习中的应用 

---
# Integration of Agentic AI with 6G Networks for Mission-Critical Applications: Use-case and Challenges 

**Title (ZH)**: 将代理人工智能与6G网络集成以应用于关键任务应用：案例研究与挑战 

**Authors**: Sunder Ali Khowaja, Kapal Dev, Muhammad Salman Pathan, Engin Zeydan, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2502.13476)  

**Abstract**: We are in a transformative era, and advances in Artificial Intelligence (AI), especially the foundational models, are constantly in the news. AI has been an integral part of many applications that rely on automation for service delivery, and one of them is mission-critical public safety applications. The problem with AI-oriented mission-critical applications is the humanin-the-loop system and the lack of adaptability to dynamic conditions while maintaining situational awareness. Agentic AI (AAI) has gained a lot of attention recently due to its ability to analyze textual data through a contextual lens while quickly adapting to conditions. In this context, this paper proposes an AAI framework for mission-critical applications. We propose a novel framework with a multi-layer architecture to realize the AAI. We also present a detailed implementation of AAI layer that bridges the gap between network infrastructure and missioncritical applications. Our preliminary analysis shows that the AAI reduces initial response time by 5.6 minutes on average, while alert generation time is reduced by 15.6 seconds on average and resource allocation is improved by up to 13.4%. We also show that the AAI methods improve the number of concurrent operations by 40, which reduces the recovery time by up to 5.2 minutes. Finally, we highlight some of the issues and challenges that need to be considered when implementing AAI frameworks. 

**Abstract (ZH)**: 面向关键任务应用的代理人工智能框架 

---
# Atomic Proximal Policy Optimization for Electric Robo-Taxi Dispatch and Charger Allocation 

**Title (ZH)**: 原子近端策略优化在电动机器人出租车调度与充电站分配中的应用 

**Authors**: Jim Dai, Manxi Wu, Zhanhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13392)  

**Abstract**: Pioneering companies such as Waymo have deployed robo-taxi services in several U.S. cities. These robo-taxis are electric vehicles, and their operations require the joint optimization of ride matching, vehicle repositioning, and charging scheduling in a stochastic environment. We model the operations of the ride-hailing system with robo-taxis as a discrete-time, average reward Markov Decision Process with infinite horizon. As the fleet size grows, the dispatching is challenging as the set of system state and the fleet dispatching action set grow exponentially with the number of vehicles. To address this, we introduce a scalable deep reinforcement learning algorithm, called Atomic Proximal Policy Optimization (Atomic-PPO), that reduces the action space using atomic action decomposition. We evaluate our algorithm using real-world NYC for-hire vehicle data and we measure the performance using the long-run average reward achieved by the dispatching policy relative to a fluid-based reward upper bound. Our experiments demonstrate the superior performance of our Atomic-PPO compared to benchmarks. Furthermore, we conduct extensive numerical experiments to analyze the efficient allocation of charging facilities and assess the impact of vehicle range and charger speed on fleet performance. 

**Abstract (ZH)**: Waymo等先驱公司在多个美国城市部署了ロbo-taxi服务。这些ロbo-taxi是电动车辆，其运营需要在一个随机环境中对行程匹配、车辆重新定位和充电调度进行联合优化。我们将配备ロbo-taxi的叫车系统建模为一个离散时间的平均奖励马尔可夫决策过程（无穷 horizons）。随着车队规模的增加，调度变得具有挑战性，因为系统状态集和调度动作集会随车辆数量的增加呈指数级增长。为此，我们引入了一种可扩展的深度强化学习算法——原子近端策略优化（Atomic-PPO），该算法通过原子动作分解减少了动作空间。我们使用纽约市的实际情况数据评估了该算法，并使用调度策略相对于基于流的奖励上界的长期平均奖励来衡量性能。我们的实验表明，相比基准算法，我们的Atomic-PPO表现出更优的性能。此外，我们进行了广泛的数值实验，分析了充电设施的高效分配，并评估了车辆续航能力和充电速度对车队性能的影响。 

---
# Fighter Jet Navigation and Combat using Deep Reinforcement Learning with Explainable AI 

**Title (ZH)**: 基于可解释AI的战斗机导航与战斗深度强化学习方法 

**Authors**: Swati Kar, Soumyabrata Dey, Mahesh K Banavar, Shahnewaz Karim Sakib  

**Link**: [PDF](https://arxiv.org/pdf/2502.13373)  

**Abstract**: This paper presents the development of an Artificial Intelligence (AI) based fighter jet agent within a customized Pygame simulation environment, designed to solve multi-objective tasks via deep reinforcement learning (DRL). The jet's primary objectives include efficiently navigating the environment, reaching a target, and selectively engaging or evading an enemy. A reward function balances these goals while optimized hyperparameters enhance learning efficiency. Results show more than 80\% task completion rate, demonstrating effective decision-making. To enhance transparency, the jet's action choices are analyzed by comparing the rewards of the actual chosen action (factual action) with those of alternate actions (counterfactual actions), providing insights into the decision-making rationale. This study illustrates DRL's potential for multi-objective problem-solving with explainable AI. Project page is available at: \href{this https URL}{Project GitHub Link}. 

**Abstract (ZH)**: 基于定制Pygame仿真环境的AI战斗机代理开发：通过深度强化学习解决多目标任务 

---
# A Training-Free Framework for Precise Mobile Manipulation of Small Everyday Objects 

**Title (ZH)**: 无需训练的框架以精确操控日常小型物体 

**Authors**: Arjun Gupta, Rishik Sathua, Saurabh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2502.13964)  

**Abstract**: Many everyday mobile manipulation tasks require precise interaction with small objects, such as grasping a knob to open a cabinet or pressing a light switch. In this paper, we develop Servoing with Vision Models (SVM), a closed-loop training-free framework that enables a mobile manipulator to tackle such precise tasks involving the manipulation of small objects. SVM employs an RGB-D wrist camera and uses visual servoing for control. Our novelty lies in the use of state-of-the-art vision models to reliably compute 3D targets from the wrist image for diverse tasks and under occlusion due to the end-effector. To mitigate occlusion artifacts, we employ vision models to out-paint the end-effector thereby significantly enhancing target localization. We demonstrate that aided by out-painting methods, open-vocabulary object detectors can serve as a drop-in module to identify semantic targets (e.g. knobs) and point tracking methods can reliably track interaction sites indicated by user clicks. This training-free method obtains an 85% zero-shot success rate on manipulating unseen objects in novel environments in the real world, outperforming an open-loop control method and an imitation learning baseline trained on 1000+ demonstrations by an absolute success rate of 50%. 

**Abstract (ZH)**: 基于视觉模型的服务机器人微观操作框架（Servoing with Vision Models for Precision Manipulation of Small Objects） 

---
# Quantifying Memorization and Retriever Performance in Retrieval-Augmented Vision-Language Models 

**Title (ZH)**: 量化 Retrieval-Augmented Vision-Language 模型中记忆化和检索器性能 

**Authors**: Peter Carragher, Abhinand Jha, R Raghav, Kathleen M. Carley  

**Link**: [PDF](https://arxiv.org/pdf/2502.13836)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities in question answering (QA), but metrics for assessing their reliance on memorization versus retrieval remain underdeveloped. Moreover, while finetuned models are state-of-the-art on closed-domain tasks, general-purpose models like GPT-4o exhibit strong zero-shot performance. This raises questions about the trade-offs between memorization, generalization, and retrieval. In this work, we analyze the extent to which multimodal retrieval-augmented VLMs memorize training data compared to baseline VLMs. Using the WebQA benchmark, we contrast finetuned models with baseline VLMs on multihop retrieval and question answering, examining the impact of finetuning on data memorization. To quantify memorization in end-to-end retrieval and QA systems, we propose several proxy metrics by investigating instances where QA succeeds despite retrieval failing. Our results reveal the extent to which finetuned models rely on memorization. In contrast, retrieval-augmented VLMs have lower memorization scores, at the cost of accuracy (72% vs 52% on WebQA test set). As such, our measures pose a challenge for future work to reconcile memorization and generalization in both Open-Domain QA and joint Retrieval-QA tasks. 

**Abstract (ZH)**: 大型语言模型在问答任务中展现出显著的能力，但评估其依赖记忆而非检索的度量仍然发展不足。此外，虽然微调模型在封闭域任务中表现卓越，通用型模型如GPT-4在零样本情况下表现出强大性能。这引发了记忆、泛化和检索间权衡关系的疑问。在本文中，我们分析了模式增强的多模态视觉语言模型在记忆训练数据方面相较于基线模型的程度。通过使用WebQA基准，我们将微调模型与基线视觉语言模型在多跳检索和问答任务中进行对比，探讨微调对数据记忆的影响。为量化端到端检索和问答系统中的记忆程度，我们提出了几种代理度量，通过调查检索失败但问答仍成功的情况。我们的结果显示，微调模型在多大程度上依赖记忆。相比之下，增强检索的视觉语言模型的记忆分数较低，但准确率较低（WebQA测试集上的准确率分别为72%和52%）。因此，我们的指标对后续研究提出了挑战，要求在开放式领域问答和联合检索-问答任务中解决记忆和泛化之间的平衡。 

---
# MILE: Model-based Intervention Learning 

**Title (ZH)**: 基于模型的干预学习 

**Authors**: Yigit Korkmaz, Erdem Bıyık  

**Link**: [PDF](https://arxiv.org/pdf/2502.13519)  

**Abstract**: Imitation learning techniques have been shown to be highly effective in real-world control scenarios, such as robotics. However, these approaches not only suffer from compounding error issues but also require human experts to provide complete trajectories. Although there exist interactive methods where an expert oversees the robot and intervenes if needed, these extensions usually only utilize the data collected during intervention periods and ignore the feedback signal hidden in non-intervention timesteps. In this work, we create a model to formulate how the interventions occur in such cases, and show that it is possible to learn a policy with just a handful of expert interventions. Our key insight is that it is possible to get crucial information about the quality of the current state and the optimality of the chosen action from expert feedback, regardless of the presence or the absence of intervention. We evaluate our method on various discrete and continuous simulation environments, a real-world robotic manipulation task, as well as a human subject study. Videos and the code can be found at this https URL . 

**Abstract (ZH)**: 模仿学习技术在实际控制场景中，如机器人领域，已被证明非常有效。然而，这些方法不仅会遇到累积误差问题，还需要人类专家提供完整的轨迹。尽管存在一种交互方法，允许专家监督机器人并在必要时介入，但这些扩展通常仅利用干预期间收集的数据，并忽略了非干预时间段中隐含的反馈信号。本文中，我们创建了一个模型来描述在这种情况下干预的发生方式，并展示了只需少量专家干预就有可能学习到策略。我们的关键见解是，无论是否存在干预，都可以从专家反馈中获取有关当前状态质量和所选动作优化性的关键信息。我们先后在各种离散和连续模拟环境、一个真实的机器人操作任务以及一项人类受控试验中评估了我们的方法。相关视频和代码可在以下链接找到：this https URL。 

---
# Unlocking Multimodal Integration in EHRs: A Prompt Learning Framework for Language and Time Series Fusion 

**Title (ZH)**: 解锁多模态集成在EHR中的潜力：一种语言与时间序列融合的提示学习框架 

**Authors**: Shuai Niu, Jing Ma, Hongzhan Lin, Liang Bai, Zhihua Wang, Wei Bi, Yida Xu, Guo Li, Xian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13509)  

**Abstract**: Large language models (LLMs) have shown remarkable performance in vision-language tasks, but their application in the medical field remains underexplored, particularly for integrating structured time series data with unstructured clinical notes. In clinical practice, dynamic time series data such as lab test results capture critical temporal patterns, while clinical notes provide rich semantic context. Merging these modalities is challenging due to the inherent differences between continuous signals and discrete text. To bridge this gap, we introduce ProMedTS, a novel self-supervised multimodal framework that employs prompt-guided learning to unify these heterogeneous data types. Our approach leverages lightweight anomaly detection to generate anomaly captions that serve as prompts, guiding the encoding of raw time series data into informative embeddings. These embeddings are aligned with textual representations in a shared latent space, preserving fine-grained temporal nuances alongside semantic insights. Furthermore, our framework incorporates tailored self-supervised objectives to enhance both intra- and inter-modal alignment. We evaluate ProMedTS on disease diagnosis tasks using real-world datasets, and the results demonstrate that our method consistently outperforms state-of-the-art approaches. 

**Abstract (ZH)**: 大型语言模型（LLMs）在视觉语言任务中展现了卓越的表现，但在医疗领域的应用仍相对未被充分探索，特别是在将结构化时间序列数据与非结构化临床笔记整合方面。在临床实践中，动态时间序列数据，如实验室检测结果，捕捉关键的时间模式，而临床笔记则提供丰富的语义背景。由于连续信号与离散文本之间的固有差异，将这些模态数据融合极具挑战性。为填补这一空白，我们提出了ProMedTS，这是一种新颖的自监督多模态框架，利用提示引导学习来统一这些异构数据类型。我们的方法利用轻量级的异常检测生成异常描述作为提示，引导原始时间序列数据的编码为信息丰富的嵌入。这些嵌入在共享的潜在空间中与文本表示对齐，从而保留细微的时间细节与语义洞察。此外，我们的框架还整合了定制的自监督目标，以增强跨模态和模态内部的对齐。我们在实际数据集上评估了ProMedTS在疾病诊断任务中的表现，结果显示我们的方法在所有测试中均优于现有最先进的方法。 

---
# Learning To Explore With Predictive World Model Via Self-Supervised Learning 

**Title (ZH)**: 基于自监督学习的预测世界模型探索学习 

**Authors**: Alana Santana, Paula P. Costa, Esther L. Colombini  

**Link**: [PDF](https://arxiv.org/pdf/2502.13200)  

**Abstract**: Autonomous artificial agents must be able to learn behaviors in complex environments without humans to design tasks and rewards. Designing these functions for each environment is not feasible, thus, motivating the development of intrinsic reward functions. In this paper, we propose using several cognitive elements that have been neglected for a long time to build an internal world model for an intrinsically motivated agent. Our agent performs satisfactory iterations with the environment, learning complex behaviors without needing previously designed reward functions. We used 18 Atari games to evaluate what cognitive skills emerge in games that require reactive and deliberative behaviors. Our results show superior performance compared to the state-of-the-art in many test cases with dense and sparse rewards. 

**Abstract (ZH)**: 自主人工代理必须能够在没有人类设计任务和奖励的情况下，在复杂环境中学习行为。由于为每个环境设计这些功能是不可行的，因此推动了内在奖励函数的开发。本文我们提出使用长期被忽略的认知元素来为内在动机代理构建内部世界模型。我们的代理能够在与环境的满意交互中学习复杂行为，无需先前设计的奖励函数。我们使用18个雅达利游戏评估了所需反应性和深思熟虑行为游戏中涌现的认知技能。实验结果表明，与最先进的方法相比，在稠密和稀疏奖励的情况下，我们的方法在多种测试案例中表现出更优的性能。 

---
# Noumenal Labs White Paper: How To Build A Brain 

**Title (ZH)**: noumenal labs 白皮书：如何构建一个大脑 

**Authors**: Maxwell J. D. Ramstead, Candice Pattisapu, Jason Fox, Jeff Beck  

**Link**: [PDF](https://arxiv.org/pdf/2502.13161)  

**Abstract**: This white paper describes some of the design principles for artificial or machine intelligence that guide efforts at Noumenal Labs. These principles are drawn from both nature and from the means by which we come to represent and understand it. The end goal of research and development in this field should be to design machine intelligences that augment our understanding of the world and enhance our ability to act in it, without replacing us. In the first two sections, we examine the core motivation for our approach: resolving the grounding problem. We argue that the solution to the grounding problem rests in the design of models grounded in the world that we inhabit, not mere word models. A machine super intelligence that is capable of significantly enhancing our understanding of the human world must represent the world as we do and be capable of generating new knowledge, building on what we already know. In other words, it must be properly grounded and explicitly designed for rational, empirical inquiry, modeled after the scientific method. A primary implication of this design principle is that agents must be capable of engaging autonomously in causal physics discovery. We discuss the pragmatic implications of this approach, and in particular, the use cases in realistic 3D world modeling and multimodal, multidimensional time series analysis. 

**Abstract (ZH)**: 本白皮书描述了 Noumenal Labs 在设计人工或机器智能时遵循的一些设计原则，这些原则来源于自然及其表现和理解的方式。这一领域研究与开发的目标应该是设计能够增强我们对世界理解并提升我们在其中行动能力的机器智能，而不取代我们。在前两部分中，我们探讨了我们方法的核心动机：解决本体论问题。我们认为本体论问题的解决方案在于建立扎根于我们所居住的世界的模型，而不仅仅是语言模型。能够显著增强我们对人类世界理解的机器超级智能必须以我们的方式表示世界，并且能够生成新的知识，建立在我们已知的基础上。换句话说，它必须是适当扎根并明确为基于理性、实证探究的设计，模仿科学方法。这一设计原则的一个主要含义是代理必须能够自主进行因果物理学发现。我们讨论了这种方法的实际意义，特别是在现实3D世界建模和多模态多维度时间序列分析方面的应用场景。 

---
