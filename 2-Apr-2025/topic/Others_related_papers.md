# Energy Weighted Learning Progress Guided Interleaved Multi-Task Learning 

**Title (ZH)**: 能量加权学习进步引导的交错多任务学习 

**Authors**: Hanne Say, Suzan Ece Ada, Emre Ugur, Erhan Oztop  

**Link**: [PDF](https://arxiv.org/pdf/2504.00707)  

**Abstract**: Humans can continuously acquire new skills and knowledge by exploiting existing ones for improved learning, without forgetting them. Similarly, 'continual learning' in machine learning aims to learn new information while preserving the previously acquired knowledge. Existing research often overlooks the nature of human learning, where tasks are interleaved due to human choice or environmental constraints. So, almost never do humans master one task before switching to the next. To investigate to what extent human-like learning can benefit the learner, we propose a method that interleaves tasks based on their 'learning progress' and energy consumption. From a machine learning perspective, our approach can be seen as a multi-task learning system that balances learning performance with energy constraints while mimicking ecologically realistic human task learning. To assess the validity of our approach, we consider a robot learning setting in simulation, where the robot learns the effect of its actions in different contexts. The conducted experiments show that our proposed method achieves better performance than sequential task learning and reduces energy consumption for learning the tasks. 

**Abstract (ZH)**: 人类可以通过利用现有技能和知识来不断获取新技能和知识，从而提高学习效果，而不至于遗忘之前的知识。类似地，机器学习中的“持续学习”旨在学习新信息的同时保留之前获得的知识。现有研究往往忽视了人类学习的特性，由于人类选择或环境约束，任务往往是交错进行的。因此，人类几乎不会在一个任务完全掌握之后才转移到下一个任务。为了探查类似人类的学习方法能给学习者带来多大程度的好处，我们提出了一种根据“学习进度”和能量消耗交错任务的方法。从机器学习的角度来看，我们的方法可以视为一种平衡学习性能与能量约束的多任务学习系统，同时模拟了生态上现实的人类任务学习。为了评估我们方法的有效性，我们在一个模拟的机器人学习设置中进行了实验，该设置中机器人在不同的情境中学习其动作的影响。实验结果表明，我们提出的方法在任务学习方面优于顺序学习方法，并且减少了学习任务的能量消耗。 

---
# Provably Stable Multi-Agent Routing with Bounded-Delay Adversaries in the Decision Loop 

**Title (ZH)**: 可验证稳定多Agent路由：决策循环中的有界延迟对手 

**Authors**: Roee M. Francos, Daniel Garces, Stephanie Gil  

**Link**: [PDF](https://arxiv.org/pdf/2504.00863)  

**Abstract**: In this work, we are interested in studying multi-agent routing settings, where adversarial agents are part of the assignment and decision loop, degrading the performance of the fleet by incurring bounded delays while servicing pickup-and-delivery requests. Specifically, we are interested in characterizing conditions on the fleet size and the proportion of adversarial agents for which a routing policy remains stable, where stability for a routing policy is achieved if the number of outstanding requests is uniformly bounded over time. To obtain this characterization, we first establish a threshold on the proportion of adversarial agents above which previously stable routing policies for fully cooperative fleets are provably unstable. We then derive a sufficient condition on the fleet size to recover stability given a maximum proportion of adversarial agents. We empirically validate our theoretical results on a case study on autonomous taxi routing, where we consider transportation requests from real San Francisco taxicab data. 

**Abstract (ZH)**: 在本工作中，我们关注包含敌对代理的多代理路径规划设置，敌对代理会影响路径规划政策的稳定性，通过引入有界延迟来降低车队的服务性能，特别是在执行取送请求时。具体来说，我们关注在何种车队规模和敌对代理占比条件下，路径规划政策仍能保持稳定，即路径规划政策稳定指的是时间上未完成请求的数量是均匀有界的。为了获得这一特性，我们首先确定了一个敌对代理占比的阈值，在此阈值以上，原本对完全合作车队稳定的路径规划政策是不稳定的。随后，我们推导了在给定最大敌对代理占比条件下，确保路径规划政策稳定所需的车队规模条件。我们通过基于真实旧金山出租车数据的案例研究，实证验证了理论结果。 

---
# In-Context Learning for Zero-Shot Speed Estimation of BLDC motors 

**Title (ZH)**: 基于上下文学习的无监督BLDC电机速度估计 

**Authors**: Alessandro Colombo, Riccardo Busetto, Valentina Breschi, Marco Forgione, Dario Piga, Simone Formentin  

**Link**: [PDF](https://arxiv.org/pdf/2504.00673)  

**Abstract**: Accurate speed estimation in sensorless brushless DC motors is essential for high-performance control and monitoring, yet conventional model-based approaches struggle with system nonlinearities and parameter uncertainties. In this work, we propose an in-context learning framework leveraging transformer-based models to perform zero-shot speed estimation using only electrical measurements. By training the filter offline on simulated motor trajectories, we enable real-time inference on unseen real motors without retraining, eliminating the need for explicit system identification while retaining adaptability to varying operating conditions. Experimental results demonstrate that our method outperforms traditional Kalman filter-based estimators, especially in low-speed regimes that are crucial during motor startup. 

**Abstract (ZH)**: 基于变压器模型的上下文学习框架实现无传感器 Brushless DC 电机的精确速度估计 

---
# Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents 

**Title (ZH)**: Agent S2: 一种计算机使用代理的组合通用专家框架 

**Authors**: Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00906)  

**Abstract**: Computer use agents automate digital tasks by directly interacting with graphical user interfaces (GUIs) on computers and mobile devices, offering significant potential to enhance human productivity by completing an open-ended space of user queries. However, current agents face significant challenges: imprecise grounding of GUI elements, difficulties with long-horizon task planning, and performance bottlenecks from relying on single generalist models for diverse cognitive tasks. To this end, we introduce Agent S2, a novel compositional framework that delegates cognitive responsibilities across various generalist and specialist models. We propose a novel Mixture-of-Grounding technique to achieve precise GUI localization and introduce Proactive Hierarchical Planning, dynamically refining action plans at multiple temporal scales in response to evolving observations. Evaluations demonstrate that Agent S2 establishes new state-of-the-art (SOTA) performance on three prominent computer use benchmarks. Specifically, Agent S2 achieves 18.9% and 32.7% relative improvements over leading baseline agents such as Claude Computer Use and UI-TARS on the OSWorld 15-step and 50-step evaluation. Moreover, Agent S2 generalizes effectively to other operating systems and applications, surpassing previous best methods by 52.8% on WindowsAgentArena and by 16.52% on AndroidWorld relatively. Code available at this https URL. 

**Abstract (ZH)**: Agent S2：一种新型的 compositional 框架，通过分配认知责任来增强数字任务自动化 

---
# Example-Based Concept Analysis Framework for Deep Weather Forecast Models 

**Title (ZH)**: 基于示例的概念分析框架：深水气象预报模型 

**Authors**: Soyeon Kim, Junho Choi, Subeen Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00831)  

**Abstract**: To improve the trustworthiness of an AI model, finding consistent, understandable representations of its inference process is essential. This understanding is particularly important in high-stakes operations such as weather forecasting, where the identification of underlying meteorological mechanisms is as critical as the accuracy of the predictions. Despite the growing literature that addresses this issue through explainable AI, the applicability of their solutions is often limited due to their AI-centric development. To fill this gap, we follow a user-centric process to develop an example-based concept analysis framework, which identifies cases that follow a similar inference process as the target instance in a target model and presents them in a user-comprehensible format. Our framework provides the users with visually and conceptually analogous examples, including the probability of concept assignment to resolve ambiguities in weather mechanisms. To bridge the gap between vector representations identified from models and human-understandable explanations, we compile a human-annotated concept dataset and implement a user interface to assist domain experts involved in the the framework development. 

**Abstract (ZH)**: 提高AI模型可信度的关键在于找到其推理过程的一致且可理解的表示形式。这一理解在如天气预报等高风险操作中尤为重要，因为识别潜在的气象机制与预测的准确性一样重要。尽管已有大量关于此问题的可解释AI研究，但其解决方案的应用受限，因为这些解决方案多以AI为中心进行开发。为弥补这一差距，我们采用用户为中心的过程，开发了一种基于实例的概念分析框架。该框架识别出与目标模型中目标实例具有相似推理过程的案例，并以用户可理解的格式呈现。我们的框架提供了可视化和概念上类比的示例，包括概念的赋值概率，以解决天气机制中的模糊性。为了弥合来自模型的向量表示与人类可理解解释之间的差距，我们构建了一个由人工标注的概念数据集，并实现了一个用户界面，以辅助参与框架开发的领域专家。 

---
# Explainable AI-Based Interface System for Weather Forecasting Model 

**Title (ZH)**: 基于可解释AI的天气预报模型界面系统 

**Authors**: Soyeon Kim, Junho Choi, Yeji Choi, Subeen Lee, Artyom Stitsyuk, Minkyoung Park, Seongyeop Jeong, Youhyun Baek, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00795)  

**Abstract**: Machine learning (ML) is becoming increasingly popular in meteorological decision-making. Although the literature on explainable artificial intelligence (XAI) is growing steadily, user-centered XAI studies have not extend to this domain yet. This study defines three requirements for explanations of black-box models in meteorology through user studies: statistical model performance for different rainfall scenarios to identify model bias, model reasoning, and the confidence of model outputs. Appropriate XAI methods are mapped to each requirement, and the generated explanations are tested quantitatively and qualitatively. An XAI interface system is designed based on user feedback. The results indicate that the explanations increase decision utility and user trust. Users prefer intuitive explanations over those based on XAI algorithms even for potentially easy-to-recognize examples. These findings can provide evidence for future research on user-centered XAI algorithms, as well as a basis to improve the usability of AI systems in practice. 

**Abstract (ZH)**: 机器学习在气象决策中的应用日益增多。尽管可解释人工智能（XAI）的相关文献在稳步增长，但用户中心的XAI研究尚未扩展到该领域。本研究通过用户研究定义了气象中黑盒模型解释的三项要求：不同降雨情景下的统计模型性能以识别模型偏差、模型推理以及模型输出的信心。合适的方法被映射到每个要求，并生成的解释进行了定量和定性的测试。基于用户反馈设计了XAI界面系统。结果显示，这些解释增加了决策的价值和用户的信任。用户更偏好直观的解释，即使对于可能容易识别的例子也是如此。这些发现可以为未来用户中心的XAI算法研究提供证据，并为改进实践中人工智能系统的可用性提供基础。 

---
# Towards Responsible and Trustworthy Educational Data Mining: Comparing Symbolic, Sub-Symbolic, and Neural-Symbolic AI Methods 

**Title (ZH)**: 负责任且可信赖的教育数据挖掘：符号、亚符号及神经符号AI方法的比较 

**Authors**: Danial Hooshyar, Eve Kikas, Yeongwook Yang, Gustav Šír, Raija Hämäläinen, Tommi Kärkkäinen, Roger Azevedo  

**Link**: [PDF](https://arxiv.org/pdf/2504.00615)  

**Abstract**: Given the demand for responsible and trustworthy AI for education, this study evaluates symbolic, sub-symbolic, and neural-symbolic AI (NSAI) in terms of generalizability and interpretability. Our extensive experiments on balanced and imbalanced self-regulated learning datasets of Estonian primary school students predicting 7th-grade mathematics national test performance showed that symbolic and sub-symbolic methods performed well on balanced data but struggled to identify low performers in imbalanced datasets. Interestingly, symbolic and sub-symbolic methods emphasized different factors in their decision-making: symbolic approaches primarily relied on cognitive and motivational factors, while sub-symbolic methods focused more on cognitive aspects, learned knowledge, and the demographic variable of gender -- yet both largely overlooked metacognitive factors. The NSAI method, on the other hand, showed advantages by: (i) being more generalizable across both classes -- even in imbalanced datasets -- as its symbolic knowledge component compensated for the underrepresented class; and (ii) relying on a more integrated set of factors in its decision-making, including motivation, (meta)cognition, and learned knowledge, thus offering a comprehensive and theoretically grounded interpretability framework. These contrasting findings highlight the need for a holistic comparison of AI methods before drawing conclusions based solely on predictive performance. They also underscore the potential of hybrid, human-centered NSAI methods to address the limitations of other AI families and move us closer to responsible AI for education. Specifically, by enabling stakeholders to contribute to AI design, NSAI aligns learned patterns with theoretical constructs, incorporates factors like motivation and metacognition, and strengthens the trustworthiness and responsibility of educational data mining. 

**Abstract (ZH)**: 负责任和可信赖的AI在教育中的应用：符号、亚符号和神经符号AI的可泛化性和可解释性评价 

---
# Rack Position Optimization in Large-Scale Heterogeneous Data Centers 

**Title (ZH)**: 大规模异构数据中心机架位置优化 

**Authors**: Chang-Lin Chen, Jiayu Chen, Tian Lan, Zhaoxia Zhao, Hongbo Dong, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2504.00277)  

**Abstract**: As rapidly growing AI computational demands accelerate the need for new hardware installation and maintenance, this work explores optimal data center resource management by balancing operational efficiency with fault tolerance through strategic rack positioning considering diverse resources and locations. Traditional mixed-integer programming (MIP) approaches often struggle with scalability, while heuristic methods may result in significant sub-optimality. To address these issues, this paper presents a novel two-tier optimization framework using a high-level deep reinforcement learning (DRL) model to guide a low-level gradient-based heuristic for local search. The high-level DRL agent employs Leader Reward for optimal rack type ordering, and the low-level heuristic efficiently maps racks to positions, minimizing movement counts and ensuring fault-tolerant resource distribution. This approach allows scalability to over 100,000 positions and 100 rack types. Our method outperformed the gradient-based heuristic by 7\% on average and the MIP solver by over 30\% in objective value. It achieved a 100\% success rate versus MIP's 97.5\% (within a 20-minute limit), completing in just 2 minutes compared to MIP's 1630 minutes (i.e., almost 4 orders of magnitude improvement). Unlike the MIP solver, which showed performance variability under time constraints and high penalties, our algorithm consistently delivered stable, efficient results - an essential feature for large-scale data center management. 

**Abstract (ZH)**: 面向新型计算需求的数据中心资源管理优化：基于高阶深度强化学习的两层优化框架 

---
# The Axiom-Based Atlas: A Structural Mapping of Theorems via Foundational Proof Vectors 

**Title (ZH)**: 基于公理的图谱：通过基础证明向量的定理结构映射 

**Authors**: Harim Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2504.00063)  

**Abstract**: The Axiom-Based Atlas is a novel framework that structurally represents mathematical theorems as proof vectors over foundational axiom systems. By mapping the logical dependencies of theorems onto vectors indexed by axioms - such as those from Hilbert geometry, Peano arithmetic, or ZFC - we offer a new way to visualize, compare, and analyze mathematical knowledge. This vector-based formalism not only captures the logical foundation of theorems but also enables quantitative similarity metrics - such as cosine distance - between mathematical results, offering a new analytic layer for structural comparison. Using heatmaps, vector clustering, and AI-assisted modeling, this atlas enables the grouping of theorems by logical structure, not just by mathematical domain. We also introduce a prototype assistant (Atlas-GPT) that interprets natural language theorems and suggests likely proof vectors, supporting future applications in automated reasoning, mathematical education, and formal verification.
This direction is partially inspired by Terence Tao's recent reflections on the convergence of symbolic and structural mathematics. The Axiom-Based Atlas aims to provide a scalable, interpretable model of mathematical reasoning that is both human-readable and AI-compatible, contributing to the future landscape of formal mathematical systems. 

**Abstract (ZH)**: 基于公理的图谱：一种将数学定理结构化表示为公理基础公理系统上的证明向量的新框架。 

---
# IntrinsiX: High-Quality PBR Generation using Image Priors 

**Title (ZH)**: IntrinsiX: 使用图像先验的高质量物理基于渲染生成 

**Authors**: Peter Kocsis, Lukas Höllein, Matthias Nießner  

**Link**: [PDF](https://arxiv.org/pdf/2504.01008)  

**Abstract**: We introduce IntrinsiX, a novel method that generates high-quality intrinsic images from text description. In contrast to existing text-to-image models whose outputs contain baked-in scene lighting, our approach predicts physically-based rendering (PBR) maps. This enables the generated outputs to be used for content creation scenarios in core graphics applications that facilitate re-lighting, editing, and texture generation tasks. In order to train our generator, we exploit strong image priors, and pre-train separate models for each PBR material component (albedo, roughness, metallic, normals). We then align these models with a new cross-intrinsic attention formulation that concatenates key and value features in a consistent fashion. This allows us to exchange information between each output modality and to obtain semantically coherent PBR predictions. To ground each intrinsic component, we propose a rendering loss which provides image-space signals to constrain the model, thus facilitating sharp details also in the output BRDF properties. Our results demonstrate detailed intrinsic generation with strong generalization capabilities that outperforms existing intrinsic image decomposition methods used with generated images by a significant margin. Finally, we show a series of applications, including re-lighting, editing, and text-conditioned room-scale PBR texture generation. 

**Abstract (ZH)**: 我们引入了IntrinsiX，一种新颖的方法，能够从文本描述生成高质量的内在图像。与现有包含内置场景照明的文本到图像模型不同，我们的方法预测基于物理的渲染（PBR）图。这使得生成的输出能够用于核心图形应用程序中的内容创作场景，方便重新照明、编辑和纹理生成任务。为了训练我们的生成器，我们利用强大的图像先验知识，并为每种PBR材质成分（反射率、粗糙度、金属度、法线）分别预训练模型。然后，我们通过一种新的跨内在注意力形式化将这些模型与一致的关键特征和价值特征连接起来。这允许我们在每个输出模态之间交换信息，并获得语义上一致的PBR预测。为了使每个内在成分落地，我们提出了一个渲染损失，提供了图像空间信号来约束模型，从而在输出BRDF属性中实现清晰的细节。我们的结果展示了详细而具有强泛化能力的内在生成，明显优于现有用于生成图像的内在图像分解方法。最后，我们展示了包括重新照明、编辑以及条件文本的房间尺度PBR纹理生成等一系列应用。 

---
# Accelerating drug discovery with Artificial: a whole-lab orchestration and scheduling system for self-driving labs 

**Title (ZH)**: 使用人工智能加速药物发现：一个自动化实验室全流程调度系统 

**Authors**: Yao Fehlis, Paul Mandel, Charles Crain, Betty Liu, David Fuller  

**Link**: [PDF](https://arxiv.org/pdf/2504.00986)  

**Abstract**: Self-driving labs are transforming drug discovery by enabling automated, AI-guided experimentation, but they face challenges in orchestrating complex workflows, integrating diverse instruments and AI models, and managing data efficiently. Artificial addresses these issues with a comprehensive orchestration and scheduling system that unifies lab operations, automates workflows, and integrates AI-driven decision-making. By incorporating AI/ML models like NVIDIA BioNeMo - which facilitates molecular interaction prediction and biomolecular analysis - Artificial enhances drug discovery and accelerates data-driven research. Through real-time coordination of instruments, robots, and personnel, the platform streamlines experiments, enhances reproducibility, and advances drug discovery. 

**Abstract (ZH)**: 自驱动实验室正在通过实现自动化的、AI引导的实验来转变药物发现，但面临复杂工作流调度、多元仪器和AI模型集成以及数据管理的挑战。Artificial通过一个全面的调度和编排系统解决了这些问题，统一了实验室运营、自动化工作流和AI驱动的决策集成，提升了药物发现能力并加速了数据驱动的研究。通过实时协调仪器、机器人和人员，该平台简化了实验流程，提高了实验的可重复性，并推动了药物发现的进步。 

---
# Resource Allocation for RIS-Assisted CoMP-NOMA Networks using Reinforcement Learning 

**Title (ZH)**: RIS辅助协作多点传输和非正交多址网络的资源分配方法研究（使用强化学习） 

**Authors**: Muhammad Umer, Muhammad Ahmed Mohsin, Huma Ghafoor, Syed Ali Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00975)  

**Abstract**: This thesis delves into the forefront of wireless communication by exploring the synergistic integration of three transformative technologies: STAR-RIS, CoMP, and NOMA. Driven by the ever-increasing demand for higher data rates, improved spectral efficiency, and expanded coverage in the evolving landscape of 6G development, this research investigates the potential of these technologies to revolutionize future wireless networks.
The thesis analyzes the performance gains achievable through strategic deployment of STAR-RIS, focusing on mitigating inter-cell interference, enhancing signal strength, and extending coverage to cell-edge users. Resource sharing strategies for STAR-RIS elements are explored, optimizing both transmission and reflection functionalities. Analytical frameworks are developed to quantify the benefits of STAR-RIS assisted CoMP-NOMA networks under realistic channel conditions, deriving key performance metrics such as ergodic rates and outage probabilities. Additionally, the research delves into energy-efficient design approaches for CoMP-NOMA networks incorporating RIS, proposing novel RIS configurations and optimization algorithms to achieve a balance between performance and energy consumption. Furthermore, the application of Deep Reinforcement Learning (DRL) techniques for intelligent and adaptive optimization in aerial RIS-assisted CoMP-NOMA networks is explored, aiming to maximize network sum rate while meeting user quality of service requirements. Through a comprehensive investigation of these technologies and their synergistic potential, this thesis contributes valuable insights into the future of wireless communication, paving the way for the development of more efficient, reliable, and sustainable networks capable of meeting the demands of our increasingly connected world. 

**Abstract (ZH)**: 本论文探讨了无线通信前沿技术，通过研究STAR-RIS、CoMP和NOMA三种 transformative 技术的协同集成。受6G发展中对更高数据速率、更佳频谱效率和更广覆盖范围的不断增长需求驱动，本研究调查了这些技术未来无线网络革命化潜力。论文分析了通过战略性部署STAR-RIS实现的性能提升，重点关注干扰抑制、信号增强以及边缘用户的覆盖扩展。探讨了STAR-RIS元素的资源共享策略，优化了传输和反射功能。开发了在实际信道条件下量化STAR-RIS辅助CoMP-NOMA网络效益的分析框架，推导出了关键性能指标，如遍历速率和 outage 概率。此外，研究了结合RIS的CoMP-NOMA网络的节能设计方法，提出了新的RIS配置和优化算法，以实现性能与能耗之间的平衡。同时，探索了在空中RIS辅助CoMP-NOMA网络中应用深度强化学习（DRL）技术进行智能自适应优化的方法，旨在最大化网络总速率并满足用户服务质量要求。通过全面研究这些技术和它们的协同潜力，本论文为未来无线通信提供了宝贵的见解，为开发更高效、可靠和可持续的网络奠定了基础，以满足我们日益互联世界的需求。 

---
# Enabling Efficient Processing of Spiking Neural Networks with On-Chip Learning on Commodity Neuromorphic Processors for Edge AI Systems 

**Title (ZH)**: 在现货神经形态处理器上实现芯片内学习以高效处理脉冲神经网络，应用于边缘AI系统 

**Authors**: Rachmad Vidya Wicaksana Putra, Pasindu Wickramasinghe, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2504.00957)  

**Abstract**: The rising demand for energy-efficient edge AI systems (e.g., mobile agents/robots) has increased the interest in neuromorphic computing, since it offers ultra-low power/energy AI computation through spiking neural network (SNN) algorithms on neuromorphic processors. However, their efficient implementation strategy has not been comprehensively studied, hence limiting SNN deployments for edge AI systems. Toward this, we propose a design methodology to enable efficient SNN processing on commodity neuromorphic processors. To do this, we first study the key characteristics of targeted neuromorphic hardware (e.g., memory and compute budgets), and leverage this information to perform compatibility analysis for network selection. Afterward, we employ a mapping strategy for efficient SNN implementation on the targeted processor. Furthermore, we incorporate an efficient on-chip learning mechanism to update the systems' knowledge for adapting to new input classes and dynamic environments. The experimental results show that the proposed methodology leads the system to achieve low latency of inference (i.e., less than 50ms for image classification, less than 200ms for real-time object detection in video streaming, and less than 1ms in keyword recognition) and low latency of on-chip learning (i.e., less than 2ms for keyword recognition), while incurring less than 250mW of processing power and less than 15mJ of energy consumption across the respective different applications and scenarios. These results show the potential of the proposed methodology in enabling efficient edge AI systems for diverse application use-cases. 

**Abstract (ZH)**: 边缘AI系统中能源效率提升的需求促使了类脑计算的兴趣增加，类脑计算通过神经形态处理器上的脉冲神经网络（SNN）算法提供了超低功耗的AI计算。然而，其高效的实现策略尚未得到全面研究，限制了SNN在边缘AI系统中的部署。为此，我们提出了一种设计方法，以在商用神经形态处理器上实现高效的SNN处理。为此，我们首先研究了目标神经形态硬件的关键特性（如存储和计算预算），并利用这些信息进行网络选择的兼容性分析。随后，我们采用了一种映射策略，以在目标处理器上高效实现SNN。此外，我们引入了一种高效的片内学习机制，以更新系统的知识，使其能够适应新的输入类别和动态环境。实验结果显示，所提出的方法使系统实现了较低的推理延迟（如图像分类少于50毫秒，视频流中实时物体检测少于200毫秒，关键词识别少于1毫秒）和较低的片内学习延迟（如关键词识别少于2毫秒），同时消耗的处理功率少于250毫瓦，能量消耗少于15毫焦，适用于不同的应用和场景。这些结果展示了所提出方法在为多种应用场景提供高效边缘AI系统方面的潜力。 

---
# Unfair Learning: GenAI Exceptionalism and Copyright Law 

**Title (ZH)**: 不公平的学习：GenAI例外主义与版权法 

**Authors**: David Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2504.00955)  

**Abstract**: This paper challenges the argument that generative artificial intelligence (GenAI) is entitled to broad immunity from copyright law for reproducing copyrighted works without authorization due to a fair use defense. It examines fair use legal arguments and eight distinct substantive arguments, contending that every legal and substantive argument favoring fair use for GenAI applies equally, if not more so, to humans. Therefore, granting GenAI exceptional privileges in this domain is legally and logically inconsistent with withholding broad fair use exemptions from individual humans. It would mean no human would need to pay for virtually any copyright work again. The solution is to take a circumspect view of any fair use claim for mass copyright reproduction by any entity and focus on the first principles of whether permitting such exceptionalism for GenAI promotes science and the arts. 

**Abstract (ZH)**: 本文挑战了生成式人工智能（GenAI）在未经授权复制受版权保护的作品时，因公平使用抗辩而享有广泛版权法律豁免权的说法。本文考察了公平使用法律论点和八项具体的实质论点，认为支持GenAI公平使用的每一项法律和实质论点同样适用于个人，甚至更为适用。因此，给予GenAI在这方面享有特权与对个人不予广泛公平使用豁免是法律和逻辑上不一致的。这样意味着人类无需再次为几乎任何版权作品付费。解决方案是谨慎对待任何实体大规模版权复制的公平使用主张，并关注允许这种例外主义是否促进科学和艺术发展。 

---
# Personalized Federated Training of Diffusion Models with Privacy Guarantees 

**Title (ZH)**: 带有隐私保证的个性化联邦扩散模型训练 

**Authors**: Kumar Kshitij Patel, Weitong Zhang, Lingxiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00952)  

**Abstract**: The scarcity of accessible, compliant, and ethically sourced data presents a considerable challenge to the adoption of artificial intelligence (AI) in sensitive fields like healthcare, finance, and biomedical research. Furthermore, access to unrestricted public datasets is increasingly constrained due to rising concerns over privacy, copyright, and competition. Synthetic data has emerged as a promising alternative, and diffusion models -- a cutting-edge generative AI technology -- provide an effective solution for generating high-quality and diverse synthetic data. In this paper, we introduce a novel federated learning framework for training diffusion models on decentralized private datasets. Our framework leverages personalization and the inherent noise in the forward diffusion process to produce high-quality samples while ensuring robust differential privacy guarantees. Our experiments show that our framework outperforms non-collaborative training methods, particularly in settings with high data heterogeneity, and effectively reduces biases and imbalances in synthetic data, resulting in fairer downstream models. 

**Abstract (ZH)**: 可访问、合规且伦理来源数据的稀缺性对医疗、金融和生物医学研究等领域中人工智能（AI）的应用构成了重大挑战。此外，由于对隐私、版权和竞争的担忧日益增加，不可限制的公共数据集的访问也越来越受到限制。合成数据作为一种有前途的替代方案已经出现，而基于最新生成AI技术的扩散模型为生成高质量和多样化的合成数据提供了一个有效的解决方案。在本文中，我们介绍了一种新的联邦学习框架，用于在分布式私有数据集上训练扩散模型。我们的框架利用个性化和正向扩散过程固有的噪声来生成高质量样本，同时确保强大的差分隐私保证。实验结果表明，我们的框架在数据异质性高的情况下优于非协作训练方法，有效减少了合成数据中的偏差和不平衡，从而提高了下游模型的公平性。 

---
# QSViT: A Methodology for Quantizing Spiking Vision Transformers 

**Title (ZH)**: QSViT：量化脉冲视觉变换器的方法ology 

**Authors**: Rachmad Vidya Wicaksana Putra, Saad Iftikhar, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2504.00948)  

**Abstract**: Vision Transformer (ViT)-based models have shown state-of-the-art performance (e.g., accuracy) in vision-based AI tasks. However, realizing their capability in resource-constrained embedded AI systems is challenging due to their inherent large memory footprints and complex computations, thereby incurring high power/energy consumption. Recently, Spiking Vision Transformer (SViT)-based models have emerged as alternate low-power ViT networks. However, their large memory footprints still hinder their applicability for resource-constrained embedded AI systems. Therefore, there is a need for a methodology to compress SViT models without degrading the accuracy significantly. To address this, we propose QSViT, a novel design methodology to compress the SViT models through a systematic quantization strategy across different network layers. To do this, our QSViT employs several key steps: (1) investigating the impact of different precision levels in different network layers, (2) identifying the appropriate base quantization settings for guiding bit precision reduction, (3) performing a guided quantization strategy based on the base settings to select the appropriate quantization setting, and (4) developing an efficient quantized network based on the selected quantization setting. The experimental results demonstrate that, our QSViT methodology achieves 22.75% memory saving and 21.33% power saving, while also maintaining high accuracy within 2.1% from that of the original non-quantized SViT model on the ImageNet dataset. These results highlight the potential of QSViT methodology to pave the way toward the efficient SViT deployments on resource-constrained embedded AI systems. 

**Abstract (ZH)**: 基于Spiking Vision Transformer (SViT)的模型压缩方法：QSViT在资源受限嵌入式AI系统的高效部署 

---
# Graph Classification and Radiomics Signature for Identification of Tuberculous Meningitis 

**Title (ZH)**: 基于图分类和放射omics特征标识结核性脑膜炎 

**Authors**: Snigdha Agarwal, Ganaraja V H, Neelam Sinha, Abhilasha Indoria, Netravathi M, Jitender Saini  

**Link**: [PDF](https://arxiv.org/pdf/2504.00943)  

**Abstract**: Introduction: Tuberculous meningitis (TBM) is a serious brain infection caused by Mycobacterium tuberculosis, characterized by inflammation of the meninges covering the brain and spinal cord. Diagnosis often requires invasive lumbar puncture (LP) and cerebrospinal fluid (CSF) analysis. Objectives: This study aims to classify TBM patients using T1-weighted (T1w) non-contrast Magnetic Resonance Imaging (MRI) scans. We hypothesize that specific brain regions, such as the interpeduncular cisterns, bone, and corpus callosum, contain visual markers that can non-invasively distinguish TBM patients from healthy controls. We propose a novel Pixel-array Graphs Classifier (PAG-Classifier) that leverages spatial relationships between neighbouring 3D pixels in a graph-based framework to extract significant features through eigen decomposition. These features are then used to train machine learning classifiers for effective patient classification. We validate our approach using a radiomics-based methodology, classifying TBM patients based on relevant radiomics features. Results: We utilized an internal dataset consisting of 52 scans, 32 from confirmed TBM patients based on mycobacteria detection in CSF, and 20 from healthy individuals. We achieved a 5-fold cross-validated average F1 score of 85.71% for cistern regions with our PAG-Classifier and 92.85% with the radiomics features classifier, surpassing current state-of-the-art benchmarks by 15% and 22%, respectively. However, bone and corpus callosum regions showed poor classification effectiveness, with average F1 scores below 50%. Conclusion: Our study suggests that algorithms like the PAG-Classifier serve as effective tools for non-invasive TBM analysis, particularly by targeting the interpeduncular cistern. Findings indicate that the bone and corpus callosum regions lack distinctive patterns for differentiation. 

**Abstract (ZH)**: Tuberculous 脑膜炎患者基于 T1 加权非对比磁共振成像的分类研究：像素阵列图形分类器的探索 

---
# Role and Use of Race in AI/ML Models Related to Health 

**Title (ZH)**: AI/ML模型在健康领域中种族的角色与应用 

**Authors**: Martin C. Were, Ang Li, Bradley A. Malin, Zhijun Yin, Joseph R. Coco, Benjamin X. Collins, Ellen Wright Clayton, Laurie L. Novak, Rachele Hendricks-Sturrup, Abiodun Oluyomi, Shilo Anders, Chao Yan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00899)  

**Abstract**: The role and use of race within health-related artificial intelligence and machine learning (AI/ML) models has sparked increasing attention and controversy. Despite the complexity and breadth of related issues, a robust and holistic framework to guide stakeholders in their examination and resolution remains lacking. This perspective provides a broad-based, systematic, and cross-cutting landscape analysis of race-related challenges, structured around the AI/ML lifecycle and framed through "points to consider" to support inquiry and decision-making. 

**Abstract (ZH)**: 健康相关人工智能和机器学习模型中种族问题的作用与应用引发了广泛关注和争议。尽管相关问题复杂且广泛，但仍缺乏一个全面和综合的框架来指导相关利益方的审查和解决。本文提供了关于种族相关挑战的广泛、系统和综合的景观分析，围绕人工智能和机器学习生命周期展开，并通过“需考虑的要点”来支持探索和决策。 

---
# Spectral Architecture Search for Neural Networks 

**Title (ZH)**: 神经网络的光谱架构搜索 

**Authors**: Gianluca Peri, Lorenzo Giambagli, Lorenzo Chicchi, Duccio Fanelli  

**Link**: [PDF](https://arxiv.org/pdf/2504.00885)  

**Abstract**: Architecture design and optimization are challenging problems in the field of artificial neural networks. Working in this context, we here present SPARCS (SPectral ARchiteCture Search), a novel architecture search protocol which exploits the spectral attributes of the inter-layer transfer matrices. SPARCS allows one to explore the space of possible architectures by spanning continuous and differentiable manifolds, thus enabling for gradient-based optimization algorithms to be eventually employed. With reference to simple benchmark models, we show that the newly proposed method yields a self-emerging architecture with a minimal degree of expressivity to handle the task under investigation and with a reduced parameter count as compared to other viable alternatives. 

**Abstract (ZH)**: 基于人工神经网络的架构设计与优化是具有挑战性的问题。在此背景下，我们提出SPARCS（SPectral ARchiteCture Search），这是一种新颖的架构搜索协议，利用层间传输矩阵的谱属性。SPARCS通过扩展连续和可微流形来探索可能架构的空间，从而使得基于梯度的优化算法得以应用。参考简单的基准模型，我们展示了新提出的 方法生成了一个自涌现架构，该架构具有处理所研究任务所需的最小表达能力，并且参数数量较少，与其他可行的替代方案相比。 

---
# Investigating the Capabilities and Limitations of Machine Learning for Identifying Bias in English Language Data with Information and Heritage Professionals 

**Title (ZH)**: 探究机器学习在识别英语语言数据中的偏见方面的能力和局限性——以信息和遗产专业人员为例 

**Authors**: Lucy Havens, Benjamin Bach, Melissa Terras, Beatrice Alex  

**Link**: [PDF](https://arxiv.org/pdf/2504.00860)  

**Abstract**: Despite numerous efforts to mitigate their biases, ML systems continue to harm already-marginalized people. While predominant ML approaches assume bias can be removed and fair models can be created, we show that these are not always possible, nor desirable, goals. We reframe the problem of ML bias by creating models to identify biased language, drawing attention to a dataset's biases rather than trying to remove them. Then, through a workshop, we evaluated the models for a specific use case: workflows of information and heritage professionals. Our findings demonstrate the limitations of ML for identifying bias due to its contextual nature, the way in which approaches to mitigating it can simultaneously privilege and oppress different communities, and its inevitability. We demonstrate the need to expand ML approaches to bias and fairness, providing a mixed-methods approach to investigating the feasibility of removing bias or achieving fairness in a given ML use case. 

**Abstract (ZH)**: 尽管付出了大量努力来减轻其偏见，机器学习系统仍继续伤害已处于不利地位的人群。尽管主流的机器学习方法假设可以消除偏见并创建公平模型，但我们表明，这并非总是可行或可取的目标。我们通过创建模型来识别有偏见的语言，重新定义机器学习偏见问题，从而将注意力集中在数据集的偏见上，而不是试图消除它们。随后，通过研讨会，我们评估了这些模型在具体用例中的适用性：信息和遗产专业人士的工作流程。研究结果证明了因上下文关系而导致的机器学习在识别偏见方面的局限性，以及缓解偏见的方法可能会同时特权和压迫不同的社群，并且偏见是不可避免的。我们展示了需要扩展机器学习在偏见和公平性方面的研究，提供混合方法来探究在特定机器学习用例中去除偏见或实现公平性的可行性。 

---
# Exploring Personalized Federated Learning Architectures for Violence Detection in Surveillance Videos 

**Title (ZH)**: 探索针对监控视频中暴力检测的个性化联邦学习架构 

**Authors**: Mohammad Kassir, Siba Haidar, Antoun Yaacoub  

**Link**: [PDF](https://arxiv.org/pdf/2504.00857)  

**Abstract**: The challenge of detecting violent incidents in urban surveillance systems is compounded by the voluminous and diverse nature of video data. This paper presents a targeted approach using Personalized Federated Learning (PFL) to address these issues, specifically employing the Federated Learning with Personalization Layers method within the Flower framework. Our methodology adapts learning models to the unique data characteristics of each surveillance node, effectively managing the heterogeneous and non-IID nature of surveillance video data. Through rigorous experiments conducted on balanced and imbalanced datasets, our PFL models demonstrated enhanced accuracy and efficiency, achieving up to 99.3% accuracy. This study underscores the potential of PFL to significantly improve the scalability and effectiveness of surveillance systems, offering a robust, privacy-preserving solution for violence detection in complex urban environments. 

**Abstract (ZH)**: 基于个性化联邦学习的城市 surveillance 系统中暴力事件检测挑战及其解决方法 

---
# ReaLitE: Enrichment of Relation Embeddings in Knowledge Graphs using Numeric Literals 

**Title (ZH)**: ReaLitE：在知识图中利用数值_LITERAL_丰富关系嵌入 

**Authors**: Antonis Klironomos, Baifan Zhou, Zhuoxun Zheng, Gad-Elrab Mohamed, Heiko Paulheim, Evgeny Kharlamov  

**Link**: [PDF](https://arxiv.org/pdf/2504.00852)  

**Abstract**: Most knowledge graph embedding (KGE) methods tailored for link prediction focus on the entities and relations in the graph, giving little attention to other literal values, which might encode important information. Therefore, some literal-aware KGE models attempt to either integrate numerical values into the embeddings of the entities or convert these numerics into entities during preprocessing, leading to information loss. Other methods concerned with creating relation-specific numerical features assume completeness of numerical data, which does not apply to real-world graphs. In this work, we propose ReaLitE, a novel relation-centric KGE model that dynamically aggregates and merges entities' numerical attributes with the embeddings of the connecting relations. ReaLitE is designed to complement existing conventional KGE methods while supporting multiple variations for numerical aggregations, including a learnable method.
We comprehensively evaluated the proposed relation-centric embedding using several benchmarks for link prediction and node classification tasks. The results showed the superiority of ReaLitE over the state of the art in both tasks. 

**Abstract (ZH)**: 关系中心的ReaLitE：一种动态聚合和融合实体数值属性的知识图嵌入模型 

---
# Global Intervention and Distillation for Federated Out-of-Distribution Generalization 

**Title (ZH)**: 全球干预与蒸馏在联邦领域外泛化的应用 

**Authors**: Zhuang Qi, Runhui Zhang, Lei Meng, Wei Wu, Yachong Zhang, Xiangxu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2504.00850)  

**Abstract**: Attribute skew in federated learning leads local models to focus on learning non-causal associations, guiding them towards inconsistent optimization directions, which inevitably results in performance degradation and unstable convergence. Existing methods typically leverage data augmentation to enhance sample diversity or employ knowledge distillation to learn invariant representations. However, the instability in the quality of generated data and the lack of domain information limit their performance on unseen samples. To address these issues, this paper presents a global intervention and distillation method, termed FedGID, which utilizes diverse attribute features for backdoor adjustment to break the spurious association between background and label. It includes two main modules, where the global intervention module adaptively decouples objects and backgrounds in images, injects background information into random samples to intervene in the sample distribution, which links backgrounds to all categories to prevent the model from treating background-label associations as causal. The global distillation module leverages a unified knowledge base to guide the representation learning of client models, preventing local models from overfitting to client-specific attributes. Experimental results on three datasets demonstrate that FedGID enhances the model's ability to focus on the main subjects in unseen data and outperforms existing methods in collaborative modeling. 

**Abstract (ZH)**: 联邦学习中属性偏差导致局部模型聚焦于学习非因果关联，引导它们朝着不一致的优化方向发展，从而不可避免地导致性能下降和收敛不稳定。现有方法通常依赖数据增强以增强样本多样性或采用知识蒸馏以学习不变表示。然而，生成数据质量的不稳定性和领域信息的缺乏限制了其在未见过样本上的性能。为解决这些问题，本文提出了一种全球干预和蒸馏方法，命名为FedGID，利用多样化的属性特征进行后门调整以打破背景与标签之间的虚假关联。该方法包括两个主要模块，其中全球干预模块自适应地将图像中的对象和背景分离，向随机样本注入背景信息以干预样本分布，将背景与所有类别联系起来，防止模型将背景-标签关联视为因果关系。全球蒸馏模块利用统一的知识库指导客户端模型的表示学习，防止局部模型过度拟合于客户端特定的属性。在三个数据集上的实验结果表明，FedGID提升了模型在未见过数据中聚焦主要主体的能力，并在联合建模中优于现有方法。 

---
# Conditional Temporal Neural Processes with Covariance Loss 

**Title (ZH)**: 条件时序神经过程及其协方差损失 

**Authors**: Boseon Yoo, Jiwoo Lee, Janghoon Ju, Seijun Chung, Soyeon Kim, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00794)  

**Abstract**: We introduce a novel loss function, Covariance Loss, which is conceptually equivalent to conditional neural processes and has a form of regularization so that is applicable to many kinds of neural networks. With the proposed loss, mappings from input variables to target variables are highly affected by dependencies of target variables as well as mean activation and mean dependencies of input and target variables. This nature enables the resulting neural networks to become more robust to noisy observations and recapture missing dependencies from prior information. In order to show the validity of the proposed loss, we conduct extensive sets of experiments on real-world datasets with state-of-the-art models and discuss the benefits and drawbacks of the proposed Covariance Loss. 

**Abstract (ZH)**: 我们介绍了一种新的损失函数——协方差损失，该损失函数在概念上等价于条件神经过程，并且具有正则化的形式，使其适用于多种类型的神经网络。这种损失函数使得输入变量到目标变量的映射不仅受到目标变量依赖性的影响，还受到输入和目标变量的均值激活及均值依赖性的影响。这种特性使得得到的神经网络能够更好地应对噪声观测，并从先验信息中还原缺失的依赖性。为了证明所提出损失函数的有效性，我们在使用最新模型的现实世界数据集上进行了广泛的实验，并讨论了所提出协方差损失的优点和缺点。 

---
# Digitally Supported Analysis of Spontaneous Speech (DigiSpon): Benchmarking NLP-Supported Language Sample Analysis of Swiss Children's Speech 

**Title (ZH)**: 数字化支持的自发口语分析 (DigiSpon): 瑞士儿童口语语言样本分析的NLP Benchmarking 

**Authors**: Anja Ryser, Yingqiang Gao, Sarah Ebling  

**Link**: [PDF](https://arxiv.org/pdf/2504.00780)  

**Abstract**: Language sample analysis (LSA) is a process that complements standardized psychometric tests for diagnosing, for example, developmental language disorder (DLD) in children. However, its labor-intensive nature has limited its use in speech-language pathology practice. We introduce an approach that leverages natural language processing (NLP) methods not based on commercial large language models (LLMs) applied to transcribed speech data from 119 children in the German speaking part of Switzerland with typical and atypical language development. The study aims to identify optimal practices that support speech-language pathologists in diagnosing DLD more efficiently within a human-in-the-loop framework, without relying on potentially unethical implementations that leverage commercial LLMs. Preliminary findings underscore the potential of integrating locally deployed NLP methods into the process of semi-automatic LSA. 

**Abstract (ZH)**: 基于自然语言处理的方法在瑞士德语区儿童典型和非典型语言发展数据中的语言样本分析 

---
# Advancements in Multimodal Differential Evolution: A Comprehensive Review and Future Perspectives 

**Title (ZH)**: 多模态差分进化算法的发展：综述与未来展望 

**Authors**: Dikshit Chauhan, Shivani, Donghwi Jung, Anupam Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2504.00717)  

**Abstract**: Multi-modal optimization involves identifying multiple global and local optima of a function, offering valuable insights into diverse optimal solutions within the search space. Evolutionary algorithms (EAs) excel at finding multiple solutions in a single run, providing a distinct advantage over classical optimization techniques that often require multiple restarts without guarantee of obtaining diverse solutions. Among these EAs, differential evolution (DE) stands out as a powerful and versatile optimizer for continuous parameter spaces. DE has shown significant success in multi-modal optimization by utilizing its population-based search to promote the formation of multiple stable subpopulations, each targeting different optima. Recent advancements in DE for multi-modal optimization have focused on niching methods, parameter adaptation, hybridization with other algorithms including machine learning, and applications across various domains. Given these developments, it is an opportune moment to present a critical review of the latest literature and identify key future research directions. This paper offers a comprehensive overview of recent DE advancements in multimodal optimization, including methods for handling multiple optima, hybridization with EAs, and machine learning, and highlights a range of real-world applications. Additionally, the paper outlines a set of compelling open problems and future research issues from multiple perspectives 

**Abstract (ZH)**: 多模态优化涉及识别函数的多个全局和局部最优解，为搜索空间内的多种最优解提供有价值的见解。进化算法（EAs）能够在单次运行中找到多个解，这在获得多样解方面远胜于需要多次重启且无法保证得到多样解的古典优化技术。在这些EAs中，差分进化（DE）因其在连续参数空间中作为强大且多功能优化器的突出表现而脱颖而出。DE通过基于群体的搜索机制促进了多个稳定子群体的形成，每个子群体都针对不同的最优解。近年来，DE在多模态优化领域的进步集中在分群方法、参数自适应以及与其他算法（包括机器学习）的结合，以及在各个领域的应用。鉴于这些发展，对最新文献进行批判性回顾并确定关键的未来研究方向恰逢其时。本文提供了最近DE在多模态优化方面的综合概述，包括处理多个最优解的方法、与其他进化算法的结合以及与机器学习的结合，并强调了各种实际应用。此外，本文还从多个视角列出了若干引人入胜的开放问题和未来研究问题。 

---
# The HCI GenAI CO2ST Calculator: A Tool for Calculating the Carbon Footprint of Generative AI Use in Human-Computer Interaction Research 

**Title (ZH)**: 面向人机交互的生成式AI碳足迹计算器：一种计算生成式AI使用碳足迹的工具 

**Authors**: Nanna Inie, Jeanette Falk, Raghavendra Selvan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00692)  

**Abstract**: Increased usage of generative AI (GenAI) in Human-Computer Interaction (HCI) research induces a climate impact from carbon emissions due to energy consumption of the hardware used to develop and run GenAI models and systems. The exact energy usage and and subsequent carbon emissions are difficult to estimate in HCI research because HCI researchers most often use cloud-based services where the hardware and its energy consumption are hidden from plain view. The HCI GenAI CO2ST Calculator is a tool designed specifically for the HCI research pipeline, to help researchers estimate the energy consumption and carbon footprint of using generative AI in their research, either a priori (allowing for mitigation strategies or experimental redesign) or post hoc (allowing for transparent documentation of carbon footprint in written reports of the research). 

**Abstract (ZH)**: 生成式AI（GenAI）在人机交互（HCI）研究中的使用增加了因硬件能耗而导致的碳排放气候影响。由于HCI研究者通常使用云基服务，其中硬件及其能耗隐藏不见，因此在HCI研究中精确估计能耗和随之而来的碳排放具有困难。HCI生成式AI碳排放计算器是专门为HCI研究流程设计的工具，旨在帮助研究人员估算使用生成式AI在研究中的能耗和碳足迹，无论是事先（允许采取缓解策略或实验重设计）还是事后（允许在研究书面报告中透明地记录碳足迹）。 

---
# Towards Adaptive AI Governance: Comparative Insights from the U.S., EU, and Asia 

**Title (ZH)**: 面向适应性AI治理：来自美国、欧盟和亚洲的比较洞察 

**Authors**: Vikram Kulothungan, Deepti Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2504.00652)  

**Abstract**: Artificial intelligence (AI) trends vary significantly across global regions, shaping the trajectory of innovation, regulation, and societal impact. This variation influences how different regions approach AI development, balancing technological progress with ethical and regulatory considerations. This study conducts a comparative analysis of AI trends in the United States (US), the European Union (EU), and Asia, focusing on three key dimensions: generative AI, ethical oversight, and industrial applications. The US prioritizes market-driven innovation with minimal regulatory constraints, the EU enforces a precautionary risk-based framework emphasizing ethical safeguards, and Asia employs state-guided AI strategies that balance rapid deployment with regulatory oversight. Although these approaches reflect different economic models and policy priorities, their divergence poses challenges to international collaboration, regulatory harmonization, and the development of global AI standards. To address these challenges, this paper synthesizes regional strengths to propose an adaptive AI governance framework that integrates risk-tiered oversight, innovation accelerators, and strategic alignment mechanisms. By bridging governance gaps, this study offers actionable insights for fostering responsible AI development while ensuring a balance between technological progress, ethical imperatives, and regulatory coherence. 

**Abstract (ZH)**: 全球不同地区的人工智能趋势差异显著，塑造了创新、监管和社会影响的轨迹。这种差异影响了不同地区在人工智能开发中的不同做法，平衡了科技进步与伦理和监管考量。本研究对比分析了美国、欧盟和亚洲的人工智能趋势，重点关注生成式人工智能、伦理监管和工业应用三个维度。美国以市场驱动的创新为主，法规约束较少；欧盟采用预防性的基于风险的框架，强调伦理保障；亚洲则采取由国家指导的人工智能战略，平衡快速部署与监管监督。尽管这些方法反映了不同的经济模式和政策优先事项，但它们之间的差异对国际协作、监管协调和全球人工智能标准的发展提出了挑战。为应对这些挑战，本文整合了区域优势，提出了一个适应性的人工智能治理框架，其中包括分级监管、创新加速器和战略对齐机制。通过弥合治理缺口，本研究提供了促进负责任的人工智能发展、确保在科技进步、伦理要求和监管一致性的平衡方面的实际建议。 

---
# CNOT-Optimal Clifford Synthesis as SAT 

**Title (ZH)**: CNOT-最优克利福德合成作为SAT问题 

**Authors**: Irfansha Shaik, Jaco van de Pol  

**Link**: [PDF](https://arxiv.org/pdf/2504.00634)  

**Abstract**: Clifford circuit optimization is an important step in the quantum compilation pipeline. Major compilers employ heuristic approaches. While they are fast, their results are often suboptimal. Minimization of noisy gates, like 2-qubit CNOT gates, is crucial for practical computing. Exact approaches have been proposed to fill the gap left by heuristic approaches. Among these are SAT based approaches that optimize gate count or depth, but they suffer from scalability issues. Further, they do not guarantee optimality on more important metrics like CNOT count or CNOT depth. A recent work proposed an exhaustive search only on Clifford circuits in a certain normal form to guarantee CNOT count optimality. But an exhaustive approach cannot scale beyond 6 qubits.
In this paper, we incorporate search restricted to Clifford normal forms in a SAT encoding to guarantee CNOT count optimality. By allowing parallel plans, we propose a second SAT encoding that optimizes CNOT depth. By taking advantage of flexibility in SAT based approaches, we also handle connectivity restrictions in hardware platforms, and allow for qubit relabeling. We have implemented the above encodings and variations in our open source tool Q-Synth.
In experiments, our encodings significantly outperform existing SAT approaches on random Clifford circuits. We consider practical VQE and Feynman benchmarks to compare with TKET and Qiskit compilers. In all-to-all connectivity, we observe reductions up to 32.1% in CNOT count and 48.1% in CNOT depth. Overall, we observe better results than TKET in the CNOT count and depth. We also experiment with connectivity restrictions of major quantum platforms. Compared to Qiskit, we observe up to 30.3% CNOT count and 35.9% CNOT depth further reduction. 

**Abstract (ZH)**: Clifford 电路优化是量子编译管道中的一个关键步骤。主要编译器采用启发式方法。尽管这些方法速度快，但结果通常不理想。减少噪声门，如 2 腰 CNOT 门，对实际计算至关重要。已提出了精确方法以填补启发式方法的不足。其中一些方法基于 SAT 的优化，可以最小化门的数量或深度，但它们存在可扩展性问题。此外，它们在如 CNOT 数量或 CNOT 深度等更重要的指标上无法保证最优性。最近的一项工作提出了一种仅在特定正常形式的 Clifford 电路中进行穷举搜索的方法，以保证 CNOT 数量优化。然而，穷举方法无法扩展超过 6 个量子位。

本文中，我们在 SAT 编码中整合了仅针对 Clifford 正规形式的搜索，以保证 CNOT 数量优化。通过允许并行计划，我们提出了一种新的 SAT 编码方法，以优化 CNOT 深度。利用 SAT 方法的灵活性，我们还处理了硬件平台的连接性限制，并允许量子位重新标记。我们已在开源工具 Q-Synth 中实现了上述编码及其变体。

在实验中，我们的编码在随机 Clifford 电路的 SAT 方法中表现出显著的优异性能。我们考虑了实用的 VQE 和费曼基准与 TKET 和 Qiskit 编译器进行比较。在全连接情况下，我们观察到 CNOT 数量最多减少了 32.1%，CNOT 深度最多减少了 48.1%。总体而言，我们的 CNOT 数量和深度表现优于 TKET。我们还实验了主要量子平台的连接性限制。与 Qiskit 相比，我们观察到 CNOT 数量最多进一步减少了 30.3%，CNOT 深度最多进一步减少了 35.9%。 

---
# Feature Subset Weighting for Distance-based Supervised Learning through Choquet Integration 

**Title (ZH)**: 基于Choquet积分的特征子集加权距离导向监督学习 

**Authors**: Adnan Theerens, Yvan Saeys, Chris Cornelis  

**Link**: [PDF](https://arxiv.org/pdf/2504.00624)  

**Abstract**: This paper introduces feature subset weighting using monotone measures for distance-based supervised learning. The Choquet integral is used to define a distance metric that incorporates these weights. This integration enables the proposed distances to effectively capture non-linear relationships and account for interactions both between conditional and decision attributes and among conditional attributes themselves, resulting in a more flexible distance measure. In particular, we show how this approach ensures that the distances remain unaffected by the addition of duplicate and strongly correlated features. Another key point of this approach is that it makes feature subset weighting computationally feasible, since only $m$ feature subset weights should be calculated each time instead of calculating all feature subset weights ($2^m$), where $m$ is the number of attributes. Next, we also examine how the use of the Choquet integral for measuring similarity leads to a non-equivalent definition of distance. The relationship between distance and similarity is further explored through dual measures. Additionally, symmetric Choquet distances and similarities are proposed, preserving the classical symmetry between similarity and distance. Finally, we introduce a concrete feature subset weighting distance, evaluate its performance in a $k$-nearest neighbors (KNN) classification setting, and compare it against Mahalanobis distances and weighted distance methods. 

**Abstract (ZH)**: 本文介绍了一种使用单调测度进行特征子集加权的距离基监督学习方法。利用Choquet积分定义包含这些权重的距离度量，这种集成使得提出的距离能够有效地捕捉非线性关系并考虑条件属性之间以及决策属性和条件属性之间的相互作用，从而获得更灵活的距离度量。特别是，我们展示了这种方法确保在添加重复和强相关特征时距离不受影响。该方法的另一个关键点是它使特征子集加权在计算上可行，每次只需要计算$m$个特征子集权重，而不是计算所有特征子集权重($2^m$)，其中$m$是属性的数量。接下来，我们还探讨了使用Choquet积分衡量相似性导致的距离非等价定义。通过双测度进一步探讨了距离与相似性的关系。此外，提出了对称Choquet距离和相似性，保持了相似性和距离的经典对称性。最后，我们引入了一个具体的特征子集加权距离，在$k$-最近邻（KNN）分类设置中评估其性能，并将其与马氏距离和加权距离方法进行比较。 

---
# PLM4NDV: Minimizing Data Access for Number of Distinct Values Estimation with Pre-trained Language Models 

**Title (ZH)**: PLM4NDV：使用预训练语言模型最小化数据访问的数量唯一值估算 

**Authors**: Xianghong Xu, Xiao He, Tieying Zhang, Lei Zhang, Rui Shi, Jianjun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00608)  

**Abstract**: Number of Distinct Values (NDV) estimation of a multiset/column is a basis for many data management tasks, especially within databases. Despite decades of research, most existing methods require either a significant amount of samples through uniform random sampling or access to the entire column to produce estimates, leading to substantial data access costs and potentially ineffective estimations in scenarios with limited data access. In this paper, we propose leveraging semantic information, i.e., schema, to address these challenges. The schema contains rich semantic information that can benefit the NDV estimation. To this end, we propose PLM4NDV, a learned method incorporating Pre-trained Language Models (PLMs) to extract semantic schema information for NDV estimation. Specifically, PLM4NDV leverages the semantics of the target column and the corresponding table to gain a comprehensive understanding of the column's meaning. By using the semantics, PLM4NDV reduces data access costs, provides accurate NDV estimation, and can even operate effectively without any data access. Extensive experiments on a large-scale real-world dataset demonstrate the superiority of PLM4NDV over baseline methods. Our code is available at this https URL. 

**Abstract (ZH)**: 基于语义信息的多集合/列的唯一值数量估计 

---
# Data Cleansing for GANs 

**Title (ZH)**: GANs的数据清洗 

**Authors**: Naoyuki Terashita, Hiroki Ohashi, Satoshi Hara  

**Link**: [PDF](https://arxiv.org/pdf/2504.00603)  

**Abstract**: As the application of generative adversarial networks (GANs) expands, it becomes increasingly critical to develop a unified approach that improves performance across various generative tasks. One effective strategy that applies to any machine learning task is identifying harmful instances, whose removal improves the performance. While previous studies have successfully estimated these harmful training instances in supervised settings, their approaches are not easily applicable to GANs. The challenge lies in two requirements of the previous approaches that do not apply to GANs. First, previous approaches require that the absence of a training instance directly affects the parameters. However, in the training for GANs, the instances do not directly affect the generator's parameters since they are only fed into the discriminator. Second, previous approaches assume that the change in loss directly quantifies the harmfulness of the instance to a model's performance, while common types of GAN losses do not always reflect the generative performance. To overcome the first challenge, we propose influence estimation methods that use the Jacobian of the generator's gradient with respect to the discriminator's parameters (and vice versa). Such a Jacobian represents the indirect effect between two models: how removing an instance from the discriminator's training changes the generator's parameters. Second, we propose an instance evaluation scheme that measures the harmfulness of each training instance based on how a GAN evaluation metric (e.g., Inception score) is expected to change by the instance's removal. Furthermore, we demonstrate that removing the identified harmful instances significantly improves the generative performance on various GAN evaluation metrics. 

**Abstract (ZH)**: 生成对抗网络（GANs）的应用扩展使得开发一种统一的方法来提高各种生成任务性能变得日益重要。一种适用于任何机器学习任务的有效策略是识别有害实例，通过移除这些实例可以提高性能。尽管先前的研究已经在监督设置中成功估计了这些有害的训练实例，但其方法不适用于GANs。先前方法的两个要求在GANs中并不适用。首先，先前方法要求训练实例的缺失直接影响模型参数，但在GANs的训练中，实例并不会直接影响生成器的参数，因为它们仅被输入到判别器中。其次，先前方法假设损失变化直接反映了实例对模型性能的有害性，而常见的GAN损失类型并不总是反映生成性能。为克服第一个挑战，我们提出了一种使用生成器梯度关于判别器参数的雅各宾矩阵（反之亦然）来估计影响的方法。此类雅各宾矩阵表示两个模型之间的间接影响：从判别器训练中移除一个实例如何改变生成器参数。其次，我们提出了一种实例评估方案，基于移除实例后预期的GAN评估指标（如Inception分数）的变化来衡量每个训练实例的有害性。此外，我们证明移除识别出的有害实例可以显著提高各种GAN评估指标的生成性能。 

---
# High-Quality Pseudo-Label Generation Based on Visual Prompt Assisted Cloud Model Update 

**Title (ZH)**: 基于视觉提示辅助云模型更新的高质量伪标签生成 

**Authors**: Xinrun Xu, Qiuhong Zhang, Jianwen Yang, Zhanbiao Lian, Jin Yan, Zhiming Ding, Shan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00526)  

**Abstract**: Generating high-quality pseudo-labels on the cloud is crucial for cloud-edge object detection, especially in dynamic traffic monitoring where data distributions evolve. Existing methods often assume reliable cloud models, neglecting potential errors or struggling with complex distribution shifts. This paper proposes Cloud-Adaptive High-Quality Pseudo-label generation (CA-HQP), addressing these limitations by incorporating a learnable Visual Prompt Generator (VPG) and dual feature alignment into cloud model updates. The VPG enables parameter-efficient adaptation by injecting visual prompts, enhancing flexibility without extensive fine-tuning. CA-HQP mitigates domain discrepancies via two feature alignment techniques: global Domain Query Feature Alignment (DQFA) capturing scene-level shifts, and fine-grained Temporal Instance-Aware Feature Embedding Alignment (TIAFA) addressing instance variations. Experiments on the Bellevue traffic dataset demonstrate that CA-HQP significantly improves pseudo-label quality compared to existing methods, leading to notable performance gains for the edge model and showcasing CA-HQP's adaptation effectiveness. Ablation studies validate each component (DQFA, TIAFA, VPG) and the synergistic effect of combined alignment strategies, highlighting the importance of adaptive cloud updates and domain adaptation for robust object detection in evolving scenarios. CA-HQP provides a promising solution for enhancing cloud-edge object detection systems in real-world applications. 

**Abstract (ZH)**: 生成高质量的云端伪标签对于云端边缘对象检测至关重要，特别是在数据分布演变的动态交通监控中。现有的方法往往假设云模型可靠，忽视潜在的错误或难以处理复杂的分布偏移。本文提出了一种云自适应高质量伪标签生成方法（CA-HQP），通过引入可学习的视觉提示生成器（VPG）和双特征对齐，解决这些问题。VPG通过注入视觉提示实现参数高效的适应，增强灵活性而无需大量的微调。CA-HQP通过两种特征对齐技术来缓解领域差异：全局领域查询特征对齐（DQFA）捕捉场景级偏移，以及细粒度的时间感知实例特征嵌入对齐（TIAFA）解决实例变异。在Bellevue交通数据集上的实验表明，CA-HQP在伪标签质量上显著优于现有方法，显著提高了边缘模型的性能，并展示了CA-HQP的适应效果。消融研究验证了每个组件（DQFA、TIAFA、VPG）及其组合对齐策略的协同效应，突显了适应性云更新和领域适应对动态场景中稳健对象检测的重要性。CA-HQP为提升实际应用中的云端边缘对象检测系统提供了有前景的解决方案。 

---
# Operator Learning with Domain Decomposition for Geometry Generalization in PDE Solving 

**Title (ZH)**: 基于领域分解的运算器学习在偏微分方程求解中的几何泛化 

**Authors**: Jianing Huang, Kaixuan Zhang, Youjia Wu, Ze Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.00510)  

**Abstract**: Neural operators have become increasingly popular in solving \textit{partial differential equations} (PDEs) due to their superior capability to capture intricate mappings between function spaces over complex domains. However, the data-hungry nature of operator learning inevitably poses a bottleneck for their widespread applications. At the core of the challenge lies the absence of transferability of neural operators to new geometries. To tackle this issue, we propose operator learning with domain decomposition, a local-to-global framework to solve PDEs on arbitrary geometries. Under this framework, we devise an iterative scheme \textit{Schwarz Neural Inference} (SNI). This scheme allows for partitioning of the problem domain into smaller subdomains, on which local problems can be solved with neural operators, and stitching local solutions to construct a global solution. Additionally, we provide a theoretical analysis of the convergence rate and error bound. We conduct extensive experiments on several representative PDEs with diverse boundary conditions and achieve remarkable geometry generalization compared to alternative methods. These analysis and experiments demonstrate the proposed framework's potential in addressing challenges related to geometry generalization and data efficiency. 

**Abstract (ZH)**: 神经算子在解决偏微分方程（PDEs）方面的应用越来越受欢迎，得益于其在复杂域上函数空间之间捕获复杂映射的优异能力。然而，算子学习对数据的高需求不可避免地为其广泛应用设置了瓶颈。挑战的核心在于神经算子在新几何结构上的不可转移性。为了解决这一问题，我们提出了基于域分解的算子学习方法，这是一种将问题域分解为小子域、并在子域上使用神经算子求解局部问题，然后再将局部解拼接成全局解的局部到全局框架。在此框架下，我们设计了一种迭代方案—— Schwarz 神经推理（SNI）。此外，我们还提供了收敛速率和误差界的相关理论分析。我们对多个具有不同边界条件的代表性 PDE 进行了广泛的实验，并在几何泛化方面取得了显著效果，优于其他替代方法。这些分析和实验表明，所提出框架在应对几何泛化和数据效率相关挑战方面具有潜在应用价值。 

---
# Enhancing stroke disease classification through machine learning models via a novel voting system by feature selection techniques 

**Title (ZH)**: 通过特征选择技术实现的新型投票系统增强中风疾病分类的机器学习模型 

**Authors**: Mahade Hasan, Farhana Yasmin, Md. Mehedi Hassan, Xue Yu, Soniya Yeasmin, Herat Joshi, Sheikh Mohammed Shariful Islam  

**Link**: [PDF](https://arxiv.org/pdf/2504.00485)  

**Abstract**: Heart disease remains a leading cause of mortality and morbidity worldwide, necessitating the development of accurate and reliable predictive models to facilitate early detection and intervention. While state of the art work has focused on various machine learning approaches for predicting heart disease, but they could not able to achieve remarkable accuracy. In response to this need, we applied nine machine learning algorithms XGBoost, logistic regression, decision tree, random forest, k-nearest neighbors (KNN), support vector machine (SVM), gaussian naïve bayes (NB gaussian), adaptive boosting, and linear regression to predict heart disease based on a range of physiological indicators. Our approach involved feature selection techniques to identify the most relevant predictors, aimed at refining the models to enhance both performance and interpretability. The models were trained, incorporating processes such as grid search hyperparameter tuning, and cross-validation to minimize overfitting. Additionally, we have developed a novel voting system with feature selection techniques to advance heart disease classification. Furthermore, we have evaluated the models using key performance metrics including accuracy, precision, recall, F1-score, and the area under the receiver operating characteristic curve (ROC AUC). Among the models, XGBoost demonstrated exceptional performance, achieving 99% accuracy, precision, F1-Score, 98% recall, and 100% ROC AUC. This study offers a promising approach to early heart disease diagnosis and preventive healthcare. 

**Abstract (ZH)**: 心臟疾病仍然是全球 Leading 的致死和致病主要因素，亟需开发准确可靠的预测模型以促进早期检测和干预。尽管最先进的研究集中在各种机器学习方法来预测心臟疾病，但它们未能达到显著的准确性。为应对这一需求，我们应用了九种机器学习算法（XGBoost、逻辑回归、决策树、随机森林、K-近邻（KNN）、支持向量机（SVM）、高斯朴素贝叶斯（Gaussian Naïve Bayes）、自适应提升和线性回归），基于一系列生理指标预测心臟疾病。我们的方法包括特征选择技术，以识别最相关的预测因子，旨在优化模型以提高性能和可解释性。模型经过训练，并采用网格搜索超参数调优和交叉验证等过程，以减少过拟合。此外，我们还开发了一种新颖的投票系统并结合特征选择技术，以推进心臟疾病分类。进一步地，我们使用关键性能指标（包括准确率、精确率、召回率、F1分数和受试者操作特征曲线下的面积（ROC AUC））来评估模型。其中，XGBoost表现出色，实现了99%的准确率、精确率、F1分数，98%的召回率和100%的ROC AUC。本研究提供了早期心臟疾病诊断和预防保健的有前途的方法。 

---
# MetaLoRA: Tensor-Enhanced Adaptive Low-Rank Fine-tuning 

**Title (ZH)**: MetaLoRA: 张量增强自适应低秩微调 

**Authors**: Maolin Wang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.00460)  

**Abstract**: There has been a significant increase in the deployment of neural network models, presenting substantial challenges in model adaptation and fine-tuning. Efficient adaptation is crucial in maintaining model performance across diverse tasks and domains. While Low-Rank Adaptation (LoRA) has emerged as a promising parameter-efficient fine-tuning method, its fixed parameter nature limits its ability to handle dynamic task requirements effectively. Adapting models to new tasks can be challenging due to the need for extensive fine-tuning. Current LoRA variants primarily focus on general parameter reduction while overlooking the importance of dynamic parameter adjustment and meta-learning capabilities. Moreover, existing approaches mainly address static adaptations, neglecting the potential benefits of task-aware parameter generation in handling diverse task distributions. To address these limitations, this Ph.D. research proposes a LoRA generation approach to model task relationships and introduces MetaLoRA, a novel parameter-efficient adaptation framework incorporating meta-learning principles. This work develops a comprehensive architecture that integrates meta-parameter generation with adaptive low-rank decomposition, enabling efficient handling of both task-specific and task-agnostic features. MetaLoRA accurately captures task patterns by incorporating meta-learning mechanisms and dynamic parameter adjustment strategies. To our knowledge, this research represents the first attempt to provide a meta-learning enhanced LoRA variant, offering improved adaptation capability while maintaining computational efficiency in model fine-tuning. 

**Abstract (ZH)**: 低秩适应增强的元学习框架：MetaLoRA 

---
# From Intuition to Understanding: Using AI Peers to Overcome Physics Misconceptions 

**Title (ZH)**: 从直觉到理解：使用AI同伴克服物理misconceptions 

**Authors**: Ruben Weijers, Denton Wu, Hannah Betts, Tamara Jacod, Yuxiang Guan, Vidya Sujaya, Kushal Dev, Toshali Goel, William Delooze, Reihaneh Rabbany, Ying Wu, Jean-François Godbout, Kellin Pelrine  

**Link**: [PDF](https://arxiv.org/pdf/2504.00408)  

**Abstract**: Generative AI has the potential to transform personalization and accessibility of education. However, it raises serious concerns about accuracy and helping students become independent critical thinkers. In this study, we designed a helpful AI "Peer" to help students correct fundamental physics misconceptions related to Newtonian mechanic concepts. In contrast to approaches that seek near-perfect accuracy to create an authoritative AI tutor or teacher, we directly inform students that this AI can answer up to 40% of questions incorrectly. In a randomized controlled trial with 165 students, those who engaged in targeted dialogue with the AI Peer achieved post-test scores that were, on average, 10.5 percentage points higher - with over 20 percentage points higher normalized gain - than a control group that discussed physics history. Qualitative feedback indicated that 91% of the treatment group's AI interactions were rated as helpful. Furthermore, by comparing student performance on pre- and post-test questions about the same concept, along with experts' annotations of the AI interactions, we find initial evidence suggesting the improvement in performance does not depend on the correctness of the AI. With further research, the AI Peer paradigm described here could open new possibilities for how we learn, adapt to, and grow with AI. 

**Abstract (ZH)**: 生成式AI有潜力变革教育的个性化和可及性，但同时也引发了关于准确性和帮助学生培养独立批判性思维的严重关切。在本研究中，我们设计了一个有益的人工智能“同伴”，帮助学生纠正与牛顿力学概念相关的物理误解。与寻求近乎完美准确度以创建权威的人工智能导师或教师的方法不同，我们直接告知学生，该人工智能可能会错误回答多达40%的问题。在涉及165名学生的随机对照试验中，与对照组讨论物理历史相比，与AI同伴进行目标对话的学生在测试中的平均得分高出10.5个百分点，标准化增益高出20个百分点以上。定性反馈显示，治疗组中有91%的人工智能互动被认为是有帮助的。此外，通过比较学生在相同概念上的前后测表现以及专家对人工智能互动的注释，我们发现了初步证据，表明性能的提升并不依赖于人工智能的正确性。通过进一步研究，此处描述的AI同伴范式有可能为如何学习、适应和成长于人工智能开辟新的可能性。 

---
# SeizureTransformer: Scaling U-Net with Transformer for Simultaneous Time-Step Level Seizure Detection from Long EEG Recordings 

**Title (ZH)**: SeizureTransformer: 通过Transformer扩展U-Net以实现长EEG记录的同步时间步长 seizures 检测 

**Authors**: Kerui Wu, Ziyue Zhao, Bülent Yener  

**Link**: [PDF](https://arxiv.org/pdf/2504.00336)  

**Abstract**: Epilepsy is a common neurological disorder that affects around 65 million people worldwide. Detecting seizures quickly and accurately is vital, given the prevalence and severity of the associated complications. Recently, deep learning-based automated seizure detection methods have emerged as solutions; however, most existing methods require extensive post-processing and do not effectively handle the crucial long-range patterns in EEG data. In this work, we propose SeizureTransformer, a simple model comprised of (i) a deep encoder comprising 1D convolutions (ii) a residual CNN stack and a transformer encoder to embed previous output into high-level representation with contextual information, and (iii) streamlined decoder which converts these features into a sequence of probabilities, directly indicating the presence or absence of seizures at every time step. Extensive experiments on public and private EEG seizure detection datasets demonstrate that our model significantly outperforms existing approaches (ranked in the first place in the 2025 "seizure detection challenge" organized in the International Conference on Artificial Intelligence in Epilepsy and Other Neurological Disorders), underscoring its potential for real-time, precise seizure detection. 

**Abstract (ZH)**: 癫痫是一种影响全球约6500万人的常见神经 disorder，快速准确地检测癫痫发作至关重要。近年来，基于深度学习的自动化癫痫发作检测方法逐渐成为解决方案；然而，现有的大多数方法需要大量的后处理，并不能很好地处理 EEG 数据中的关键长范围模式。在本工作中，我们提出了 SeizureTransformer，这是一种简单的模型，包括（i）一个由1D卷积构成的深度编码器；（ii）一个残差 CNN 层叠和变压器编码器，用于将先前输出嵌入到包含上下文信息的高层表示；（iii）一个精简的解码器，将这些特征转化为时间步骤序列的概率，直接指示每个时间步骤是否存在癫痫发作。在公开和私有 EEG 癫痫发作检测数据集上的广泛实验表明，我们的模型在国际癫痫与其他神经疾病人工智能会议组织的2025年“癫痫发作检测挑战赛”中排名首位，突显了其在实时、精确的癫痫发作检测方面的潜力。 

---
# FedPaI: Achieving Extreme Sparsity in Federated Learning via Pruning at Initialization 

**Title (ZH)**: FedPaI: 在初始化时剪枝以实现联邦学习中的极端稀疏性 

**Authors**: Haonan Wang, Zeli Liu, Kajimusugura Hoshino, Tuo Zhang, John Paul Walters, Stephen Crago  

**Link**: [PDF](https://arxiv.org/pdf/2504.00308)  

**Abstract**: Federated Learning (FL) enables distributed training on edge devices but faces significant challenges due to resource constraints in edge environments, impacting both communication and computational efficiency. Existing iterative pruning techniques improve communication efficiency but are limited by their centralized design, which struggles with FL's decentralized and data-imbalanced nature, resulting in suboptimal sparsity levels. To address these issues, we propose FedPaI, a novel efficient FL framework that leverages Pruning at Initialization (PaI) to achieve extreme sparsity. FedPaI identifies optimal sparse connections at an early stage, maximizing model capacity and significantly reducing communication and computation overhead by fixing sparsity patterns at the start of training. To adapt to diverse hardware and software environments, FedPaI supports both structured and unstructured pruning. Additionally, we introduce personalized client-side pruning mechanisms for improved learning capacity and sparsity-aware server-side aggregation for enhanced efficiency. Experimental results demonstrate that FedPaI consistently outperforms existing efficient FL that applies conventional iterative pruning with significant leading in efficiency and model accuracy. For the first time, our proposed FedPaI achieves an extreme sparsity level of up to 98% without compromising the model accuracy compared to unpruned baselines, even under challenging non-IID settings. By employing our FedPaI with joint optimization of model learning capacity and sparsity, FL applications can benefit from faster convergence and accelerate the training by 6.4 to 7.9 times. 

**Abstract (ZH)**: Federated Learning中基于初始化剪枝的高效框架FedPaI：实现极端稀疏性的同时保持模型准确性 

---
# Digital Twins in Biopharmaceutical Manufacturing: Review and Perspective on Human-Machine Collaborative Intelligence 

**Title (ZH)**: 生物制药制造中的数字孪生：人类-机器协作智能的回顾与展望 

**Authors**: Mohammed Aatif Shahab, Francesco Destro, Richard D. Braatz  

**Link**: [PDF](https://arxiv.org/pdf/2504.00286)  

**Abstract**: The biopharmaceutical industry is increasingly developing digital twins to digitalize and automate the manufacturing process in response to the growing market demands. However, this shift presents significant challenges for human operators, as the complexity and volume of information can overwhelm their ability to manage the process effectively. These issues are compounded when digital twins are designed without considering interaction and collaboration with operators, who are responsible for monitoring processes and assessing situations, particularly during abnormalities. Our review of current trends in biopharma digital twin development reveals a predominant focus on technology and often overlooks the critical role of human operators. To bridge this gap, this article proposes a collaborative intelligence framework that emphasizes the integration of operators with digital twins. Approaches to system design that can enhance operator trust and human-machine interface usability are presented. Moreover, innovative training programs for preparing operators to understand and utilize digital twins are discussed. The framework outlined in this article aims to enhance collaboration between operators and digital twins effectively by using their full capabilities to boost resilience and productivity in biopharmaceutical manufacturing. 

**Abstract (ZH)**: 生物制药行业正在越来越多地开发数字孪生以实现制造过程的数字化和自动化，以应对市场需求的增长。然而，这一转变对人类操作人员提出了重大挑战，因为信息的复杂性和数量可能超出他们有效管理过程的能力。当数字孪生的设计忽视了与操作人员的互动和协作时，这些问题会进一步加剧，尤其是操作人员在异常情况下的监控和评估工作。我们对生物制药数字孪生发展现状的回顾表明，当前主要集中在技术上，往往忽视了操作人员的关键作用。为解决这一差距，本文提出了一种协同智能框架，强调将操作人员与数字孪生集成。本文提出了增强操作人员信任和人机界面易用性的系统设计方法，并讨论了创新的操作人员培训计划，以便他们能够理解并利用数字孪生。本文概述的框架旨在通过充分利用操作人员和数字孪生的能力，有效提升生物制药制造的韧性和生产效率。 

---
# ElaLoRA: Elastic & Learnable Low-Rank Adaptation for Efficient Model Fine-Tuning 

**Title (ZH)**: ElaLoRA: 弹性可学习低秩适应以实现高效的模型微调 

**Authors**: Huandong Chang, Zicheng Ma, Mingyuan Ma, Zhenting Qi, Andrew Sabot, Hong Jiang, H. T. Kung  

**Link**: [PDF](https://arxiv.org/pdf/2504.00254)  

**Abstract**: Low-Rank Adaptation (LoRA) has become a widely adopted technique for fine-tuning large-scale pre-trained models with minimal parameter updates. However, existing methods rely on fixed ranks or focus solely on either rank pruning or expansion, failing to adapt ranks dynamically to match the importance of different layers during training. In this work, we propose ElaLoRA, an adaptive low-rank adaptation framework that dynamically prunes and expands ranks based on gradient-derived importance scores. To the best of our knowledge, ElaLoRA is the first method that enables both rank pruning and expansion during fine-tuning. Experiments across multiple benchmarks demonstrate that ElaLoRA consistently outperforms existing PEFT methods across different parameter budgets. Furthermore, our studies validate that layers receiving higher rank allocations contribute more significantly to model performance, providing theoretical justification for our adaptive strategy. By introducing a principled and adaptive rank allocation mechanism, ElaLoRA offers a scalable and efficient fine-tuning solution, particularly suited for resource-constrained environments. 

**Abstract (ZH)**: ElaLoRA: An Adaptive Low-Rank Adaptation Framework for Fine-Tuning Large-Scale Pre-Trained Models 

---
# MultiMorph: On-demand Atlas Construction 

**Title (ZH)**: MultiMorph: 按需成图方法 

**Authors**: S. Mazdak Abulnaga, Andrew Hoopes, Neel Dey, Malte Hoffmann, Marianne Rakic, Bruce Fischl, John Guttag, Adrian Dalca  

**Link**: [PDF](https://arxiv.org/pdf/2504.00247)  

**Abstract**: We present MultiMorph, a fast and efficient method for constructing anatomical atlases on the fly. Atlases capture the canonical structure of a collection of images and are essential for quantifying anatomical variability across populations. However, current atlas construction methods often require days to weeks of computation, thereby discouraging rapid experimentation. As a result, many scientific studies rely on suboptimal, precomputed atlases from mismatched populations, negatively impacting downstream analyses. MultiMorph addresses these challenges with a feedforward model that rapidly produces high-quality, population-specific atlases in a single forward pass for any 3D brain dataset, without any fine-tuning or optimization. MultiMorph is based on a linear group-interaction layer that aggregates and shares features within the group of input images. Further, by leveraging auxiliary synthetic data, MultiMorph generalizes to new imaging modalities and population groups at test-time. Experimentally, MultiMorph outperforms state-of-the-art optimization-based and learning-based atlas construction methods in both small and large population settings, with a 100-fold reduction in time. This makes MultiMorph an accessible framework for biomedical researchers without machine learning expertise, enabling rapid, high-quality atlas generation for diverse studies. 

**Abstract (ZH)**: MultiMorph: 一种快速高效的大脑结构图谱构建方法 

---
# Can Diffusion Models Disentangle? A Theoretical Perspective 

**Title (ZH)**: 扩散模型能否解耦？一个理论视角 

**Authors**: Liming Wang, Muhammad Jehanzeb Mirza, Yishu Gong, Yuan Gong, Jiaqi Zhang, Brian H. Tracey, Katerina Placek, Marco Vilela, James R. Glass  

**Link**: [PDF](https://arxiv.org/pdf/2504.00220)  

**Abstract**: This paper presents a novel theoretical framework for understanding how diffusion models can learn disentangled representations. Within this framework, we establish identifiability conditions for general disentangled latent variable models, analyze training dynamics, and derive sample complexity bounds for disentangled latent subspace models. To validate our theory, we conduct disentanglement experiments across diverse tasks and modalities, including subspace recovery in latent subspace Gaussian mixture models, image colorization, image denoising, and voice conversion for speech classification. Additionally, our experiments show that training strategies inspired by our theory, such as style guidance regularization, consistently enhance disentanglement performance. 

**Abstract (ZH)**: 本文提出了一种新的理论框架，用于理解扩散模型如何学习解耦表示。在此框架内，我们建立了通用解耦潜在变量模型的可识别性条件，分析了训练动力学，并推导出了解耦潜在子空间模型的样本复杂性边界。为了验证我们的理论，我们在多任务和多种模态下进行了解耦实验，包括潜在子空间高斯混合模型中的子空间恢复、图像颜色化、图像去噪以及用于语音分类的语音转换。此外，我们的实验表明，受我们的理论启发的训练策略，如风格引导正则化，始终能提升解耦性能。 

---
# Identifying Sparsely Active Circuits Through Local Loss Landscape Decomposition 

**Title (ZH)**: 通过局部损失景观分解识别稀疏激活电路 

**Authors**: Brianna Chrisman, Lucius Bushnaq, Lee Sharkey  

**Link**: [PDF](https://arxiv.org/pdf/2504.00194)  

**Abstract**: Much of mechanistic interpretability has focused on understanding the activation spaces of large neural networks. However, activation space-based approaches reveal little about the underlying circuitry used to compute features. To better understand the circuits employed by models, we introduce a new decomposition method called Local Loss Landscape Decomposition (L3D). L3D identifies a set of low-rank subnetworks: directions in parameter space of which a subset can reconstruct the gradient of the loss between any sample's output and a reference output vector. We design a series of progressively more challenging toy models with well-defined subnetworks and show that L3D can nearly perfectly recover the associated subnetworks. Additionally, we investigate the extent to which perturbing the model in the direction of a given subnetwork affects only the relevant subset of samples. Finally, we apply L3D to a real-world transformer model and a convolutional neural network, demonstrating its potential to identify interpretable and relevant circuits in parameter space. 

**Abstract (ZH)**: 基于局部损失景观分解的方法揭示神经网络模型中使用的电路结构 

---
# Are Domain Generalization Benchmarks with Accuracy on the Line Misspecified? 

**Title (ZH)**: 泛化基准的准确度是否失准？ 

**Authors**: Olawale Salaudeen, Nicole Chiou, Shiny Weng, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2504.00186)  

**Abstract**: Spurious correlations are unstable statistical associations that hinder robust decision-making. Conventional wisdom suggests that models relying on such correlations will fail to generalize out-of-distribution (OOD), especially under strong distribution shifts. However, empirical evidence challenges this view as naive in-distribution empirical risk minimizers often achieve the best OOD accuracy across popular OOD generalization benchmarks. In light of these results, we propose a different perspective: many widely used benchmarks for evaluating robustness to spurious correlations are misspecified. Specifically, they fail to include shifts in spurious correlations that meaningfully impact OOD generalization, making them unsuitable for evaluating the benefit of removing such correlations. We establish conditions under which a distribution shift can reliably assess a model's reliance on spurious correlations. Crucially, under these conditions, we should not observe a strong positive correlation between in-distribution and OOD accuracy, often called "accuracy on the line." Yet, most state-of-the-art benchmarks exhibit this pattern, suggesting they do not effectively assess robustness. Our findings expose a key limitation in current benchmarks used to evaluate domain generalization algorithms, that is, models designed to avoid spurious correlations. We highlight the need to rethink how robustness to spurious correlations is assessed, identify well-specified benchmarks the field should prioritize, and enumerate strategies for designing future benchmarks that meaningfully reflect robustness under distribution shift. 

**Abstract (ZH)**: 虚假相关性是不稳定的统计关联，妨碍了稳健决策的制定。传统智慧认为依赖此类关联的模型在出分布（OOD）外泛化能力较差，特别是在分布强烈变化时。然而，实证证据挑战了这一观点，因为在分布内的经验风险最小化方法往往能达到最佳的OOD准确性。鉴于这些结果，我们提出了一种不同的视角：许多用于评估对虚假相关性稳健性的常用基准可能是不恰当的。具体而言，它们未能包含对OOD泛化有实质性影响的虚假相关性的变化，使得它们不适合评估去除这些虚假相关性的益处。我们确立了在哪些条件下分布变化可以可靠地评估模型对虚假相关性的依赖性。至关重要的是，在这些条件下，我们不应该观察到在分布内和OOD准确性之间存在强烈的正相关，即所谓的“准确线性”。然而，大多数最先进的基准都表现出这种模式，表明它们未能有效评估稳健性。我们的研究揭示了当前用于评估领域泛化算法的基准的一个关键局限性，即设计为避免虚假相关性的模型。我们强调需要重新思考如何评估对虚假相关性的稳健性，指出了领域应优先考虑的恰当基准，并列举了设计未来的基准以更好地反映分布变化下的稳健性的策略。 

---
# Boundless Byte Pair Encoding: Breaking the Pre-tokenization Barrier 

**Title (ZH)**: 无界字对编码：突破预分词障碍 

**Authors**: Craig W. Schmidt, Varshini Reddy, Chris Tanner, Yuval Pinter  

**Link**: [PDF](https://arxiv.org/pdf/2504.00178)  

**Abstract**: Pre-tokenization, the initial step in many modern tokenization pipelines, segments text into smaller units called pretokens, typically splitting on whitespace and punctuation. While this process encourages having full, individual words as tokens, it introduces a fundamental limitation in most tokenization algorithms such as Byte Pair Encoding (BPE). Specifically, pre-tokenization causes the distribution of tokens in a corpus to heavily skew towards common, full-length words. This skewed distribution limits the benefits of expanding to larger vocabularies, since the additional tokens appear with progressively lower counts. To overcome this barrier, we propose BoundlessBPE, a modified BPE algorithm that relaxes the pretoken boundary constraint. Our approach selectively merges two complete pretokens into a larger unit we term a superword. Superwords are not necessarily semantically cohesive. For example, the pretokens " of" and " the" might be combined to form the superword " of the". This merging strategy results in a substantially more uniform distribution of tokens across a corpus than standard BPE, and compresses text more effectively, with an approximate 20% increase in bytes per token. 

**Abstract (ZH)**: 无界BPE：一种放松预词化边界约束的BPE算法 

---
# MetaCLBench: Meta Continual Learning Benchmark on Resource-Constrained Edge Devices 

**Title (ZH)**: MetaCLBench: 有限资源边缘设备上的元连续学习基准 

**Authors**: Sijia Li, Young D. Kwon, Lik-Hang Lee, Pan Hui  

**Link**: [PDF](https://arxiv.org/pdf/2504.00174)  

**Abstract**: Meta-Continual Learning (Meta-CL) has emerged as a promising approach to minimize manual labeling efforts and system resource requirements by enabling Continual Learning (CL) with limited labeled samples. However, while existing methods have shown success in image-based tasks, their effectiveness remains unexplored for sequential time-series data from sensor systems, particularly audio inputs. To address this gap, we conduct a comprehensive benchmark study evaluating six representative Meta-CL approaches using three network architectures on five datasets from both image and audio modalities. We develop MetaCLBench, an end-to-end Meta-CL benchmark framework for edge devices to evaluate system overheads and investigate trade-offs among performance, computational costs, and memory requirements across various Meta-CL methods. Our results reveal that while many Meta-CL methods enable to learn new classes for both image and audio modalities, they impose significant computational and memory costs on edge devices. Also, we find that pre-training and meta-training procedures based on source data before deployment improve Meta-CL performance. Finally, to facilitate further research, we provide practical guidelines for researchers and machine learning practitioners implementing Meta-CL on resource-constrained environments and make our benchmark framework and tools publicly available, enabling fair evaluation across both accuracy and system-level metrics. 

**Abstract (ZH)**: 元持续学习（Meta-Continual Learning, Meta-CL）已 emerge 作为通过使用有限标注样本实现持续学习（Continual Learning, CL）以减轻手动标注努力和系统资源需求的颇有前景的方法。然而，尽管现有的方法在图像任务上已显示出成功，它们在来自传感器系统的顺序时间序列数据，特别是音频输入上的有效性仍未被探索。为弥补这一差距，我们使用三种网络架构在五个来自图像和音频模态的数据集中，对六个代表性 Meta-CL 方法进行了全面的基准研究。我们开发了针对边缘设备的端到端 Meta-CL 基准框架 MetaCLBench，用于评估系统开销，并探讨 Meta-CL 方法之间性能、计算成本和内存需求之间的权衡。结果显示，许多 Meta-CL 方法能够同时学习图像和音频模态的新类，但在边缘设备上却会产生明显的计算和内存成本。我们还发现，在部署前基于源数据进行预训练和元训练可以改善 Meta-CL 性能。最后，为了促进进一步研究，我们提供了在资源受限环境中实现 Meta-CL 的实用指南，并将基准框架和工具公开发布，支持基于准确性和系统级指标之间的公平评估。 

---
# Backdoor Detection through Replicated Execution of Outsourced Training 

**Title (ZH)**: 外包训练副本执行的后门检测 

**Authors**: Hengrui Jia, Sierra Wyllie, Akram Bin Sediq, Ahmed Ibrahim, Nicolas Papernot  

**Link**: [PDF](https://arxiv.org/pdf/2504.00170)  

**Abstract**: It is common practice to outsource the training of machine learning models to cloud providers. Clients who do so gain from the cloud's economies of scale, but implicitly assume trust: the server should not deviate from the client's training procedure. A malicious server may, for instance, seek to insert backdoors in the model. Detecting a backdoored model without prior knowledge of both the backdoor attack and its accompanying trigger remains a challenging problem. In this paper, we show that a client with access to multiple cloud providers can replicate a subset of training steps across multiple servers to detect deviation from the training procedure in a similar manner to differential testing. Assuming some cloud-provided servers are benign, we identify malicious servers by the substantial difference between model updates required for backdooring and those resulting from clean training. Perhaps the strongest advantage of our approach is its suitability to clients that have limited-to-no local compute capability to perform training; we leverage the existence of multiple cloud providers to identify malicious updates without expensive human labeling or heavy computation. We demonstrate the capabilities of our approach on an outsourced supervised learning task where $50\%$ of the cloud providers insert their own backdoor; our approach is able to correctly identify $99.6\%$ of them. In essence, our approach is successful because it replaces the signature-based paradigm taken by existing approaches with an anomaly-based detection paradigm. Furthermore, our approach is robust to several attacks from adaptive adversaries utilizing knowledge of our detection scheme. 

**Abstract (ZH)**: 将机器学习模型的训练外包给云提供商是常见的做法。客户端通过这种方式利用了云的成本优势，但前提是信任云服务器不会偏离客户端的训练流程。恶意服务器可能会试图在模型中植入后门。检测带有后门的模型仍是一个挑战性问题，尤其是缺乏关于后门攻击及其触发条件先验知识的情况下。本文展示了客户端如何通过访问多个云提供商，复制部分训练步骤并在多个服务器上执行这些步骤，以类似于差异测试的方式检测训练流程的偏差。假设一些云服务器是无害的，我们可以通过模型更新之间的显著差异来识别恶意服务器，这些差异反映了植入后门所需的更新与干净训练产生的更新不同。我们的方法的一大优势是对训练计算能力有限的客户端而言更为适合；我们利用多个云提供商的存在，无需昂贵的人工标注或大量计算即可识别恶意更新。通过在50%的云提供商中植入后门的一个外包监督学习任务中展示了我们的方法的能力，我们的方法能够正确识别其中99.6%的后门。简而言之，我们的方法成功之处在于将其与现有依赖签名的方法相比，转变为基于异常检测的方法。此外，我们的方法对善于利用我们检测方案知识的适应性攻击具有鲁棒性。 

---
# Lorentzian Graph Isomorphic Network 

**Title (ZH)**: 洛伦兹图形同构网络 

**Authors**: Srinitish Srinivasan, Omkumar CU  

**Link**: [PDF](https://arxiv.org/pdf/2504.00142)  

**Abstract**: We introduce the Lorentzian Graph Isomorphic Network (LGIN), a novel graph neural network (GNN) designed to operate in hyperbolic spaces, leveraging the Lorentzian model to enhance graph representation learning. Existing GNNs primarily operate in Euclidean spaces, which can limit their ability to capture hierarchical and multi-relational structures inherent to complex graphs. LGIN addresses this by incorporating curvature-aware aggregation functions that preserve the Lorentzian metric tensor, ensuring embeddings remain constrained within the hyperbolic space by proposing a new update rule that effectively captures both local neighborhood interactions and global structural properties, enabling LGIN to distinguish non-isomorphic graphs with expressiveness at least as powerful as the Weisfeiler-Lehman test. Through extensive evaluation across nine benchmark datasets, including molecular and protein structures, LGIN consistently outperforms or matches state-of-the-art GNNs, demonstrating its robustness and efficacy in modeling complex graph structures. To the best of our knowledge, this is the first study to extend the concept of a powerful graph neural network to Riemannian manifolds, paving the way for future advancements in hyperbolic graph learning. The code for our paper can be found at this https URL. 

**Abstract (ZH)**: 洛伦兹图同构网络：一种基于洛伦兹模型的新型 hyperbolic 图神经网络 

---
# Data-driven Power Loss Identification through Physics-Based Thermal Model Backpropagation 

**Title (ZH)**: 基于物理热模型反向传播的数据驱动功率损失识别 

**Authors**: Mattia Scarpa, Francesco Pase, Ruggero Carli, Mattia Bruschetta, Franscesco Toso  

**Link**: [PDF](https://arxiv.org/pdf/2504.00133)  

**Abstract**: Digital twins for power electronics require accurate power losses whose direct measurements are often impractical or impossible in real-world applications. This paper presents a novel hybrid framework that combines physics-based thermal modeling with data-driven techniques to identify and correct power losses accurately using only temperature measurements. Our approach leverages a cascaded architecture where a neural network learns to correct the outputs of a nominal power loss model by backpropagating through a reduced-order thermal model. We explore two neural architectures, a bootstrapped feedforward network, and a recurrent neural network, demonstrating that the bootstrapped feedforward approach achieves superior performance while maintaining computational efficiency for real-time applications. Between the interconnection, we included normalization strategies and physics-guided training loss functions to preserve stability and ensure physical consistency. Experimental results show that our hybrid model reduces both temperature estimation errors (from 7.2+-6.8°C to 0.3+-0.3°C) and power loss prediction errors (from 5.4+-6.6W to 0.2+-0.3W) compared to traditional physics-based approaches, even in the presence of thermal model uncertainties. This methodology allows us to accurately estimate power losses without direct measurements, making it particularly helpful for real-time industrial applications where sensor placement is hindered by cost and physical limitations. 

**Abstract (ZH)**: 数字孪生技术在电力电子中的应用要求准确的功率损耗数据，而在实际应用中直接测量这些数据往往不现实或不可能。本文提出了一种新的混合框架，该框架结合了基于物理的热学建模与数据驱动技术，仅通过温度测量即可准确识别和校正功率损耗。该方法采用嵌套架构，其中神经网络通过反向传播通过降阶热模型来学习校正名义功率损耗模型的输出。我们探索了两种神经网络架构——自助前向网络和递归神经网络，结果显示自助前向网络在保持实时应用所需计算效率的同时，实现了更优的性能。在两者之间，我们还包含了归一化策略和基于物理的训练损失函数，以保持模型的稳定性和物理一致性。实验结果显示，与传统的基于物理的方法相比，我们的混合模型在存在热模型不确定性的情况下，能够降低温度估计误差（从7.2±6.8°C降至0.3±0.3°C）和功率损耗预测误差（从5.4±6.6W降至0.2±0.3W），从而特别适用于由于成本和物理限制而受到传感器放置限制的实时工业应用。 

---
# Times2D: Multi-Period Decomposition and Derivative Mapping for General Time Series Forecasting 

**Title (ZH)**: Times2D: 多时期分解和导数映射通用时间序列预测 

**Authors**: Reza Nematirad, Anil Pahwa, Balasubramaniam Natarajan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00118)  

**Abstract**: Time series forecasting is an important application in various domains such as energy management, traffic planning, financial markets, meteorology, and medicine. However, real-time series data often present intricate temporal variability and sharp fluctuations, which pose significant challenges for time series forecasting. Previous models that rely on 1D time series representations usually struggle with complex temporal variations. To address the limitations of 1D time series, this study introduces the Times2D method that transforms the 1D time series into 2D space. Times2D consists of three main parts: first, a Periodic Decomposition Block (PDB) that captures temporal variations within a period and between the same periods by converting the time series into a 2D tensor in the frequency domain. Second, the First and Second Derivative Heatmaps (FSDH) capture sharp changes and turning points, respectively. Finally, an Aggregation Forecasting Block (AFB) integrates the output tensors from PDB and FSDH for accurate forecasting. This 2D transformation enables the utilization of 2D convolutional operations to effectively capture long and short characteristics of the time series. Comprehensive experimental results across large-scale data in the literature demonstrate that the proposed Times2D model achieves state-of-the-art performance in both short-term and long-term forecasting. The code is available in this repository: this https URL. 

**Abstract (ZH)**: 时间序列 forecasting 是能源管理、交通规划、金融市场、气象学和医学等领域中的一项重要应用。然而，真实世界的时间序列数据 often 呈现出复杂的时域变异性以及尖锐的波动，这对时间序列 forecasting 带来了巨大挑战。依赖于 1D 时间序列表示的先前模型通常难以处理复杂的时域变化。为了应对 1D 时间序列的局限性，本文提出了 Times2D 方法，将 1D 时间序列转换为 2D 空间。Times2D 由三个主要部分组成：首先，周期分解块 (PDB) 将时间序列在频域中转换为 2D 张量，以捕捉同一周期内和不同周期之间的时域变化。其次，一阶和二阶导数热图 (FSDH) 分别捕捉尖锐变化和转折点。最后，聚合预测块 (AFB) 将 PDB 和 FSDH 的输出张量进行集成，以实现准确的预测。这种 2D 转换使利用 2D 卷积操作能够有效捕捉时间序列的长期和短期特征。在文献中的大规模数据集上进行的综合实验证明，所提出的 Times2D 模型在短、长期 forecasting 领域均达到了最先进的性能。代码见本仓库：this https URL。 

---
# CF-CAM: Gradient Perturbation Mitigation and Feature Stabilization for Reliable Interpretability 

**Title (ZH)**: CF-CAM: 梯度扰动缓解与特征稳定化以实现可靠的可解释性 

**Authors**: Hongjie He, Xu Pan, Yudong Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.00060)  

**Abstract**: As deep learning continues to advance, the opacity of neural network decision-making remains a critical challenge, limiting trust and applicability in high-stakes domains. Class Activation Mapping (CAM) techniques have emerged as a key approach to visualizing model decisions, yet existing methods face inherent trade-offs. Gradient-based CAM variants suffer from sensitivity to gradient perturbations, leading to unstable and unreliable explanations. Conversely, gradient-free approaches mitigate gradient instability but incur significant computational overhead and inference latency. To address these limitations, we propose Cluster Filter Class Activation Map (CF-CAM), a novel framework that reintroduces gradient-based weighting while enhancing robustness against gradient noise. CF-CAM employs a hierarchical importance weighting strategy to balance discriminative feature preservation and noise elimination. A density-aware channel clustering via Density-Based Spatial Clustering of Applications with Noise (DBSCAN) groups semantically relevant feature channels and discard noise-prone activations. Additionally, cluster-conditioned gradient filtering leverages bilateral filters to refine gradient signals, preserving edge-aware localization while suppressing noise impact. Experiment results demonstrate that CF-CAM achieves superior interpretability performance while maintaining resilience to gradient perturbations, outperforming state-of-the-art CAM methods in faithfulness and robustness. By effectively mitigating gradient instability without excessive computational cost, CF-CAM provides a reliable solution for enhancing the interpretability of deep neural networks in critical applications such as medical diagnosis and autonomous driving. 

**Abstract (ZH)**: 基于聚类滤波的类激活图（Cluster Filter Class Activation Map, CF-CAM）：增强深层神经网络的可解释性与鲁棒性 

---
# GAL-MAD: Towards Explainable Anomaly Detection in Microservice Applications Using Graph Attention Networks 

**Title (ZH)**: GAL-MAD：基于图注意力网络的可解释微服务应用异常检测 

**Authors**: Lahiru Akmeemana, Chamodya Attanayake, Husni Faiz, Sandareka Wickramanayake  

**Link**: [PDF](https://arxiv.org/pdf/2504.00058)  

**Abstract**: The transition to microservices has revolutionized software architectures, offering enhanced scalability and modularity. However, the distributed and dynamic nature of microservices introduces complexities in ensuring system reliability, making anomaly detection crucial for maintaining performance and functionality. Anomalies stemming from network and performance issues must be swiftly identified and addressed. Existing anomaly detection techniques often rely on statistical models or machine learning methods that struggle with the high-dimensional, interdependent data inherent in microservice applications. Current techniques and available datasets predominantly focus on system traces and logs, limiting their ability to support advanced detection models. This paper addresses these gaps by introducing the RS-Anomic dataset generated using the open-source RobotShop microservice application. The dataset captures multivariate performance metrics and response times under normal and anomalous conditions, encompassing ten types of anomalies. We propose a novel anomaly detection model called Graph Attention and LSTM-based Microservice Anomaly Detection (GAL-MAD), leveraging Graph Attention and Long Short-Term Memory architectures to capture spatial and temporal dependencies in microservices. We utilize SHAP values to localize anomalous services and identify root causes to enhance explainability. Experimental results demonstrate that GAL-MAD outperforms state-of-the-art models on the RS-Anomic dataset, achieving higher accuracy and recall across varying anomaly rates. The explanations provide actionable insights into service anomalies, which benefits system administrators. 

**Abstract (ZH)**: 微服务转型重塑了软件架构，提升了可扩展性和模块性。然而，微服务的分布式和动态特性带来了确保系统可靠性的复杂性，这使得异常检测对于保持性能和功能至关重要。源自网络和性能问题的异常必须迅速被识别和处理。现有异常检测技术通常依赖于统计模型或机器学习方法，这些方法在处理微服务应用中存在的高维和相互依赖的数据时表现不佳。当前的技术和可用的数据集主要集中在系统跟踪和日志上，限制了它们支持高级检测模型的能力。本文通过引入基于开源RobotShop微服务应用生成的RS-Anomic数据集来填补这些空白。该数据集在正常和异常条件下捕获了多变量性能指标和响应时间，并包括了十种类型的异常。我们提出了一种新的异常检测模型——基于图注意力和长短期记忆的微服务异常检测（GAL-MAD），利用图注意力和长短期记忆架构来捕捉微服务中的空间和时间依赖性。我们使用SHAP值来定位异常服务并识别根本原因，以提高解释性。实验结果表明，GAL-MAD在RS-Anomic数据集上优于最先进的模型，即使在不同异常率的情况下也能实现更高的准确性和召回率。这些解释为系统管理员提供了可操作的洞察，帮助他们更好地理解和处理服务异常。 

---
# Quantum Methods for Managing Ambiguity in Natural Language Processing 

**Title (ZH)**: 量子方法在自然语言处理中管理歧义的应用 

**Authors**: Jurek Eisinger, Ward Gauderis, Lin de Huybrecht, Geraint A. Wiggins  

**Link**: [PDF](https://arxiv.org/pdf/2504.00040)  

**Abstract**: The Categorical Compositional Distributional (DisCoCat) framework models meaning in natural language using the mathematical framework of quantum theory, expressed as formal diagrams. DisCoCat diagrams can be associated with tensor networks and quantum circuits. DisCoCat diagrams have been connected to density matrices in various contexts in Quantum Natural Language Processing (QNLP). Previous use of density matrices in QNLP entails modelling ambiguous words as probability distributions over more basic words (the word \texttt{queen}, e.g., might mean the reigning queen or the chess piece). In this article, we investigate using probability distributions over processes to account for syntactic ambiguity in sentences. The meanings of these sentences are represented by density matrices. We show how to create probability distributions on quantum circuits that represent the meanings of sentences and explain how this approach generalises tasks from the literature. We conduct an experiment to validate the proposed theory. 

**Abstract (ZH)**: 基于量子理论的分类组合分布（DisCoCat）框架使用形式图表来表示自然语言的意义。DisCoCat图表可以与张量网络和量子电路相联系。在量子自然语言处理（QNLP）的多种背景下，DisCoCat图表与密度矩阵相关联。在先前于QNLP中的密度矩阵使用中，模糊词被建模为更基本词的概率分布（例如，“queen”这个词可能指的是在位的女王或棋盘上的棋子）。在本文中，我们研究使用过程的概率分布来解释句子的句法模糊性。这些句子的意义由密度矩阵表示。我们展示了如何在表示句子意义的量子电路上创建概率分布，并解释了这种方法如何推广文献中的任务。我们进行了一项实验来验证提出理论的有效性。 

---
# Revisiting the Relationship between Adversarial and Clean Training: Why Clean Training Can Make Adversarial Training Better 

**Title (ZH)**: 重新审视对抗训练与干净训练之间的关系：为什么干净训练可以使对抗训练更好 

**Authors**: MingWei Zhou, Xiaobing Pei  

**Link**: [PDF](https://arxiv.org/pdf/2504.00038)  

**Abstract**: Adversarial training (AT) is an effective technique for enhancing adversarial robustness, but it usually comes at the cost of a decline in generalization ability. Recent studies have attempted to use clean training to assist adversarial training, yet there are contradictions among the conclusions. We comprehensively summarize the representative strategies and, with a focus on the multi - view hypothesis, provide a unified explanation for the contradictory phenomena among different studies. In addition, we conduct an in - depth analysis of the knowledge combinations transferred from clean - trained models to adversarially - trained models in previous studies, and find that they can be divided into two categories: reducing the learning difficulty and providing correct guidance. Based on this finding, we propose a new idea of leveraging clean training to further improve the performance of advanced AT this http URL reveal that the problem of generalization degradation faced by AT partly stems from the difficulty of adversarial training in learning certain sample features, and this problem can be alleviated by making full use of clean training. 

**Abstract (ZH)**: 对抗训练（AT）是一种有效的增强对抗鲁棒性的技术，但通常会以牺牲泛化能力为代价。最近的研究试图通过干净的训练来辅助对抗训练，然而不同研究的结论之间存在矛盾。我们全面总结了代表性策略，并以多视角假说为重点，提供了不同研究中矛盾现象的统一解释。此外，我们深入分析了之前研究中干净训练模型向对抗训练模型转移的知识组合，并发现它们可以分为两类：减少学习难度和提供正确指导。基于这一发现，我们提出了一个新的思路，即利用干净训练进一步提高高级AT的性能：对抗训练面临的泛化能力下降问题部分源自于学习某些样本特征的难度，这个问题可以通过充分利用干净训练来缓解。 

---
# Improving Diseases Predictions Utilizing External Bio-Banks 

**Title (ZH)**: 利用外部生物银行提高疾病预测准确性 

**Authors**: Hido Pinto, Eran Segal  

**Link**: [PDF](https://arxiv.org/pdf/2504.00036)  

**Abstract**: Machine learning has been successfully used in critical domains, such as medicine. However, extracting meaningful insights from biomedical data is often constrained by the lack of their available disease labels. In this research, we demonstrate how machine learning can be leveraged to enhance explainability and uncover biologically meaningful associations, even when predictive improvements in disease modeling are limited. We train LightGBM models from scratch on our dataset (10K) to impute metabolomics features and apply them to the UK Biobank (UKBB) for downstream analysis. The imputed metabolomics features are then used in survival analysis to assess their impact on disease-related risk factors. As a result, our approach successfully identified biologically relevant connections that were not previously known to the predictive models. Additionally, we applied a genome-wide association study (GWAS) on key metabolomics features, revealing a link between vascular dementia and smoking. Although being a well-established epidemiological relationship, this link was not embedded in the model's training data, which validated the method's ability to extract meaningful signals. Furthermore, by integrating survival models as inputs in the 10K data, we uncovered associations between metabolic substances and obesity, demonstrating the ability to infer disease risk for future patients without requiring direct outcome labels. These findings highlight the potential of leveraging external bio-banks to extract valuable biomedical insights, even in data-limited scenarios. Our results demonstrate that machine learning models trained on smaller datasets can still be used to uncover real biological associations when carefully integrated with survival analysis and genetic studies. 

**Abstract (ZH)**: 机器学习在生物医学数据解释中的应用：即使在疾病标签有限的情况下也能揭示生物意义关联 

---
# Opioid Named Entity Recognition (ONER-2025) from Reddit 

**Title (ZH)**: Opioid Named Entity Recognition (ONER-2025) from Reddit 

**Authors**: Muhammad Ahmad, Humaira Farid, Iqra Ameer, Muhammad Muzamil, Ameer Hamza Muhammad Jalal, Ildar Batyrshin, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2504.00027)  

**Abstract**: The opioid overdose epidemic remains a critical public health crisis, particularly in the United States, leading to significant mortality and societal costs. Social media platforms like Reddit provide vast amounts of unstructured data that offer insights into public perceptions, discussions, and experiences related to opioid use. This study leverages Natural Language Processing (NLP), specifically Opioid Named Entity Recognition (ONER-2025), to extract actionable information from these platforms. Our research makes four key contributions. First, we created a unique, manually annotated dataset sourced from Reddit, where users share self-reported experiences of opioid use via different administration routes. This dataset contains 331,285 tokens and includes eight major opioid entity categories. Second, we detail our annotation process and guidelines while discussing the challenges of labeling the ONER-2025 dataset. Third, we analyze key linguistic challenges, including slang, ambiguity, fragmented sentences, and emotionally charged language, in opioid discussions. Fourth, we propose a real-time monitoring system to process streaming data from social media, healthcare records, and emergency services to identify overdose events. Using 5-fold cross-validation in 11 experiments, our system integrates machine learning, deep learning, and transformer-based language models with advanced contextual embeddings to enhance understanding. Our transformer-based models (bert-base-NER and roberta-base) achieved 97% accuracy and F1-score, outperforming baselines by 10.23% (RF=0.88). 

**Abstract (ZH)**: opioid过量危机仍然是一个关键的公共卫生危机，特别是在美国，导致大量的死亡和社会成本。像Reddit这样的社交媒体平台提供了大量的非结构化数据，这些数据提供了关于公众对 opioids 使用的看法、讨论和经验的见解。本研究利用自然语言处理（NLP），特别是Opioid Named Entity Recognition (ONER-2025)，从这些平台中提取有用信息。我们的研究做出了四项关键贡献。首先，我们创建了一个独特的、人工标注的数据集，数据来源为Reddit，用户在此平台上分享不同给药途径的 opioid 自我报告使用经历。该数据集包含331,285个令牌，并包括八大主要 opioid 实体类别。其次，我们详细介绍了我们的标注过程和指南，同时讨论了标注ONER-2025数据集所面临的挑战。第三，我们分析了 opioid 讨论中的关键语言挑战，包括俚语、歧义、断裂的句子和情感化的语言。第四，我们提出了一种实时监测系统，用于处理来自社交媒体、医疗记录和紧急服务的流式数据，以识别过量事件。通过在11次实验中使用5折交叉验证，我们的系统结合了机器学习、深度学习和基于变换器的语言模型以及高级上下文嵌入，增强了对事件的理解。基于变换器的模型（bert-base-NER和roberta-base）实现了97%的准确率和F1分数，优于基线10.23%（RF=0.88）。 

---
# A multi-locus predictiveness curve and its summary assessment for genetic risk prediction 

**Title (ZH)**: 多 locus 风险预测曲线及其遗传风险预测综合评估 

**Authors**: Changshuai Wei, Ming Li, Yalu Wen, Chengyin Ye, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00024)  

**Abstract**: With the advance of high-throughput genotyping and sequencing technologies, it becomes feasible to comprehensive evaluate the role of massive genetic predictors in disease prediction. There exists, therefore, a critical need for developing appropriate statistical measurements to access the combined effects of these genetic variants in disease prediction. Predictiveness curve is commonly used as a graphical tool to measure the predictive ability of a risk prediction model on a single continuous biomarker. Yet, for most complex diseases, risk prediciton models are formed on multiple genetic variants. We therefore propose a multi-marker predictiveness curve and provide a non-parametric method to construct the curve for case-control studies. We further introduce a global predictiveness U and a partial predictiveness U to summarize prediction curve across the whole population and sub-population of clinical interest, respectively. We also demonstrate the connections of predictiveness curve with ROC curve and Lorenz curve. Through simulation, we compared the performance of the predictiveness U to other three summary indices: R square, Total Gain, and Average Entropy, and showed that Predictiveness U outperformed the other three indexes in terms of unbiasedness and robustness. Moreover, we simulated a series of rare-variants disease model, found partial predictiveness U performed better than global predictiveness U. Finally, we conducted a real data analysis, using predictiveness curve and predictiveness U to evaluate a risk prediction model for Nicotine Dependence. 

**Abstract (ZH)**: 高通量基因分型和测序技术的发展使得全面评估大量遗传预测因子在疾病预测中的作用成为可能。因此，迫切需要开发适当的统计测量方法来评估这些遗传变异的综合效应对疾病预测的影响。预测曲线常被用作图形工具来衡量风险预测模型在单一连续生物标志物上的预测能力。然而，对于大多数复杂的疾病，风险预测模型是基于多个遗传变异形成的。因此，我们提出了一种多标记预测曲线，并提供了一种非参数方法来构建这种曲线，适用于病例对照研究。我们还引入了全局预测曲线U和部分预测曲线U来分别总结整个群体和临床兴趣子群体的预测曲线。我们还展示了预测曲线与ROC曲线和洛伦兹曲线之间的联系。通过模拟，我们将预测曲线U的性能与其他三个汇总指标（决定系数R平方、总增益和平均熵）进行了比较，并展示了预测曲线U在无偏性和稳健性方面的优越性。此外，我们模拟了一系列稀有变异疾病模型，发现部分预测曲线U在某些情况下优于全局预测曲线U。最后，我们进行了一项实际数据分析，使用预测曲线和预测曲线U来评估尼古丁依赖的风险预测模型。 

---
# Deep Learning-Based Hypoglycemia Classification Across Multiple Prediction Horizons 

**Title (ZH)**: 基于深度学习的多预测 horizons 低血糖分类 

**Authors**: Beyza Cinar, Jennifer Daniel Onwuchekwa, Maria Maleshkova  

**Link**: [PDF](https://arxiv.org/pdf/2504.00009)  

**Abstract**: Type 1 diabetes (T1D) management can be significantly enhanced through the use of predictive machine learning (ML) algorithms, which can mitigate the risk of adverse events like hypoglycemia. Hypoglycemia, characterized by blood glucose levels below 70 mg/dL, is a life-threatening condition typically caused by excessive insulin administration, missed meals, or physical activity. Its asymptomatic nature impedes timely intervention, making ML models crucial for early detection. This study integrates short- (up to 2h) and long-term (up to 24h) prediction horizons (PHs) within a single classification model to enhance decision support. The predicted times are 5-15 min, 15-30 min, 30 min-1h, 1-2h, 2-4h, 4-8h, 8-12h, and 12-24h before hypoglycemia. In addition, a simplified model classifying up to 4h before hypoglycemia is compared. We trained ResNet and LSTM models on glucose levels, insulin doses, and acceleration data. The results demonstrate the superiority of the LSTM models when classifying nine classes. In particular, subject-specific models yielded better performance but achieved high recall only for classes 0, 1, and 2 with 98%, 72%, and 50%, respectively. A population-based six-class model improved the results with at least 60% of events detected. In contrast, longer PHs remain challenging with the current approach and may be considered with different models. 

**Abstract (ZH)**: 通过使用预测机器学习算法，1型糖尿病管理可以显著增强，从而减轻低血糖等不良事件的风险。 

---
# Tensor Generalized Approximate Message Passing 

**Title (ZH)**: 张量广义消息传递逼近 

**Authors**: Yinchuan Li, Guangchen Lan, Xiaodong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00008)  

**Abstract**: We propose a tensor generalized approximate message passing (TeG-AMP) algorithm for low-rank tensor inference, which can be used to solve tensor completion and decomposition problems. We derive TeG-AMP algorithm as an approximation of the sum-product belief propagation algorithm in high dimensions where the central limit theorem and Taylor series approximations are applicable. As TeG-AMP is developed based on a general TR decomposition model, it can be directly applied to many low-rank tensor types. Moreover, our TeG-AMP can be simplified based on the CP decomposition model and a tensor simplified AMP is proposed for low CP-rank tensor inference problems. Experimental results demonstrate that the proposed methods significantly improve recovery performances since it takes full advantage of tensor structures. 

**Abstract (ZH)**: 我们提出了一种张量广义消息传递（TeG-AMP）算法用于低秩张量推理，可以用于解决张量填充和分解问题。 

---
