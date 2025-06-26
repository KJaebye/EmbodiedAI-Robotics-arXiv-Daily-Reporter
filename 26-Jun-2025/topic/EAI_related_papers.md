# DemoDiffusion: One-Shot Human Imitation using pre-trained Diffusion Policy 

**Title (ZH)**: DemoDiffusion：基于预训练扩散策略的一次性人体模仿 

**Authors**: Sungjae Park, Homanga Bharadhwaj, Shubham Tulsiani  

**Link**: [PDF](https://arxiv.org/pdf/2506.20668)  

**Abstract**: We propose DemoDiffusion, a simple and scalable method for enabling robots to perform manipulation tasks in natural environments by imitating a single human demonstration. Our approach is based on two key insights. First, the hand motion in a human demonstration provides a useful prior for the robot's end-effector trajectory, which we can convert into a rough open-loop robot motion trajectory via kinematic retargeting. Second, while this retargeted motion captures the overall structure of the task, it may not align well with plausible robot actions in-context. To address this, we leverage a pre-trained generalist diffusion policy to modify the trajectory, ensuring it both follows the human motion and remains within the distribution of plausible robot actions. Our approach avoids the need for online reinforcement learning or paired human-robot data, enabling robust adaptation to new tasks and scenes with minimal manual effort. Experiments in both simulation and real-world settings show that DemoDiffusion outperforms both the base policy and the retargeted trajectory, enabling the robot to succeed even on tasks where the pre-trained generalist policy fails entirely. Project page: this https URL 

**Abstract (ZH)**: 我们提出了一种名为DemoDiffusion的简单可扩展方法，通过模仿单个人类示范，使机器人能够在自然环境中执行操作任务。我们的方法基于两个关键见解。首先，人类示范中的手部运动为机器人的末端执行器轨迹提供了有用的先验知识，我们可以通过运动目标转换将其转换为粗糙的开环机器人运动轨迹。其次，虽然这种目标转换的运动捕捉了任务的整体结构，但在具体情境下可能不符合合理的机器人动作。为此，我们利用预先训练的一般扩散策略来修改轨迹，确保其既能遵循人类运动，又能保持在合理的机器人动作分布之内。我们的方法避免了在线强化学习或人-机器人配对数据的需求，使得机器人能够在最少的人工努力下，对新任务和场景进行稳健的适应。实验结果表明，DemoDiffusion在仿真和真实环境设置中均优于基线策略和目标转换轨迹，即使在预训练的一般策略完全失败的任务中，也能使机器人成功执行。项目页面: 这里 

---
# HRIBench: Benchmarking Vision-Language Models for Real-Time Human Perception in Human-Robot Interaction 

**Title (ZH)**: HRIBench: 用于人类-机器人交互实时视觉-语言模型感知基准测试 

**Authors**: Zhonghao Shi, Enyu Zhao, Nathaniel Dennler, Jingzhen Wang, Xinyang Xu, Kaleen Shrestha, Mengxue Fu, Daniel Seita, Maja Matarić  

**Link**: [PDF](https://arxiv.org/pdf/2506.20566)  

**Abstract**: Real-time human perception is crucial for effective human-robot interaction (HRI). Large vision-language models (VLMs) offer promising generalizable perceptual capabilities but often suffer from high latency, which negatively impacts user experience and limits VLM applicability in real-world scenarios. To systematically study VLM capabilities in human perception for HRI and performance-latency trade-offs, we introduce HRIBench, a visual question-answering (VQA) benchmark designed to evaluate VLMs across a diverse set of human perceptual tasks critical for HRI. HRIBench covers five key domains: (1) non-verbal cue understanding, (2) verbal instruction understanding, (3) human-robot object relationship understanding, (4) social navigation, and (5) person identification. To construct HRIBench, we collected data from real-world HRI environments to curate questions for non-verbal cue understanding, and leveraged publicly available datasets for the remaining four domains. We curated 200 VQA questions for each domain, resulting in a total of 1000 questions for HRIBench. We then conducted a comprehensive evaluation of both state-of-the-art closed-source and open-source VLMs (N=11) on HRIBench. Our results show that, despite their generalizability, current VLMs still struggle with core perceptual capabilities essential for HRI. Moreover, none of the models within our experiments demonstrated a satisfactory performance-latency trade-off suitable for real-time deployment, underscoring the need for future research on developing smaller, low-latency VLMs with improved human perception capabilities. HRIBench and our results can be found in this Github repository: this https URL. 

**Abstract (ZH)**: 实时人类感知对于有效的机器人人类交互（HRI）至关重要。大型视觉-语言模型（VLMs）提供了广泛适用的感知能力，但往往遭受高延迟的影响，这会负面影响用户体验，并限制VLM在实际场景中的应用。为了系统地研究VLM在HRI中的人类感知能力及其性能-延迟权衡，我们引入了HRIBench这一视觉问答（VQA）基准，旨在评估VLM在HRI中关键的人类感知任务上的表现。HRIBench涵盖了五个关键领域：（1）非语言暗示理解，（2）口头指示理解，（3）人机器人物体关系理解，（4）社会导航，（5）人物识别。为了构建HRIBench，我们从实际的HRI环境收集数据以构建非语言暗示领域的问答问题，并利用公开可用的数据集构建其余四个领域。我们为每个领域构建了200个视觉问答问题，总共构建了1000个问题。然后，我们在HRIBench上对11个最先进的商用闭源和开源VLM进行了全面评估。结果显示，尽管VLMs具有普遍适用性，但它们仍然在HRI中核心的感知能力方面存在不足。此外，我们实验中的所有模型均未表现出适合实时部署的令人满意的性能-延迟权衡，突显了未来研究开发更小的低延迟、具有改进的人类感知能力的VLM的重要性。HRIBench和我们的结果可在以下GitHub仓库中找到：this https URL。 

---
# Behavior Foundation Model: Towards Next-Generation Whole-Body Control System of Humanoid Robots 

**Title (ZH)**: 行为基础模型：面向下一代类人机器人全身控制系统的探索 

**Authors**: Mingqi Yuan, Tao Yu, Wenqi Ge, Xiuyong Yao, Dapeng Li, Huijiang Wang, Jiayu Chen, Xin Jin, Bo Li, Hua Chen, Wei Zhang, Wenjun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.20487)  

**Abstract**: Humanoid robots are drawing significant attention as versatile platforms for complex motor control, human-robot interaction, and general-purpose physical intelligence. However, achieving efficient whole-body control (WBC) in humanoids remains a fundamental challenge due to sophisticated dynamics, underactuation, and diverse task requirements. While learning-based controllers have shown promise for complex tasks, their reliance on labor-intensive and costly retraining for new scenarios limits real-world applicability. To address these limitations, behavior(al) foundation models (BFMs) have emerged as a new paradigm that leverages large-scale pretraining to learn reusable primitive skills and behavioral priors, enabling zero-shot or rapid adaptation to a wide range of downstream tasks. In this paper, we present a comprehensive overview of BFMs for humanoid WBC, tracing their development across diverse pre-training pipelines. Furthermore, we discuss real-world applications, current limitations, urgent challenges, and future opportunities, positioning BFMs as a key approach toward scalable and general-purpose humanoid intelligence. Finally, we provide a curated and long-term list of BFM papers and projects to facilitate more subsequent research, which is available at this https URL. 

**Abstract (ZH)**: 类人机器人作为复杂运动控制、人机交互和通用物理智能的多功能平台正吸引着广泛关注。然而，由于复杂的动力学、欠驱动以及多样的任务要求，实现高效的整体身体控制（WBC）仍然是一个根本性的挑战。虽然基于学习的控制器在复杂任务中显示出潜力，但它们依赖于劳动密集型且昂贵的新场景重新训练限制了其实用性。为了解决这些局限性，行为基础模型（BFMs）作为一种新的范式出现，利用大规模预训练学习可重用的基本技能和行为先验，从而能够对各种下游任务实现零样本或快速适应。在本文中，我们提供了一种全面的BFMs概述，追溯了它们在不同预训练管道中的发展过程。此外，我们还讨论了BFMs的实际应用、当前局限性、紧迫挑战和未来机会，将BFMs定位为走向可扩展和通用型类人智能的关键方法。最后，我们提供了一份精心策划的长期BFMs论文和项目列表，以促进后续研究，该列表可在以下链接访问：this https URL。 

---
# A Review of Personalisation in Human-Robot Collaboration and Future Perspectives Towards Industry 5.0 

**Title (ZH)**: 人类与机器人协作中的个性化综述及面向工业4.0的未来展望 

**Authors**: James Fant-Male, Roel Pieters  

**Link**: [PDF](https://arxiv.org/pdf/2506.20447)  

**Abstract**: The shift in research focus from Industry 4.0 to Industry 5.0 (I5.0) promises a human-centric workplace, with social and well-being values at the centre of technological implementation. Human-Robot Collaboration (HRC) is a core aspect of I5.0 development, with an increase in adaptive and personalised interactions and behaviours. This review investigates recent advancements towards personalised HRC, where user-centric adaption is key. There is a growing trend for adaptable HRC research, however there lacks a consistent and unified approach. The review highlights key research trends on which personal factors are considered, workcell and interaction design, and adaptive task completion. This raises various key considerations for future developments, particularly around the ethical and regulatory development of personalised systems, which are discussed in detail. 

**Abstract (ZH)**: 从 Industry 4.0 到 Industry 5.0 (I5.0) 的研究重点转移：以人类为中心的工作场所及其技术实施，强调社会和福祉价值。人机协作 (HRC) 是 I5.0 发展的核心方面，涉及适应性和个性化交互与行为的增加。本文回顾了近期朝着个性化 HRC 的进步，其中用户中心的适应性是关键。尽管可适应的 HRC 研究呈增长趋势，但缺乏一致和统一的方法。本文总结了关键的研究趋势，重点考虑个人因素、工作单元和交互设计以及适应性任务完成。这提出了对未来发展的各种关键考虑，特别是在个性化系统的伦理和法规发展方面，详细讨论了这些问题。 

---
# Multimodal Behaviour Trees for Robotic Laboratory Task Automation 

**Title (ZH)**: 多模态行为树在机器人实验室任务自动化中的应用 

**Authors**: Hatem Fakhruldeen, Arvind Raveendran Nambiar, Satheeshkumar Veeramani, Bonilkumar Vijaykumar Tailor, Hadi Beyzaee Juneghani, Gabriella Pizzuto, Andrew Ian Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2506.20399)  

**Abstract**: Laboratory robotics offer the capability to conduct experiments with a high degree of precision and reproducibility, with the potential to transform scientific research. Trivial and repeatable tasks; e.g., sample transportation for analysis and vial capping are well-suited for robots; if done successfully and reliably, chemists could contribute their efforts towards more critical research activities. Currently, robots can perform these tasks faster than chemists, but how reliable are they? Improper capping could result in human exposure to toxic chemicals which could be fatal. To ensure that robots perform these tasks as accurately as humans, sensory feedback is required to assess the progress of task execution. To address this, we propose a novel methodology based on behaviour trees with multimodal perception. Along with automating robotic tasks, this methodology also verifies the successful execution of the task, a fundamental requirement in safety-critical environments. The experimental evaluation was conducted on two lab tasks: sample vial capping and laboratory rack insertion. The results show high success rate, i.e., 88% for capping and 92% for insertion, along with strong error detection capabilities. This ultimately proves the robustness and reliability of our approach and that using multimodal behaviour trees should pave the way towards the next generation of robotic chemists. 

**Abstract (ZH)**: 实验室机器人提供高精度和可重复性的实验能力，有望变革科学研究。基于行为树的多模态感知方法在样本瓶封盖和实验室架插入等任务中的应用验证了其可靠性和鲁棒性，推动了新一代机器人化学家的发展。 

---
# SPARK: Graph-Based Online Semantic Integration System for Robot Task Planning 

**Title (ZH)**: SPARK：基于图的机器人任务规划在线语义集成系统 

**Authors**: Mimo Shirasaka, Yuya Ikeda, Tatsuya Matsushima, Yutaka Matsuo, Yusuke Iwasawa  

**Link**: [PDF](https://arxiv.org/pdf/2506.20394)  

**Abstract**: The ability to update information acquired through various means online during task execution is crucial for a general-purpose service robot. This information includes geometric and semantic data. While SLAM handles geometric updates on 2D maps or 3D point clouds, online updates of semantic information remain unexplored. We attribute the challenge to the online scene graph representation, for its utility and scalability. Building on previous works regarding offline scene graph representations, we study online graph representations of semantic information in this work. We introduce SPARK: Spatial Perception and Robot Knowledge Integration. This framework extracts semantic information from environment-embedded cues and updates the scene graph accordingly, which is then used for subsequent task planning. We demonstrate that graph representations of spatial relationships enhance the robot system's ability to perform tasks in dynamic environments and adapt to unconventional spatial cues, like gestures. 

**Abstract (ZH)**: 基于各种在线手段获取的信息更新能力对于通用服务机器人在任务执行过程中至关重要。这些信息包括几何和语义数据。虽然SLAM处理二维地图或三维点云的几何更新，但语义信息的在线更新尚未受到探索。我们归因于在线场景图表示的实用性和可扩展性带来的挑战。基于关于离线场景图表示的前期工作，我们在本文中研究语义信息的在线图表示。我们引入了SPARK：空间感知与机器人知识整合框架。该框架从环境嵌入的线索中提取语义信息，并相应地更新场景图，然后用于后续任务规划。我们展示了图表示的空间关系增强了机器人系统在动态环境中执行任务并在遇到不寻常的空间线索（如手势）时进行适应的能力。 

---
# CARMA: Context-Aware Situational Grounding of Human-Robot Group Interactions by Combining Vision-Language Models with Object and Action Recognition 

**Title (ZH)**: CARMA：结合视觉语言模型、物体识别与动作识别的基于上下文的情境化人类-机器人组交互接地技术 

**Authors**: Joerg Deigmoeller, Stephan Hasler, Nakul Agarwal, Daniel Tanneberg, Anna Belardinelli, Reza Ghoddoosian, Chao Wang, Felix Ocker, Fan Zhang, Behzad Dariush, Michael Gienger  

**Link**: [PDF](https://arxiv.org/pdf/2506.20373)  

**Abstract**: We introduce CARMA, a system for situational grounding in human-robot group interactions. Effective collaboration in such group settings requires situational awareness based on a consistent representation of present persons and objects coupled with an episodic abstraction of events regarding actors and manipulated objects. This calls for a clear and consistent assignment of instances, ensuring that robots correctly recognize and track actors, objects, and their interactions over time. To achieve this, CARMA uniquely identifies physical instances of such entities in the real world and organizes them into grounded triplets of actors, objects, and actions.
To validate our approach, we conducted three experiments, where multiple humans and a robot interact: collaborative pouring, handovers, and sorting. These scenarios allow the assessment of the system's capabilities as to role distinction, multi-actor awareness, and consistent instance identification. Our experiments demonstrate that the system can reliably generate accurate actor-action-object triplets, providing a structured and robust foundation for applications requiring spatiotemporal reasoning and situated decision-making in collaborative settings. 

**Abstract (ZH)**: CARMA：人类-机器人团队互动中的情境接地系统 

---
# PIMBS: Efficient Body Schema Learning for Musculoskeletal Humanoids with Physics-Informed Neural Networks 

**Title (ZH)**: PIMBS：基于物理信息神经网络的高效身体方案学习方法用于肌骨骼类人机器人 

**Authors**: Kento Kawaharazuka, Takahiro Hattori, Keita Yoneda, Kei Okada  

**Link**: [PDF](https://arxiv.org/pdf/2506.20343)  

**Abstract**: Musculoskeletal humanoids are robots that closely mimic the human musculoskeletal system, offering various advantages such as variable stiffness control, redundancy, and flexibility. However, their body structure is complex, and muscle paths often significantly deviate from geometric models. To address this, numerous studies have been conducted to learn body schema, particularly the relationships among joint angles, muscle tension, and muscle length. These studies typically rely solely on data collected from the actual robot, but this data collection process is labor-intensive, and learning becomes difficult when the amount of data is limited. Therefore, in this study, we propose a method that applies the concept of Physics-Informed Neural Networks (PINNs) to the learning of body schema in musculoskeletal humanoids, enabling high-accuracy learning even with a small amount of data. By utilizing not only data obtained from the actual robot but also the physical laws governing the relationship between torque and muscle tension under the assumption of correct joint structure, more efficient learning becomes possible. We apply the proposed method to both simulation and an actual musculoskeletal humanoid and discuss its effectiveness and characteristics. 

**Abstract (ZH)**: 基于物理约束的神经网络在肌骨骼人形机器人身体模式学习中的应用 

---
# Building Forest Inventories with Autonomous Legged Robots -- System, Lessons, and Challenges Ahead 

**Title (ZH)**: 使用自主腿式机器人建立森林inventory系统、经验与未来挑战 

**Authors**: Matías Mattamala, Nived Chebrolu, Jonas Frey, Leonard Freißmuth, Haedam Oh, Benoit Casseau, Marco Hutter, Maurice Fallon  

**Link**: [PDF](https://arxiv.org/pdf/2506.20315)  

**Abstract**: Legged robots are increasingly being adopted in industries such as oil, gas, mining, nuclear, and agriculture. However, new challenges exist when moving into natural, less-structured environments, such as forestry applications. This paper presents a prototype system for autonomous, under-canopy forest inventory with legged platforms. Motivated by the robustness and mobility of modern legged robots, we introduce a system architecture which enabled a quadruped platform to autonomously navigate and map forest plots. Our solution involves a complete navigation stack for state estimation, mission planning, and tree detection and trait estimation. We report the performance of the system from trials executed over one and a half years in forests in three European countries. Our results with the ANYmal robot demonstrate that we can survey plots up to 1 ha plot under 30 min, while also identifying trees with typical DBH accuracy of 2cm. The findings of this project are presented as five lessons and challenges. Particularly, we discuss the maturity of hardware development, state estimation limitations, open problems in forest navigation, future avenues for robotic forest inventory, and more general challenges to assess autonomous systems. By sharing these lessons and challenges, we offer insight and new directions for future research on legged robots, navigation systems, and applications in natural environments. Additional videos can be found in this https URL 

**Abstract (ZH)**: 基于腿式机器人在欧洲三国森林环境下自主林分调查的原型系统研究 

---
# Why Robots Are Bad at Detecting Their Mistakes: Limitations of Miscommunication Detection in Human-Robot Dialogue 

**Title (ZH)**: 为什么机器人不擅长检测其错误：人类-机器人对话中错误沟通检测的局限性 

**Authors**: Ruben Janssens, Jens De Bock, Sofie Labat, Eva Verhelst, Veronique Hoste, Tony Belpaeme  

**Link**: [PDF](https://arxiv.org/pdf/2506.20268)  

**Abstract**: Detecting miscommunication in human-robot interaction is a critical function for maintaining user engagement and trust. While humans effortlessly detect communication errors in conversations through both verbal and non-verbal cues, robots face significant challenges in interpreting non-verbal feedback, despite advances in computer vision for recognizing affective expressions. This research evaluates the effectiveness of machine learning models in detecting miscommunications in robot dialogue. Using a multi-modal dataset of 240 human-robot conversations, where four distinct types of conversational failures were systematically introduced, we assess the performance of state-of-the-art computer vision models. After each conversational turn, users provided feedback on whether they perceived an error, enabling an analysis of the models' ability to accurately detect robot mistakes. Despite using state-of-the-art models, the performance barely exceeds random chance in identifying miscommunication, while on a dataset with more expressive emotional content, they successfully identified confused states. To explore the underlying cause, we asked human raters to do the same. They could also only identify around half of the induced miscommunications, similarly to our model. These results uncover a fundamental limitation in identifying robot miscommunications in dialogue: even when users perceive the induced miscommunication as such, they often do not communicate this to their robotic conversation partner. This knowledge can shape expectations of the performance of computer vision models and can help researchers to design better human-robot conversations by deliberately eliciting feedback where needed. 

**Abstract (ZH)**: 检测人类与机器人交互中的沟通错误对于维持用户参与度和信任至关重要。尽管人类可以通过口头和非口头线索轻松检测交流错误，机器人在解读非口头反馈方面仍面临重大挑战，尽管在通过计算机视觉识别情感表达方面取得了进展。本研究评估了机器学习模型在检测机器人对话中的沟通错误方面的有效性。使用包含240轮人类与机器人对话的多模态数据集，系统地引入了四种不同类型的对话失败，评估了最先进的计算机视觉模型的性能。在每次对话回合后，用户提供了他们是否察觉到错误的反馈，从而分析了模型准确检测机器人错误的能力。尽管使用了最先进的模型，但在识别沟通错误方面的性能几乎没有超过随机猜测，在具有更多表达性情感内容的数据集上，它们能够识别出困惑状态。为了探索背后的原因，我们让人类评分者也做了同样的事情。他们也只能识别出大约一半诱导的沟通错误，与我们的模型类似。这些结果揭示了检测对话中机器人沟通错误的基本局限性：即使用户察觉到诱导的沟通错误，他们也往往不会将其传达给其机器人对话伙伴。这些知识可以塑造对计算机视觉模型性能的期望，并有助于研究人员通过适时诱发反馈来设计更好的人机对话。 

---
# Personalized Mental State Evaluation in Human-Robot Interaction using Federated Learning 

**Title (ZH)**: 基于联邦学习的人机交互个性化心理状态评估 

**Authors**: Andrea Bussolan, Oliver Avram, Andrea Pignata, Gianvito Urgese, Stefano Baraldo, Anna Valente  

**Link**: [PDF](https://arxiv.org/pdf/2506.20212)  

**Abstract**: With the advent of Industry 5.0, manufacturers are increasingly prioritizing worker well-being alongside mass customization. Stress-aware Human-Robot Collaboration (HRC) plays a crucial role in this paradigm, where robots must adapt their behavior to human mental states to improve collaboration fluency and safety. This paper presents a novel framework that integrates Federated Learning (FL) to enable personalized mental state evaluation while preserving user privacy. By leveraging physiological signals, including EEG, ECG, EDA, EMG, and respiration, a multimodal model predicts an operator's stress level, facilitating real-time robot adaptation. The FL-based approach allows distributed on-device training, ensuring data confidentiality while improving model generalization and individual customization. Results demonstrate that the deployment of an FL approach results in a global model with performance in stress prediction accuracy comparable to a centralized training approach. Moreover, FL allows for enhancing personalization, thereby optimizing human-robot interaction in industrial settings, while preserving data privacy. The proposed framework advances privacy-preserving, adaptive robotics to enhance workforce well-being in smart manufacturing. 

**Abstract (ZH)**: 随着 industrie 5.0 的到来，制造商越来越重视在大规模个性化生产的同时保障工人的福祉。具备压力感知能力的人机协作（HRC）在此范式中发挥着关键作用，其中机器人必须根据人类的心理状态调整其行为，以提高协作流畅性和安全性。本文提出了一种将联邦学习（FL）集成的新框架，以实现个性化心理健康评估并同时保护用户隐私。通过利用包括EEG、ECG、EDA、EMG和呼吸在内的生理信号，一个多模态模型预测操作员的压力水平，从而实现实时机器人适应。基于联邦学习的方法允许分布式设备上训练，确保数据机密性的同时提高模型泛化能力和个体定制能力。结果显示，部署基于联邦学习的方法能够获得与集中训练方法在压力预测准确性方面相当的全局模型。此外，联邦学习还有助于提高个性化水平，从而优化工业环境中的人机交互，同时保护数据隐私。所提出的框架促进了保护隐私的人机适应性技术的发展，以提高智能制造业的工作环境质量。 

---
# PSALM-V: Automating Symbolic Planning in Interactive Visual Environments with Large Language Models 

**Title (ZH)**: PSALM-V: 在大型语言模型指导下自动在交互式视觉环境中进行符号规划 

**Authors**: Wang Bill Zhu, Miaosen Chai, Ishika Singh, Robin Jia, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2506.20097)  

**Abstract**: We propose PSALM-V, the first autonomous neuro-symbolic learning system able to induce symbolic action semantics (i.e., pre- and post-conditions) in visual environments through interaction. PSALM-V bootstraps reliable symbolic planning without expert action definitions, using LLMs to generate heuristic plans and candidate symbolic semantics. Previous work has explored using large language models to generate action semantics for Planning Domain Definition Language (PDDL)-based symbolic planners. However, these approaches have primarily focused on text-based domains or relied on unrealistic assumptions, such as access to a predefined problem file, full observability, or explicit error messages. By contrast, PSALM-V dynamically infers PDDL problem files and domain action semantics by analyzing execution outcomes and synthesizing possible error explanations. The system iteratively generates and executes plans while maintaining a tree-structured belief over possible action semantics for each action, iteratively refining these beliefs until a goal state is reached. Simulated experiments of task completion in ALFRED demonstrate that PSALM-V increases the plan success rate from 37% (Claude-3.7) to 74% in partially observed setups. Results on two 2D game environments, RTFM and Overcooked-AI, show that PSALM-V improves step efficiency and succeeds in domain induction in multi-agent settings. PSALM-V correctly induces PDDL pre- and post-conditions for real-world robot BlocksWorld tasks, despite low-level manipulation failures from the robot. 

**Abstract (ZH)**: 我们提出PSALM-V，这是一种能够在视觉环境中通过交互自动诱导符号动作语义（即前件和后件）的第一种自主神经符号学习系统。PSALM-V 无需专家动作定义即可启动可靠的符号规划，使用大语言模型生成启发式计划和候选符号语义。此前的工作探索了使用大语言模型为基于PDDL的符号规划器生成动作语义。然而，这些方法主要集中在文本基础领域，或者依赖于不切实际的假设，如访问预定义的问题文件、完全可观测性或明确的错误消息。相比之下，PSALM-V 动态推断 PDDL 问题文件和领域动作语义，通过分析执行结果并综合可能的错误解释。该系统迭代生成和执行计划，同时维护每个动作可能的动作语义的树状信念结构，直到达到目标状态。在 ALFRED 的模拟任务完成实验中，PSALM-V 在部分可观测设置中的计划成功率从 Claude-3.7 的 37% 提高到 74%。对两个 2D 游戏环境 RTFM 和 Overcooked-AI 的结果表明，PSALM-V 在多智能体设置中提高了步骤效率并成功进行了领域诱导。尽管机器人在低级操作中出现故障，PSALM-V 仍能正确诱导 PDDL 前件和后件，以完成真实世界的机器人 BlocksWorld 任务。 

---
# Robust Robotic Exploration and Mapping Using Generative Occupancy Map Synthesis 

**Title (ZH)**: 基于生成占用地图合成的鲁棒机器人探索与建图 

**Authors**: Lorin Achey, Alec Reed, Brendan Crowe, Bradley Hayes, Christoffer Heckman  

**Link**: [PDF](https://arxiv.org/pdf/2506.20049)  

**Abstract**: We present a novel approach for enhancing robotic exploration by using generative occupancy mapping. We introduce SceneSense, a diffusion model designed and trained for predicting 3D occupancy maps given partial observations. Our proposed approach probabilistically fuses these predictions into a running occupancy map in real-time, resulting in significant improvements in map quality and traversability. We implement SceneSense onboard a quadruped robot and validate its performance with real-world experiments to demonstrate the effectiveness of the model. In these experiments, we show that occupancy maps enhanced with SceneSense predictions better represent our fully observed ground truth data (24.44% FID improvement around the robot and 75.59% improvement at range). We additionally show that integrating SceneSense-enhanced maps into our robotic exploration stack as a "drop-in" map improvement, utilizing an existing off-the-shelf planner, results in improvements in robustness and traversability time. Finally we show results of full exploration evaluations with our proposed system in two dissimilar environments and find that locally enhanced maps provide more consistent exploration results than maps constructed only from direct sensor measurements. 

**Abstract (ZH)**: 我们提出了一种利用生成式占用映射增强机器人探索的新方法。我们引入了SceneSense，这是一种用于预测3D占用映射的扩散模型，该模型经过设计和训练，可以处理部分观测数据。我们的方法将这些预测以概率方式实时融合到运行中的占用映射中，从而显著提高了地图的质量和可通行性。我们在腿式机器人上实现了SceneSense，并通过实地实验验证了该模型的性能，展示了其有效性。在这些实验中，我们展示了使用SceneSense预测增强的占用映射更好地代表了我们完全观测到的真实数据（机器人周围区域FID改进24.44%，远距离区域改进75.59%）。此外，我们将SceneSense增强的地图作为“即插即用”的地图改进集成到我们的机器人探索框架中，利用现有的现成规划器，结果表明这提高了鲁棒性和可通行性时间。最后，我们在两种不同的环境中对所提出的系统进行了全面探索评估，并发现局部增强的地图提供了比仅从直接传感器测量构建的地图更一致的探索结果。 

---
# Hierarchical Reinforcement Learning and Value Optimization for Challenging Quadruped Locomotion 

**Title (ZH)**: 层次强化学习与值优化在挑战性 quadruped 运动控制中的应用 

**Authors**: Jeremiah Coholich, Muhammad Ali Murtaza, Seth Hutchinson, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2506.20036)  

**Abstract**: We propose a novel hierarchical reinforcement learning framework for quadruped locomotion over challenging terrain. Our approach incorporates a two-layer hierarchy in which a high-level policy (HLP) selects optimal goals for a low-level policy (LLP). The LLP is trained using an on-policy actor-critic RL algorithm and is given footstep placements as goals. We propose an HLP that does not require any additional training or environment samples and instead operates via an online optimization process over the learned value function of the LLP. We demonstrate the benefits of this framework by comparing it with an end-to-end reinforcement learning (RL) approach. We observe improvements in its ability to achieve higher rewards with fewer collisions across an array of different terrains, including terrains more difficult than any encountered during training. 

**Abstract (ZH)**: 我们提出了一种新型分层强化学习框架，用于在挑战性地形上实现四足运动。该方法采用两层结构，高层策略（HLP）选择适合低层策略（LLP）执行的任务目标。LLP使用基于策略的演员-评论家RL算法进行训练，并由脚印放置位置作为目标。我们提出了一种HLP，它不需要额外的训练或环境样本，而是通过在线优化过程在LLP学习的价值函数上运行。我们通过将其与端到端的强化学习（RL）方法进行比较，展示了该框架的优势。我们观察到，该框架在不同地形（包括训练中遇到的更难的地形）上实现了更高的奖励并减少了碰撞。 

---
# Robust Embodied Self-Identification of Morphology in Damaged Multi-Legged Robots 

**Title (ZH)**: 受损多足机器人鲁棒体体现有结构自我识别 

**Authors**: Sahand Farghdani, Mili Patel, Robin Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2506.19984)  

**Abstract**: Multi-legged robots (MLRs) are vulnerable to leg damage during complex missions, which can impair their performance. This paper presents a self-modeling and damage identification algorithm that enables autonomous adaptation to partial or complete leg loss using only data from a low-cost IMU. A novel FFT-based filter is introduced to address time-inconsistent signals, improving damage detection by comparing body orientation between the robot and its model. The proposed method identifies damaged legs and updates the robot's model for integration into its control system. Experiments on uneven terrain validate its robustness and computational efficiency. 

**Abstract (ZH)**: 多腿机器人在复杂任务中易遭受腿部损伤，影响其性能。本文提出了一种自建模和损伤识别算法，仅使用低成本IMU的数据，实现对部分或完全腿部损失的自主适应。引入了一种新型FFTベース的滤波器来处理时间不一致的信号，通过比较机器人与其模型之间的身体姿态来提高损伤检测的准确性。所提出的方法能够识别受损腿部并更新机器人的模型，以便将其集成到控制系统中。实验结果在不平地面上验证了其稳健性和计算效率。 

---
# Evolutionary Gait Reconfiguration in Damaged Legged Robots 

**Title (ZH)**: 受损腿式机器人中的进化步态重构 

**Authors**: Sahand Farghdani, Robin Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2506.19968)  

**Abstract**: Multi-legged robots deployed in complex missions are susceptible to physical damage in their legs, impairing task performance and potentially compromising mission success. This letter presents a rapid, training-free damage recovery algorithm for legged robots subject to partial or complete loss of functional legs. The proposed method first stabilizes locomotion by generating a new gait sequence and subsequently optimally reconfigures leg gaits via a developed differential evolution algorithm to maximize forward progression while minimizing body rotation and lateral drift. The algorithm successfully restores locomotion in a 24-degree-of-freedom hexapod within one hour, demonstrating both high efficiency and robustness to structural damage. 

**Abstract (ZH)**: 具有复杂任务需求的多足机器人容易遭受腿部物理损伤，影响任务性能并可能危及任务成功。本信提出了一种无需训练的快速损伤恢复算法，用于在部分或完全丧失功能性腿部的情况下优化腿足运动。该方法首先通过生成新的步态序列来稳定运动，然后利用开发的差分进化算法重新配置腿足运动，以最大化前进距离的同时最小化身体旋转和侧向漂移。该算法在一小时内成功恢复了一种具有24个自由度的六足机器人的运动，显示出高效率和对结构损伤的鲁棒性。 

---
# Mixtures of Neural Cellular Automata: A Stochastic Framework for Growth Modelling and Self-Organization 

**Title (ZH)**: 神经细胞自动机的混合模型：一种生长建模和自我组织的随机框架 

**Authors**: Salvatore Milite, Giulio Caravagna, Andrea Sottoriva  

**Link**: [PDF](https://arxiv.org/pdf/2506.20486)  

**Abstract**: Neural Cellular Automata (NCAs) are a promising new approach to model self-organizing processes, with potential applications in life science. However, their deterministic nature limits their ability to capture the stochasticity of real-world biological and physical systems.
We propose the Mixture of Neural Cellular Automata (MNCA), a novel framework incorporating the idea of mixture models into the NCA paradigm. By combining probabilistic rule assignments with intrinsic noise, MNCAs can model diverse local behaviors and reproduce the stochastic dynamics observed in biological processes.
We evaluate the effectiveness of MNCAs in three key domains: (1) synthetic simulations of tissue growth and differentiation, (2) image morphogenesis robustness, and (3) microscopy image segmentation. Results show that MNCAs achieve superior robustness to perturbations, better recapitulate real biological growth patterns, and provide interpretable rule segmentation. These findings position MNCAs as a promising tool for modeling stochastic dynamical systems and studying self-growth processes. 

**Abstract (ZH)**: 混合神经细胞自动机（Mixture of Neural Cellular Automata, MNCA）：一种新的自组织过程建模框架 

---
# Mobile-R1: Towards Interactive Reinforcement Learning for VLM-Based Mobile Agent via Task-Level Rewards 

**Title (ZH)**: Mobile-R1：面向基于VLM的移动代理的交互式强化学习方法及其任务级奖励机制 

**Authors**: Jihao Gu, Qihang Ai, Yingyao Wang, Pi Bu, Jingxuan Xing, Zekun Zhu, Wei Jiang, Ziming Wang, Yingxiu Zhao, Ming-Liang Zhang, Jun Song, Yuning Jiang, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.20332)  

**Abstract**: Vision-language model-based mobile agents have gained the ability to not only understand complex instructions and mobile screenshots, but also optimize their action outputs via thinking and reasoning, benefiting from reinforcement learning, such as Group Relative Policy Optimization (GRPO). However, existing research centers on offline reinforcement learning training or online optimization using action-level rewards, which limits the agent's dynamic interaction with the environment. This often results in agents settling into local optima, thereby weakening their ability for exploration and error action correction. To address these challenges, we introduce an approach called Mobile-R1, which employs interactive multi-turn reinforcement learning with task-level rewards for mobile agents. Our training framework consists of three stages: initial format finetuning, single-step online training via action-level reward, followed by online training via task-level reward based on multi-turn trajectories. This strategy is designed to enhance the exploration and error correction capabilities of Mobile-R1, leading to significant performance improvements. Moreover, we have collected a dataset covering 28 Chinese applications with 24,521 high-quality manual annotations and established a new benchmark with 500 trajectories. We will open source all resources, including the dataset, benchmark, model weight, and codes: this https URL. 

**Abstract (ZH)**: 基于视觉-语言模型的移动代理通过交互式多轮强化学习和任务级奖励优化探索与错误纠正能力 

---
# TRACED: Transition-aware Regret Approximation with Co-learnability for Environment Design 

**Title (ZH)**: TRACED: 考虑转换的遗憾近似与协同学习的环境设计 

**Authors**: Geonwoo Cho, Jaegyun Im, Jihwan Lee, Hojun Yi, Sejin Kim, Sundong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.19997)  

**Abstract**: Generalizing deep reinforcement learning agents to unseen environments remains a significant challenge. One promising solution is Unsupervised Environment Design (UED), a co-evolutionary framework in which a teacher adaptively generates tasks with high learning potential, while a student learns a robust policy from this evolving curriculum. Existing UED methods typically measure learning potential via regret, the gap between optimal and current performance, approximated solely by value-function loss. Building on these approaches, we introduce the transition prediction error as an additional term in our regret approximation. To capture how training on one task affects performance on others, we further propose a lightweight metric called co-learnability. By combining these two measures, we present Transition-aware Regret Approximation with Co-learnability for Environment Design (TRACED). Empirical evaluations show that TRACED yields curricula that improve zero-shot generalization across multiple benchmarks while requiring up to 2x fewer environment interactions than strong baselines. Ablation studies confirm that the transition prediction error drives rapid complexity ramp-up and that co-learnability delivers additional gains when paired with the transition prediction error. These results demonstrate how refined regret approximation and explicit modeling of task relationships can be leveraged for sample-efficient curriculum design in UED. 

**Abstract (ZH)**: 基于转换预测误差与共学习能力的环境设计后悔近似方法（TRACED） 

---
# Causal-Aware Intelligent QoE Optimization for VR Interaction with Adaptive Keyframe Extraction 

**Title (ZH)**: 基于因果意识的自适应关键帧提取以优化VR交互的QoE 

**Authors**: Ziru Zhang, Jiadong Yu, Danny H.K. Tsang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19890)  

**Abstract**: The optimization of quality of experience (QoE) in multi-user virtual reality (VR) interactions demands a delicate balance between ultra-low latency, high-fidelity motion synchronization, and equitable resource allocation. While adaptive keyframe extraction mitigates transmission overhead, existing approaches often overlook the causal relationships among allocated bandwidth, CPU frequency, and user perception, limiting QoE gains. This paper proposes an intelligent framework to maximize QoE by integrating adaptive keyframe extraction with causal-aware reinforcement learning (RL). First, a novel QoE metric is formulated using the Weber-Fechner Law, combining perceptual sensitivity, attention-driven priorities, and motion reconstruction accuracy. The QoE optimization problem is then modeled as a mixed integer programming (MIP) task, jointly optimizing keyframe ratios, bandwidth, and computational resources under horizon-fairness constraints. We propose Partial State Causal Deep Deterministic Policy Gradient (PS-CDDPG), which integrates the Deep Deterministic Policy Gradient (DDPG) method with causal influence detection. By leveraging causal information regarding how QoE is influenced and determined by various actions, we explore actions guided by weights calculated from causal inference (CI), which in turn improves training efficiency. Experiments conducted with the CMU Motion Capture Database demonstrate that our framework significantly reduces interactive latency, enhances QoE, and maintains fairness, achieving superior performance compared to benchmark methods. 

**Abstract (ZH)**: 多用户虚拟现实（VR）交互中体验质量（QoE）的优化需要在超低延迟、高保真运动同步和资源公平分配之间实现精细平衡。现有方法通过自适应关键帧提取减少传输开销，但往往忽视分配带宽、CPU频率与用户感知之间的因果关系，限制了QoE的提升。本文提出了一种智能框架，通过结合自适应关键帧提取与因果感知强化学习（RL）来最大化QoE。首先，利用韦伯-费希纳定律（Weber-Fechner Law）制定一个新型QoE度量标准，结合感知敏感度、注意力驱动的优先级与运动重构精度。随后，将QoE优化问题建模为混合整数规划（MIP）任务，在时间公平约束下联合优化关键帧比例、带宽与计算资源。我们提出了部分状态因果深度确定性策略梯度（PS-CDDPG），将深度确定性策略梯度（DDPG）方法与因果影响检测相结合。通过利用因果信息来指导QoE受不同动作影响的方式，我们探索由因果推断（CI）计算权重引导的动作，从而提高训练效率。实验结果表明，与基准方法相比，我们的框架显著降低了交互延迟、提升了QoE并保持了公平性，性能更优。 

---
