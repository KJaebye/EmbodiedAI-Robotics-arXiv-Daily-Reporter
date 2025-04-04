# Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets 

**Title (ZH)**: 统一的世界模型：结合视频和动作扩散的大规模机器人数据预训练 

**Authors**: Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, Abhishek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2504.02792)  

**Abstract**: Imitation learning has emerged as a promising approach towards building generalist robots. However, scaling imitation learning for large robot foundation models remains challenging due to its reliance on high-quality expert demonstrations. Meanwhile, large amounts of video data depicting a wide range of environments and diverse behaviors are readily available. This data provides a rich source of information about real-world dynamics and agent-environment interactions. Leveraging this data directly for imitation learning, however, has proven difficult due to the lack of action annotation required for most contemporary methods. In this work, we present Unified World Models (UWM), a framework that allows for leveraging both video and action data for policy learning. Specifically, a UWM integrates an action diffusion process and a video diffusion process within a unified transformer architecture, where independent diffusion timesteps govern each modality. We show that by simply controlling each diffusion timestep, UWM can flexibly represent a policy, a forward dynamics, an inverse dynamics, and a video generator. Through simulated and real-world experiments, we show that: (1) UWM enables effective pretraining on large-scale multitask robot datasets with both dynamics and action predictions, resulting in more generalizable and robust policies than imitation learning, (2) UWM naturally facilitates learning from action-free video data through independent control of modality-specific diffusion timesteps, further improving the performance of finetuned policies. Our results suggest that UWM offers a promising step toward harnessing large, heterogeneous datasets for scalable robot learning, and provides a simple unification between the often disparate paradigms of imitation learning and world modeling. Videos and code are available at this https URL. 

**Abstract (ZH)**: 统一世界模型：结合视频和动作数据进行策略学习的方法 

---
# BT-ACTION: A Test-Driven Approach for Modular Understanding of User Instruction Leveraging Behaviour Trees and LLMs 

**Title (ZH)**: 基于行为树和大语言模型的测试驱动模块化用户指令理解方法：BT-ACTION 

**Authors**: Alexander Leszczynski, Sarah Gillet, Iolanda Leite, Fethiye Irmak Dogan  

**Link**: [PDF](https://arxiv.org/pdf/2504.02779)  

**Abstract**: Natural language instructions are often abstract and complex, requiring robots to execute multiple subtasks even for seemingly simple queries. For example, when a user asks a robot to prepare avocado toast, the task involves several sequential steps. Moreover, such instructions can be ambiguous or infeasible for the robot or may exceed the robot's existing knowledge. While Large Language Models (LLMs) offer strong language reasoning capabilities to handle these challenges, effectively integrating them into robotic systems remains a key challenge. To address this, we propose BT-ACTION, a test-driven approach that combines the modular structure of Behavior Trees (BT) with LLMs to generate coherent sequences of robot actions for following complex user instructions, specifically in the context of preparing recipes in a kitchen-assistance setting. We evaluated BT-ACTION in a comprehensive user study with 45 participants, comparing its performance to direct LLM prompting. Results demonstrate that the modular design of BT-ACTION helped the robot make fewer mistakes and increased user trust, and participants showed a significant preference for the robot leveraging BT-ACTION. The code is publicly available at this https URL. 

**Abstract (ZH)**: 自然语言指令往往是抽象且复杂的，即使对于看似简单的查询，机器人也需要执行多个子任务。例如，当用户要求机器人准备牛油果吐司时，该任务涉及多个连续的步骤。此外，这样的指令对于机器人来说可能是模糊的或不可行的，或者超出了机器人的现有知识范围。尽管大型语言模型（LLMs）提供了强大的语言推理能力以应对这些挑战，但有效地将它们集成到机器人系统中仍然是一个关键挑战。为了解决这一问题，我们提出了一种基于测试的方法BT-ACTION，该方法结合了行为树（BT）的模块化结构与LLMs，以生成针对复杂用户指令的一致序列机器人动作，特别是在厨房辅助环境下准备食谱的情境中。我们通过包括45名参与者的全面用户研究评估了BT-ACTION，并将其性能与直接LLM提示进行了对比。结果表明，BT-ACTION的模块化设计帮助机器人减少了错误并提高了用户的信任度，参与者明显更偏好利用BT-ACTION的机器人。代码已公开，可从这个链接获取。 

---
# Robot-Led Vision Language Model Wellbeing Assessment of Children 

**Title (ZH)**: 机器人引领的视觉语言模型儿童福祉评估 

**Authors**: Nida Itrat Abbasi, Fethiye Irmak Dogan, Guy Laban, Joanna Anderson, Tamsin Ford, Peter B. Jones, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2504.02765)  

**Abstract**: This study presents a novel robot-led approach to assessing children's mental wellbeing using a Vision Language Model (VLM). Inspired by the Child Apperception Test (CAT), the social robot NAO presented children with pictorial stimuli to elicit their verbal narratives of the images, which were then evaluated by a VLM in accordance with CAT assessment guidelines. The VLM's assessments were systematically compared to those provided by a trained psychologist. The results reveal that while the VLM demonstrates moderate reliability in identifying cases with no wellbeing concerns, its ability to accurately classify assessments with clinical concern remains limited. Moreover, although the model's performance was generally consistent when prompted with varying demographic factors such as age and gender, a significantly higher false positive rate was observed for girls, indicating potential sensitivity to gender attribute. These findings highlight both the promise and the challenges of integrating VLMs into robot-led assessments of children's wellbeing. 

**Abstract (ZH)**: 本研究提出了一种使用视觉语言模型（VLM）评估儿童心理健康的新型机器人主导方法。受儿童投射测试（CAT）启发，社会机器人NAO向儿童展示了图示刺激，以引发他们对图片的口头叙述，然后根据CAT评估指南由VLM进行评估。VLM的评估结果系统地与受训心理学家提供的评估结果进行了比较。结果表明，虽然VLM在识别无健康问题的情况方面显示出一定的可靠性，但在准确分类存在临床关注的评估方面仍有限制。此外，尽管该模型在面对不同人口统计因素（如年龄和性别）时的一般表现一致，但女性的假阳性率显著较高，这表明模型可能对性别特征较为敏感。这些发现强调了将VLM整合到机器人主导的儿童福祉评估中的潜力与挑战。 

---
# Autonomous Human-Robot Interaction via Operator Imitation 

**Title (ZH)**: 自主人体-机器人交互通过操作员模仿 

**Authors**: Sammy Christen, David Müller, Agon Serifi, Ruben Grandia, Georg Wiedebach, Michael A. Hopkins, Espen Knoop, Moritz Bächer  

**Link**: [PDF](https://arxiv.org/pdf/2504.02724)  

**Abstract**: Teleoperated robotic characters can perform expressive interactions with humans, relying on the operators' experience and social intuition. In this work, we propose to create autonomous interactive robots, by training a model to imitate operator data. Our model is trained on a dataset of human-robot interactions, where an expert operator is asked to vary the interactions and mood of the robot, while the operator commands as well as the pose of the human and robot are recorded. Our approach learns to predict continuous operator commands through a diffusion process and discrete commands through a classifier, all unified within a single transformer architecture. We evaluate the resulting model in simulation and with a user study on the real system. We show that our method enables simple autonomous human-robot interactions that are comparable to the expert-operator baseline, and that users can recognize the different robot moods as generated by our model. Finally, we demonstrate a zero-shot transfer of our model onto a different robotic platform with the same operator interface. 

**Abstract (ZH)**: 基于生成模型和分类器的自主交互机器人训练方法及其零样本迁移 

---
# Industrial Internet Robot Collaboration System and Edge Computing Optimization 

**Title (ZH)**: 工业互联网机器人协作系统与边缘计算优化 

**Authors**: Qian Zuo, Dajun Tao, Tian Qi, Jieyi Xie, Zijie Zhou, Zhen Tian, Yu Mingyu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02492)  

**Abstract**: In a complex environment, for a mobile robot to safely and collision - free avoid all obstacles, it poses high requirements for its intelligence level. Given that the information such as the position and geometric characteristics of obstacles is random, the control parameters of the robot, such as velocity and angular velocity, are also prone to random deviations. To address this issue in the framework of the Industrial Internet Robot Collaboration System, this paper proposes a global path control scheme for mobile robots based on deep learning. First of all, the dynamic equation of the mobile robot is established. According to the linear velocity and angular velocity of the mobile robot, its motion behaviors are divided into obstacle - avoidance behavior, target - turning behavior, and target approaching behavior. Subsequently, the neural network method in deep learning is used to build a global path planning model for the robot. On this basis, a fuzzy controller is designed with the help of a fuzzy control algorithm to correct the deviations that occur during path planning, thereby achieving optimized control of the robot's global path. In addition, considering edge computing optimization, the proposed model can process local data at the edge device, reducing the communication burden between the robot and the central server, and improving the real time performance of path planning. The experimental results show that for the mobile robot controlled by the research method in this paper, the deviation distance of the path angle is within 5 cm, the deviation convergence can be completed within 10 ms, and the planned path is shorter. This indicates that the proposed scheme can effectively improve the global path planning ability of mobile robots in the industrial Internet environment and promote the collaborative operation of robots through edge computing optimization. 

**Abstract (ZH)**: 基于深度学习的工业互联网移动机器人全局路径控制方案 

---
# Multimodal Fusion and Vision-Language Models: A Survey for Robot Vision 

**Title (ZH)**: 多模态融合与视觉-语言模型：机器人视觉综述 

**Authors**: Xiaofeng Han, Shunpeng Chen, Zenghuang Fu, Zhe Feng, Lue Fan, Dong An, Changwei Wang, Li Guo, Weiliang Meng, Xiaopeng Zhang, Rongtao Xu, Shibiao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02477)  

**Abstract**: Robot vision has greatly benefited from advancements in multimodal fusion techniques and vision-language models (VLMs). We systematically review the applications of multimodal fusion in key robotic vision tasks, including semantic scene understanding, simultaneous localization and mapping (SLAM), 3D object detection, navigation and localization, and robot manipulation. We compare VLMs based on large language models (LLMs) with traditional multimodal fusion methods, analyzing their advantages, limitations, and synergies. Additionally, we conduct an in-depth analysis of commonly used datasets, evaluating their applicability and challenges in real-world robotic scenarios. Furthermore, we identify critical research challenges such as cross-modal alignment, efficient fusion strategies, real-time deployment, and domain adaptation, and propose future research directions, including self-supervised learning for robust multimodal representations, transformer-based fusion architectures, and scalable multimodal frameworks. Through a comprehensive review, comparative analysis, and forward-looking discussion, we provide a valuable reference for advancing multimodal perception and interaction in robotic vision. A comprehensive list of studies in this survey is available at this https URL. 

**Abstract (ZH)**: 机器人视觉在多模态融合技术和视觉语言模型的进步中获益匪浅。我们系统地回顾了多模态融合在关键机器人视觉任务中的应用，包括语义场景理解、同步定位与mapping（SLAM）、3D物体检测、导航与定位、以及机器人操作。我们将基于大型语言模型（LLMs）的视觉语言模型与传统多模态融合方法进行了比较，分析了它们的优势、局限性和协同效应。此外，我们对常用的数据集进行了深入分析，评估了它们在真实世界机器人场景中的适用性和挑战。我们还确定了跨模态对齐、高效融合策略、实时部署和领域适应等关键研究挑战，并提出了未来的研究方向，包括稳健的多模态表示的自监督学习、基于变换器的融合架构以及可扩展的多模态框架。通过全面回顾、比较分析和前瞻性的讨论，我们为提升机器人视觉中的多模态感知与交互提供了宝贵的参考。本综述中涉及的研究列表可在以下链接找到：this https URL。 

---
# CHARMS: Cognitive Hierarchical Agent with Reasoning and Motion Styles 

**Title (ZH)**: CHARMS：认知层次代理及其推理与运动风格 

**Authors**: Jingyi Wang, Duanfeng Chu, Zejian Deng, Liping Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02450)  

**Abstract**: To address the current challenges of low intelligence and simplistic vehicle behavior modeling in autonomous driving simulation scenarios, this paper proposes the Cognitive Hierarchical Agent with Reasoning and Motion Styles (CHARMS). The model can reason about the behavior of other vehicles like a human driver and respond with different decision-making styles, thereby improving the intelligence and diversity of the surrounding vehicles in the driving scenario. By introducing the Level-k behavioral game theory, the paper models the decision-making process of human drivers and employs deep reinforcement learning to train the models with diverse decision styles, simulating different reasoning approaches and behavioral characteristics. Building on the Poisson cognitive hierarchy theory, this paper also presents a novel driving scenario generation method. The method controls the proportion of vehicles with different driving styles in the scenario using Poisson and binomial distributions, thus generating controllable and diverse driving environments. Experimental results demonstrate that CHARMS not only exhibits superior decision-making capabilities as ego vehicles, but also generates more complex and diverse driving scenarios as surrounding vehicles. We will release code for CHARMS at this https URL. 

**Abstract (ZH)**: Cognitive Hierarchical Agent with Reasoning and Motion Styles (CHARMS): Improving Vehicle Behavior Modeling in Autonomous Driving Simulation Scenarios 

---
# Estimating Scene Flow in Robot Surroundings with Distributed Miniaturized Time-of-Flight Sensors 

**Title (ZH)**: 使用分布式微型飞行时间传感器估计机器人环境中的场景流 

**Authors**: Jack Sander, Giammarco Caroleo, Alessandro Albini, Perla Maiolino  

**Link**: [PDF](https://arxiv.org/pdf/2504.02439)  

**Abstract**: Tracking motions of humans or objects in the surroundings of the robot is essential to improve safe robot motions and reactions. In this work, we present an approach for scene flow estimation from low-density and noisy point clouds acquired from miniaturized Time of Flight (ToF) sensors distributed on the robot body. The proposed method clusters points from consecutive frames and applies Iterative Closest Point (ICP) to estimate a dense motion flow, with additional steps introduced to mitigate the impact of sensor noise and low-density data points. Specifically, we employ a fitness-based classification to distinguish between stationary and moving points and an inlier removal strategy to refine geometric correspondences. The proposed approach is validated in an experimental setup where 24 ToF are used to estimate the velocity of an object moving at different controlled speeds. Experimental results show that the method consistently approximates the direction of the motion and its magnitude with an error which is in line with sensor noise. 

**Abstract (ZH)**: 基于miniaturized Time of Flight传感器的低密度噪声点云场景流估计方法 

---
# On learning racing policies with reinforcement learning 

**Title (ZH)**: 基于强化学习的学习赛车策略 

**Authors**: Grzegorz Czechmanowski, Jan Węgrzynowski, Piotr Kicki, Krzysztof Walas  

**Link**: [PDF](https://arxiv.org/pdf/2504.02420)  

**Abstract**: Fully autonomous vehicles promise enhanced safety and efficiency. However, ensuring reliable operation in challenging corner cases requires control algorithms capable of performing at the vehicle limits. We address this requirement by considering the task of autonomous racing and propose solving it by learning a racing policy using Reinforcement Learning (RL). Our approach leverages domain randomization, actuator dynamics modeling, and policy architecture design to enable reliable and safe zero-shot deployment on a real platform. Evaluated on the F1TENTH race car, our RL policy not only surpasses a state-of-the-art Model Predictive Control (MPC), but, to the best of our knowledge, also represents the first instance of an RL policy outperforming expert human drivers in RC racing. This work identifies the key factors driving this performance improvement, providing critical insights for the design of robust RL-based control strategies for autonomous vehicles. 

**Abstract (ZH)**: 完全自主车辆承诺提高安全性和效率。然而，确保在复杂边缘情况下的可靠运行需要能够在车辆极限范围内执行的控制算法。为应对这一要求，我们通过考虑自主赛车任务，并提出使用强化学习（RL）学习赛车策略来解决该问题。我们的方法利用领域随机化、执行器动力学建模和策略架构设计，以在实际平台上实现可靠且安全的零样本部署。在F1TENTH赛车上的评估表明，我们的RL策略不仅超越了最先进的模型预测控制（MPC），据我们所知，也是首次有RL策略在遥控赛车比赛中超越专业真人赛车手的表现。这项工作识别了推动这种性能提升的关键因素，为自主车辆的鲁棒RL控制策略设计提供了宝贵的洞察。 

---
# Model Predictive Control with Visibility Graphs for Humanoid Path Planning and Tracking Against Adversarial Opponents 

**Title (ZH)**: 基于可视化图形的模型预测控制在对抗对手时的人形机器人路径规划与跟踪 

**Authors**: Ruochen Hou, Gabriel I. Fernandez, Mingzhang Zhu, Dennis W. Hong  

**Link**: [PDF](https://arxiv.org/pdf/2504.02184)  

**Abstract**: In this paper we detail the methods used for obstacle avoidance, path planning, and trajectory tracking that helped us win the adult-sized, autonomous humanoid soccer league in RoboCup 2024. Our team was undefeated for all seated matches and scored 45 goals over 6 games, winning the championship game 6 to 1. During the competition, a major challenge for collision avoidance was the measurement noise coming from bipedal locomotion and a limited field of view (FOV). Furthermore, obstacles would sporadically jump in and out of our planned trajectory. At times our estimator would place our robot inside a hard constraint. Any planner in this competition must also be be computationally efficient enough to re-plan and react in real time. This motivated our approach to trajectory generation and tracking. In many scenarios long-term and short-term planning is needed. To efficiently find a long-term general path that avoids all obstacles we developed DAVG (Dynamic Augmented Visibility Graphs). DAVG focuses on essential path planning by setting certain regions to be active based on obstacles and the desired goal pose. By augmenting the states in the graph, turning angles are considered, which is crucial for a large soccer playing robot as turning may be more costly. A trajectory is formed by linearly interpolating between discrete points generated by DAVG. A modified version of model predictive control (MPC) is used to then track this trajectory called cf-MPC (Collision-Free MPC). This ensures short-term planning. Without having to switch formulations cf-MPC takes into account the robot dynamics and collision free constraints. Without a hard switch the control input can smoothly transition in cases where the noise places our robot inside a constraint boundary. The nonlinear formulation runs at approximately 120 Hz, while the quadratic version achieves around 400 Hz. 

**Abstract (ZH)**: 我们在RoboCup 2024成人自主人形足球联赛中所使用的目标避免、路径规划和轨迹跟踪方法详细研究：我们的团队在所有坐位比赛中未尝一败，并在6场比赛中打进45球，最终在冠军比赛中以6比1获胜。比赛中主要的碰撞避免挑战包括两足运动产生的测量噪声和有限的视野，此外，障碍物会间歇性地出现在预规划的轨迹上。我们的估计器有时会将机器人置于硬约束之内，因此任何参赛计划都必须足够高效，能够在实时情况下重新规划和应对。这促使我们制定了轨迹生成和跟踪的方法。在许多情况下，需要长期和短期规划。为了高效地找到一条避开所有障碍物的整体路径，我们开发了DAVG（动态增强可见性图）。DAVG 通过根据障碍物和目标姿态设置某些区域为活动状态，专注于关键路径规划。通过扩展图中的状态，考虑拐角角度，这对于大型足球机器人来说至关重要，因为拐角可能更加昂贵。轨迹由DAVG生成的离散点之间线性插值形成。我们使用了一种修改后的模型预测控制（MPC），称为cf-MPC（碰撞免费MPC），以进行短期规划。不需要切换公式，cf-MPC 考虑了机器人动力学和碰撞自由约束，因此在噪声将机器人置于约束边界内的情况下，控制输入可以平滑过渡。非线性公式运行速度约为每秒120次，二次版本则达到约每秒400次。 

---
# Preference-Driven Active 3D Scene Representation for Robotic Inspection in Nuclear Decommissioning 

**Title (ZH)**: 基于偏好驱动的主动3D场景表示在核废墟清理中机器人检测的应用 

**Authors**: Zhen Meng, Kan Chen, Xiangmin Xu, Erwin Jose Lopez Pulgarin, Emma Li, Philip G. Zhao, David Flynn  

**Link**: [PDF](https://arxiv.org/pdf/2504.02161)  

**Abstract**: Active 3D scene representation is pivotal in modern robotics applications, including remote inspection, manipulation, and telepresence. Traditional methods primarily optimize geometric fidelity or rendering accuracy, but often overlook operator-specific objectives, such as safety-critical coverage or task-driven viewpoints. This limitation leads to suboptimal viewpoint selection, particularly in constrained environments such as nuclear decommissioning. To bridge this gap, we introduce a novel framework that integrates expert operator preferences into the active 3D scene representation pipeline. Specifically, we employ Reinforcement Learning from Human Feedback (RLHF) to guide robotic path planning, reshaping the reward function based on expert input. To capture operator-specific priorities, we conduct interactive choice experiments that evaluate user preferences in 3D scene representation. We validate our framework using a UR3e robotic arm for reactor tile inspection in a nuclear decommissioning scenario. Compared to baseline methods, our approach enhances scene representation while optimizing trajectory efficiency. The RLHF-based policy consistently outperforms random selection, prioritizing task-critical details. By unifying explicit 3D geometric modeling with implicit human-in-the-loop optimization, this work establishes a foundation for adaptive, safety-critical robotic perception systems, paving the way for enhanced automation in nuclear decommissioning, remote maintenance, and other high-risk environments. 

**Abstract (ZH)**: 主动三维场景表示在现代机器人应用中至关重要，包括远程检查、操作和远程存在。传统的方法主要优化几何保真度或渲染准确性，但往往忽视了操作员特定的目标，如安全关键覆盖或任务驱动的视角。这一限制导致在受限环境（如核退役）中视点选择不足。为弥补这一差距，我们引入了一种新的框架，将专家操作员的偏好整合到主动三维场景表示管道中。具体而言，我们采用基于人类反馈的强化学习（RLHF）来指导机器人路径规划，根据专家输入重新调整奖励函数。为了捕捉操作员特定的优先级，我们进行了交互选择实验，评估用户在三维场景表示中的偏好。我们使用UR3e机械臂在核退役场景中的反应堆砖块检查中验证了我们的框架。与基线方法相比，该方法在优化路径效率的同时提升了场景表示。基于RLHF的策略始终优于随机选择，优先考虑任务关键细节。通过将显式的三维几何建模与隐式的在环人机优化统一起来，本工作建立了适应性、安全关键的机器人感知系统的基础，为核退役、远程维护和其他高风险环境中的增强自动化铺平了道路。 

---
# Let's move on: Topic Change in Robot-Facilitated Group Discussions 

**Title (ZH)**: 让我们继续：机器人介导的小组讨论的议题转换 

**Authors**: Georgios Hadjiantonis, Sarah Gillet, Marynel Vázquez, Iolanda Leite, Fethiye Irmak Dogan  

**Link**: [PDF](https://arxiv.org/pdf/2504.02123)  

**Abstract**: Robot-moderated group discussions have the potential to facilitate engaging and productive interactions among human participants. Previous work on topic management in conversational agents has predominantly focused on human engagement and topic personalization, with the agent having an active role in the discussion. Also, studies have shown the usefulness of including robots in groups, yet further exploration is still needed for robots to learn when to change the topic while facilitating discussions. Accordingly, our work investigates the suitability of machine-learning models and audiovisual non-verbal features in predicting appropriate topic changes. We utilized interactions between a robot moderator and human participants, which we annotated and used for extracting acoustic and body language-related features. We provide a detailed analysis of the performance of machine learning approaches using sequential and non-sequential data with different sets of features. The results indicate promising performance in classifying inappropriate topic changes, outperforming rule-based approaches. Additionally, acoustic features exhibited comparable performance and robustness compared to the complete set of multimodal features. Our annotated data is publicly available at this https URL. 

**Abstract (ZH)**: 机器人调节的群体讨论有可能促进人类参与者之间的 engaging 和 productive 互动。先前关于对话代理的主题管理研究主要集中在人类参与和主题个性化上，代理在讨论中扮演着积极角色。此外，研究表明在群体中包含机器人具有 usefulness，但仍需进一步探索机器人在促进讨论时如何学习在适当时候切换话题。因此，我们的研究探讨了机器学习模型和音频视觉非言语特征在预测适当话题变化方面的适用性。我们利用机器人调节人参与者之间的交互，并对其进行标注以提取声学和肢体语言相关的特征。我们使用不同的特征集对序列和非序列数据进行了机器学习方法性能的详细分析。结果表明，在分类不适当的话题变化方面表现出有希望的性能，并优于基于规则的方法。此外，声学特征在性能和鲁棒性方面与多模态特征集相当。我们的标注数据可在以下网址公开访问：this https URL。 

---
# RoboAct-CLIP: Video-Driven Pre-training of Atomic Action Understanding for Robotics 

**Title (ZH)**: RoboAct-CLIP: 由视频驱动的机器人原子动作理解预训练 

**Authors**: Zhiyuan Zhang, Yuxin He, Yong Sun, Junyu Shi, Lijiang Liu, Qiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2504.02069)  

**Abstract**: Visual Language Models (VLMs) have emerged as pivotal tools for robotic systems, enabling cross-task generalization, dynamic environmental interaction, and long-horizon planning through multimodal perception and semantic reasoning. However, existing open-source VLMs predominantly trained for generic vision-language alignment tasks fail to model temporally correlated action semantics that are crucial for robotic manipulation effectively. While current image-based fine-tuning methods partially adapt VLMs to robotic applications, they fundamentally disregard temporal evolution patterns in video sequences and suffer from visual feature entanglement between robotic agents, manipulated objects, and environmental contexts, thereby limiting semantic decoupling capability for atomic actions and compromising model this http URL overcome these challenges, this work presents RoboAct-CLIP with dual technical contributions: 1) A dataset reconstruction framework that performs semantic-constrained action unit segmentation and re-annotation on open-source robotic videos, constructing purified training sets containing singular atomic actions (e.g., "grasp"); 2) A temporal-decoupling fine-tuning strategy based on Contrastive Language-Image Pretraining (CLIP) architecture, which disentangles temporal action features across video frames from object-centric characteristics to achieve hierarchical representation learning of robotic atomic this http URL results in simulated environments demonstrate that the RoboAct-CLIP pretrained model achieves a 12% higher success rate than baseline VLMs, along with superior generalization in multi-object manipulation tasks. 

**Abstract (ZH)**: Visual语言模型（VLMs）已成为机器人系统的关键工具，通过多模态感知和语义推理实现跨任务通用性、动态环境交互和长期规划。然而，现有的开源VLMs主要针对通用视觉-语言对齐任务进行训练，未能有效建模对于机器人操作至关重要的时间相关动作语义。尽管现有的基于图像的微调方法部分地使VLMs适应机器人应用，但他们从根本上忽略了视频序列中的时间演化模式，并且受制于机器人代理、操作对象和环境上下文之间的视觉特征纠缠，从而限制了原子动作的语义解耦能力并损害了模型的泛化能力。为了解决这些挑战，本文提出了RoboAct-CLIP，并贡献了两项关键技术：1）一个数据集重构框架，对开源机器人视频进行语义约束的动作单元分割和重新注释，构建包含单一原子动作（如“抓取”）的净化训练集；2）基于对比视觉-语言预训练（CLIP）架构的时间解耦微调策略，该策略从以对象为中心的特性中解纠缠时间动作特征，以实现层次化的机器人原子操作的表示学习。模拟环境的实验结果表明，RoboAct-CLIP预训练模型的成功率比基线VLMs高出12%，并且在多对象操作任务中的泛化能力更优。 

---
# Semantic SLAM with Rolling-Shutter Cameras and Low-Precision INS in Outdoor Environments 

**Title (ZH)**: 户外环境中滚筒快门相机和低精度INS的语义SLAM 

**Authors**: Yuchen Zhang, Miao Fan, Shengtong Xu, Xiangzeng Liu, Haoyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2504.01997)  

**Abstract**: Accurate localization and mapping in outdoor environments remains challenging when using consumer-grade hardware, particularly with rolling-shutter cameras and low-precision inertial navigation systems (INS). We present a novel semantic SLAM approach that leverages road elements such as lane boundaries, traffic signs, and road markings to enhance localization accuracy. Our system integrates real-time semantic feature detection with a graph optimization framework, effectively handling both rolling-shutter effects and INS drift. Using a practical hardware setup which consists of a rolling-shutter camera (3840*2160@30fps), IMU (100Hz), and wheel encoder (50Hz), we demonstrate significant improvements over existing methods. Compared to state-of-the-art approaches, our method achieves higher recall (up to 5.35\%) and precision (up to 2.79\%) in semantic element detection, while maintaining mean relative error (MRE) within 10cm and mean absolute error (MAE) around 1m. Extensive experiments in diverse urban environments demonstrate the robust performance of our system under varying lighting conditions and complex traffic scenarios, making it particularly suitable for autonomous driving applications. The proposed approach provides a practical solution for high-precision localization using affordable hardware, bridging the gap between consumer-grade sensors and production-level performance requirements. 

**Abstract (ZH)**: 基于道路元素的高精度室外环境语义SLAM方法 

---
# CaLiV: LiDAR-to-Vehicle Calibration of Arbitrary Sensor Setups via Object Reconstruction 

**Title (ZH)**: CaLiV: 任意传感器布置下基于对象重建的LiDAR-to-Vehicle标定 

**Authors**: Ilir Tahiraj, Markus Edinger, Dominik Kulmer, Markus Lienkamp  

**Link**: [PDF](https://arxiv.org/pdf/2504.01987)  

**Abstract**: In autonomous systems, sensor calibration is essential for a safe and efficient navigation in dynamic environments. Accurate calibration is a prerequisite for reliable perception and planning tasks such as object detection and obstacle avoidance. Many existing LiDAR calibration methods require overlapping fields of view, while others use external sensing devices or postulate a feature-rich environment. In addition, Sensor-to-Vehicle calibration is not supported by the vast majority of calibration algorithms. In this work, we propose a novel target-based technique for extrinsic Sensor-to-Sensor and Sensor-to-Vehicle calibration of multi-LiDAR systems called CaLiV. This algorithm works for non-overlapping FoVs, as well as arbitrary calibration targets, and does not require any external sensing devices. First, we apply motion to produce FoV overlaps and utilize a simple unscented Kalman filter to obtain vehicle poses. Then, we use the Gaussian mixture model-based registration framework GMMCalib to align the point clouds in a common calibration frame. Finally, we reduce the task of recovering the sensor extrinsics to a minimization problem. We show that both translational and rotational Sensor-to-Sensor errors can be solved accurately by our method. In addition, all Sensor-to-Vehicle rotation angles can also be calibrated with high accuracy. We validate the simulation results in real-world experiments. The code is open source and available on this https URL. 

**Abstract (ZH)**: 基于目标导向的多LiDAR系统外参标定方法CaLiV 

---
# Adapting World Models with Latent-State Dynamics Residuals 

**Title (ZH)**: 使用潜在状态动力学残差适应世界模型 

**Authors**: JB Lanier, Kyungmin Kim, Armin Karamzade, Yifei Liu, Ankita Sinha, Kat He, Davide Corsi, Roy Fox  

**Link**: [PDF](https://arxiv.org/pdf/2504.02252)  

**Abstract**: Simulation-to-reality reinforcement learning (RL) faces the critical challenge of reconciling discrepancies between simulated and real-world dynamics, which can severely degrade agent performance. A promising approach involves learning corrections to simulator forward dynamics represented as a residual error function, however this operation is impractical with high-dimensional states such as images. To overcome this, we propose ReDRAW, a latent-state autoregressive world model pretrained in simulation and calibrated to target environments through residual corrections of latent-state dynamics rather than of explicit observed states. Using this adapted world model, ReDRAW enables RL agents to be optimized with imagined rollouts under corrected dynamics and then deployed in the real world. In multiple vision-based MuJoCo domains and a physical robot visual lane-following task, ReDRAW effectively models changes to dynamics and avoids overfitting in low data regimes where traditional transfer methods fail. 

**Abstract (ZH)**: 模拟到现实的强化学习（RL）面临着 reconciling 虚拟环境和真实世界动力学差异的关键挑战，这可能导致代理性能严重下降。一种有前景的方法是学习修正仿真的前向动力学，这种修正以残差误差函数的形式表示，然而当状态维度高，如图像时，这种操作是不切实际的。为克服这一问题，我们提出了一种名为 ReDRAW 的潜在状态自回归世界模型，在仿真中预先训练，并通过潜在状态动力学的残差修正而非显式观测状态的修正来对准目标环境。利用这种适应的世界模型，ReDRAW 可以使 RL 代理在修正后的动力学下进行想象中的 rollout 优化，并部署到现实世界中。在多个基于视觉的 MuJoCo 领域和一个物理机器人视觉车道跟随任务中，ReDRAW 有效地模型化了动力学的变化，并在传统迁移方法失败的数据稀少情况下避免了过拟合。 

---
# SymDQN: Symbolic Knowledge and Reasoning in Neural Network-based Reinforcement Learning 

**Title (ZH)**: SymDQN: 基于神经网络的强化学习中符号知识与推理GORITHM 

**Authors**: Ivo Amador, Nina Gierasimczuk  

**Link**: [PDF](https://arxiv.org/pdf/2504.02654)  

**Abstract**: We propose a learning architecture that allows symbolic control and guidance in reinforcement learning with deep neural networks. We introduce SymDQN, a novel modular approach that augments the existing Dueling Deep Q-Networks (DuelDQN) architecture with modules based on the neuro-symbolic framework of Logic Tensor Networks (LTNs). The modules guide action policy learning and allow reinforcement learning agents to display behaviour consistent with reasoning about the environment. Our experiment is an ablation study performed on the modules. It is conducted in a reinforcement learning environment of a 5x5 grid navigated by an agent that encounters various shapes, each associated with a given reward. The underlying DuelDQN attempts to learn the optimal behaviour of the agent in this environment, while the modules facilitate shape recognition and reward prediction. We show that our architecture significantly improves learning, both in terms of performance and the precision of the agent. The modularity of SymDQN allows reflecting on the intricacies and complexities of combining neural and symbolic approaches in reinforcement learning. 

**Abstract (ZH)**: 我们提出一种学习架构，允许在深度神经网络中进行符号控制和指导的强化学习。我们引入了SymDQN，这是一种新颖的模块化方法，将基于逻辑张量网络（LTNs）的神经符号框架模块集成到现有的 Dueling Deep Q-Networks（DuelDQN）架构中。这些模块指导动作策略学习，并允许强化学习代理表现出与环境推理一致的行为。我们的实验是对模块的消融研究，在一个5x5网格环境中进行，该环境中有一个代理遇到各种形状，每种形状对应一个给定的奖励。DuelingDQN试图学习代理在该环境中的最优行为，而模块则促进形状识别和奖励预测。我们证明，我们的架构在性能和代理精度方面显著提高了学习效果。SymDQN的模块化设计允许我们探讨在强化学习中结合神经和符号方法的复杂性和细微之处。 

---
# Advances and Challenges in Foundation Agents: From Brain-Inspired Intelligence to Evolutionary, Collaborative, and Safe Systems 

**Title (ZH)**: 基于脑启发智能的基座代理进展与挑战：从进化协作到安全系统的探索 

**Authors**: Bang Liu, Xinfeng Li, Jiayi Zhang, Jinlin Wang, Tanjin He, Sirui Hong, Hongzhang Liu, Shaokun Zhang, Kaitao Song, Kunlun Zhu, Yuheng Cheng, Suyuchen Wang, Xiaoqiang Wang, Yuyu Luo, Haibo Jin, Peiyan Zhang, Ollie Liu, Jiaqi Chen, Huan Zhang, Zhaoyang Yu, Haochen Shi, Boyan Li, Dekun Wu, Fengwei Teng, Xiaojun Jia, Jiawei Xu, Jinyu Xiang, Yizhang Lin, Tianming Liu, Tongliang Liu, Yu Su, Huan Sun, Glen Berseth, Jianyun Nie, Ian Foster, Logan Ward, Qingyun Wu, Yu Gu, Mingchen Zhuge, Xiangru Tang, Haohan Wang, Jiaxuan You, Chi Wang, Jian Pei, Qiang Yang, Xiaoliang Qi, Chenglin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01990)  

**Abstract**: The advent of large language models (LLMs) has catalyzed a transformative shift in artificial intelligence, paving the way for advanced intelligent agents capable of sophisticated reasoning, robust perception, and versatile action across diverse domains. As these agents increasingly drive AI research and practical applications, their design, evaluation, and continuous improvement present intricate, multifaceted challenges. This survey provides a comprehensive overview, framing intelligent agents within a modular, brain-inspired architecture that integrates principles from cognitive science, neuroscience, and computational research. We structure our exploration into four interconnected parts. First, we delve into the modular foundation of intelligent agents, systematically mapping their cognitive, perceptual, and operational modules onto analogous human brain functionalities, and elucidating core components such as memory, world modeling, reward processing, and emotion-like systems. Second, we discuss self-enhancement and adaptive evolution mechanisms, exploring how agents autonomously refine their capabilities, adapt to dynamic environments, and achieve continual learning through automated optimization paradigms, including emerging AutoML and LLM-driven optimization strategies. Third, we examine collaborative and evolutionary multi-agent systems, investigating the collective intelligence emerging from agent interactions, cooperation, and societal structures, highlighting parallels to human social dynamics. Finally, we address the critical imperative of building safe, secure, and beneficial AI systems, emphasizing intrinsic and extrinsic security threats, ethical alignment, robustness, and practical mitigation strategies necessary for trustworthy real-world deployment. 

**Abstract (ZH)**: 大型语言模型的出现推动了人工智能领域的 transformative变革，为高级智能代理的发展铺平了道路，这些代理能够在多样化的领域中进行复杂的推理、 robust的感知和灵活的行动。随着这些代理在人工智能研究和实际应用中发挥越来越重要的作用，它们的设计、评估和持续改进面临着复杂多维的挑战。本综述提供了全面的概述，在具模块化、脑启发式架构中将智能代理整合进认知科学、神经科学和计算研究的原则中。我们将探索分为四个紧密相连的部分。首先，我们探讨智能代理的模块化基础，系统地将认知、感知和操作模块映射到类人的大脑功能，并阐明核心组件，如记忆、世界建模、奖赏处理和类似情感的系统。其次，我们讨论自我增强和适应性进化机制，探讨智能代理如何自主提升其能力、适应动态环境并通过自动优化范式实现持续学习，包括新兴的自动化机器学习和以大型语言模型驱动的优化策略。第三，我们研究协作和进化的多智能体系统，调查来自代理互动、合作和社会结构的集体智能，突出与人类社会动态的相似之处。最后，我们应对构建安全、安全和有益的人工智能系统的关键需求，强调内在和外在安全威胁、伦理对齐、鲁棒性和实际缓解策略，以确保可信赖的实际部署。 

---
# Hierarchical Policy-Gradient Reinforcement Learning for Multi-Agent Shepherding Control of Non-Cohesive Targets 

**Title (ZH)**: 层次化策略梯度强化学习在非凝聚力目标多Agent放牧控制中的应用 

**Authors**: Stefano Covone, Italo Napolitano, Francesco De Lellis, Mario di Bernardo  

**Link**: [PDF](https://arxiv.org/pdf/2504.02479)  

**Abstract**: We propose a decentralized reinforcement learning solution for multi-agent shepherding of non-cohesive targets using policy-gradient methods. Our architecture integrates target-selection with target-driving through Proximal Policy Optimization, overcoming discrete-action constraints of previous Deep Q-Network approaches and enabling smoother agent trajectories. This model-free framework effectively solves the shepherding problem without prior dynamics knowledge. Experiments demonstrate our method's effectiveness and scalability with increased target numbers and limited sensing capabilities. 

**Abstract (ZH)**: 我们提出了一种基于策略梯度方法的去中心化强化学习多-Agent非粘聚目标驱赶解决方案。该架构通过近端策略优化将目标选择与目标引导集成在一起，克服了之前基于Deep Q-Network方法的离散动作限制，使Agent的轨迹更加平滑。该无模型框架在无需先验动力学知识的情况下有效解决了驱赶问题。实验表明，该方法在目标数量增加和传感器能力受限的情况下仍具有有效性和可扩展性。 

---
