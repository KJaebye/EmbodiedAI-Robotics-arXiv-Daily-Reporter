# Dex1B: Learning with 1B Demonstrations for Dexterous Manipulation 

**Title (ZH)**: Dex1B: 使用1亿示范样本进行灵巧操作学习 

**Authors**: Jianglong Ye, Keyi Wang, Chengjing Yuan, Ruihan Yang, Yiquan Li, Jiyue Zhu, Yuzhe Qin, Xueyan Zou, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17198)  

**Abstract**: Generating large-scale demonstrations for dexterous hand manipulation remains challenging, and several approaches have been proposed in recent years to address this. Among them, generative models have emerged as a promising paradigm, enabling the efficient creation of diverse and physically plausible demonstrations. In this paper, we introduce Dex1B, a large-scale, diverse, and high-quality demonstration dataset produced with generative models. The dataset contains one billion demonstrations for two fundamental tasks: grasping and articulation. To construct it, we propose a generative model that integrates geometric constraints to improve feasibility and applies additional conditions to enhance diversity. We validate the model on both established and newly introduced simulation benchmarks, where it significantly outperforms prior state-of-the-art methods. Furthermore, we demonstrate its effectiveness and robustness through real-world robot experiments. Our project page is at this https URL 

**Abstract (ZH)**: 生成灵巧手操作的大规模演示仍然具有挑战性，近年来提出了一些方法来应对这一挑战。其中，生成模型作为一种有前景的范式 emerged as a promising paradigm，使得高效创建多样且物理上合理的演示成为可能。在本文中，我们介绍了 Dex1B，这是一个使用生成模型构建的大规模、多样且高质量的演示数据集。该数据集包含了一百亿个演示，用于两个基础任务：抓取和动作。为构建该数据集，我们提出了一种生成模型，该模型整合了几何约束以提高可行性，并应用额外条件以增强多样性。我们在多个常用和新引入的仿真基准上验证了该模型，其性能显著优于先前的最佳方法。此外，我们通过实际的机器人实验展示了其有效性和鲁棒性。项目页面链接为：这个 https URL。 

---
# Learning Accurate Whole-body Throwing with High-frequency Residual Policy and Pullback Tube Acceleration 

**Title (ZH)**: 基于高频残差策略和拉回管加速的学习精确全身投掷方法 

**Authors**: Yuntao Ma, Yang Liu, Kaixian Qu, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2506.16986)  

**Abstract**: Throwing is a fundamental skill that enables robots to manipulate objects in ways that extend beyond the reach of their arms. We present a control framework that combines learning and model-based control for prehensile whole-body throwing with legged mobile manipulators. Our framework consists of three components: a nominal tracking policy for the end-effector, a high-frequency residual policy to enhance tracking accuracy, and an optimization-based module to improve end-effector acceleration control. The proposed controller achieved the average of 0.28 m landing error when throwing at targets located 6 m away. Furthermore, in a comparative study with university students, the system achieved a velocity tracking error of 0.398 m/s and a success rate of 56.8%, hitting small targets randomly placed at distances of 3-5 m while throwing at a specified speed of 6 m/s. In contrast, humans have a success rate of only 15.2%. This work provides an early demonstration of prehensile throwing with quantified accuracy on hardware, contributing to progress in dynamic whole-body manipulation. 

**Abstract (ZH)**: 基于学习与模型导向控制的腿式操作机器人预抓握全身投掷控制框架 

---
# Learning Dexterous Object Handover 

**Title (ZH)**: 学习灵巧的物体传递 

**Authors**: Daniel Frau-Alfaro, Julio Castaño-Amoros, Santiago Puente, Pablo Gil, Roberto Calandra  

**Link**: [PDF](https://arxiv.org/pdf/2506.16822)  

**Abstract**: Object handover is an important skill that we use daily when interacting with other humans. To deploy robots in collaborative setting, like houses, being able to receive and handing over objects safely and efficiently becomes a crucial skill. In this work, we demonstrate the use of Reinforcement Learning (RL) for dexterous object handover between two multi-finger hands. Key to this task is the use of a novel reward function based on dual quaternions to minimize the rotation distance, which outperforms other rotation representations such as Euler and rotation matrices. The robustness of the trained policy is experimentally evaluated by testing w.r.t. objects that are not included in the training distribution, and perturbations during the handover process. The results demonstrate that the trained policy successfully perform this task, achieving a total success rate of 94% in the best-case scenario after 100 experiments, thereby showing the robustness of our policy with novel objects. In addition, the best-case performance of the policy decreases by only 13.8% when the other robot moves during the handover, proving that our policy is also robust to this type of perturbation, which is common in real-world object handovers. 

**Abstract (ZH)**: 基于双四元数的强化学习在多指手之间进行灵巧物体传递的研究 

---
# DRARL: Disengagement-Reason-Augmented Reinforcement Learning for Efficient Improvement of Autonomous Driving Policy 

**Title (ZH)**: DRARL: 回退原因增强的 reinforcement 学习以提高自主驾驶策略效率 

**Authors**: Weitao Zhou, Bo Zhang, Zhong Cao, Xiang Li, Qian Cheng, Chunyang Liu, Yaqin Zhang, Diange Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16720)  

**Abstract**: With the increasing presence of automated vehicles on open roads under driver supervision, disengagement cases are becoming more prevalent. While some data-driven planning systems attempt to directly utilize these disengagement cases for policy improvement, the inherent scarcity of disengagement data (often occurring as a single instances) restricts training effectiveness. Furthermore, some disengagement data should be excluded since the disengagement may not always come from the failure of driving policies, e.g. the driver may casually intervene for a while. To this end, this work proposes disengagement-reason-augmented reinforcement learning (DRARL), which enhances driving policy improvement process according to the reason of disengagement cases. Specifically, the reason of disengagement is identified by a out-of-distribution (OOD) state estimation model. When the reason doesn't exist, the case will be identified as a casual disengagement case, which doesn't require additional policy adjustment. Otherwise, the policy can be updated under a reason-augmented imagination environment, improving the policy performance of disengagement cases with similar reasons. The method is evaluated using real-world disengagement cases collected by autonomous driving robotaxi. Experimental results demonstrate that the method accurately identifies policy-related disengagement reasons, allowing the agent to handle both original and semantically similar cases through reason-augmented training. Furthermore, the approach prevents the agent from becoming overly conservative after policy adjustments. Overall, this work provides an efficient way to improve driving policy performance with disengagement cases. 

**Abstract (ZH)**: 基于脱离原因增强的强化学习方法（DRARL）以提升驾驶策略性能 

---
# VLM-Empowered Multi-Mode System for Efficient and Safe Planetary Navigation 

**Title (ZH)**: VLM赋能的多模式系统及其在高效与安全行星导航中的应用 

**Authors**: Sinuo Cheng, Ruyi Zhou, Wenhao Feng, Huaiguang Yang, Haibo Gao, Zongquan Deng, Liang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2506.16703)  

**Abstract**: The increasingly complex and diverse planetary exploration environment requires more adaptable and flexible rover navigation strategy. In this study, we propose a VLM-empowered multi-mode system to achieve efficient while safe autonomous navigation for planetary rovers. Vision-Language Model (VLM) is used to parse scene information by image inputs to achieve a human-level understanding of terrain complexity. Based on the complexity classification, the system switches to the most suitable navigation mode, composing of perception, mapping and planning modules designed for different terrain types, to traverse the terrain ahead before reaching the next waypoint. By integrating the local navigation system with a map server and a global waypoint generation module, the rover is equipped to handle long-distance navigation tasks in complex scenarios. The navigation system is evaluated in various simulation environments. Compared to the single-mode conservative navigation method, our multi-mode system is able to bootstrap the time and energy efficiency in a long-distance traversal with varied type of obstacles, enhancing efficiency by 79.5%, while maintaining its avoidance capabilities against terrain hazards to guarantee rover safety. More system information is shown at this https URL. 

**Abstract (ZH)**: 行星探索环境日益复杂多变，需要更加适应灵活的火星车导航策略。本研究提出一种基于VLM的多模式系统，以实现高效安全的自主导航。视觉语言模型（VLM）通过图像输入解析场景信息，实现对地形复杂性的类人类理解。基于复杂性分类，系统切换到最适合的导航模式，该模式由设计适用于不同地形类型的感知、制图和规划模块组成，以便在到达下一个航点前穿越前方地形。通过整合局部导航系统、地图服务器和全球航点生成模块，火星车能够处理复杂场景下的远程导航任务。导航系统在多种仿真环境中进行了评估。与单一模式保守导航方法相比，本多模式系统能够在不同类型障碍物存在的远程穿越中提高79.5%的时间和能量效率，同时保持对地形危险的规避能力，确保火星车的安全。更多系统信息详见此链接：[更多系统信息链接]。 

---
# Compliant Residual DAgger: Improving Real-World Contact-Rich Manipulation with Human Corrections 

**Title (ZH)**: compliant residual DAgger: 通过人类纠正提高实际场景中接触丰富的 manipulation 技能 

**Authors**: Xiaomeng Xu, Yifan Hou, Zeyi Liu, Shuran Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.16685)  

**Abstract**: We address key challenges in Dataset Aggregation (DAgger) for real-world contact-rich manipulation: how to collect informative human correction data and how to effectively update policies with this new data. We introduce Compliant Residual DAgger (CR-DAgger), which contains two novel components: 1) a Compliant Intervention Interface that leverages compliance control, allowing humans to provide gentle, accurate delta action corrections without interrupting the ongoing robot policy execution; and 2) a Compliant Residual Policy formulation that learns from human corrections while incorporating force feedback and force control. Our system significantly enhances performance on precise contact-rich manipulation tasks using minimal correction data, improving base policy success rates by over 50\% on two challenging tasks (book flipping and belt assembly) while outperforming both retraining-from-scratch and finetuning approaches. Through extensive real-world experiments, we provide practical guidance for implementing effective DAgger in real-world robot learning tasks. Result videos are available at: this https URL 

**Abstract (ZH)**: 我们研究了现实世界中接触丰富的操作数据集聚合（Dataset Aggregation, DAgger）的关键挑战：如何收集有用的人类更正数据，以及如何有效利用这些新数据更新策略。我们引入了Compliant Residual DAgger (CR-DAgger)，其包含两个新颖的组成部分：1) 一种顺应性干预接口，利用顺应性控制，使人类能够在不中断机器人策略执行的情况下提供温柔准确的增量动作更正；2) 一种顺应性残差策略形式化，该形式化从人类更正中学习，同时结合力反馈和力控制。我们的系统使用极少的更正数据显著提升了精确接触丰富操作任务的表现，相对于两个具有挑战性的任务（书本翻转和带子装配），基线策略的成功率提高了超过50%，并且优于从头训练和微调的方法。通过广泛的现实世界实验，我们提供了在实际机器人学习任务中实施有效DAgger的实用指导。结果视频可在以下链接查看：this https URL 

---
# CodeDiffuser: Attention-Enhanced Diffusion Policy via VLM-Generated Code for Instruction Ambiguity 

**Title (ZH)**: CodeDiffuser: 通过VLM生成代码增强注意力的扩散策略以解决指令歧义 

**Authors**: Guang Yin, Yitong Li, Yixuan Wang, Dale McConachie, Paarth Shah, Kunimatsu Hashimoto, Huan Zhang, Katherine Liu, Yunzhu Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16652)  

**Abstract**: Natural language instructions for robotic manipulation tasks often exhibit ambiguity and vagueness. For instance, the instruction "Hang a mug on the mug tree" may involve multiple valid actions if there are several mugs and branches to choose from. Existing language-conditioned policies typically rely on end-to-end models that jointly handle high-level semantic understanding and low-level action generation, which can result in suboptimal performance due to their lack of modularity and interpretability. To address these challenges, we introduce a novel robotic manipulation framework that can accomplish tasks specified by potentially ambiguous natural language. This framework employs a Vision-Language Model (VLM) to interpret abstract concepts in natural language instructions and generates task-specific code - an interpretable and executable intermediate representation. The generated code interfaces with the perception module to produce 3D attention maps that highlight task-relevant regions by integrating spatial and semantic information, effectively resolving ambiguities in instructions. Through extensive experiments, we identify key limitations of current imitation learning methods, such as poor adaptation to language and environmental variations. We show that our approach excels across challenging manipulation tasks involving language ambiguity, contact-rich manipulation, and multi-object interactions. 

**Abstract (ZH)**: 自然语言指令在机器人操作任务中的表达往往具有歧义性和模糊性。例如，“将茶杯挂在茶杯树上”这一指令在存在多个茶杯和分支的情况下，可能涉及多种有效的操作方式。现有的基于语言的策略通常依赖于端到端模型，这些模型能够同时处理高层次语义理解和低层次动作生成，但由于缺乏模块化和可解释性，可能会导致性能不佳。为应对这些挑战，我们提出了一种新的机器人操作框架，能够完成由可能含糊不清的自然语言指定的任务。该框架利用视觉-语言模型（VLM）解释自然语言指令中的抽象概念，并生成特定任务的代码——一种具有可解释性和可执行性的中间表示。生成的代码与感知模块接口，通过整合空间和语义信息生成3D注意图，突出显示任务相关的区域，从而有效解决指令中的歧义性。通过广泛实验证明，我们的方法在涉及语言歧义、丰富接触操作及多物体交互等具有挑战性的操作任务中表现优异。 

---
# See What I Mean? Expressiveness and Clarity in Robot Display Design 

**Title (ZH)**: 你看得明白吗？机器人展示设计中的表达力与清晰度研究 

**Authors**: Matthew Ebisu, Hang Yu, Reuben Aronson, Elaine Short  

**Link**: [PDF](https://arxiv.org/pdf/2506.16643)  

**Abstract**: Nonverbal visual symbols and displays play an important role in communication when humans and robots work collaboratively. However, few studies have investigated how different types of non-verbal cues affect objective task performance, especially in a dynamic environment that requires real time decision-making. In this work, we designed a collaborative navigation task where the user and the robot only had partial information about the map on each end and thus the users were forced to communicate with a robot to complete the task. We conducted our study in a public space and recruited 37 participants who randomly passed by our setup. Each participant collaborated with a robot utilizing either animated anthropomorphic eyes and animated icons, or static anthropomorphic eyes and static icons. We found that participants that interacted with a robot with animated displays reported the greatest level of trust and satisfaction; that participants interpreted static icons the best; and that participants with a robot with static eyes had the highest completion success. These results suggest that while animation can foster trust with robots, human-robot communication can be optimized by the addition of familiar static icons that may be easier for users to interpret. We published our code, designed symbols, and collected results online at: this https URL. 

**Abstract (ZH)**: 非言语视觉符号和显示在人机协作沟通中扮演重要角色，但在动态环境下的实时决策中，不同类型的非言语线索如何影响客观任务绩效的研究较少。在本研究中，我们设计了一个协作导航任务，用户和机器人仅掌握地图的部分信息，从而迫使用户与机器人进行沟通以完成任务。我们在公共场所进行了研究，招募了37名随机路过该设置的参与者。每位参与者与机器人协作，使用的信号方式为带有动画拟人眼睛和动画图标，或带有静态拟人眼睛和静态图标。我们发现，与具有动画显示的机器人互动的参与者报告了最高的信任和满意度；参与者对静态图标理解最好；与具有静态眼睛的机器人合作的参与者完成任务的成功率最高。这些结果表明，虽然动画可以促进对机器人的信任，但通过添加易于用户理解的熟悉的静态图标，可以优化人机沟通。我们在网上发布了我们的代码、设计的符号和收集的结果：https://yourlinkhere。 

---
# History-Augmented Vision-Language Models for Frontier-Based Zero-Shot Object Navigation 

**Title (ZH)**: 基于历史信息的视觉-语言模型在前沿导向的零样本物体导航中的应用 

**Authors**: Mobin Habibpour, Fatemeh Afghah  

**Link**: [PDF](https://arxiv.org/pdf/2506.16623)  

**Abstract**: Object Goal Navigation (ObjectNav) challenges robots to find objects in unseen environments, demanding sophisticated reasoning. While Vision-Language Models (VLMs) show potential, current ObjectNav methods often employ them superficially, primarily using vision-language embeddings for object-scene similarity checks rather than leveraging deeper reasoning. This limits contextual understanding and leads to practical issues like repetitive navigation behaviors. This paper introduces a novel zero-shot ObjectNav framework that pioneers the use of dynamic, history-aware prompting to more deeply integrate VLM reasoning into frontier-based exploration. Our core innovation lies in providing the VLM with action history context, enabling it to generate semantic guidance scores for navigation actions while actively avoiding decision loops. We also introduce a VLM-assisted waypoint generation mechanism for refining the final approach to detected objects. Evaluated on the HM3D dataset within Habitat, our approach achieves a 46% Success Rate (SR) and 24.8% Success weighted by Path Length (SPL). These results are comparable to state-of-the-art zero-shot methods, demonstrating the significant potential of our history-augmented VLM prompting strategy for more robust and context-aware robotic navigation. 

**Abstract (ZH)**: 基于对象的目标导航（ObjectNav）挑战机器人在未见环境中寻找物体，要求具备复杂的推理能力。尽管视觉-语言模型（VLMs）展现出潜力，当前的ObjectNav方法往往仅浅表地使用它们，主要通过视觉-语言嵌入进行物体-场景相似性检查，而未能充分利用深入的推理。这限制了上下文理解并导致重复的导航行为。本文介绍了一种新颖的零样本ObjectNav框架， pioneering 使用动态的历史感知提示以更深入地将VLM推理集成到前沿探索中。我们核心的创新在于为VLM提供动作历史上下文，使其能够生成导航动作的语义指导分数并主动避免决策循环。我们还引入了一种VLM辅助的航点生成机制，以细化对检测到物体的最终接近方式。在Habitat的HM3D数据集上进行评估，我们的方法实现了46%的成功率（SR）和24.8%的成功加权路径长度（SPL）。这些结果与最新的零样本方法相当，表明我们的历史增强VLM提示策略在实现更 robust 和上下文感知的机器人导航方面具有显著潜力。 

---
# Reimagination with Test-time Observation Interventions: Distractor-Robust World Model Predictions for Visual Model Predictive Control 

**Title (ZH)**: 基于测试时观测干预的重塑：视觉模型预测控制中的干扰物鲁棒世界模型预测 

**Authors**: Yuxin Chen, Jianglan Wei, Chenfeng Xu, Boyi Li, Masayoshi Tomizuka, Andrea Bajcsy, Ran Tian  

**Link**: [PDF](https://arxiv.org/pdf/2506.16565)  

**Abstract**: World models enable robots to "imagine" future observations given current observations and planned actions, and have been increasingly adopted as generalized dynamics models to facilitate robot learning. Despite their promise, these models remain brittle when encountering novel visual distractors such as objects and background elements rarely seen during training. Specifically, novel distractors can corrupt action outcome predictions, causing downstream failures when robots rely on the world model imaginations for planning or action verification. In this work, we propose Reimagination with Observation Intervention (ReOI), a simple yet effective test-time strategy that enables world models to predict more reliable action outcomes in open-world scenarios where novel and unanticipated visual distractors are inevitable. Given the current robot observation, ReOI first detects visual distractors by identifying which elements of the scene degrade in physically implausible ways during world model prediction. Then, it modifies the current observation to remove these distractors and bring the observation closer to the training distribution. Finally, ReOI "reimagines" future outcomes with the modified observation and reintroduces the distractors post-hoc to preserve visual consistency for downstream planning and verification. We validate our approach on a suite of robotic manipulation tasks in the context of action verification, where the verifier needs to select desired action plans based on predictions from a world model. Our results show that ReOI is robust to both in-distribution and out-of-distribution visual distractors. Notably, it improves task success rates by up to 3x in the presence of novel distractors, significantly outperforming action verification that relies on world model predictions without imagination interventions. 

**Abstract (ZH)**: Reimagination with Observation Intervention for Robust Action Outcome Prediction in Open-World Scenarios 

---
# BIDA: A Bi-level Interaction Decision-making Algorithm for Autonomous Vehicles in Dynamic Traffic Scenarios 

**Title (ZH)**: BIDA：动态交通场景中自主车辆的多层次交互决策算法 

**Authors**: Liyang Yu, Tianyi Wang, Junfeng Jiao, Fengwu Shan, Hongqing Chu, Bingzhao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16546)  

**Abstract**: In complex real-world traffic environments, autonomous vehicles (AVs) need to interact with other traffic participants while making real-time and safety-critical decisions accordingly. The unpredictability of human behaviors poses significant challenges, particularly in dynamic scenarios, such as multi-lane highways and unsignalized T-intersections. To address this gap, we design a bi-level interaction decision-making algorithm (BIDA) that integrates interactive Monte Carlo tree search (MCTS) with deep reinforcement learning (DRL), aiming to enhance interaction rationality, efficiency and safety of AVs in dynamic key traffic scenarios. Specifically, we adopt three types of DRL algorithms to construct a reliable value network and policy network, which guide the online deduction process of interactive MCTS by assisting in value update and node selection. Then, a dynamic trajectory planner and a trajectory tracking controller are designed and implemented in CARLA to ensure smooth execution of planned maneuvers. Experimental evaluations demonstrate that our BIDA not only enhances interactive deduction and reduces computational costs, but also outperforms other latest benchmarks, which exhibits superior safety, efficiency and interaction rationality under varying traffic conditions. 

**Abstract (ZH)**: 在复杂现实交通环境中，自动驾驶车辆（AVs）需要在进行实时和安全关键决策的同时与其它交通参与者互动。人类行为的不可预测性在动态场景下，如多车道高速公路和无信号T形交叉路口等，提出了重大挑战。为应对这一挑战，我们设计了一种双层互动决策算法（BIDA），将交互蒙特卡洛树搜索（MCTS）与深度强化学习（DRL）相结合，旨在提升自动驾驶车辆在动态关键交通场景中的互动合理性、效率和安全性。具体而言，我们采用了三种类型的DRL算法构建了可靠的值网络和策略网络，通过协助价值更新和节点选择来引导交互MCTS的在线推断过程。然后，在CARLA中设计并实现了动态轨迹规划器和轨迹跟踪控制器，以确保计划机动动作的平滑执行。实验评估表明，我们的BIDA不仅提高了互动推断的合理性和降低了计算成本，还在各种交通条件下超越了其他最新基准，显示出优越的安全性、效率和互动合理性。 

---
# Grounding Language Models with Semantic Digital Twins for Robotic Planning 

**Title (ZH)**: 基于语义数字孪生的语言模型在机器人规划中的应用 

**Authors**: Mehreen Naeem, Andrew Melnik, Michael Beetz  

**Link**: [PDF](https://arxiv.org/pdf/2506.16493)  

**Abstract**: We introduce a novel framework that integrates Semantic Digital Twins (SDTs) with Large Language Models (LLMs) to enable adaptive and goal-driven robotic task execution in dynamic environments. The system decomposes natural language instructions into structured action triplets, which are grounded in contextual environmental data provided by the SDT. This semantic grounding allows the robot to interpret object affordances and interaction rules, enabling action planning and real-time adaptability. In case of execution failures, the LLM utilizes error feedback and SDT insights to generate recovery strategies and iteratively revise the action plan. We evaluate our approach using tasks from the ALFRED benchmark, demonstrating robust performance across various household scenarios. The proposed framework effectively combines high-level reasoning with semantic environment understanding, achieving reliable task completion in the face of uncertainty and failure. 

**Abstract (ZH)**: 我们提出一种将语义数字孪生（SDT）与大型语言模型（LLM）集成的新框架，以在动态环境中实现适应性和目标驱动的机器人任务执行。该系统将自然语言指令分解为结构化的动作三元组，这些三元组基于SDT提供的上下文环境数据进行语义绑定。这种语义绑定使机器人能够解释物体功能和交互规则，从而实现动作规划和实时适应。在执行失败时，LLM利用错误反馈和SDT的见解生成恢复策略，并迭代地修改动作计划。我们使用ALFRED基准中的任务对我们的方法进行评估，展示了在各种家庭场景中的稳健性能。所提出框架有效地结合了高层次推理与语义环境理解，能够在不确定性与失败面前实现可靠的任务完成。 

---
# Human2LocoMan: Learning Versatile Quadrupedal Manipulation with Human Pretraining 

**Title (ZH)**: Human2LocoMan: 人体预训练下的多功能四足Manipulation学习 

**Authors**: Yaru Niu, Yunzhe Zhang, Mingyang Yu, Changyi Lin, Chenhao Li, Yikai Wang, Yuxiang Yang, Wenhao Yu, Tingnan Zhang, Bingqing Chen, Jonathan Francis, Zhenzhen Li, Jie Tan, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16475)  

**Abstract**: Quadrupedal robots have demonstrated impressive locomotion capabilities in complex environments, but equipping them with autonomous versatile manipulation skills in a scalable way remains a significant challenge. In this work, we introduce a cross-embodiment imitation learning system for quadrupedal manipulation, leveraging data collected from both humans and LocoMan, a quadruped equipped with multiple manipulation modes. Specifically, we develop a teleoperation and data collection pipeline, which unifies and modularizes the observation and action spaces of the human and the robot. To effectively leverage the collected data, we propose an efficient modularized architecture that supports co-training and pretraining on structured modality-aligned data across different embodiments. Additionally, we construct the first manipulation dataset for the LocoMan robot, covering various household tasks in both unimanual and bimanual modes, supplemented by a corresponding human dataset. We validate our system on six real-world manipulation tasks, where it achieves an average success rate improvement of 41.9% overall and 79.7% under out-of-distribution (OOD) settings compared to the baseline. Pretraining with human data contributes a 38.6% success rate improvement overall and 82.7% under OOD settings, enabling consistently better performance with only half the amount of robot data. Our code, hardware, and data are open-sourced at: this https URL. 

**Abstract (ZH)**: 四足机器人在复杂环境中的运动能力令人印象深刻，但以可扩展的方式为其配备自主通用操作技能仍然是一个重大挑战。在本工作中，我们引入了一种跨体态模仿学习系统，利用来自人类和LocoMan（一种配备了多种操作模式的四足机器人）的数据。具体来说，我们开发了一种远程操作和数据收集流水线，将人类和机器人的观察空间和行动空间统一和模块化。为了有效利用收集的数据，我们提出了一种高效的模块化架构，支持在不同体态的结构化模态对齐数据上进行联合训练和预训练。此外，我们为LocoMan机器人构建了首个操作数据集，涵盖了多种单手和双手家庭任务，并补充了相应的人类数据集。我们在六个真实世界的操作任务上验证了我们的系统，整体成功率提高了41.9%，在分布外（OOD）设置下提高了79.7%，使用人类数据预训练的整体成功率提高了38.6%，在OOD设置下提高了82.7%，仅使用一半的机器人数据就实现了持续的更好性能。我们的代码、硬件和数据已开源：this https URL。 

---
# Goal-conditioned Hierarchical Reinforcement Learning for Sample-efficient and Safe Autonomous Driving at Intersections 

**Title (ZH)**: 面向交叉口高效安全自主驾驶的条件导向分层强化学习 

**Authors**: Yiou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16336)  

**Abstract**: Reinforcement learning (RL) exhibits remarkable potential in addressing autonomous driving tasks. However, it is difficult to train a sample-efficient and safe policy in complex scenarios. In this article, we propose a novel hierarchical reinforcement learning (HRL) framework with a goal-conditioned collision prediction (GCCP) module. In the hierarchical structure, the GCCP module predicts collision risks according to different potential subgoals of the ego vehicle. A high-level decision-maker choose the best safe subgoal. A low-level motion-planner interacts with the environment according to the subgoal. Compared to traditional RL methods, our algorithm is more sample-efficient, since its hierarchical structure allows reusing the policies of subgoals across similar tasks for various navigation scenarios. In additional, the GCCP module's ability to predict both the ego vehicle's and surrounding vehicles' future actions according to different subgoals, ensures the safety of the ego vehicle throughout the decision-making process. Experimental results demonstrate that the proposed method converges to an optimal policy faster and achieves higher safety than traditional RL methods. 

**Abstract (ZH)**: 强化学习（RL）在应对自动驾驶任务方面展现了显著潜力。然而，在复杂场景下训练高效且安全的策略颇具挑战。本文提出了一种新颖的分层强化学习（HRL）框架，包含目标引导的碰撞预测（GCCP）模块。在分层次结构中，GCCP模块根据自主车辆的不同潜在子目标预测碰撞风险，高层次决策制定者选择最佳安全子目标，低层次运动规划者根据子目标与环境交互。与传统的RL方法相比，我们的算法更具样本效率，因为其分层结构允许在相似任务中重用子目标的策略，以适应各种导航场景。此外，GCCP模块根据不同子目标预测自主车辆及其周围车辆的未来行动，确保在决策过程中自主车辆的安全性。实验结果表明，所提出的方法比传统RL方法更快地收敛到最优策略，并且安全性更高。 

---
# CapsDT: Diffusion-Transformer for Capsule Robot Manipulation 

**Title (ZH)**: CapsDT: 扩散变换器在胶囊机器人操作中的应用 

**Authors**: Xiting He, Mingwu Su, Xinqi Jiang, Long Bai, Jiewen Lai, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2506.16263)  

**Abstract**: Vision-Language-Action (VLA) models have emerged as a prominent research area, showcasing significant potential across a variety of applications. However, their performance in endoscopy robotics, particularly endoscopy capsule robots that perform actions within the digestive system, remains unexplored. The integration of VLA models into endoscopy robots allows more intuitive and efficient interactions between human operators and medical devices, improving both diagnostic accuracy and treatment outcomes. In this work, we design CapsDT, a Diffusion Transformer model for capsule robot manipulation in the stomach. By processing interleaved visual inputs, and textual instructions, CapsDT can infer corresponding robotic control signals to facilitate endoscopy tasks. In addition, we developed a capsule endoscopy robot system, a capsule robot controlled by a robotic arm-held magnet, addressing different levels of four endoscopy tasks and creating corresponding capsule robot datasets within the stomach simulator. Comprehensive evaluations on various robotic tasks indicate that CapsDT can serve as a robust vision-language generalist, achieving state-of-the-art performance in various levels of endoscopy tasks while achieving a 26.25% success rate in real-world simulation manipulation. 

**Abstract (ZH)**: Vision-Language-Action (VLA)模型已成为一个重要的研究领域，展示出在多种应用中的巨大潜力。然而，这些模型在内窥镜机器人，特别是消化系统内的胶囊机器人执行操作方面的性能仍未被探索。将VLA模型集成到内窥镜机器人中，可以实现更直观和高效的医护人员与医疗设备之间的交互，提高诊断准确性和治疗效果。在本工作中，我们设计了CapsDT，一种用于胃部胶囊机器人操作的扩散变换器模型。通过处理交错的视觉输入和文本指令，CapsDT可以推断出相应的机器人控制信号，以辅助内窥镜任务。此外，我们还开发了一种胶囊内窥镜机器人系统，该系统采用手持磁铁的机械臂控制胶囊机器人，并在胃部模拟器中针对四种不同的内窥镜任务的不同层级建立了相应的胶囊机器人数据集。各种机器人任务的全面评估表明，CapsDT可以作为稳健的视觉-语言通用模型，在不同层级的内窥镜任务中实现最先进的性能，同时在现实世界的模拟操作中实现26.25%的成功率。 

---
# ControlVLA: Few-shot Object-centric Adaptation for Pre-trained Vision-Language-Action Models 

**Title (ZH)**: ControlVLA：预训练视觉-语言-行动模型的少样本对象中心适应方法 

**Authors**: Puhao Li, Yingying Wu, Ziheng Xi, Wanlin Li, Yuzhe Huang, Zhiyuan Zhang, Yinghan Chen, Jianan Wang, Song-Chun Zhu, Tengyu Liu, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16211)  

**Abstract**: Learning real-world robotic manipulation is challenging, particularly when limited demonstrations are available. Existing methods for few-shot manipulation often rely on simulation-augmented data or pre-built modules like grasping and pose estimation, which struggle with sim-to-real gaps and lack extensibility. While large-scale imitation pre-training shows promise, adapting these general-purpose policies to specific tasks in data-scarce settings remains unexplored. To achieve this, we propose ControlVLA, a novel framework that bridges pre-trained VLA models with object-centric representations via a ControlNet-style architecture for efficient fine-tuning. Specifically, to introduce object-centric conditions without overwriting prior knowledge, ControlVLA zero-initializes a set of projection layers, allowing them to gradually adapt the pre-trained manipulation policies. In real-world experiments across 6 diverse tasks, including pouring cubes and folding clothes, our method achieves a 76.7% success rate while requiring only 10-20 demonstrations -- a significant improvement over traditional approaches that require more than 100 demonstrations to achieve comparable success. Additional experiments highlight ControlVLA's extensibility to long-horizon tasks and robustness to unseen objects and backgrounds. 

**Abstract (ZH)**: 学习现实世界中的机器人 manipulation 挑战重重，尤其在示例有限的情况下。现有的少样本 manipulation 方法往往依赖于模拟增强的数据或预构建的模块（如抓取和姿态估计），这些方法难以解决模拟与现实之间的差距，并且缺乏可扩展性。尽管大规模模仿预训练前景广阔，但在数据稀缺的情况下将这些通用策略适应特定任务仍待探索。为此，我们提出了 ControlVLA，这是一种新的框架，通过 ControlNet 风格的架构将预训练的 VLA 模型与物体中心表示相结合，以实现高效的微调。具体而言，为了引入物体中心条件而不覆盖先前知识，ControlVLA 将一系列投影层初始化为零，使其能够逐步适应预训练的 manipulation 策略。在包括倒立方体和折叠衣物在内的 6 个不同任务的现实世界实验中，我们的方法在仅需要 10-20 个示例的情况下实现了 76.7% 的成功率，这远优于传统方法需要超过 100 个示例才能达到类似成功率的情形。额外的实验还突显了 ControlVLA 在长时任务上的可扩展性和对未见过的物体和背景的鲁棒性。 

---
# FlowRAM: Grounding Flow Matching Policy with Region-Aware Mamba Framework for Robotic Manipulation 

**Title (ZH)**: FlowRAM：基于区域意识Mamba框架的流动匹配策略接地方法 

**Authors**: Sen Wang, Le Wang, Sanping Zhou, Jingyi Tian, Jiayi Li, Haowen Sun, Wei Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16201)  

**Abstract**: Robotic manipulation in high-precision tasks is essential for numerous industrial and real-world applications where accuracy and speed are required. Yet current diffusion-based policy learning methods generally suffer from low computational efficiency due to the iterative denoising process during inference. Moreover, these methods do not fully explore the potential of generative models for enhancing information exploration in 3D environments. In response, we propose FlowRAM, a novel framework that leverages generative models to achieve region-aware perception, enabling efficient multimodal information processing. Specifically, we devise a Dynamic Radius Schedule, which allows adaptive perception, facilitating transitions from global scene comprehension to fine-grained geometric details. Furthermore, we integrate state space models to integrate multimodal information, while preserving linear computational complexity. In addition, we employ conditional flow matching to learn action poses by regressing deterministic vector fields, simplifying the learning process while maintaining performance. We verify the effectiveness of the FlowRAM in the RLBench, an established manipulation benchmark, and achieve state-of-the-art performance. The results demonstrate that FlowRAM achieves a remarkable improvement, particularly in high-precision tasks, where it outperforms previous methods by 12.0% in average success rate. Additionally, FlowRAM is able to generate physically plausible actions for a variety of real-world tasks in less than 4 time steps, significantly increasing inference speed. 

**Abstract (ZH)**: 基于流”的生成模型在高精度任务中的机器人操作：实现区域感知的高效多模态信息处理 

---
# Investigating Lagrangian Neural Networks for Infinite Horizon Planning in Quadrupedal Locomotion 

**Title (ZH)**: 基于拉格朗日神经网络的四足行走无限_horizon规划研究 

**Authors**: Prakrut Kotecha, Aditya Shirwatkar, Shishir Kolathaya  

**Link**: [PDF](https://arxiv.org/pdf/2506.16079)  

**Abstract**: Lagrangian Neural Networks (LNNs) present a principled and interpretable framework for learning the system dynamics by utilizing inductive biases. While traditional dynamics models struggle with compounding errors over long horizons, LNNs intrinsically preserve the physical laws governing any system, enabling accurate and stable predictions essential for sustainable locomotion. This work evaluates LNNs for infinite horizon planning in quadrupedal robots through four dynamics models: (1) full-order forward dynamics (FD) training and inference, (2) diagonalized representation of Mass Matrix in full order FD, (3) full-order inverse dynamics (ID) training with FD inference, (4) reduced-order modeling via torso centre-of-mass (CoM) dynamics. Experiments demonstrate that LNNs bring improvements in sample efficiency (10x) and superior prediction accuracy (up to 2-10x) compared to baseline methods. Notably, the diagonalization approach of LNNs reduces computational complexity while retaining some interpretability, enabling real-time receding horizon control. These findings highlight the advantages of LNNs in capturing the underlying structure of system dynamics in quadrupeds, leading to improved performance and efficiency in locomotion planning and control. Additionally, our approach achieves a higher control frequency than previous LNN methods, demonstrating its potential for real-world deployment on quadrupeds. 

**Abstract (ZH)**: Lagrangian 神经网络 (LNNs) 为通过利用归纳偏置学习系统动力学提供了一个原则性和可解释性的框架。尽管传统动力模型在长期预测中面临着累积误差的问题，LNNs 本质上能保持任何系统所遵循的物理法则，从而实现对可持续运动至关重要的准确且稳定的预测。本文通过四种动力学模型评估 LNNs 在四足机器人无限 horizon 规划中的应用：(1) 完全套数前向动力学 (FD) 训练和推理，(2) 完全套数质量矩阵对角化表示，(3) 完全套数逆动力学 (ID) 训练与 FD 推理，(4) 通过躯干质心 (CoM) 动力学的降维建模。实验结果表明，与基准方法相比，LNNs 在样本效率 (10 倍) 和预测精度 (最多 2-10 倍) 上带来了改进。值得注意的是，LNNs 的对角化方法降低了计算复杂度并保持了一定的可解释性，使其能够实现实时后退预测视野控制。这些发现突显了 LNNs 在捕获四足动物系统动力学底层结构方面的优势，从而提高了运动规划和控制的性能和效率。此外，我们的方法实现的控制频率高于之前的 LNN 方法，展示了其在四足动物的实际部署中的潜力。 

---
# DualTHOR: A Dual-Arm Humanoid Simulation Platform for Contingency-Aware Planning 

**Title (ZH)**: DualTHOR：一种面向 contingencies 意识规划的双臂仿人模拟平台 

**Authors**: Boyu Li, Siyuan He, Hang Xu, Haoqi Yuan, Yu Zang, Liwei Hu, Junpeng Yue, Zhenxiong Jiang, Pengbo Hu, Börje F. Karlsson, Yehui Tang, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.16012)  

**Abstract**: Developing embodied agents capable of performing complex interactive tasks in real-world scenarios remains a fundamental challenge in embodied AI. Although recent advances in simulation platforms have greatly enhanced task diversity to train embodied Vision Language Models (VLMs), most platforms rely on simplified robot morphologies and bypass the stochastic nature of low-level execution, which limits their transferability to real-world robots. To address these issues, we present a physics-based simulation platform DualTHOR for complex dual-arm humanoid robots, built upon an extended version of AI2-THOR. Our simulator includes real-world robot assets, a task suite for dual-arm collaboration, and inverse kinematics solvers for humanoid robots. We also introduce a contingency mechanism that incorporates potential failures through physics-based low-level execution, bridging the gap to real-world scenarios. Our simulator enables a more comprehensive evaluation of the robustness and generalization of VLMs in household environments. Extensive evaluations reveal that current VLMs struggle with dual-arm coordination and exhibit limited robustness in realistic environments with contingencies, highlighting the importance of using our simulator to develop more capable VLMs for embodied tasks. The code is available at this https URL. 

**Abstract (ZH)**: 在现实世界场景中开发能够执行复杂交互任务的具身代理仍然是具身AI领域的基本挑战。尽管近期在模拟平台方面的进展极大地增强了训练具身视觉语言模型（VLMs）的任务多样性，但大多数平台依赖于简化的人形机器人形态，并规避了低级执行的随机性质，这限制了它们向实际机器人转移的能力。为了解决这些问题，我们提出了一个基于物理的模拟平台DualTHOR，该平台适用于复杂双臂人形机器人，基于AI2-THOR的扩展版本。我们的模拟器包括真实机器人资产、双臂协作任务套件以及人形机器人逆运动学求解器。我们还引入了一种应急机制，通过基于物理的低级执行来整合潜在的失败情况，从而填补与现实世界场景之间的差距。我们的模拟器使得更具身环境中的VLMs的鲁棒性和泛化能力评估更加全面。广泛评估表明，当前的VLMs难以应对双臂协调，并且在具有应急情况的现实环境中表现出有限的鲁棒性，强调了使用我们的模拟器开发更加能够胜任的VLMs的重要性。代码可在该网址获取。 

---
# ViTacFormer: Learning Cross-Modal Representation for Visuo-Tactile Dexterous Manipulation 

**Title (ZH)**: ViTacFormer: 学习跨模态表示以实现视觉-触觉灵巧操作 

**Authors**: Liang Heng, Haoran Geng, Kaifeng Zhang, Pieter Abbeel, Jitendra Malik  

**Link**: [PDF](https://arxiv.org/pdf/2506.15953)  

**Abstract**: Dexterous manipulation is a cornerstone capability for robotic systems aiming to interact with the physical world in a human-like manner. Although vision-based methods have advanced rapidly, tactile sensing remains crucial for fine-grained control, particularly in unstructured or visually occluded settings. We present ViTacFormer, a representation-learning approach that couples a cross-attention encoder to fuse high-resolution vision and touch with an autoregressive tactile prediction head that anticipates future contact signals. Building on this architecture, we devise an easy-to-challenging curriculum that steadily refines the visual-tactile latent space, boosting both accuracy and robustness. The learned cross-modal representation drives imitation learning for multi-fingered hands, enabling precise and adaptive manipulation. Across a suite of challenging real-world benchmarks, our method achieves approximately 50% higher success rates than prior state-of-the-art systems. To our knowledge, it is also the first to autonomously complete long-horizon dexterous manipulation tasks that demand highly precise control with an anthropomorphic hand, successfully executing up to 11 sequential stages and sustaining continuous operation for 2.5 minutes. 

**Abstract (ZH)**: 灵巧操作是旨在以类人方式与物理世界互动的机器人系统的一个核心能力。尽管基于视觉的方法已经迅速发展，但在精细控制方面，特别是在无结构或视觉遮挡的环境中，触觉感知仍然至关重要。我们提出了一种名为ViTacFormer的表示学习方法，该方法结合了跨注意力编码器以融合高分辨率视觉和触觉信息，并配备了一个自回归触觉预测头部，可以预见未来的接触信号。在此架构基础上，我们设计了一种从易到难的课程学习，逐步优化视觉-触觉潜在空间，从而提升准确性和鲁棒性。学习到的跨模态表示驱动多指手的模仿学习，使其能够实现精确和适应性的操作。在一系列具有挑战性的现实世界基准测试中，我们的方法实现了比之前的最先进的系统约50%更高的成功率。据我们所知，这也是第一个能够自主完成需要具有高度精确控制的长时间 horizon 灵巧操作任务的人形手，成功执行了多达11个连续阶段，并持续运行2.5分钟。 

---
# KARL: Kalman-Filter Assisted Reinforcement Learner for Dynamic Object Tracking and Grasping 

**Title (ZH)**: KARL：卡尔曼滤波辅助强化学习者用于动态物体跟踪与抓取 

**Authors**: Kowndinya Boyalakuntla, Abdeslam Boularias, Jingjin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15945)  

**Abstract**: We present Kalman-filter Assisted Reinforcement Learner (KARL) for dynamic object tracking and grasping over eye-on-hand (EoH) systems, significantly expanding such systems capabilities in challenging, realistic environments. In comparison to the previous state-of-the-art, KARL (1) incorporates a novel six-stage RL curriculum that doubles the system's motion range, thereby greatly enhancing the system's grasping performance, (2) integrates a robust Kalman filter layer between the perception and reinforcement learning (RL) control modules, enabling the system to maintain an uncertain but continuous 6D pose estimate even when the target object temporarily exits the camera's field-of-view or undergoes rapid, unpredictable motion, and (3) introduces mechanisms to allow retries to gracefully recover from unavoidable policy execution failures. Extensive evaluations conducted in both simulation and real-world experiments qualitatively and quantitatively corroborate KARL's advantage over earlier systems, achieving higher grasp success rates and faster robot execution speed. Source code and supplementary materials for KARL will be made available at: this https URL. 

**Abstract (ZH)**: 基于卡尔曼滤波辅助强化学习的动态物体跟踪与抓取系统（KARL）：扩展眼随手系统在挑战性现实环境中的能力 

---
# CooperRisk: A Driving Risk Quantification Pipeline with Multi-Agent Cooperative Perception and Prediction 

**Title (ZH)**: CooperRisk: 基于多智能体协同感知与预测的驾驶风险量化管道 

**Authors**: Mingyue Lei, Zewei Zhou, Hongchen Li, Jia Hu, Jiaqi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.15868)  

**Abstract**: Risk quantification is a critical component of safe autonomous driving, however, constrained by the limited perception range and occlusion of single-vehicle systems in complex and dense scenarios. Vehicle-to-everything (V2X) paradigm has been a promising solution to sharing complementary perception information, nevertheless, how to ensure the risk interpretability while understanding multi-agent interaction with V2X remains an open question. In this paper, we introduce the first V2X-enabled risk quantification pipeline, CooperRisk, to fuse perception information from multiple agents and quantify the scenario driving risk in future multiple timestamps. The risk is represented as a scenario risk map to ensure interpretability based on risk severity and exposure, and the multi-agent interaction is captured by the learning-based cooperative prediction model. We carefully design a risk-oriented transformer-based prediction model with multi-modality and multi-agent considerations. It aims to ensure scene-consistent future behaviors of multiple agents and avoid conflicting predictions that could lead to overly conservative risk quantification and cause the ego vehicle to become overly hesitant to drive. Then, the temporal risk maps could serve to guide a model predictive control planner. We evaluate the CooperRisk pipeline in a real-world V2X dataset V2XPnP, and the experiments demonstrate its superior performance in risk quantification, showing a 44.35% decrease in conflict rate between the ego vehicle and background traffic participants. 

**Abstract (ZH)**: 基于V2X的风险量化管道CooperRisk：多代理交互的可解释风险评估与预测 

---
# PRISM-Loc: a Lightweight Long-range LiDAR Localization in Urban Environments with Topological Maps 

**Title (ZH)**: PRISM-Loc：基于拓扑地图的城市环境轻量级远距离LiDAR定位 

**Authors**: Kirill Muravyev, Vasily Yuryev, Oleg Bulichev, Dmitry Yudin, Konstantin Yakovlev  

**Link**: [PDF](https://arxiv.org/pdf/2506.15849)  

**Abstract**: Localization in the environment is one of the crucial tasks of navigation of a mobile robot or a self-driving vehicle. For long-range routes, performing localization within a dense global lidar map in real time may be difficult, and the creation of such a map may require much memory. To this end, leveraging topological maps may be useful. In this work, we propose PRISM-Loc -- a topological map-based approach for localization in large environments. The proposed approach leverages a twofold localization pipeline, which consists of global place recognition and estimation of the local pose inside the found location. For local pose estimation, we introduce an original lidar scan matching algorithm, which is based on 2D features and point-based optimization. We evaluate the proposed method on the ITLP-Campus dataset on a 3 km route, and compare it against the state-of-the-art metric map-based and place recognition-based competitors. The results of the experiments show that the proposed method outperforms its competitors both quality-wise and computationally-wise. 

**Abstract (ZH)**: 基于拓扑图的大型环境定位方法PRISM-Loc 

---
# SafeMimic: Towards Safe and Autonomous Human-to-Robot Imitation for Mobile Manipulation 

**Title (ZH)**: SafeMimic：迈向安全自主的人机模仿移动操作 

**Authors**: Arpit Bahety, Arnav Balaji, Ben Abbatematteo, Roberto Martín-Martín  

**Link**: [PDF](https://arxiv.org/pdf/2506.15847)  

**Abstract**: For robots to become efficient helpers in the home, they must learn to perform new mobile manipulation tasks simply by watching humans perform them. Learning from a single video demonstration from a human is challenging as the robot needs to first extract from the demo what needs to be done and how, translate the strategy from a third to a first-person perspective, and then adapt it to be successful with its own morphology. Furthermore, to mitigate the dependency on costly human monitoring, this learning process should be performed in a safe and autonomous manner. We present SafeMimic, a framework to learn new mobile manipulation skills safely and autonomously from a single third-person human video. Given an initial human video demonstration of a multi-step mobile manipulation task, SafeMimic first parses the video into segments, inferring both the semantic changes caused and the motions the human executed to achieve them and translating them to an egocentric reference. Then, it adapts the behavior to the robot's own morphology by sampling candidate actions around the human ones, and verifying them for safety before execution in a receding horizon fashion using an ensemble of safety Q-functions trained in simulation. When safe forward progression is not possible, SafeMimic backtracks to previous states and attempts a different sequence of actions, adapting both the trajectory and the grasping modes when required for its morphology. As a result, SafeMimic yields a strategy that succeeds in the demonstrated behavior and learns task-specific actions that reduce exploration in future attempts. Our experiments show that our method allows robots to safely and efficiently learn multi-step mobile manipulation behaviors from a single human demonstration, from different users, and in different environments, with improvements over state-of-the-art baselines across seven tasks 

**Abstract (ZH)**: 家用机器人通过观看人类演示学习新移动操作任务以成为高效的助手 

---
# Steering Your Diffusion Policy with Latent Space Reinforcement Learning 

**Title (ZH)**: 使用潜空间强化学习引导你的扩散策略 

**Authors**: Andrew Wagenmaker, Mitsuhiko Nakamoto, Yunchu Zhang, Seohong Park, Waleed Yagoub, Anusha Nagabandi, Abhishek Gupta, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2506.15799)  

**Abstract**: Robotic control policies learned from human demonstrations have achieved impressive results in many real-world applications. However, in scenarios where initial performance is not satisfactory, as is often the case in novel open-world settings, such behavioral cloning (BC)-learned policies typically require collecting additional human demonstrations to further improve their behavior -- an expensive and time-consuming process. In contrast, reinforcement learning (RL) holds the promise of enabling autonomous online policy improvement, but often falls short of achieving this due to the large number of samples it typically requires. In this work we take steps towards enabling fast autonomous adaptation of BC-trained policies via efficient real-world RL. Focusing in particular on diffusion policies -- a state-of-the-art BC methodology -- we propose diffusion steering via reinforcement learning (DSRL): adapting the BC policy by running RL over its latent-noise space. We show that DSRL is highly sample efficient, requires only black-box access to the BC policy, and enables effective real-world autonomous policy improvement. Furthermore, DSRL avoids many of the challenges associated with finetuning diffusion policies, obviating the need to modify the weights of the base policy at all. We demonstrate DSRL on simulated benchmarks, real-world robotic tasks, and for adapting pretrained generalist policies, illustrating its sample efficiency and effective performance at real-world policy improvement. 

**Abstract (ZH)**: 基于强化学习的快速自主适应行为克隆训练策略 

---
# AnyTraverse: An off-road traversability framework with VLM and human operator in the loop 

**Title (ZH)**: AnyTraverse: 一种结合VLM和人工操作者的离路穿越框架 

**Authors**: Sattwik Sahu, Agamdeep Singh, Karthik Nambiar, Srikanth Saripalli, P.B. Sujit  

**Link**: [PDF](https://arxiv.org/pdf/2506.16826)  

**Abstract**: Off-road traversability segmentation enables autonomous navigation with applications in search-and-rescue, military operations, wildlife exploration, and agriculture. Current frameworks struggle due to significant variations in unstructured environments and uncertain scene changes, and are not adaptive to be used for different robot types. We present AnyTraverse, a framework combining natural language-based prompts with human-operator assistance to determine navigable regions for diverse robotic vehicles. The system segments scenes for a given set of prompts and calls the operator only when encountering previously unexplored scenery or unknown class not part of the prompt in its region-of-interest, thus reducing active supervision load while adapting to varying outdoor scenes. Our zero-shot learning approach eliminates the need for extensive data collection or retraining. Our experimental validation includes testing on RELLIS-3D, Freiburg Forest, and RUGD datasets and demonstrate real-world deployment on multiple robot platforms. The results show that AnyTraverse performs better than GA-NAV and Off-seg while offering a vehicle-agnostic approach to off-road traversability that balances automation with targeted human supervision. 

**Abstract (ZH)**: 基于自然语言提示与操作员辅助的离路通行性分割使能自主导航应用于搜索救援、军事操作、野生动物探索和农业等领域 

---
# PPTP: Performance-Guided Physiological Signal-Based Trust Prediction in Human-Robot Collaboration 

**Title (ZH)**: 基于生理信号的性能指导型人类与机器人协作信任预测 

**Authors**: Hao Guo, Wei Fan, Shaohui Liu, Feng Jiang, Chunzhi Yi  

**Link**: [PDF](https://arxiv.org/pdf/2506.16677)  

**Abstract**: Trust prediction is a key issue in human-robot collaboration, especially in construction scenarios where maintaining appropriate trust calibration is critical for safety and efficiency. This paper introduces the Performance-guided Physiological signal-based Trust Prediction (PPTP), a novel framework designed to improve trust assessment. We designed a human-robot construction scenario with three difficulty levels to induce different trust states. Our approach integrates synchronized multimodal physiological signals (ECG, GSR, and EMG) with collaboration performance evaluation to predict human trust levels. Individual physiological signals are processed using collaboration performance information as guiding cues, leveraging the standardized nature of collaboration performance to compensate for individual variations in physiological responses. Extensive experiments demonstrate the efficacy of our cross-modality fusion method in significantly improving trust classification performance. Our model achieves over 81% accuracy in three-level trust classification, outperforming the best baseline method by 6.7%, and notably reaches 74.3% accuracy in high-resolution seven-level classification, which is a first in trust prediction research. Ablation experiments further validate the superiority of physiological signal processing guided by collaboration performance assessment. 

**Abstract (ZH)**: 基于绩效引导的生理信号信任预测（PPTP）：一种提高信任评估的方法 

---
# IS-Bench: Evaluating Interactive Safety of VLM-Driven Embodied Agents in Daily Household Tasks 

**Title (ZH)**: IS-Bench: 评估基于VLM的具身代理在日常家务任务中互动安全性eci-bench: 评估基于VLM的具身代理在日常家务任务中互动安全性 

**Authors**: Xiaoya Lu, Zeren Chen, Xuhao Hu, Yijin Zhou, Weichen Zhang, Dongrui Liu, Lu Sheng, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16402)  

**Abstract**: Flawed planning from VLM-driven embodied agents poses significant safety hazards, hindering their deployment in real-world household tasks. However, existing static, non-interactive evaluation paradigms fail to adequately assess risks within these interactive environments, since they cannot simulate dynamic risks that emerge from an agent's actions and rely on unreliable post-hoc evaluations that ignore unsafe intermediate steps. To bridge this critical gap, we propose evaluating an agent's interactive safety: its ability to perceive emergent risks and execute mitigation steps in the correct procedural order. We thus present IS-Bench, the first multi-modal benchmark designed for interactive safety, featuring 161 challenging scenarios with 388 unique safety risks instantiated in a high-fidelity simulator. Crucially, it facilitates a novel process-oriented evaluation that verifies whether risk mitigation actions are performed before/after specific risk-prone steps. Extensive experiments on leading VLMs, including the GPT-4o and Gemini-2.5 series, reveal that current agents lack interactive safety awareness, and that while safety-aware Chain-of-Thought can improve performance, it often compromises task completion. By highlighting these critical limitations, IS-Bench provides a foundation for developing safer and more reliable embodied AI systems. 

**Abstract (ZH)**: 基于VLM驱动的实体智能代理的规划缺陷带来了重大的安全风险，阻碍了它们在现实家庭任务中的应用。然而，现有的静态、非交互式评估范式无法充分评估这些交互环境中出现的风险，因为它们无法模拟由代理行为引发的动态风险，并且依赖于忽视中间不安全步骤的不可靠事后评估。为了弥合这一关键差距，我们提出评估代理的交互安全性：其感知新兴风险并按正确程序顺序执行缓解步骤的能力。因此，我们提出了IS-Bench，这是首个针对交互安全性的多模态基准，包含161个具有388种独特安全风险的高保真模拟器中的挑战性场景。至关重要的是，它促进了一种新的过程导向评估，验证风险缓解行动是否在特定风险易发步骤之前/之后执行。对领先的人类躯体模型，包括GPT-4o和Gemini-2.5系列，进行的广泛实验表明，当前的代理缺乏交互安全性意识，尽管安全意识的逐步推理可以改善性能，但往往会牺牲任务完成度。通过揭示这些关键局限性，IS-Bench为开发更安全、更可靠的实体AI系统提供了基础。 

---
# Human-Centered Shared Autonomy for Motor Planning, Learning, and Control Applications 

**Title (ZH)**: 以人为本的共享自治在运动规划、学习与控制中的应用 

**Authors**: MH Farhadi, Ali Rabiee, Sima Ghafoori, Anna Cetera, Wei Xu, Reza Abiri  

**Link**: [PDF](https://arxiv.org/pdf/2506.16044)  

**Abstract**: With recent advancements in AI and computational tools, intelligent paradigms have emerged to enhance fields like shared autonomy and human-machine teaming in healthcare. Advanced AI algorithms (e.g., reinforcement learning) can autonomously make decisions to achieve planning and motion goals. However, in healthcare, where human intent is crucial, fully independent machine decisions may not be ideal. This chapter presents a comprehensive review of human-centered shared autonomy AI frameworks, focusing on upper limb biosignal-based machine interfaces and associated motor control systems, including computer cursors, robotic arms, and planar platforms. We examine motor planning, learning (rehabilitation), and control, covering conceptual foundations of human-machine teaming in reach-and-grasp tasks and analyzing both theoretical and practical implementations. Each section explores how human and machine inputs can be blended for shared autonomy in healthcare applications. Topics include human factors, biosignal processing for intent detection, shared autonomy in brain-computer interfaces (BCI), rehabilitation, assistive robotics, and Large Language Models (LLMs) as the next frontier. We propose adaptive shared autonomy AI as a high-performance paradigm for collaborative human-AI systems, identify key implementation challenges, and outline future directions, particularly regarding AI reasoning agents. This analysis aims to bridge neuroscientific insights with robotics to create more intuitive, effective, and ethical human-machine teaming frameworks. 

**Abstract (ZH)**: 近年来，随着人工智能和计算工具的进步，智能 paradigms 已在增强医疗健康领域的共享自主性和人机协同方面崭露头角。高级 AI 算法（例如强化学习）可以自主作出决策以实现规划和运动目标。然而，在医疗健康领域，由于人类意图至关重要，完全独立的机器决策可能并不理想。本章综述了以人类为中心的共享自主性 AI 框架，重点关注基于上肢生物信号的机器界面及相关运动控制系统，包括计算机光标、机器人手臂和平面平台。本文探讨了运动规划、学习（康复）和控制，涵盖接近-抓取任务中的人机协同概念基础，并分析了理论和实践实现。每个部分都探讨了如何在医疗应用中将人类和机器输入结合起来实现共享自主性。主题包括人因工程、生物信号处理以探测意图、脑机接口（BCI）中的共享自主性、康复、辅助机器人以及大语言模型（LLMs）作为下一个前沿领域。我们提出了适应性共享自主性 AI 作为协作人-机系统高性能范式，确定了关键实现挑战，并概述了未来方向，特别是关于 AI 推理代理。本分析旨在将神经科学洞察与机器人技术相结合，创建更具直观性、有效性与伦理性的医疗人机协同框架。 

---
# Quantum Artificial Intelligence for Secure Autonomous Vehicle Navigation: An Architectural Proposal 

**Title (ZH)**: 量子人工智能在安全自主车辆导航中的应用：一种架构提案 

**Authors**: Hemanth Kannamarlapudi, Sowmya Chintalapudi  

**Link**: [PDF](https://arxiv.org/pdf/2506.16000)  

**Abstract**: Navigation is a very crucial aspect of autonomous vehicle ecosystem which heavily relies on collecting and processing large amounts of data in various states and taking a confident and safe decision to define the next vehicle maneuver. In this paper, we propose a novel architecture based on Quantum Artificial Intelligence by enabling quantum and AI at various levels of navigation decision making and communication process in Autonomous vehicles : Quantum Neural Networks for multimodal sensor fusion, Nav-Q for Quantum reinforcement learning for navigation policy optimization and finally post-quantum cryptographic protocols for secure communication. Quantum neural networks uses quantum amplitude encoding to fuse data from various sensors like LiDAR, radar, camera, GPS and weather etc., This approach gives a unified quantum state representation between heterogeneous sensor modalities. Nav-Q module processes the fused quantum states through variational quantum circuits to learn optimal navigation policies under swift dynamic and complex conditions. Finally, post quantum cryptographic protocols are used to secure communication channels for both within vehicle communication and V2X (Vehicle to Everything) communications and thus secures the autonomous vehicle communication from both classical and quantum security threats. Thus, the proposed framework addresses fundamental challenges in autonomous vehicles navigation by providing quantum performance and future proof security. Index Terms Quantum Computing, Autonomous Vehicles, Sensor Fusion 

**Abstract (ZH)**: 基于量子人工智能的自主车辆导航新型架构：量子神经网络多模传感器融合、Nav-Q量子强化学习导航策略优化及后量子加密协议安全通信 

---
# Optimal Navigation in Microfluidics via the Optimization of a Discrete Loss 

**Title (ZH)**: 通过离散损失优化实现的微流控最优导航 

**Authors**: Petr Karnakov, Lucas Amoudruz, Petros Koumoutsakos  

**Link**: [PDF](https://arxiv.org/pdf/2506.15902)  

**Abstract**: Optimal path planning and control of microscopic devices navigating in fluid environments is essential for applications ranging from targeted drug delivery to environmental monitoring. These tasks are challenging due to the complexity of microdevice-flow interactions. We introduce a closed-loop control method that optimizes a discrete loss (ODIL) in terms of dynamics and path objectives. In comparison with reinforcement learning, ODIL is more robust, up to three orders faster, and excels in high-dimensional action/state spaces, making it a powerful tool for navigating complex flow environments. 

**Abstract (ZH)**: 微观设备在流体环境中导航的最优路径规划与控制对于从靶向药物递送到环境监测等应用至关重要。由于微设备-流体相互作用的复杂性，这些任务具有挑战性。我们提出了一种闭环控制方法，基于动力学和路径目标优化离散损失（ODIL）。与强化学习相比，ODIL更稳健，速度快三个数量级，并且在高维动作/状态空间中表现出色，使其成为导航复杂流场环境的有力工具。 

---
# When Can Model-Free Reinforcement Learning be Enough for Thinking? 

**Title (ZH)**: 无模型强化学习何时足以进行思考？ 

**Authors**: Josiah P. Hanna, Nicholas E. Corrado  

**Link**: [PDF](https://arxiv.org/pdf/2506.17124)  

**Abstract**: Recent work on large language models has demonstrated the use of model-free reinforcement learning (RL) to train reasoning-like capabilities. The emergence of "thinking" through model-free RL is interesting as thinking actions neither produce reward nor change the external world state to one where the agent is more likely to get reward. This paper seeks to build a domain-independent understanding of when model-free RL will lead to "thinking" as a strategy for reward maximization. To build this understanding, we first introduce a theoretical model which we call a \textit{thought Markov decision process} (MDP). Thought MDPs minimally extend the classical MDP model to include an abstract notion of thought state and thought action. Using the thought MDP model, we prove the importance of policy initialization in determining whether or not thinking emerges and show formally that thought actions are equivalent to the agent choosing to perform a step of policy improvement before continuing to act. We then show that open-source LLMs satisfy the conditions that our theory predicts are necessary for model-free RL to produce thinking-like behavior. Finally, we hypothesize sufficient conditions that would enable thinking to be learned outside of language generation and introduce a toy domain where a combination of multi-task pre-training and designated thought actions enable more data-efficient RL compared to non-thinking agents. 

**Abstract (ZH)**: recent 工作表明，通过无模型强化学习（RL）训练具有类推理能力是可行的。无模型 RL 中的“思考”行为令人感兴趣，因为思考动作既不产生奖励，也不改变外部世界状态使得代理更有可能获得奖励。本文旨在构建一种独立于领域的理解和判断，在哪种情况下无模型 RL 将导致“思考”作为一种奖励最大化策略。为了构建这种理解，我们首先引入了一个称为“思考马尔可夫决策过程” (MDP) 的理论模型。思考 MDP 仅最小地扩展了经典的 MDP 模型，以包括抽象的概念——思考状态和思考动作。利用思考 MDP 模型，我们证明了策略初始化在决定是否会出现思考方面的重要性，并且正式证明了思考动作等价于代理选择执行一次策略改进步骤后再继续行动。随后，我们展示了开源语言模型满足理论预测的必要条件，从而使无模型 RL 产生类似思考的行为。最后，我们提出了允许思考在语言生成之外被学到的充分条件，并引入了一个玩具领域，在该领域中，多任务预训练与特定设计的思考动作的结合使相对于非思考代理更高效的学习成为可能。 

---
# Elevating Styled Mahjong Agents with Learning from Demonstration 

**Title (ZH)**: 基于示范学习提升风格化麻将代理 

**Authors**: Lingfeng Li, Yunlong Lu, Yongyi Wang, Wenxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16995)  

**Abstract**: A wide variety of bots in games enriches the gameplay experience and enhances replayability. Recent advancements in game artificial intelligence have predominantly focused on improving the proficiency of bots. Nevertheless, developing highly competent bots with a wide range of distinct play styles remains a relatively under-explored area. We select the Mahjong game environment as a case study. The high degree of randomness inherent in the Mahjong game and the prevalence of out-of-distribution states lead to suboptimal performance of existing offline learning and Learning-from-Demonstration (LfD) algorithms. In this paper, we leverage the gameplay histories of existing Mahjong agents and put forward a novel LfD algorithm that necessitates only minimal modifications to the Proximal Policy Optimization algorithm. The comprehensive empirical results illustrate that our proposed method not only significantly enhances the proficiency of the agents but also effectively preserves their unique play styles. 

**Abstract (ZH)**: 游戏中的各种 bots 丰富了游戏体验并增强了重玩价值。近年来，游戏人工智能的发展主要集中在提高 bots 的熟练程度上。然而，开发具有广泛不同游戏风格的高水平 bots 仍然是一个相对未被充分探索的领域。我们以麻将游戏环境为例。麻将游戏固有的高随机性和离分布状态的普遍存在导致现有离线学习和学习从演示（LfD）算法的性能不佳。在本文中，我们利用现有麻将代理的游戏历史记录，提出了一种仅需对策略优化（PPO）算法进行少量修改的新LfD算法。综合的实证结果表明，我们提出的方法不仅显著提高了代理的熟练程度，还有效地保留了它们的独特游戏风格。 

---
# Reinforcement learning for hybrid charging stations planning and operation considering fixed and mobile chargers 

**Title (ZH)**: 基于固定和移动充电器考虑的混合充电站规划与运行的强化学习方法 

**Authors**: Yanchen Zhu, Honghui Zou, Chufan Liu, Yuyu Luo, Yuankai Wu, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16764)  

**Abstract**: The success of vehicle electrification, which brings significant societal and environmental benefits, is contingent upon the availability of efficient and adaptable charging infrastructure. Traditional fixed-location charging stations often face issues like underutilization or congestion due to the dynamic nature of charging demand. Mobile chargers have emerged as a flexible solution, capable of relocating to align with these demand fluctuations. This paper addresses the optimal planning and operation of hybrid charging infrastructures, integrating both fixed and mobile chargers within urban road networks. We introduce the Hybrid Charging Station Planning and Operation (HCSPO) problem, which simultaneously optimizes the location and configuration of fixed charging stations and schedules mobile chargers for dynamic operations. Our approach incorporates a charging demand prediction model grounded in Model Predictive Control (MPC) to enhance decision-making. To solve the HCSPO problem, we propose a deep reinforcement learning method, augmented with heuristic scheduling techniques, to effectively bridge the planning of fixed chargers with the real-time operation of mobile chargers. Extensive case studies using real-world urban scenarios demonstrate that our method significantly improves the availability of charging infrastructure and reduces user inconvenience compared to existing solutions and baselines. 

**Abstract (ZH)**: 车辆电气化成功的实现，带来了重要的社会和环境效益，取决于高效且灵活的充电基础设施的可用性。传统的固定位置充电站常因充电需求的动态变化而面临利用率低或拥堵的问题。移动充电器作为一项灵活的解决方案，能够根据需求变化重新部署。本文探讨了混合充电基础设施的最优规划与运营问题，将固定和移动充电站整合到城市道路网络中。我们提出了混合充电站规划与运营（HCSPO）问题，该问题同时优化了固定充电站的位置和配置，并为移动充电器安排动态操作。我们的方法结合了基于模型预测控制（MPC）的充电需求预测模型，以增强决策制定能力。为了解决HCSPO问题，我们提出了一种增强学习方法，并结合启发式调度技术，有效地将固定充电器的规划与移动充电器的实时操作相结合。通过对现实世界城市场景的广泛案例研究，我们的方法显著提高了充电基础设施的可用性，并减少了用户的不便，相较于现有解决方案和基线方法。 

---
# Deep Reinforcement Learning Xiangqi Player with Monte Carlo Tree Search 

**Title (ZH)**: 基于蒙特卡洛树搜索的深度强化学习象棋玩家 

**Authors**: Berk Yilmaz, Junyu Hu, Jinsong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15880)  

**Abstract**: This paper presents a Deep Reinforcement Learning (DRL) system for Xiangqi (Chinese Chess) that integrates neural networks with Monte Carlo Tree Search (MCTS) to enable strategic self-play and self-improvement. Addressing the underexplored complexity of Xiangqi, including its unique board layout, piece movement constraints, and victory conditions, our approach combines policy-value networks with MCTS to simulate move consequences and refine decision-making. By overcoming challenges such as Xiangqi's high branching factor and asymmetrical piece dynamics, our work advances AI capabilities in culturally significant strategy games while providing insights for adapting DRL-MCTS frameworks to domain-specific rule systems. 

**Abstract (ZH)**: 基于深度强化学习的中国象棋系统：结合神经网络与蒙特卡洛树搜索实现策略自我对弈与提升 

---
# OAgents: An Empirical Study of Building Effective Agents 

**Title (ZH)**: OAgents：构建有效代理的实证研究 

**Authors**: He Zhu, Tianrui Qin, King Zhu, Heyuan Huang, Yeyi Guan, Jinxiang Xia, Yi Yao, Hanhao Li, Ningning Wang, Pai Liu, Tianhao Peng, Xin Gui, Xiaowan Li, Yuhui Liu, Yuchen Eleanor Jiang, Jun Wang, Changwang Zhang, Xiangru Tang, Ge Zhang, Jian Yang, Minghao Liu, Xitong Gao, Wangchunshu Zhou, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15741)  

**Abstract**: Recently, Agentic AI has become an increasingly popular research field. However, we argue that current agent research practices lack standardization and scientific rigor, making it hard to conduct fair comparisons among methods. As a result, it is still unclear how different design choices in agent frameworks affect effectiveness, and measuring their progress remains challenging. In this work, we conduct a systematic empirical study on GAIA benchmark and BrowseComp to examine the impact of popular design choices in key agent components in a fair and rigorous manner. We find that the lack of a standard evaluation protocol makes previous works, even open-sourced ones, non-reproducible, with significant variance between random runs. Therefore, we introduce a more robust evaluation protocol to stabilize comparisons. Our study reveals which components and designs are crucial for effective agents, while others are redundant, despite seeming logical. Based on our findings, we build and open-source OAgents, a new foundation agent framework that achieves state-of-the-art performance among open-source projects. OAgents offers a modular design for various agent components, promoting future research in Agentic AI. 

**Abstract (ZH)**: 最近，代理型人工智能成为了一个日益流行的研究领域。然而，我们认为目前的代理研究实践缺乏标准化和科学严谨性，使得不同方法之间难以进行公正比较。因此，仍不清楚不同代理框架设计选择如何影响其有效性，衡量其进步也颇具挑战性。在本文中，我们对GAIA基准和BrowseComp进行了系统性的 empirical 研究，以公平和严谨的方式考察关键代理组件中流行设计选择的影响。我们发现，缺乏标准评估协议使得以往的研究工作，即使开源的，也不可重现，随机运行之间存在显著差异。因此，我们引入了一种更稳健的评估协议以稳定比较。我们的研究揭示了哪些组件和设计对于有效代理是至关重要的，而哪些则是冗余的，尽管这些设计看起来合乎逻辑。基于我们的发现，我们构建并开源了OA_agents，这是一个新的基础代理框架，其中包含了最先进的开源项目性能。OA_agents 提供了各种代理组件的模块化设计，促进了代理型人工智能领域的未来研究。 

---
# TransDreamerV3: Implanting Transformer In DreamerV3 

**Title (ZH)**: TransDreamerV3: 在DreamerV3中植入Transformer 

**Authors**: Shruti Sadanand Dongare, Amun Kharel, Jonathan Samuel, Xiaona Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17103)  

**Abstract**: This paper introduces TransDreamerV3, a reinforcement learning model that enhances the DreamerV3 architecture by integrating a transformer encoder. The model is designed to improve memory and decision-making capabilities in complex environments. We conducted experiments on Atari-Boxing, Atari-Freeway, Atari-Pong, and Crafter tasks, where TransDreamerV3 demonstrated improved performance over DreamerV3, particularly in the Atari-Freeway and Crafter tasks. While issues in the Minecraft task and limited training across all tasks were noted, TransDreamerV3 displays advancement in world model-based reinforcement learning, leveraging transformer architectures. 

**Abstract (ZH)**: TransDreamerV3：一种通过集成变压器编码器增强的DreamerV3架构的强化学习模型 

---
# ParkFormer: A Transformer-Based Parking Policy with Goal Embedding and Pedestrian-Aware Control 

**Title (ZH)**: ParkFormer：基于目标嵌入和行人感知控制的变压器停车策略 

**Authors**: Jun Fu, Bin Tian, Haonan Chen, Shi Meng, Tingting Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16856)  

**Abstract**: Autonomous parking plays a vital role in intelligent vehicle systems, particularly in constrained urban environments where high-precision control is required. While traditional rule-based parking systems struggle with environmental uncertainties and lack adaptability in crowded or dynamic scenes, human drivers demonstrate the ability to park intuitively without explicit modeling. Inspired by this observation, we propose a Transformer-based end-to-end framework for autonomous parking that learns from expert demonstrations. The network takes as input surround-view camera images, goal-point representations, ego vehicle motion, and pedestrian trajectories. It outputs discrete control sequences including throttle, braking, steering, and gear selection. A novel cross-attention module integrates BEV features with target points, and a GRU-based pedestrian predictor enhances safety by modeling dynamic obstacles. We validate our method on the CARLA 0.9.14 simulator in both vertical and parallel parking scenarios. Experiments show our model achieves a high success rate of 96.57\%, with average positional and orientation errors of 0.21 meters and 0.41 degrees, respectively. The ablation studies further demonstrate the effectiveness of key modules such as pedestrian prediction and goal-point attention fusion. The code and dataset will be released at: this https URL. 

**Abstract (ZH)**: 自主泊车在智能车辆系统中扮演着重要角色，特别是在受限制的都市环境中，需要高精度控制。虽然传统的基于规则的泊车系统难以应对环境不确定性并缺乏在拥挤或动态场景中的适应性，但人类驾驶员能够直观地泊车而无需显式的建模。受此启发，我们提出了一种基于Transformer的端到端自主泊车框架，该框架从专家示范中学习。网络将全景摄像头图像、目标点表示、ego车辆运动和行人的轨迹作为输入，输出包括油门、刹车、转向和换挡的离散控制序列。一个新颖的交叉注意力模块将BEV特征与目标点集成，基于GRU的行人预测器通过建模动态障碍物来增强安全性。我们在CARLA 0.9.14模拟器上对垂直泊车和平行泊车场景进行了实验验证，结果显示我们的模型的成功率为96.57%，平均位置和方向误差分别为0.21米和0.41度。进一步的消融研究展示了行人预测和目标点注意力融合等关键模块的有效性。相关代码和数据集将在以下链接中发布：this https URL。 

---
# Robust Dynamic Material Handling via Adaptive Constrained Evolutionary Reinforcement Learning 

**Title (ZH)**: 适应性约束进化强化学习下的稳健动态物料处理 

**Authors**: Chengpeng Hu, Ziming Wang, Bo Yuan, Jialin Liu, Chengqi Zhang, Xin Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16795)  

**Abstract**: Dynamic material handling (DMH) involves the assignment of dynamically arriving material transporting tasks to suitable vehicles in real time for minimising makespan and tardiness. In real-world scenarios, historical task records are usually available, which enables the training of a decision policy on multiple instances consisting of historical records. Recently, reinforcement learning has been applied to solve DMH. Due to the occurrence of dynamic events such as new tasks, adaptability is highly required. Solving DMH is challenging since constraints including task delay should be satisfied. A feedback is received only when all tasks are served, which leads to sparse reward. Besides, making the best use of limited computational resources and historical records for training a robust policy is crucial. The time allocated to different problem instances would highly impact the learning process. To tackle those challenges, this paper proposes a novel adaptive constrained evolutionary reinforcement learning (ACERL) approach, which maintains a population of actors for diverse exploration. ACERL accesses each actor for tackling sparse rewards and constraint violation to restrict the behaviour of the policy. Moreover, ACERL adaptively selects the most beneficial training instances for improving the policy. Extensive experiments on eight training and eight unseen test instances demonstrate the outstanding performance of ACERL compared with several state-of-the-art algorithms. Policies trained by ACERL can schedule the vehicles while fully satisfying the constraints. Additional experiments on 40 unseen noised instances show the robust performance of ACERL. Cross-validation further presents the overall effectiveness of ACREL. Besides, a rigorous ablation study highlights the coordination and benefits of each ingredient of ACERL. 

**Abstract (ZH)**: 动态物料搬运中的自适应约束进化强化学习（ACERL）方法 

---
# Language-Informed Synthesis of Rational Agent Models for Grounded Theory-of-Mind Reasoning On-The-Fly 

**Title (ZH)**: 基于语言指导的合理代理模型合成用于即时嵌地心智理论推理 

**Authors**: Lance Ying, Ryan Truong, Katherine M. Collins, Cedegao E. Zhang, Megan Wei, Tyler Brooke-Wilson, Tan Zhi-Xuan, Lionel Wong, Joshua B. Tenenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2506.16755)  

**Abstract**: Drawing real world social inferences usually requires taking into account information from multiple modalities. Language is a particularly powerful source of information in social settings, especially in novel situations where language can provide both abstract information about the environment dynamics and concrete specifics about an agent that cannot be easily visually observed. In this paper, we propose Language-Informed Rational Agent Synthesis (LIRAS), a framework for drawing context-specific social inferences that integrate linguistic and visual inputs. LIRAS frames multimodal social reasoning as a process of constructing structured but situation-specific agent and environment representations - leveraging multimodal language models to parse language and visual inputs into unified symbolic representations, over which a Bayesian inverse planning engine can be run to produce granular probabilistic judgments. On a range of existing and new social reasoning tasks derived from cognitive science experiments, we find that our model (instantiated with a comparatively lightweight VLM) outperforms ablations and state-of-the-art models in capturing human judgments across all domains. 

**Abstract (ZH)**: 基于语言指导的理性代理合成（LIRAS）：一种结合语言和视觉输入进行情境特定社会推理的框架 

---
# Do We Talk to Robots Like Therapists, and Do They Respond Accordingly? Language Alignment in AI Emotional Support 

**Title (ZH)**: 我们像对待治疗师一样与机器人交谈，它们也会相应地回应吗？AI情感支持中的语言对齐 

**Authors**: Sophie Chiang, Guy Laban, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2506.16473)  

**Abstract**: As conversational agents increasingly engage in emotionally supportive dialogue, it is important to understand how closely their interactions resemble those in traditional therapy settings. This study investigates whether the concerns shared with a robot align with those shared in human-to-human (H2H) therapy sessions, and whether robot responses semantically mirror those of human therapists. We analyzed two datasets: one of interactions between users and professional therapists (Hugging Face's NLP Mental Health Conversations), and another involving supportive conversations with a social robot (QTrobot from LuxAI) powered by a large language model (LLM, GPT-3.5). Using sentence embeddings and K-means clustering, we assessed cross-agent thematic alignment by applying a distance-based cluster-fitting method that evaluates whether responses from one agent type map to clusters derived from the other, and validated it using Euclidean distances. Results showed that 90.88% of robot conversation disclosures could be mapped to clusters from the human therapy dataset, suggesting shared topical structure. For matched clusters, we compared the subjects as well as therapist and robot responses using Transformer, Word2Vec, and BERT embeddings, revealing strong semantic overlap in subjects' disclosures in both datasets, as well as in the responses given to similar human disclosure themes across agent types (robot vs. human therapist). These findings highlight both the parallels and boundaries of robot-led support conversations and their potential for augmenting mental health interventions. 

**Abstract (ZH)**: 随着对话代理越来越多地参与情感支持对话，了解其互动与传统治疗环境中互动的相似性变得尤为重要。本研究探讨了与机器人分享的顾虑是否与人对人（H2H） therapy会话中分享的顾虑一致，以及机器人回复是否在语义上类似于人类治疗师。我们分析了两个数据集：一个是用户与专业治疗师互动的数据集（Hugging Face的NLP心理健康对话），另一个是与社会机器人（由大型语言模型GPT-3.5驱动的LuxAI的QTrobot）进行支持性对话的数据集。通过使用句子嵌入和K-means聚类，并应用基于距离的聚类拟合方法来评估一种代理类型回复是否映射到另一种代理类型衍生的聚类，并使用欧几里得距离进行验证。结果显示，90.88%的机器人对话披露可以映射到人类治疗数据集的聚类中，表明主题结构的共享。对匹配的聚类，我们使用Transformer、Word2Vec和BERT嵌入比较了话题以及治疗师和机器人的回复，揭示了两个数据集的话题披露以及不同代理类型（机器人 vs. 人类治疗师）对类似人类披露主题的回复具有强烈的语义重叠。这些发现突出了机器人引导支持对话的相似性和界限，并探讨了其在补充心理健康干预方面的潜力。 

---
# GRPO-CARE: Consistency-Aware Reinforcement Learning for Multimodal Reasoning 

**Title (ZH)**: GRPO-CARE：一致性导向的多模态强化学习 

**Authors**: Yi Chen, Yuying Ge, Rui Wang, Yixiao Ge, Junhao Cheng, Ying Shan, Xihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.16141)  

**Abstract**: Recent reinforcement learning approaches, such as outcome-supervised GRPO, have advanced Chain-of-Thought reasoning in large language models (LLMs), yet their adaptation to multimodal LLMs (MLLMs) is unexplored. To address the lack of rigorous evaluation for MLLM post-training methods, we introduce SEED-Bench-R1, a benchmark with complex real-world videos requiring balanced perception and reasoning. It offers a large training set and evaluates generalization across three escalating challenges: in-distribution, cross-environment, and cross-environment-task scenarios. Using SEED-Bench-R1, we find that standard GRPO, while improving answer accuracy, often reduces logical coherence between reasoning steps and answers, with only a 57.9% consistency rate. This stems from reward signals focusing solely on final answers, encouraging shortcuts, and strict KL penalties limiting this http URL address this, we propose GRPO-CARE, a consistency-aware RL framework optimizing both answer correctness and reasoning coherence without explicit supervision. GRPO-CARE introduces a two-tiered reward: (1) a base reward for answer correctness, and (2) an adaptive consistency bonus, computed by comparing the model's reasoning-to-answer likelihood (via a slowly-evolving reference model) against group this http URL dual mechanism amplifies rewards for reasoning paths that are both correct and logically consistent. Replacing KL penalties with this adaptive bonus, GRPO-CARE outperforms standard GRPO on SEED-Bench-R1, achieving a 6.7% performance gain on the hardest evaluation level and a 24.5% improvement in consistency. It also shows strong transferability, improving model performance across diverse video understanding benchmarks. Our work contributes a systematically designed benchmark and a generalizable post-training framework, advancing the development of more interpretable and robust MLLMs. 

**Abstract (ZH)**: Recent Reinforcement Learning Approaches for Enhancing Reasoning in Multimodal Large Language Models: Introducing SEED-Bench-R1 and GRPO-CARE 

---
