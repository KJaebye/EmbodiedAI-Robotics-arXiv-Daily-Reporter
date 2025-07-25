# WorldVLA: Towards Autoregressive Action World Model 

**Title (ZH)**: WorldVLA: 向 toward 自回归行动世界模型模型逼近 

**Authors**: Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, Deli Zhao, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.21539)  

**Abstract**: We present WorldVLA, an autoregressive action world model that unifies action and image understanding and generation. Our WorldVLA intergrates Vision-Language-Action (VLA) model and world model in one single framework. The world model predicts future images by leveraging both action and image understanding, with the purpose of learning the underlying physics of the environment to improve action generation. Meanwhile, the action model generates the subsequent actions based on image observations, aiding in visual understanding and in turn helps visual generation of the world model. We demonstrate that WorldVLA outperforms standalone action and world models, highlighting the mutual enhancement between the world model and the action model. In addition, we find that the performance of the action model deteriorates when generating sequences of actions in an autoregressive manner. This phenomenon can be attributed to the model's limited generalization capability for action prediction, leading to the propagation of errors from earlier actions to subsequent ones. To address this issue, we propose an attention mask strategy that selectively masks prior actions during the generation of the current action, which shows significant performance improvement in the action chunk generation task. 

**Abstract (ZH)**: WorldVLA：统一动作与图像理解与生成的自回归世界模型 

---
# ACTLLM: Action Consistency Tuned Large Language Model 

**Title (ZH)**: ACTLLM: 行动一致性调优的大语言模型 

**Authors**: Jing Bi, Lianggong Bruce Wen, Zhang Liu, Chenliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21250)  

**Abstract**: This paper introduces ACTLLM (Action Consistency Tuned Large Language Model), a novel approach for robot manipulation in dynamic environments. Traditional vision-based systems often struggle to learn visual representations that excel in both task execution and spatial reasoning, thereby limiting their adaptability in dynamic environments. ACTLLM addresses these challenges by harnessing language to craft structured scene descriptors, providing a uniform interface for both spatial understanding and task performance through flexible language instructions. Moreover, we introduce a novel action consistency constraint that aligns visual perception with corresponding actions, thereby enhancing the learning of actionable visual representations. Additionally, we have reformulated the Markov decision process for manipulation tasks into a multi-turn visual dialogue framework. This approach enables the modeling of long-term task execution with enhanced contextual relevance derived from the history of task execution. During our evaluation, ACTLLM excels in diverse scenarios, proving its effectiveness on challenging vision-based robot manipulation tasks. 

**Abstract (ZH)**: ACTLLM：动作一致性调优的大语言模型在动态环境中的机器人操作方法 

---
# UAIbot: Beginner-friendly web-based simulator for interactive robotics learning and research 

**Title (ZH)**: UAIbot：面向初学者的基于Web的交互式机器人学习与研究模拟器 

**Authors**: Johnata Brayan, Armando Alves Neto, Pavel Petrovič, Gustavo M Freitas, Vinicius Mariano Gonçalves  

**Link**: [PDF](https://arxiv.org/pdf/2506.21178)  

**Abstract**: This paper presents UAIbot, a free and open-source web-based robotics simulator designed to address the educational and research challenges conventional simulation platforms generally face. The Python and JavaScript interfaces of UAIbot enable accessible hands-on learning experiences without cumbersome installations. By allowing users to explore fundamental mathematical and physical principles interactively, ranging from manipulator kinematics to pedestrian flow dynamics, UAIbot provides an effective tool for deepening student understanding, facilitating rapid experimentation, and enhancing research dissemination. 

**Abstract (ZH)**: UAIbot：一种免费开源的基于Web的机器人模拟器，用于解决传统模拟平台普遍面临的教育教学和研究挑战 

---
# Knowledge-Driven Imitation Learning: Enabling Generalization Across Diverse Conditions 

**Title (ZH)**: 知识驱动的模仿学习：在多样条件下实现泛化 

**Authors**: Zhuochen Miao, Jun Lv, Hongjie Fang, Yang Jin, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21057)  

**Abstract**: Imitation learning has emerged as a powerful paradigm in robot manipulation, yet its generalization capability remains constrained by object-specific dependencies in limited expert demonstrations. To address this challenge, we propose knowledge-driven imitation learning, a framework that leverages external structural semantic knowledge to abstract object representations within the same category. We introduce a novel semantic keypoint graph as a knowledge template and develop a coarse-to-fine template-matching algorithm that optimizes both structural consistency and semantic similarity. Evaluated on three real-world robotic manipulation tasks, our method achieves superior performance, surpassing image-based diffusion policies with only one-quarter of the expert demonstrations. Extensive experiments further demonstrate its robustness across novel objects, backgrounds, and lighting conditions. This work pioneers a knowledge-driven approach to data-efficient robotic learning in real-world settings. Code and more materials are available on this https URL. 

**Abstract (ZH)**: 基于知识驱动的模仿学习：一种利用外部结构语义知识的框架 

---
# V2X-REALM: Vision-Language Model-Based Robust End-to-End Cooperative Autonomous Driving with Adaptive Long-Tail Modeling 

**Title (ZH)**: V2X-REALM：基于视觉-语言模型的鲁棒端到端协同自动驾驶及自适应长尾建模 

**Authors**: Junwei You, Pei Li, Zhuoyu Jiang, Zilin Huang, Rui Gan, Haotian Shi, Bin Ran  

**Link**: [PDF](https://arxiv.org/pdf/2506.21041)  

**Abstract**: Ensuring robust planning and decision-making under rare, diverse, and visually degraded long-tail scenarios remains a fundamental challenge for autonomous driving in urban environments. This issue becomes more critical in cooperative settings, where vehicles and infrastructure jointly perceive and reason across complex environments. To address this challenge, we propose V2X-REALM, a vision-language model (VLM)-based framework with adaptive multimodal learning for robust cooperative autonomous driving under long-tail scenarios. V2X-REALM introduces three core innovations: (i) a prompt-driven long-tail scenario generation and evaluation pipeline that leverages foundation models to synthesize realistic long-tail conditions such as snow and fog across vehicle- and infrastructure-side views, enriching training diversity efficiently; (ii) a gated multi-scenario adaptive attention module that modulates the visual stream using scenario priors to recalibrate ambiguous or corrupted features; and (iii) a multi-task scenario-aware contrastive learning objective that improves multimodal alignment and promotes cross-scenario feature separability. Extensive experiments demonstrate that V2X-REALM significantly outperforms existing baselines in robustness, semantic reasoning, safety, and planning accuracy under complex, challenging driving conditions, advancing the scalability of end-to-end cooperative autonomous driving. 

**Abstract (ZH)**: 确保在罕见、多样且视觉退化的长尾场景下实现稳健的规划与决策是城市环境中自动驾驶面临的基本挑战。在协同场景中，这一问题尤为重要，因为车辆和基础设施需要共同感知和推理复杂环境。为应对这一挑战，我们提出了一种基于视觉语言模型（VLM）且具备自适应多模态学习的框架V2X-REALM，以在长尾场景下实现稳健的协同自动驾驶。V2X-REALM 包含三项核心创新：（i）一种基于提示驱动的长尾场景生成和评估流水线，利用基础模型合成如雪和雾等真实的长尾条件，有效丰富训练多样性；（ii）一种门控多场景自适应注意力模块，利用场景先验调节视觉流，重新校准不确定或损坏的特征；（iii）一种多任务场景感知对比学习目标，提高多模态对齐并促进跨场景特征可分离性。大量实验结果表明，V2X-REALM 在复杂且具有挑战性的驾驶条件下，在稳健性、语义推理、安全性和规划准确性方面显著优于现有基线，推动了端到端协同自动驾驶的可扩展性。 

---
# STEP Planner: Constructing cross-hierarchical subgoal tree as an embodied long-horizon task planner 

**Title (ZH)**: STEP规划器：构建跨层次子目标树作为具身长时序任务规划器 

**Authors**: Zhou Tianxing, Wang Zhirui, Ao Haojia, Chen Guangyan, Xing Boyang, Cheng Jingwen, Yang Yi, Yue Yufeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.21030)  

**Abstract**: The ability to perform reliable long-horizon task planning is crucial for deploying robots in real-world environments. However, directly employing Large Language Models (LLMs) as action sequence generators often results in low success rates due to their limited reasoning ability for long-horizon embodied tasks. In the STEP framework, we construct a subgoal tree through a pair of closed-loop models: a subgoal decomposition model and a leaf node termination model. Within this framework, we develop a hierarchical tree structure that spans from coarse to fine resolutions. The subgoal decomposition model leverages a foundation LLM to break down complex goals into manageable subgoals, thereby spanning the subgoal tree. The leaf node termination model provides real-time feedback based on environmental states, determining when to terminate the tree spanning and ensuring each leaf node can be directly converted into a primitive action. Experiments conducted in both the VirtualHome WAH-NL benchmark and on real robots demonstrate that STEP achieves long-horizon embodied task completion with success rates up to 34% (WAH-NL) and 25% (real robot) outperforming SOTA methods. 

**Abstract (ZH)**: 具备可靠长时间任务规划能力对于将机器人部署到实际环境至关重要。在STEP框架中，我们通过一对闭环模型——子目标分解模型和叶节点终止模型，构建子目标树，并开发了一种从粗到细的分层树结构。子目标分解模型利用基础大模型将复杂目标分解为可管理的子目标，从而构建子目标树。叶节点终止模型根据环境状态提供实时反馈，确定停止扩展树的时间，并确保每个叶节点可以直接转换为基本动作。在VirtualHome WAH-NL基准和真实机器人上的实验表明，STEP在WAH-NL的完成成功率高达34%，在真实机器人上的完成成功率高达25%，超过当前最先进的方法。 

---
# Parallels Between VLA Model Post-Training and Human Motor Learning: Progress, Challenges, and Trends 

**Title (ZH)**: VLA模型后训练与人类运动学习之间的 parallel：进展、挑战与趋势 

**Authors**: Tian-Yu Xiang, Ao-Qun Jin, Xiao-Hu Zhou, Mei-Jiang Gui, Xiao-Liang Xie, Shi-Qi Liu, Shuang-Yi Wang, Sheng-Bin Duan, Fu-Chao Xie, Wen-Kai Wang, Si-Cheng Wang, Ling-Yun Li, Tian Tu, Zeng-Guang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2506.20966)  

**Abstract**: Vision-language-action (VLA) models extend vision-language models (VLM) by integrating action generation modules for robotic manipulation. Leveraging strengths of VLM in vision perception and instruction understanding, VLA models exhibit promising generalization across diverse manipulation tasks. However, applications demanding high precision and accuracy reveal performance gaps without further adaptation. Evidence from multiple domains highlights the critical role of post-training to align foundational models with downstream applications, spurring extensive research on post-training VLA models. VLA model post-training aims to address the challenge of improving an embodiment's ability to interact with the environment for the given tasks, analogous to the process of humans motor skills acquisition. Accordingly, this paper reviews post-training strategies for VLA models through the lens of human motor learning, focusing on three dimensions: environments, embodiments, and tasks. A structured taxonomy is introduced aligned with human learning mechanisms: (1) enhancing environmental perception, (2) improving embodiment awareness, (3) deepening task comprehension, and (4) multi-component integration. Finally, key challenges and trends in post-training VLA models are identified, establishing a conceptual framework to guide future research. This work delivers both a comprehensive overview of current VLA model post-training methods from a human motor learning perspective and practical insights for VLA model development. (Project website: this https URL) 

**Abstract (ZH)**: Vision-语言-行动（VLA）模型通过集成动作生成模块扩展了视觉-语言模型（VLM），以实现机器人的操作 manipulation。利用VLM在视觉感知和指令理解方面的优势，VLA模型在不同类型的操作任务中展现出良好的泛化能力。然而，对于要求高精度和高准确性的应用场景，表明需要进一步适应才能提高性能。跨多个领域的实验证据强调了后训练对调整基础模型以匹配下游应用的重要性，推动了对后训练VLA模型的广泛研究。VLA模型的后训练旨在通过改进实体与给定任务环境的互动能力来应对挑战，类似于人类运动技能的习得过程。本文从人类运动学习的角度回顾了VLA模型的后训练策略，重点关注三个维度：环境、实体和任务，并介绍了与人类学习机制相一致的结构化分类：1）增强环境感知，2）提高实体意识，3）深化任务理解，4）多组件集成。最后，本文指出了后训练VLA模型的关键挑战和趋势，为未来的研究提供了概念框架。这项工作不仅提供了从人类运动学习视角对当前VLA模型后训练方法的全面回顾，还为VLA模型开发提供了实用见解。（项目网站：this https URL） 

---
# Whole-Body Conditioned Egocentric Video Prediction 

**Title (ZH)**: 全身条件下的 propioceptive 视频预测 

**Authors**: Yutong Bai, Danny Tran, Amir Bar, Yann LeCun, Trevor Darrell, Jitendra Malik  

**Link**: [PDF](https://arxiv.org/pdf/2506.21552)  

**Abstract**: We train models to Predict Ego-centric Video from human Actions (PEVA), given the past video and an action represented by the relative 3D body pose. By conditioning on kinematic pose trajectories, structured by the joint hierarchy of the body, our model learns to simulate how physical human actions shape the environment from a first-person point of view. We train an auto-regressive conditional diffusion transformer on Nymeria, a large-scale dataset of real-world egocentric video and body pose capture. We further design a hierarchical evaluation protocol with increasingly challenging tasks, enabling a comprehensive analysis of the model's embodied prediction and control abilities. Our work represents an initial attempt to tackle the challenges of modeling complex real-world environments and embodied agent behaviors with video prediction from the perspective of a human. 

**Abstract (ZH)**: 基于人体动作预测以自我为中心视频（PEVA）：给定过去视频和由相对3D肢体姿态表示的动作 

---
# "Who Should I Believe?": User Interpretation and Decision-Making When a Family Healthcare Robot Contradicts Human Memory 

**Title (ZH)**: “我该相信谁？”：当家庭医疗服务机器人与人类记忆冲突时，用户如何解读和决策 

**Authors**: Hong Wang, Natalia Calvo-Barajas, Katie Winkle, Ginevra Castellano  

**Link**: [PDF](https://arxiv.org/pdf/2506.21322)  

**Abstract**: Advancements in robotic capabilities for providing physical assistance, psychological support, and daily health management are making the deployment of intelligent healthcare robots in home environments increasingly feasible in the near future. However, challenges arise when the information provided by these robots contradicts users' memory, raising concerns about user trust and decision-making. This paper presents a study that examines how varying a robot's level of transparency and sociability influences user interpretation, decision-making and perceived trust when faced with conflicting information from a robot. In a 2 x 2 between-subjects online study, 176 participants watched videos of a Furhat robot acting as a family healthcare assistant and suggesting a fictional user to take medication at a different time from that remembered by the user. Results indicate that robot transparency influenced users' interpretation of information discrepancies: with a low transparency robot, the most frequent assumption was that the user had not correctly remembered the time, while with the high transparency robot, participants were more likely to attribute the discrepancy to external factors, such as a partner or another household member modifying the robot's information. Additionally, participants exhibited a tendency toward overtrust, often prioritizing the robot's recommendations over the user's memory, even when suspecting system malfunctions or third-party interference. These findings highlight the impact of transparency mechanisms in robotic systems, the complexity and importance associated with system access control for multi-user robots deployed in home environments, and the potential risks of users' over reliance on robots in sensitive domains such as healthcare. 

**Abstract (ZH)**: 家用环境中智能医疗机器人的发展使其日益可行，但当机器人提供的信息与用户记忆相矛盾时，用户信任和决策制定方面的问题也随之出现。本文研究了机器人透明度和社交性水平变化对用户在面对机器人矛盾信息时解释、决策和感知信任的影响。在一项包含两个变量的双向在线研究中，共有176名参与者观看了Furhat机器人作为家庭健康助手，建议虚构用户在不同于用户记忆的时间服用药物的视频。研究结果表明，机器人的透明度影响了用户对信息差异的理解：在透明度较低的机器人中，用户最常假设自己没有正确记住时间；而在透明度较高的机器人中，参与者更倾向于将差异归因于外部因素，如伴侣或其他家庭成员修改了机器人的信息。此外，参与者倾向于过度信任机器人，经常优先考虑机器人的建议而非自己的记忆，即使怀疑系统故障或第三方干扰。这些发现强调了机器人系统中透明机制的影响，多用户家庭环境中部署的系统访问控制的复杂性和重要性，以及用户在医疗等敏感领域过度依赖机器人可能带来的风险。 

---
# World-aware Planning Narratives Enhance Large Vision-Language Model Planner 

**Title (ZH)**: 世界意识规划叙事增强大型视觉语言模型规划者 

**Authors**: Junhao Shi, Zhaoye Fei, Siyin Wang, Qipeng Guo, Jingjing Gong, Xipeng QIu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21230)  

**Abstract**: Large Vision-Language Models (LVLMs) show promise for embodied planning tasks but struggle with complex scenarios involving unfamiliar environments and multi-step goals. Current approaches rely on environment-agnostic imitation learning that disconnects instructions from environmental contexts, causing models to struggle with context-sensitive instructions and rely on supplementary cues rather than visual reasoning during long-horizon interactions. In this work, we propose World-Aware Planning Narrative Enhancement (WAP), a framework that infuses LVLMs with comprehensive environmental understanding through four cognitive capabilities (visual appearance modeling, spatial reasoning, functional abstraction, and syntactic grounding) while developing and evaluating models using only raw visual observations through curriculum learning. Evaluations on the EB-ALFRED benchmark demonstrate substantial improvements, with Qwen2.5-VL achieving a 60.7 absolute improvement in task success rates, particularly in commonsense reasoning (+60.0) and long-horizon planning (+70.0). Notably, our enhanced open-source models outperform proprietary systems like GPT-4o and Claude-3.5-Sonnet by a large margin. 

**Abstract (ZH)**: Large Vision-Language Models with World-Aware Planning Narrative Enhancement show Promising Improvements in Complex Embodied Planning Tasks 

---
# Our Coding Adventure: Using LLMs to Personalise the Narrative of a Tangible Programming Robot for Preschoolers 

**Title (ZH)**: 我们的编码冒险：使用大规模语言模型为学龄前儿童个性化物理编程机器人的叙事 

**Authors**: Martin Ruskov  

**Link**: [PDF](https://arxiv.org/pdf/2506.20982)  

**Abstract**: Finding balanced ways to employ Large Language Models (LLMs) in education is a challenge due to inherent risks of poor understanding of the technology and of a susceptible audience. This is particularly so with younger children, who are known to have difficulties with pervasive screen time. Working with a tangible programming robot called Cubetto, we propose an approach to benefit from the capabilities of LLMs by employing such models in the preparation of personalised storytelling, necessary for preschool children to get accustomed to the practice of commanding the robot. We engage in action research to develop an early version of a formalised process to rapidly prototype game stories for Cubetto. Our approach has both reproducible results, because it employs open weight models, and is model-agnostic, because we test it with 5 different LLMs. We document on one hand the process, the used materials and prompts, and on the other the learning experience and outcomes. We deem the generation successful for the intended purposes of using the results as a teacher aid. Testing the models on 4 different task scenarios, we encounter issues of consistency and hallucinations and document the corresponding evaluation process and attempts (some successful and some not) to overcome these issues. Importantly, the process does not expose children to LLMs directly. Rather, the technology is used to help teachers easily develop personalised narratives on children's preferred topics. We believe our method is adequate for preschool classes and we are planning to further experiment in real-world educational settings. 

**Abstract (ZH)**: 在教育中寻� Reeves 大型语言模型 (LLMs) 的平衡应用：鉴于技术理解不足和易受影响的受众固有风险，特别是在年轻儿童中，他们在广泛使用屏幕时间方面存在问题。通过使用名为 Cubetto 的实体编程机器人，我们提出了一种方法，即利用 LLMs 的能力来准备个性化的故事情节，这是幼儿熟悉控制机器人实践所必需的。我们开展行动研究以开发 Cubetto 游戏故事快速原型设计的早期正式化流程。我们的方法既具有可重复性，因为使用的是开放权重模型，又具有模型无关性，因为使用了 5 种不同 LLM 进行测试。我们一方面记录流程、使用的材料和提示，另一方面记录学习体验和成果。我们认为生成的内容适合作为教师辅助材料。在四个不同任务场景中测试模型时，我们遇到了一致性问题和幻觉问题，并记录了相应的评估过程和克服这些问题的尝试（部分成功，部分不成功）。重要的是，该过程不会直接将儿童暴露于 LLMs，而是利用技术帮助教师轻松创建符合儿童偏好主题的个性化叙事。我们认为我们的方法适合幼儿园班级，并计划在实际教育环境中进一步试验。 

---
# Effect of Haptic Feedback on Avoidance Behavior and Visual Exploration in Dynamic VR Pedestrian Environment 

**Title (ZH)**: 动态VR行人环境中触觉反馈对回避行为和视觉探索的影响 

**Authors**: Kyosuke Ishibashi, Atsushi Saito, Zin Y. Tun, Lucas Ray, Megan C. Coram, Akihiro Sakurai, Allison M. Okamura, Ko Yamamoto  

**Link**: [PDF](https://arxiv.org/pdf/2506.20952)  

**Abstract**: Human crowd simulation in virtual reality (VR) is a powerful tool with potential applications including emergency evacuation training and assessment of building layout. While haptic feedback in VR enhances immersive experience, its effect on walking behavior in dense and dynamic pedestrian flows is unknown. Through a user study, we investigated how haptic feedback changes user walking motion in crowded pedestrian flows in VR. The results indicate that haptic feedback changed users' collision avoidance movements, as measured by increased walking trajectory length and change in pelvis angle. The displacements of users' lateral position and pelvis angle were also increased in the instantaneous response to a collision with a non-player character (NPC), even when the NPC was inside the field of view. Haptic feedback also enhanced users' awareness and visual exploration when an NPC approached from the side and back. Furthermore, variation in walking speed was increased by the haptic feedback. These results suggested that the haptic feedback enhanced users' sensitivity to a collision in VR environment. 

**Abstract (ZH)**: 虚拟现实（VR）中的人群仿真：触觉反馈对密集动态行人流中行走行为的影响 

---
# Mind2Web 2: Evaluating Agentic Search with Agent-as-a-Judge 

**Title (ZH)**: Mind2Web 2: 以代理为评审者的能动性搜索评估 

**Authors**: Boyu Gou, Zanming Huang, Yuting Ning, Yu Gu, Michael Lin, Weijian Qi, Andrei Kopanev, Botao Yu, Bernal Jiménez Gutiérrez, Yiheng Shu, Chan Hee Song, Jiaman Wu, Shijie Chen, Hanane Nour Moussa, Tianshu Zhang, Jian Xie, Yifei Li, Tianci Xue, Zeyi Liao, Kai Zhang, Boyuan Zheng, Zhaowei Cai, Viktor Rozgic, Morteza Ziyadi, Huan Sun, Yu Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.21506)  

**Abstract**: Agentic search such as Deep Research systems, where large language models autonomously browse the web, synthesize information, and return comprehensive citation-backed answers, represents a major shift in how users interact with web-scale information. While promising greater efficiency and cognitive offloading, the growing complexity and open-endedness of agentic search have outpaced existing evaluation benchmarks and methodologies, which largely assume short search horizons and static answers. In this paper, we introduce Mind2Web 2, a benchmark of 130 realistic, high-quality, and long-horizon tasks that require real-time web browsing and extensive information synthesis, constructed with over 1,000 hours of human labor. To address the challenge of evaluating time-varying and complex answers, we propose a novel Agent-as-a-Judge framework. Our method constructs task-specific judge agents based on a tree-structured rubric design to automatically assess both answer correctness and source attribution. We conduct a comprehensive evaluation of nine frontier agentic search systems and human performance, along with a detailed error analysis to draw insights for future development. The best-performing system, OpenAI Deep Research, can already achieve 50-70% of human performance while spending half the time, showing a great potential. Altogether, Mind2Web 2 provides a rigorous foundation for developing and benchmarking the next generation of agentic search systems. 

**Abstract (ZH)**: 代理搜索：Mind2Web 2——一种包含 130 个实时网页浏览和大量信息综合的现实、高质量和长期任务基准 

---
# Active Inference AI Systems for Scientific Discovery 

**Title (ZH)**: 基于活跃推断的AI系统及其在科学研究中的应用 

**Authors**: Karthik Duraisamy  

**Link**: [PDF](https://arxiv.org/pdf/2506.21329)  

**Abstract**: The rapid evolution of artificial intelligence has led to expectations of transformative scientific discovery, yet current systems remain fundamentally limited by their operational architectures, brittle reasoning mechanisms, and their separation from experimental reality. Building on earlier work, we contend that progress in AI-driven science now depends on closing three fundamental gaps -- the abstraction gap, the reasoning gap, and the reality gap -- rather than on model size/data/test time compute. Scientific reasoning demands internal representations that support simulation of actions and response, causal structures that distinguish correlation from mechanism, and continuous calibration. We define active inference AI systems for scientific discovery as those that (i) maintain long-lived research memories grounded in causal self-supervised foundation models, (ii) symbolic or neuro-symbolic planners equipped with Bayesian guardrails, (iii) grow persistent knowledge graphs where thinking generates novel conceptual nodes, reasoning establishes causal edges, and real-world interaction prunes false connections while strengthening verified pathways, and (iv) refine their internal representations through closed-loop interaction with both high-fidelity simulators and automated laboratories - an operational loop where mental simulation guides action and empirical surprise reshapes understanding. In essence, we outline an architecture where discovery arises from the interplay between internal models that enable counterfactual reasoning and external validation that grounds hypotheses in reality. It is also argued that the inherent ambiguity in feedback from simulations and experiments, and underlying uncertainties makes human judgment indispensable, not as a temporary scaffold but as a permanent architectural component. 

**Abstract (ZH)**: 人工智能的快速演化带来了变革性科学发现的期望，但当前系统依然在运行架构、脆弱的推理机制以及与实验现实的分离方面存在根本局限。基于前期工作，我们主张，AI驱动的科学研究的进步现在依赖于弥合三类根本性的差距——抽象差距、推理差距和现实差距，而不是依赖于模型规模、数据和测试时间计算能力。科学推理要求能够支持行动和响应模拟的内部表示，能够区分相关性与机制的因果结构，以及持续的校准。我们定义了进行科学发现的主动推断AI系统，这些系统需具备：(i) 基于因果自监督基础模型的长期研究记忆；(ii) 配备贝叶斯护栏的符号或神经符号规划者；(iii) 知识图谱的增长，其中思考生成新颖的概念节点，推理建立因果边，而现实世界交互去除错误连接并加强验证路径；(iv) 通过与高保真模拟器和自动化实验室的闭环交互来精炼其内部表示——一种操作回路，其中心理模拟指导行动而实证惊讶重塑理解。本质上，我们勾勒出一种架构，其中发现源自内部模型所支持的反事实推理与外部验证的相互作用，将实验现实作为假设的基础。我们也认为，从模拟和实验获得的反馈固有的模糊性以及内在的不确定性使人为判断不可或缺，而不仅仅是作为暂时的架构支撑，而是作为永久性的架构组成部分。 

---
# Process mining-driven modeling and simulation to enhance fault diagnosis in cyber-physical systems 

**Title (ZH)**: 基于过程挖掘的建模与仿真以增强网络物理系统故障诊断 

**Authors**: Francesco Vitale, Nicola Dall'Ora, Sebastiano Gaiardelli, Enrico Fraccaroli, Nicola Mazzocca, Franco Fummi  

**Link**: [PDF](https://arxiv.org/pdf/2506.21502)  

**Abstract**: Fault diagnosis in Cyber-Physical Systems (CPSs) is essential for ensuring system dependability and operational efficiency by accurately detecting anomalies and identifying their root causes. However, the manual modeling of faulty behaviors often demands extensive domain expertise and produces models that are complex, error-prone, and difficult to interpret. To address this challenge, we present a novel unsupervised fault diagnosis methodology that integrates collective anomaly detection in multivariate time series, process mining, and stochastic simulation. Initially, collective anomalies are detected from low-level sensor data using multivariate time-series analysis. These anomalies are then transformed into structured event logs, enabling the discovery of interpretable process models through process mining. By incorporating timing distributions into the extracted Petri nets, the approach supports stochastic simulation of faulty behaviors, thereby enhancing root cause analysis and behavioral understanding. The methodology is validated using the Robotic Arm Dataset (RoAD), a widely recognized benchmark in smart manufacturing. Experimental results demonstrate its effectiveness in modeling, simulating, and classifying faulty behaviors in CPSs. This enables the creation of comprehensive fault dictionaries that support predictive maintenance and the development of digital twins for industrial environments. 

**Abstract (ZH)**: Cyber-物理系统（CPSs）中的故障诊断对于确保系统的可靠性和操作效率至关重要，通过精确检测异常并识别其根本原因。然而，手动建模故障行为往往需要大量的领域专业知识，并会产生复杂、易出错且难以解释的模型。为了解决这一挑战，我们提出了一种新颖的无监督故障诊断方法，该方法结合了多变量时间序列集体异常检测、过程挖掘和随机仿真。首先，使用多变量时间序列分析从低级传感器数据中检测集体异常。随后，将这些异常转换为结构化的事件日志，通过过程挖掘发现可解释的过程模型。通过将时间分布融入提取的Petri网中，该方法支持故障行为的随机仿真，从而增强根本原因分析和行为理解。该方法使用广泛认可的智能制造基准数据集Robotic Arm Dataset (RoAD) 进行验证。实验结果表明，该方法在模型构建、仿真和故障行为分类方面具有有效性，从而支持预测性维护并为工业环境开发数字孪生体。 

---
# Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents 

**Title (ZH)**: Agent-RewardBench：朝向跨感知、规划和安全的现实世界多模态智能体 reward 模型统一基准 

**Authors**: Tianyi Men, Zhuoran Jin, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21252)  

**Abstract**: As Multimodal Large Language Models (MLLMs) advance, multimodal agents show promise in real-world tasks like web navigation and embodied intelligence. However, due to limitations in a lack of external feedback, these agents struggle with self-correction and generalization. A promising approach is to use reward models as external feedback, but there is no clear on how to select reward models for agents. Thus, there is an urgent need to build a reward bench targeted at agents. To address these challenges, we propose Agent-RewardBench, a benchmark designed to evaluate reward modeling ability in MLLMs. The benchmark is characterized by three key features: (1) Multiple dimensions and real-world agent scenarios evaluation. It covers perception, planning, and safety with 7 scenarios; (2) Step-level reward evaluation. It allows for the assessment of agent capabilities at the individual steps of a task, providing a more granular view of performance during the planning process; and (3) Appropriately difficulty and high-quality. We carefully sample from 10 diverse models, difficulty control to maintain task challenges, and manual verification to ensure the integrity of the data. Experiments demonstrate that even state-of-the-art multimodal models show limited performance, highlighting the need for specialized training in agent reward modeling. Code is available at github. 

**Abstract (ZH)**: 随着多模态大型语言模型（MLLMs）的发展，多模态代理在网页导航和体态智能等真实世界任务中展现出潜力。然而，由于缺乏外部反馈，这些代理在自我纠正和泛化方面存在问题。一种有希望的方法是使用奖励模型作为外部反馈，但尚不清楚如何选择适合代理的奖励模型。因此，迫切需要建立一个针对代理的奖励基准。为应对这些挑战，我们提出了Agent-RewardBench，一种旨在评估MLLMs奖励建模能力的基准。该基准具有三个关键特征：（1）多维度和真实世界代理场景评估，涵盖感知、规划和安全性，包括7种场景；（2）步骤级奖励评估，允许在任务的每个步骤上评估代理能力，提供更详细的规划过程中的性能视图；（3）适配难度和高质量。我们精心从10个不同的模型中采样，控制难度以保持任务挑战性，并通过人工验证确保数据完整性。实验表明，即使是最先进的多模态模型在代理奖励建模方面也表现出有限的表现，突显出在代理奖励建模方面的专业化训练需求。代码可在github上获取。 

---
# Curriculum-Guided Antifragile Reinforcement Learning for Secure UAV Deconfliction under Observation-Space Attacks 

**Title (ZH)**: 基于课程引导的抗毁强化学习及观测空间攻击下的无人机冲突缓解 

**Authors**: Deepak Kumar Panda, Adolfo Perrusquia, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.21129)  

**Abstract**: Reinforcement learning (RL) policies deployed in safety-critical systems, such as unmanned aerial vehicle (UAV) navigation in dynamic airspace, are vulnerable to out-ofdistribution (OOD) adversarial attacks in the observation space. These attacks induce distributional shifts that significantly degrade value estimation, leading to unsafe or suboptimal decision making rendering the existing policy fragile. To address this vulnerability, we propose an antifragile RL framework designed to adapt against curriculum of incremental adversarial perturbations. The framework introduces a simulated attacker which incrementally increases the strength of observation-space perturbations which enables the RL agent to adapt and generalize across a wider range of OOD observations and anticipate previously unseen attacks. We begin with a theoretical characterization of fragility, formally defining catastrophic forgetting as a monotonic divergence in value function distributions with increasing perturbation strength. Building on this, we define antifragility as the boundedness of such value shifts and derive adaptation conditions under which forgetting is stabilized. Our method enforces these bounds through iterative expert-guided critic alignment using Wasserstein distance minimization across incrementally perturbed observations. We empirically evaluate the approach in a UAV deconfliction scenario involving dynamic 3D obstacles. Results show that the antifragile policy consistently outperforms standard and robust RL baselines when subjected to both projected gradient descent (PGD) and GPS spoofing attacks, achieving up to 15% higher cumulative reward and over 30% fewer conflict events. These findings demonstrate the practical and theoretical viability of antifragile reinforcement learning for secure and resilient decision-making in environments with evolving threat scenarios. 

**Abstract (ZH)**: 在动态空域中部署于无人驾驶航空车辆(UAV)导航的安全关键系统中的强化学习(Reinforcement Learning, RL)策略容易受到观测空间中的离分布攻击(out-of-distribution adversarial attacks)的威胁。这些攻击引起分布偏移，显著恶化价值估计，导致不安全或次优的决策，使现有策略变得脆弱。为此，我们提出了一种抗脆弱的RL框架，该框架旨在对抗逐步增加的 adversarial 干扰。该框架引入了一个模拟攻击者，逐步增强观测空间中的干扰强度，使RL智能体能够适应和泛化到更广泛的离分布观测，并预见到先前未见过的攻击。我们从理论上对脆弱性进行了刻画，正式定义了灾难性遗忘作为价值函数分布随干扰强度增加的单调发散。在此基础上，我们定义抗脆弱性为这些价值变化的有界性，并导出了使遗忘被稳定化的适应条件。该方法通过 Wasserstein 距离最小化逐步干扰观测中的专家指导批评者对齐来实现这些边界。我们通过一个涉及动态3D障碍的无人驾驶航空车辆(UAV)冲突缓解场景，对该方法进行了实证评估。结果表明，当受到项目梯度下降(PGD)和GPS欺骗攻击时，抗脆弱策略始终优于标准RL基线和鲁棒RL基线，累计奖励提高了高达15%，冲突事件减少了超过30%。这些发现证明了在不断演变的威胁场景中，抗脆弱强化学习在安全和鲁棒决策制定方面的实用性和理论可行性。 

---
# Robust Policy Switching for Antifragile Reinforcement Learning for UAV Deconfliction in Adversarial Environments 

**Title (ZH)**: 鲁棒的策略切换以实现抗脆弱的无人机避障学习在对抗环境中的应用 

**Authors**: Deepak Kumar Panda, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.21127)  

**Abstract**: The increasing automation of navigation for unmanned aerial vehicles (UAVs) has exposed them to adversarial attacks that exploit vulnerabilities in reinforcement learning (RL) through sensor manipulation. Although existing robust RL methods aim to mitigate such threats, their effectiveness has limited generalization to out-of-distribution shifts from the optimal value distribution, as they are primarily designed to handle fixed perturbation. To address this limitation, this paper introduces an antifragile RL framework that enhances adaptability to broader distributional shifts by incorporating a switching mechanism based on discounted Thompson sampling (DTS). This mechanism dynamically selects among multiple robust policies to minimize adversarially induced state-action-value distribution shifts. The proposed approach first derives a diverse ensemble of action robust policies by accounting for a range of perturbations in the policy space. These policies are then modeled as a multiarmed bandit (MAB) problem, where DTS optimally selects policies in response to nonstationary Bernoulli rewards, effectively adapting to evolving adversarial strategies. Theoretical framework has also been provided where by optimizing the DTS to minimize the overall regrets due to distributional shift, results in effective adaptation against unseen adversarial attacks thus inducing antifragility. Extensive numerical simulations validate the effectiveness of the proposed framework in complex navigation environments with multiple dynamic three-dimensional obstacles and with stronger projected gradient descent (PGD) and spoofing attacks. Compared to conventional robust, non-adaptive RL methods, the antifragile approach achieves superior performance, demonstrating shorter navigation path lengths and a higher rate of conflict-free navigation trajectories compared to existing robust RL techniques 

**Abstract (ZH)**: 无人飞行器导航自动化日益增加暴露了它们在感应器操控下利用强化学习 Vulnerabilities 的对抗性攻击风险。尽管现有的鲁棒强化学习方法旨在减轻这些威胁，但它们在应对最优价值分布之外的域外变化时效果有限，因为它们主要设计用于处理固定的扰动。为解决这一局限性，本文提出了一种抗脆强化学习框架，通过引入基于折扣曲杆采样的切换机制来增强对更广泛分布变化的适应性。该机制动态选择多个鲁棒策略，以最小化对抗性诱导的状态-动作-价值分布变化。该方法首先通过考虑策略空间中多种扰动的范围，推导出一个多样化的行动鲁棒策略集。这些策略被建模为一个多臂 bandit (MAB) 问题，其中折扣曲杆采样 (DTS) 优化地在响应非平稳伯努利奖励时选择策略，从而有效地适应不断变化的对抗性策略。通过优化 DTS 以最小化因分布变化引起的总体遗憾，从而实现对未见过的对抗性攻击的有效适应，增强系统的抗脆性。广泛的数值仿真验证了该框架在具有多个动态三维障碍物的复杂导航环境中对更强大投影梯度下降 (PGD) 和欺骗性攻击的有效性。与传统的非适应性鲁棒强化学习方法相比，抗脆方法实现了更优的性能，导航路径长度更短，并且冲突-free 导航轨迹的比例更高，优于现有的鲁棒强化学习技术。 

---
# EgoAdapt: Adaptive Multisensory Distillation and Policy Learning for Efficient Egocentric Perception 

**Title (ZH)**: EgoAdapt: 自适应多模态知识蒸馏与策略学习以实现高效的第一人称感知 

**Authors**: Sanjoy Chowdhury, Subrata Biswas, Sayan Nag, Tushar Nagarajan, Calvin Murdock, Ishwarya Ananthabhotla, Yijun Qian, Vamsi Krishna Ithapu, Dinesh Manocha, Ruohan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21080)  

**Abstract**: Modern perception models, particularly those designed for multisensory egocentric tasks, have achieved remarkable performance but often come with substantial computational costs. These high demands pose challenges for real-world deployment, especially in resource-constrained environments. In this paper, we introduce EgoAdapt, a framework that adaptively performs cross-modal distillation and policy learning to enable efficient inference across different egocentric perception tasks, including egocentric action recognition, active speaker localization, and behavior anticipation. Our proposed policy module is adaptable to task-specific action spaces, making it broadly applicable. Experimental results on three challenging egocentric datasets EPIC-Kitchens, EasyCom, and Aria Everyday Activities demonstrate that our method significantly enhances efficiency, reducing GMACs by up to 89.09%, parameters up to 82.02%, and energy up to 9.6x, while still on-par and in many cases outperforming, the performance of corresponding state-of-the-art models. 

**Abstract (ZH)**: 现代感知模型在多模态第一人称任务中取得了显著性能，但往往伴随着高昂的计算成本。这些高需求对实际部署构成了挑战，尤其是在资源受限的环境中。本文介绍了一种名为EgoAdapt的框架，该框架适应性地进行跨模态蒸馏和策略学习，以在不同的第一人称感知任务中实现高效的推理，包括第一人称动作识别、主动说话人定位和行为预测。我们提出的策略模块适用于特定任务的动作空间，使其具有广泛的适用性。在三个具有挑战性的一人称视角数据集中（EPIC-Kitchens、EasyCom和Aria Everyday Activities）的实验结果表明，我们的方法显著提高了效率，分别降低了89.09%的GMACs、82.02%的参数量和9.6倍的能耗，同时在许多情况下超过了对应的最佳模型的性能。 

---
# Efficient Skill Discovery via Regret-Aware Optimization 

**Title (ZH)**: 基于后悔意识优化的高效技能发现 

**Authors**: He Zhang, Ming Zhou, Shaopeng Zhai, Ying Sun, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.21044)  

**Abstract**: Unsupervised skill discovery aims to learn diverse and distinguishable behaviors in open-ended reinforcement learning. For existing methods, they focus on improving diversity through pure exploration, mutual information optimization, and learning temporal representation. Despite that they perform well on exploration, they remain limited in terms of efficiency, especially for the high-dimensional situations. In this work, we frame skill discovery as a min-max game of skill generation and policy learning, proposing a regret-aware method on top of temporal representation learning that expands the discovered skill space along the direction of upgradable policy strength. The key insight behind the proposed method is that the skill discovery is adversarial to the policy learning, i.e., skills with weak strength should be further explored while less exploration for the skills with converged strength. As an implementation, we score the degree of strength convergence with regret, and guide the skill discovery with a learnable skill generator. To avoid degeneration, skill generation comes from an up-gradable population of skill generators. We conduct experiments on environments with varying complexities and dimension sizes. Empirical results show that our method outperforms baselines in both efficiency and diversity. Moreover, our method achieves a 15% zero shot improvement in high-dimensional environments, compared to existing methods. 

**Abstract (ZH)**: 无监督技能发现旨在在开放强化学习中学习多样且可区分的行为。现有方法侧重于通过纯粹探索、互信息优化以及学习时间表示来提升多样性。尽管这些方法在探索方面表现良好，但在效率方面仍然受限，尤其是在高维情况下。在本文中，我们将技能发现框定为技能生成和策略学习的最小最大博弈，并在此基础上提出了一种基于时间表示学习的方法，该方法沿着可升级策略强度的方向扩展发现的技能空间。提出方法的核心洞察是，技能发现与策略学习是对抗性的：即弱技能应进一步探索，而收敛技能则较少探索。作为实现，我们用遗憾分数衡量强度收敛的程度，并使用可学习的技能生成器引导技能发现。为了避免退化，技能生成来自于可升级的技能生成器群体。我们在不同复杂性和维度大小的环境中进行了实验。实验证明，我们的方法在效率和多样性方面均优于基线方法。此外，在高维环境中，我们的方法相比现有方法实现了15%的零样本改进。 

---
# Evidence-based diagnostic reasoning with multi-agent copilot for human pathology 

**Title (ZH)**: 基于证据的多代理伴飞诊断推理在人类病理学中的应用 

**Authors**: Chengkuan Chen, Luca L. Weishaupt, Drew F. K. Williamson, Richard J. Chen, Tong Ding, Bowen Chen, Anurag Vaidya, Long Phi Le, Guillaume Jaume, Ming Y. Lu, Faisal Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2506.20964)  

**Abstract**: Pathology is experiencing rapid digital transformation driven by whole-slide imaging and artificial intelligence (AI). While deep learning-based computational pathology has achieved notable success, traditional models primarily focus on image analysis without integrating natural language instruction or rich, text-based context. Current multimodal large language models (MLLMs) in computational pathology face limitations, including insufficient training data, inadequate support and evaluation for multi-image understanding, and a lack of autonomous, diagnostic reasoning capabilities. To address these limitations, we introduce PathChat+, a new MLLM specifically designed for human pathology, trained on over 1 million diverse, pathology-specific instruction samples and nearly 5.5 million question answer turns. Extensive evaluations across diverse pathology benchmarks demonstrated that PathChat+ substantially outperforms the prior PathChat copilot, as well as both state-of-the-art (SOTA) general-purpose and other pathology-specific models. Furthermore, we present SlideSeek, a reasoning-enabled multi-agent AI system leveraging PathChat+ to autonomously evaluate gigapixel whole-slide images (WSIs) through iterative, hierarchical diagnostic reasoning, reaching high accuracy on DDxBench, a challenging open-ended differential diagnosis benchmark, while also capable of generating visually grounded, humanly-interpretable summary reports. 

**Abstract (ZH)**: 病理学正经历由全视野显微成像和人工智能（AI）驱动的快速数字化转型。虽然基于深度学习的计算病理学取得了显著成就，但传统模型主要侧重于图像分析，未整合自然语言指令或丰富文本背景信息。当前计算病理学中的多模态大型语言模型（MLLMs）面临局限性，包括训练数据不足、对多图像理解的支持和评估不足，以及缺乏自主诊断推理能力。为解决这些局限性，我们提出PathChat+这一专门设计用于人类病理学的新MMLM，其基于超过100万份多元且病理特异性的指令样本及近550万轮问题-答案交互训练。在多种病理基准测试中的广泛评估表明，PathChat+显著优于先前的PathChat副驾模型，同时也优于最先进的通用和特定于病理学的其他模型。此外，我们介绍SlideSeek，一种利用PathChat+实现自主诊断推理的多智能体AI系统，能够通过迭代分层诊断推理自主评估 gigapixel 全视野显微图像（WSIs），在具有挑战性的开放性鉴别诊断基准 DDxBench 上达到高准确率，同时还能生成基于视觉、可由人类解释的总结报告。 

---
