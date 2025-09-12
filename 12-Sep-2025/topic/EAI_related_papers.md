# SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning 

**Title (ZH)**: SimpleVLA-RL：通过强化学习拓展VLA训练 

**Authors**: Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, Dehui Wang, Dingxiang Luo, Yuchen Fan, Youbang Sun, Jia Zeng, Jiangmiao Pang, Shanghang Zhang, Yu Wang, Yao Mu, Bowen Zhou, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.09674)  

**Abstract**: Vision-Language-Action (VLA) models have recently emerged as a powerful paradigm for robotic manipulation. Despite substantial progress enabled by large-scale pretraining and supervised fine-tuning (SFT), these models face two fundamental challenges: (i) the scarcity and high cost of large-scale human-operated robotic trajectories required for SFT scaling, and (ii) limited generalization to tasks involving distribution shift. Recent breakthroughs in Large Reasoning Models (LRMs) demonstrate that reinforcement learning (RL) can dramatically enhance step-by-step reasoning capabilities, raising a natural question: Can RL similarly improve the long-horizon step-by-step action planning of VLA? In this work, we introduce SimpleVLA-RL, an efficient RL framework tailored for VLA models. Building upon veRL, we introduce VLA-specific trajectory sampling, scalable parallelization, multi-environment rendering, and optimized loss computation. When applied to OpenVLA-OFT, SimpleVLA-RL achieves SoTA performance on LIBERO and even outperforms $\pi_0$ on RoboTwin 1.0\&2.0 with the exploration-enhancing strategies we introduce. SimpleVLA-RL not only reduces dependence on large-scale data and enables robust generalization, but also remarkably surpasses SFT in real-world tasks. Moreover, we identify a novel phenomenon ``pushcut'' during RL training, wherein the policy discovers previously unseen patterns beyond those seen in the previous training process. Github: this https URL 

**Abstract (ZH)**: 基于视觉-语言-动作的强化学习框架（SimpleVLA-RL）：提升机器人长期步骤规划能力 

---
# Dexplore: Scalable Neural Control for Dexterous Manipulation from Reference-Scoped Exploration 

**Title (ZH)**: Dexplore: 面向参考范围探索的可扩展神经控制方法用于灵巧操作 

**Authors**: Sirui Xu, Yu-Wei Chao, Liuyu Bian, Arsalan Mousavian, Yu-Xiong Wang, Liang-Yan Gui, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09671)  

**Abstract**: Hand-object motion-capture (MoCap) repositories offer large-scale, contact-rich demonstrations and hold promise for scaling dexterous robotic manipulation. Yet demonstration inaccuracies and embodiment gaps between human and robot hands limit the straightforward use of these data. Existing methods adopt a three-stage workflow, including retargeting, tracking, and residual correction, which often leaves demonstrations underused and compound errors across stages. We introduce Dexplore, a unified single-loop optimization that jointly performs retargeting and tracking to learn robot control policies directly from MoCap at scale. Rather than treating demonstrations as ground truth, we use them as soft guidance. From raw trajectories, we derive adaptive spatial scopes, and train with reinforcement learning to keep the policy in-scope while minimizing control effort and accomplishing the task. This unified formulation preserves demonstration intent, enables robot-specific strategies to emerge, improves robustness to noise, and scales to large demonstration corpora. We distill the scaled tracking policy into a vision-based, skill-conditioned generative controller that encodes diverse manipulation skills in a rich latent representation, supporting generalization across objects and real-world deployment. Taken together, these contributions position Dexplore as a principled bridge that transforms imperfect demonstrations into effective training signals for dexterous manipulation. 

**Abstract (ZH)**: Dexplore: A Unified Optimization for Scaling Hand-Object Motion-Capture Demonstrations to Dexterous Robotic Manipulation 

---
# MOFU: Development of a MOrphing Fluffy Unit with Expansion and Contraction Capabilities and Evaluation of the Animacy of Its Movements 

**Title (ZH)**: MOFU：具有扩展与收缩能力的变形毛绒单元开发及其运动拟人性评价 

**Authors**: Taisei Mogi, Mari Saito, Yoshihiro Nakata  

**Link**: [PDF](https://arxiv.org/pdf/2509.09613)  

**Abstract**: Robots for therapy and social interaction are often intended to evoke "animacy" in humans. While many robots imitate appearance and joint movements, little attention has been given to whole-body expansion-contraction, volume-changing movements observed in living organisms, and their effect on animacy perception. We developed a mobile robot called "MOFU (Morphing Fluffy Unit)," capable of whole-body expansion-contraction with a single motor and covered with a fluffy exterior. MOFU employs a "Jitterbug" structure, a geometric transformation mechanism that enables smooth volume change in diameter from 210 to 280 mm using one actuator. It is also equipped with a differential two-wheel drive mechanism for locomotion. To evaluate the effect of expansion-contraction movements, we conducted an online survey using videos of MOFU's behavior. Participants rated impressions with the Godspeed Questionnaire Series. First, we compared videos of MOFU in a stationary state with and without expansion-contraction and turning, finding that expansion-contraction significantly increased perceived animacy. Second, we hypothesized that presenting two MOFUs would increase animacy compared with a single robot; however, this was not supported, as no significant difference emerged. Exploratory analyses further compared four dual-robot motion conditions. Third, when expansion-contraction was combined with locomotion, animacy ratings were higher than locomotion alone. These results suggest that volume-changing movements such as expansion and contraction enhance perceived animacy in robots and should be considered an important design element in future robot development aimed at shaping human impressions. 

**Abstract (ZH)**: 用于治疗和社会互动的机器人常常旨在唤起人类的“生命力”。尽管许多机器人模仿外观和关节运动，但很少有关注全身扩张收缩、体积变化的运动及其对生命力感知的影响。我们开发了一种名为“MOFU（形态绒毛单元）”的移动机器人，能够通过单一马达实现全身扩张收缩，并覆盖有绒毛外层。MOFU采用“Jitterbug”结构，这是一种几何变换机制，可在直径从210毫米至280毫米之间平滑变化，仅需一个执行器。它还配备了差动双轮驱动机制以实现移动。为了评估扩张收缩运动的效果，我们使用MOFU行为的视频进行了在线调查，并使用Godspeed问卷系列对参与者进行了评价。首先，我们将MOFU在静止状态和有无扩张收缩及转向的视频进行比较，发现扩张收缩显著提高了感知的生命力。其次，我们假设展示两个MOFU会比单个机器人增加生命力，但这一假设未得到支持，因为没有显著差异。进一步的探索性分析比较了四种双机器人运动条件。第三，当扩张收缩与移动结合时，生命力评价高于仅移动的情况。这些结果表明，体积变化运动如扩张和收缩能够增强对机器人生命力的感知，应被视为未来旨在塑造人类印象的机器人设计中一个重要设计元素。 

---
# VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model 

**Title (ZH)**: VLA-适配器：一种有效的细粒度多模态模型范式 

**Authors**: Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui, Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao, Pengxu Hou, Siteng Huang, Yifan Tang, Wenhui Wang, Ru Zhang, Jianyi Liu, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09372)  

**Abstract**: Vision-Language-Action (VLA) models typically bridge the gap between perceptual and action spaces by pre-training a large-scale Vision-Language Model (VLM) on robotic data. While this approach greatly enhances performance, it also incurs significant training costs. In this paper, we investigate how to effectively bridge vision-language (VL) representations to action (A). We introduce VLA-Adapter, a novel paradigm designed to reduce the reliance of VLA models on large-scale VLMs and extensive pre-training. To this end, we first systematically analyze the effectiveness of various VL conditions and present key findings on which conditions are essential for bridging perception and action spaces. Based on these insights, we propose a lightweight Policy module with Bridge Attention, which autonomously injects the optimal condition into the action space. In this way, our method achieves high performance using only a 0.5B-parameter backbone, without any robotic data pre-training. Extensive experiments on both simulated and real-world robotic benchmarks demonstrate that VLA-Adapter not only achieves state-of-the-art level performance, but also offers the fast inference speed reported to date. Furthermore, thanks to the proposed advanced bridging paradigm, VLA-Adapter enables the training of a powerful VLA model in just 8 hours on a single consumer-grade GPU, greatly lowering the barrier to deploying the VLA model. Project page: this https URL. 

**Abstract (ZH)**: Vision-Language-Action (VLA) 模型通常通过在机器人数据上预训练大规模视觉-语言模型（VLM）来弥合感知和动作空间的差距。虽然这种方法大大提升了性能，但也带来了显著的训练成本。本文研究了如何有效将视觉-语言（VL）表示桥接到动作（A）。我们引入了VLA-Adapter，这是一种新型的范式，旨在减少VLA模型对大规模VLM和长时间预训练的依赖。为此，我们首先系统分析了各种VL条件的有效性，并提出了对于弥合感知和动作空间的关键发现。基于这些见解，我们提出了一个轻量级的Policy模块，其中包含Bridge Attention，该模块可自主注入最合适的条件到动作空间中。通过这种方式，我们的方法仅使用一个0.5B参数的骨干模型即可实现高性能，无需任何机器人数据预训练。广泛实验证明，VLA-Adapter不仅达到了最先进的性能，而且还提供了迄今为止报告的最快推理速度。此外，由于提出的先进桥接范式，VLA-Adapter使得仅在一个消费级GPU上只需8小时即可训练出强大的VLA模型，大大降低了部署VLA模型的门槛。 

---
# AGILOped: Agile Open-Source Humanoid Robot for Research 

**Title (ZH)**: AGILOped：敏捷开源人形机器人用于研究 

**Authors**: Grzegorz Ficht, Luis Denninger, Sven Behnke  

**Link**: [PDF](https://arxiv.org/pdf/2509.09364)  

**Abstract**: With academic and commercial interest for humanoid robots peaking, multiple platforms are being developed. Through a high level of customization, they showcase impressive performance. Most of these systems remain closed-source or have high acquisition and maintenance costs, however. In this work, we present AGILOped - an open-source humanoid robot that closes the gap between high performance and accessibility. Our robot is driven by off-the-shelf backdrivable actuators with high power density and uses standard electronic components. With a height of 110 cm and weighing only 14.5 kg, AGILOped can be operated without a gantry by a single person. Experiments in walking, jumping, impact mitigation and getting-up demonstrate its viability for use in research. 

**Abstract (ZH)**: 基于通用智能的开源人形机器人AGILOped：高性能与易用性的桥梁 

---
# OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning 

**Title (ZH)**: OmniEVA：基于任务自适应三维接地和体态意识推理的通用 embodied 计划器 

**Authors**: Yuecheng Liu, Dafeng Chi, Shiguang Wu, Zhanguang Zhang, Yuzheng Zhuang, Bowen Yang, He Zhu, Lingfeng Zhang, Pengwei Xie, David Gamaliel Arcos Bravo, Yingxue Zhang, Jianye Hao, Xingyue Quan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09332)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have opened new opportunities for embodied intelligence, enabling multimodal understanding, reasoning, and interaction, as well as continuous spatial decision-making. Nevertheless, current MLLM-based embodied systems face two critical limitations. First, Geometric Adaptability Gap: models trained solely on 2D inputs or with hard-coded 3D geometry injection suffer from either insufficient spatial information or restricted 2D generalization, leading to poor adaptability across tasks with diverse spatial demands. Second, Embodiment Constraint Gap: prior work often neglects the physical constraints and capacities of real robots, resulting in task plans that are theoretically valid but practically this http URL address these gaps, we introduce OmniEVA -- an embodied versatile planner that enables advanced embodied reasoning and task planning through two pivotal innovations: (1) a Task-Adaptive 3D Grounding mechanism, which introduces a gated router to perform explicit selective regulation of 3D fusion based on contextual requirements, enabling context-aware 3D grounding for diverse embodied tasks. (2) an Embodiment-Aware Reasoning framework that jointly incorporates task goals and embodiment constraints into the reasoning loop, resulting in planning decisions that are both goal-directed and executable. Extensive experimental results demonstrate that OmniEVA not only achieves state-of-the-art general embodied reasoning performance, but also exhibits a strong ability across a wide range of downstream scenarios. Evaluations of a suite of proposed embodied benchmarks, including both primitive and composite tasks, confirm its robust and versatile planning capabilities. Project page: this https URL 

**Abstract (ZH)**: 最近多模态大型语言模型的进展为具身智能开辟了新机会，使其能够实现多模态理解、推理和交互，以及持续的空间决策。然而，当前基于多模态大型语言模型的具身系统面临两大关键限制：几何适应性缺口和具身约束缺口。为了解决这些问题，我们介绍了OmniEVA——一种具身通用规划器，通过两项关键创新实现高级具身推理和任务规划：（1）任务自适应3D关联机制，该机制引入门控路由器，根据上下文需求显式选择性调节3D融合，实现多种具身任务的上下文感知3D关联。（2）具身感知推理框架，该框架将任务目标和具身约束同时纳入推理循环，从而产生既目标导向又可执行的规划决策。广泛的经验结果表明，OmniEVA不仅实现了最先进的综合具身推理性能，还广泛展现出强大的规划能力。一系列提出的具身基准评估，包括基本和复合任务，证实了其稳健且多功能的规划能力。项目页面：[此链接地址]。 

---
# RENet: Fault-Tolerant Motion Control for Quadruped Robots via Redundant Estimator Networks under Visual Collapse 

**Title (ZH)**: RENet: 视觉失效情况下 quadruped 机器人冗余估计算法的容错运动控制 

**Authors**: Yueqi Zhang, Quancheng Qian, Taixian Hou, Peng Zhai, Xiaoyi Wei, Kangmai Hu, Jiafu Yi, Lihua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09283)  

**Abstract**: Vision-based locomotion in outdoor environments presents significant challenges for quadruped robots. Accurate environmental prediction and effective handling of depth sensor noise during real-world deployment remain difficult, severely restricting the outdoor applications of such algorithms. To address these deployment challenges in vision-based motion control, this letter proposes the Redundant Estimator Network (RENet) framework. The framework employs a dual-estimator architecture that ensures robust motion performance while maintaining deployment stability during onboard vision failures. Through an online estimator adaptation, our method enables seamless transitions between estimation modules when handling visual perception uncertainties. Experimental validation on a real-world robot demonstrates the framework's effectiveness in complex outdoor environments, showing particular advantages in scenarios with degraded visual perception. This framework demonstrates its potential as a practical solution for reliable robotic deployment in challenging field conditions. Project website: this https URL 

**Abstract (ZH)**: 基于视觉的四肢机器人在室外环境中的运动控制面临显著挑战。在现场部署中，准确的环境预测和深度传感器噪声的有效处理依然困难，严重限制了此类算法在室外的应用。为解决基于视觉的运动控制在实际部署中的这些挑战，本文提出了冗余估计网络（RENet）框架。该框架采用双估计器架构，确保在机载视觉故障时仍能保持鲁棒的运动性能和部署稳定性。通过在线估计器适应，我们的方法能在处理视觉感知不确定性时实现估计模块的无缝过渡。在真实机器人上的实验验证表明，该框架在复杂室外环境中的有效性，特别是在视觉感知退化的场景中显示出明显优势。该框架展示了解决在恶劣现场条件下可靠的机器人部署问题的潜在实用方案。项目网站：这个 https URL。 

---
# LIPM-Guided Reinforcement Learning for Stable and Perceptive Locomotion in Bipedal Robots 

**Title (ZH)**: 基于LIPM的强化学习方法实现 bipedal 机器人稳定且具备感知能力的运动控制 

**Authors**: Haokai Su, Haoxiang Luo, Shunpeng Yang, Kaiwen Jiang, Wei Zhang, Hua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.09106)  

**Abstract**: Achieving stable and robust perceptive locomotion for bipedal robots in unstructured outdoor environments remains a critical challenge due to complex terrain geometry and susceptibility to external disturbances. In this work, we propose a novel reward design inspired by the Linear Inverted Pendulum Model (LIPM) to enable perceptive and stable locomotion in the wild. The LIPM provides theoretical guidance for dynamic balance by regulating the center of mass (CoM) height and the torso orientation. These are key factors for terrain-aware locomotion, as they help ensure a stable viewpoint for the robot's camera. Building on this insight, we design a reward function that promotes balance and dynamic stability while encouraging accurate CoM trajectory tracking. To adaptively trade off between velocity tracking and stability, we leverage the Reward Fusion Module (RFM) approach that prioritizes stability when needed. A double-critic architecture is adopted to separately evaluate stability and locomotion objectives, improving training efficiency and robustness. We validate our approach through extensive experiments on a bipedal robot in both simulation and real-world outdoor environments. The results demonstrate superior terrain adaptability, disturbance rejection, and consistent performance across a wide range of speeds and perceptual conditions. 

**Abstract (ZH)**: 在未结构化户外环境中的 bipedal 机器人感知性稳定行走 remains a critical challenge due to complex terrain geometry and susceptibility to external disturbances. In this work, we propose a novel reward design inspired by the Linear Inverted Pendulum Model (LIPM) to enable perceptive and stable locomotion in the wild. 

---
# Curriculum-Based Multi-Tier Semantic Exploration via Deep Reinforcement Learning 

**Title (ZH)**: 基于课程的多层语义探索深度强化学习方法 

**Authors**: Abdel Hakim Drid, Vincenzo Suriani, Daniele Nardi, Abderrezzak Debilou  

**Link**: [PDF](https://arxiv.org/pdf/2509.09356)  

**Abstract**: Navigating and understanding complex and unknown environments autonomously demands more than just basic perception and movement from embodied agents. Truly effective exploration requires agents to possess higher-level cognitive abilities, the ability to reason about their surroundings, and make more informed decisions regarding exploration strategies. However, traditional RL approaches struggle to balance efficient exploration and semantic understanding due to limited cognitive capabilities embedded in the small policies for the agents, leading often to human drivers when dealing with semantic exploration. In this paper, we address this challenge by presenting a novel Deep Reinforcement Learning (DRL) architecture that is specifically designed for resource efficient semantic exploration. A key methodological contribution is the integration of a Vision-Language Model (VLM) common-sense through a layered reward function. The VLM query is modeled as a dedicated action, allowing the agent to strategically query the VLM only when deemed necessary for gaining external guidance, thereby conserving resources. This mechanism is combined with a curriculum learning strategy designed to guide learning at different levels of complexity to ensure robust and stable learning. Our experimental evaluation results convincingly demonstrate that our agent achieves significantly enhanced object discovery rates and develops a learned capability to effectively navigate towards semantically rich regions. Furthermore, it also shows a strategic mastery of when to prompt for external environmental information. By demonstrating a practical and scalable method for embedding common-sense semantic reasoning with autonomous agents, this research provides a novel approach to pursuing a fully intelligent and self-guided exploration in robotics. 

**Abstract (ZH)**: 自主导航和理解复杂未知环境需要不仅具备基本的感知和运动能力，还要求代理拥有更高层次的认知能力，能够对其周围环境进行推理，并做出更为明智的探索策略选择。传统的强化学习方法由于代理内嵌的认知能力有限，难以平衡高效的探索和语义理解，往往需要人类驾驶员处理语义探索任务。本文通过提出一种专为资源高效语义探索设计的新型深度强化学习（DRL）架构，应对这一挑战。一个关键的方法论贡献是通过层次化的奖励函数集成一种视觉语言模型（VLM）的常识。VLM 查询被建模为专门的动作，使代理能够在必要时战略性地查询VLM 以获取外部指导，从而节省资源。该机制结合了一种分级学习策略，以在不同复杂度的层面上引导学习，确保学习的稳健性和稳定性。实验评估结果表明，我们的代理显著提高了物体发现率，并发展了有效导航至语义丰富区域的能力。此外，还展示了何时求助外部环境信息的策略性掌握。通过展示如何将常识性语义推理嵌入自主代理的实用且可扩展的方法，本文为实现机器人中的全面智能和自我引导探索提供了一种新方法。 

---
# Boosting Embodied AI Agents through Perception-Generation Disaggregation and Asynchronous Pipeline Execution 

**Title (ZH)**: 通过感知-生成分歧和异步流水线执行提升具身AI代理 

**Authors**: Shulai Zhang, Ao Xu, Quan Chen, Han Zhao, Weihao Cui, Ningxin Zheng, Haibin Lin, Xin Liu, Minyi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.09560)  

**Abstract**: Embodied AI systems operate in dynamic environments, requiring seamless integration of perception and generation modules to process high-frequency input and output demands. Traditional sequential computation patterns, while effective in ensuring accuracy, face significant limitations in achieving the necessary "thinking" frequency for real-world applications. In this work, we present Auras, an algorithm-system co-designed inference framework to optimize the inference frequency of embodied AI agents. Auras disaggregates the perception and generation and provides controlled pipeline parallelism for them to achieve high and stable throughput. Faced with the data staleness problem that appears when the parallelism is increased, Auras establishes a public context for perception and generation to share, thereby promising the accuracy of embodied agents. Experimental results show that Auras improves throughput by 2.54x on average while achieving 102.7% of the original accuracy, demonstrating its efficacy in overcoming the constraints of sequential computation and providing high throughput. 

**Abstract (ZH)**: 嵌入式AI系统在动态环境中运行，需要无缝集成感知和生成模块以处理高频率的输入和输出需求。传统的时间序列计算模式虽然在确保准确性方面有效，但在实现现实世界应用所需要的“思考”频率方面存在显著限制。本文提出Auras，一种算法-系统协同设计的推理框架，以优化嵌入式AI代理的推理频率。Auras拆分感知和生成模块，并提供受控的流水线并行性，以实现高且稳定的吞吐量。面对并行度增加时出现的数据陈旧问题，Auras建立了一个公共上下文，使感知和生成模块能够共享，从而保证嵌入式代理的准确性。实验结果表明，Auras在吞吐量平均提高2.54倍的同时，实现了原始准确度的102.7%，证明了其有效克服时间序列计算约束并提供高吞吐量的能力。 

---
# Mind Meets Space: Rethinking Agentic Spatial Intelligence from a Neuroscience-inspired Perspective 

**Title (ZH)**: 心灵与空间的交汇：从神经科学启发视角重新思考行动者空间智能 

**Authors**: Bui Duc Manh, Soumyaratna Debnath, Zetong Zhang, Shriram Damodaran, Arvind Kumar, Yueyi Zhang, Lu Mi, Erik Cambria, Lin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09154)  

**Abstract**: Recent advances in agentic AI have led to systems capable of autonomous task execution and language-based reasoning, yet their spatial reasoning abilities remain limited and underexplored, largely constrained to symbolic and sequential processing. In contrast, human spatial intelligence, rooted in integrated multisensory perception, spatial memory, and cognitive maps, enables flexible, context-aware decision-making in unstructured environments. Therefore, bridging this gap is critical for advancing Agentic Spatial Intelligence toward better interaction with the physical 3D world. To this end, we first start from scrutinizing the spatial neural models as studied in computational neuroscience, and accordingly introduce a novel computational framework grounded in neuroscience principles. This framework maps core biological functions to six essential computation modules: bio-inspired multimodal sensing, multi-sensory integration, egocentric-allocentric conversion, an artificial cognitive map, spatial memory, and spatial reasoning. Together, these modules form a perspective landscape for agentic spatial reasoning capability across both virtual and physical environments. On top, we conduct a framework-guided analysis of recent methods, evaluating their relevance to each module and identifying critical gaps that hinder the development of more neuroscience-grounded spatial reasoning modules. We further examine emerging benchmarks and datasets and explore potential application domains ranging from virtual to embodied systems, such as robotics. Finally, we outline potential research directions, emphasizing the promising roadmap that can generalize spatial reasoning across dynamic or unstructured environments. We hope this work will benefit the research community with a neuroscience-grounded perspective and a structured pathway. Our project page can be found at Github. 

**Abstract (ZH)**: 近期代理型AI的进步使得系统具备了自主任务执行和基于语言的推理能力，但其空间推理能力仍有限且未得到充分探索，主要受限于符号和序列处理。相比之下，人类的空间智能基于多感官整合、空间记忆和认知地图等，能够在非结构化环境中实现灵活、情境相关决策。因此，弥合这一差距对于推动代理型空间智能更好地与物理三维世界交互至关重要。为此，我们首先从计算神经科学研究的空间神经网络入手，引入一个基于神经科学原理的新型计算框架。该框架将核心生物学功能映射到六个关键计算模块：生物启发式多模态感知、多感官整合、以己为中心到以物为中心的转换、人工认知地图、空间记忆和空间推理。这些模块共同形成了一种视角景观，涵盖了虚拟和物理环境中的代理型空间推理能力。同时，我们根据框架对近期方法进行了分析，评估其与每个模块的相关性，并识别阻碍基于神经科学的空间推理模块发展的关键差距。我们进一步探讨了新兴基准和数据集，并探索了从虚拟到实体系统的潜在应用领域，如机器人技术。最后，我们提出了潜在的研究方向，强调了一个有希望的发展路线图，能够使空间推理适用于动态或非结构化环境。我们希望这项工作能够为研究社区提供一个基于神经科学的视角和一个结构化的途径。版权所有页可在Github上找到。 

---
# Automated Unity Game Template Generation from GDDs via NLP and Multi-Modal LLMs 

**Title (ZH)**: 基于NLP和多模态大语言模型从GDD自动生成Unity游戏模板 

**Authors**: Amna Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2509.08847)  

**Abstract**: This paper presents a novel framework for automated game template generation by transforming Game Design Documents (GDDs) into functional Unity game prototypes using Natural Language Processing (NLP) and multi-modal Large Language Models (LLMs). We introduce an end-to-end system that parses GDDs, extracts structured game specifications, and synthesizes Unity-compatible C# code that implements the core mechanics, systems, and architecture defined in the design documentation. Our approach combines a fine-tuned LLaMA-3 model specialized for Unity code generation with a custom Unity integration package that streamlines the implementation process. Evaluation results demonstrate significant improvements over baseline models, with our fine-tuned model achieving superior performance (4.8/5.0 average score) compared to state-of-the-art LLMs across compilation success, GDD adherence, best practices adoption, and code modularity metrics. The generated templates demonstrate high adherence to GDD specifications across multiple game genres. Our system effectively addresses critical gaps in AI-assisted game development, positioning LLMs as valuable tools in streamlining the transition from game design to implementation. 

**Abstract (ZH)**: 本文提出了一种新颖的框架，通过自然语言处理（NLP）和多模态大型语言模型（LLMs）将游戏设计文档（GDDs）转换为功能性的Unity游戏原型，实现了自动化游戏模板生成。我们引入了一个端到端的系统，该系统解析GDDs，提取结构化的游戏规范，并合成Unity兼容的C#代码，以实现设计文档中定义的核心机制、系统和架构。该方法结合了专为Unity代码生成细调的LLaMA-3模型和一个自定义的Unity集成包，以简化实现过程。评估结果表明，与基线模型相比，我们细调的模型在编译成功率、GDD一致性、最佳实践采用和代码模块性指标方面表现出显著改进，评分平均为4.8/5.0，优于最新的LLMs。生成的模板在多种游戏类型中都高度符合GDD规范。我们的系统有效地填补了辅助游戏开发中的关键空白，将LLMs定位为简化从游戏设计到实现过渡的宝贵工具。 

---
# Vejde: A Framework for Inductive Deep Reinforcement Learning Based on Factor Graph Color Refinement 

**Title (ZH)**: Vejde：基于因子图颜色细分的归纳深度强化学习框架 

**Authors**: Jakob Nyberg, Pontus Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2509.09219)  

**Abstract**: We present and evaluate Vejde; a framework which combines data abstraction, graph neural networks and reinforcement learning to produce inductive policy functions for decision problems with richly structured states, such as object classes and relations. MDP states are represented as data bases of facts about entities, and Vejde converts each state to a bipartite graph, which is mapped to latent states through neural message passing. The factored representation of both states and actions allows Vejde agents to handle problems of varying size and structure. We tested Vejde agents on eight problem domains defined in RDDL, with ten problem instances each, where policies were trained using both supervised and reinforcement learning. To test policy generalization, we separate problem instances in two sets, one for training and the other solely for testing. Test results on unseen instances for the Vejde agents were compared to MLP agents trained on each problem instance, as well as the online planning algorithm Prost. Our results show that Vejde policies in average generalize to the test instances without a significant loss in score. Additionally, the inductive agents received scores on unseen test instances that on average were close to the instance-specific MLP agents. 

**Abstract (ZH)**: Vejde：一种结合数据抽象、图神经网络和强化学习的框架，用于生成决策问题中的归纳策略函数 

---
# Adaptive Pareto-Optimal Token Merging for Edge Transformer Models in Semantic Communication 

**Title (ZH)**: 边缘变压器模型中语义通信的自适应帕累托最优标记合并 

**Authors**: Omar Erak, Omar Alhussein, Hatem Abou-Zeid, Mehdi Bennis  

**Link**: [PDF](https://arxiv.org/pdf/2509.09168)  

**Abstract**: Large-scale transformer models have emerged as a powerful tool for semantic communication systems, enabling edge devices to extract rich representations for robust inference across noisy wireless channels. However, their substantial computational demands remain a major barrier to practical deployment in resource-constrained 6G networks. In this paper, we present a training-free framework for adaptive token merging in pretrained vision transformers to jointly reduce inference time and transmission resource usage. We formulate the selection of per-layer merging proportions as a multi-objective optimization problem to balance accuracy and computational cost. We employ Gaussian process-based Bayesian optimization to construct a Pareto frontier of optimal configurations, enabling flexible runtime adaptation to dynamic application requirements and channel conditions. Extensive experiments demonstrate that our method consistently outperforms other baselines and achieves significant reductions in floating-point operations while maintaining competitive accuracy across a wide range of signal-to-noise ratio (SNR) conditions. Additional results highlight the effectiveness of adaptive policies that adjust merging aggressiveness in response to channel quality, providing a practical mechanism to trade off latency and semantic fidelity on demand. These findings establish a scalable and efficient approach for deploying transformer-based semantic communication in future edge intelligence systems. 

**Abstract (ZH)**: 大规模变压器模型已成为语义通信系统中强大工具，能够使边缘设备在嘈杂的无线信道中提取丰富的表示以进行稳健的推理。然而，它们巨大的计算需求仍然是在资源受限的6G网络中实际部署的主要障碍。本文提出了一种无需训练的自适应令牌合并框架，用于预训练视觉变压器，以联合减少推理时间和传输资源使用量。我们将每层合并比例的选择形式化为多目标优化问题，以平衡准确性和计算成本。我们采用基于高斯过程的贝叶斯优化来构建最优配置的帕累托前沿，从而在运行时灵活适应动态应用需求和信道条件。大量实验证明，我们的方法在各种信噪比（SNR）条件下始终优于其他基线方法，在保持竞争力的同时显著减少了浮点运算量。此外，结果还突显了根据信道质量调整合并激进性的自适应策略的有效性，提供了一种在需求基础上权衡延迟和语义保真度的实用机制。这些发现确立了在未来的边缘智能系统中部署基于变压器的语义通信的可扩展和高效方法。 

---
