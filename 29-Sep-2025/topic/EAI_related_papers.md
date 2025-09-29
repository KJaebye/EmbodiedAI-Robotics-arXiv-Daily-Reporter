# Pixel Motion Diffusion is What We Need for Robot Control 

**Title (ZH)**: 像素运动扩散是机器人控制所需的方法 

**Authors**: E-Ro Nguyen, Yichi Zhang, Kanchana Ranasinghe, Xiang Li, Michael S. Ryoo  

**Link**: [PDF](https://arxiv.org/pdf/2509.22652)  

**Abstract**: We present DAWN (Diffusion is All We Need for robot control), a unified diffusion-based framework for language-conditioned robotic manipulation that bridges high-level motion intent and low-level robot action via structured pixel motion representation. In DAWN, both the high-level and low-level controllers are modeled as diffusion processes, yielding a fully trainable, end-to-end system with interpretable intermediate motion abstractions. DAWN achieves state-of-the-art results on the challenging CALVIN benchmark, demonstrating strong multi-task performance, and further validates its effectiveness on MetaWorld. Despite the substantial domain gap between simulation and reality and limited real-world data, we demonstrate reliable real-world transfer with only minimal finetuning, illustrating the practical viability of diffusion-based motion abstractions for robotic control. Our results show the effectiveness of combining diffusion modeling with motion-centric representations as a strong baseline for scalable and robust robot learning. Project page: this https URL 

**Abstract (ZH)**: 我们提出了DAWN（无需复杂机制只需扩散即可实现机器人控制），这是一种统一的基于扩散的框架，通过结构化像素运动表示将高层运动意图与低层机器人动作连接起来，实现语言条件下的机器人操作。在DAWN中，高层控制器和低层控制器都被建模为扩散过程，从而形成一个完全可训练、端到端的系统，并具有可解释的中间运动抽象。DAWN在具有挑战性的CALVIN基准测试中实现了最先进的结果，展示了其强大的多任务性能，并进一步在MetaWorld中验证了其有效性。尽管在模拟与现实之间的领域差距较大且缺乏真实世界数据的情况下，我们仅通过少量微调就实现了可靠的现实世界转移，这表明基于扩散的运动抽象对于机器人控制具有实际可行性。我们的结果证明了将扩散建模与以运动为中心的表示相结合作为可扩展且稳健的机器人学习强大基线的有效性。项目页面: this https URL 

---
# VLA-Reasoner: Empowering Vision-Language-Action Models with Reasoning via Online Monte Carlo Tree Search 

**Title (ZH)**: VLA-Reasoner：通过在线蒙特卡洛树搜索增强视觉-语言-行动模型的推理能力 

**Authors**: Wenkai Guo, Guanxing Lu, Haoyuan Deng, Zhenyu Wu, Yansong Tang, Ziwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22643)  

**Abstract**: Vision-Language-Action models (VLAs) achieve strong performance in general robotic manipulation tasks by scaling imitation learning. However, existing VLAs are limited to predicting short-sighted next-action, which struggle with long-horizon trajectory tasks due to incremental deviations. To address this problem, we propose a plug-in framework named VLA-Reasoner that effectively empowers off-the-shelf VLAs with the capability of foreseeing future states via test-time scaling. Specifically, VLA-Reasoner samples and rolls out possible action trajectories where involved actions are rationales to generate future states via a world model, which enables VLA-Reasoner to foresee and reason potential outcomes and search for the optimal actions. We further leverage Monte Carlo Tree Search (MCTS) to improve search efficiency in large action spaces, where stepwise VLA predictions seed the root. Meanwhile, we introduce a confidence sampling mechanism based on Kernel Density Estimation (KDE), to enable efficient exploration in MCTS without redundant VLA queries. We evaluate intermediate states in MCTS via an offline reward shaping strategy, to score predicted futures and correct deviations with long-term feedback. We conducted extensive experiments in both simulators and the real world, demonstrating that our proposed VLA-Reasoner achieves significant improvements over the state-of-the-art VLAs. Our method highlights a potential pathway toward scalable test-time computation of robotic manipulation. 

**Abstract (ZH)**: Vision-Language-Action模型（VLAs）通过扩展示例学习在通用机器人操作任务中取得优异性能。然而，现有的VLAs仅限于预测短期下一步动作，在长轨迹任务中由于逐步偏差而面临挑战。为解决此问题，我们提出了一种插件框架VLA-Reasoner，该框架能够通过测试时间扩展示例来增强现成的VLAs，使其具备预见未来状态的能力。具体而言，VLA-Reasoner通过世界模型采样和展开可能的行动轨迹，生成未来状态，从而帮助其预见和推理潜在结果并搜索最优动作。我们进一步利用蒙特卡洛树搜索（MCTS）提高在大规模动作空间中搜索的效率，其中逐步的VLAs预测作为根节点。同时，我们引入基于核密度估计（KDE）的置信度采样机制，以在MCTS中实现高效的探索而无需冗余的VLAs查询。我们通过离线奖励塑造策略评估MCTS中的中间状态，以评分预测的未来场景并利用长期反馈纠正偏差。我们在模拟器和现实世界中进行了广泛实验，证明了我们提出的VLA-Reasoner在现有最先进VLAs的基础上取得了显著改善。我们的方法展示了通向可扩展的测试时间机器人操作计算潜在路径的可能性。 

---
# WoW: Towards a World omniscient World model Through Embodied Interaction 

**Title (ZH)**: WoW: 通过具身交互 towards 万物全知的世界模型 

**Authors**: Xiaowei Chi, Peidong Jia, Chun-Kai Fan, Xiaozhu Ju, Weishi Mi, Kevin Zhang, Zhiyuan Qin, Wanxin Tian, Kuangzhi Ge, Hao Li, Zezhong Qian, Anthony Chen, Qiang Zhou, Yueru Jia, Jiaming Liu, Yong Dai, Qingpo Wuwu, Chengyu Bai, Yu-Kai Wang, Ying Li, Lizhang Chen, Yong Bao, Zhiyuan Jiang, Jiacheng Zhu, Kai Tang, Ruichuan An, Yulin Luo, Qiuxuan Feng, Siyuan Zhou, Chi-min Chan, Chengkai Hou, Wei Xue, Sirui Han, Yike Guo, Shanghang Zhang, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22642)  

**Abstract**: Humans develop an understanding of intuitive physics through active interaction with the world. This approach is in stark contrast to current video models, such as Sora, which rely on passive observation and therefore struggle with grasping physical causality. This observation leads to our central hypothesis: authentic physical intuition of the world model must be grounded in extensive, causally rich interactions with the real world. To test this hypothesis, we present WoW, a 14-billion-parameter generative world model trained on 2 million robot interaction trajectories. Our findings reveal that the model's understanding of physics is a probabilistic distribution of plausible outcomes, leading to stochastic instabilities and physical hallucinations. Furthermore, we demonstrate that this emergent capability can be actively constrained toward physical realism by SOPHIA, where vision-language model agents evaluate the DiT-generated output and guide its refinement by iteratively evolving the language instructions. In addition, a co-trained Inverse Dynamics Model translates these refined plans into executable robotic actions, thus closing the imagination-to-action loop. We establish WoWBench, a new benchmark focused on physical consistency and causal reasoning in video, where WoW achieves state-of-the-art performance in both human and autonomous evaluation, demonstrating strong ability in physical causality, collision dynamics, and object permanence. Our work provides systematic evidence that large-scale, real-world interaction is a cornerstone for developing physical intuition in AI. Models, data, and benchmarks will be open-sourced. 

**Abstract (ZH)**: 人类通过主动与世界互动来发展直觉物理理解，这一方法与当前依赖被动观察的视频模型（如Sora）存在明显差异，因此难以捉摸物理因果关系。这一观察使我们提出中心假设：世界模型中的真正物理直觉必须基于与真实世界进行广泛的因果丰富互动。为了验证这一假设，我们提出了WoW，一个在200万机器人交互轨迹上训练的140亿参数生成世界模型。我们的研究发现模型对物理的理解是一个可能结果的概率分布，导致随机不稳定性以及物理幻觉。此外，我们展示了通过SOPHIA这一方法可以主动限制这种新兴能力向物理现实靠拢，其中视觉-语言模型代理评估DiT生成的输出，并通过迭代演化语言指令引导其完善。此外，通过协同训练的反向动力学模型将这些优化的计划转化为可执行的机器人行动，从而完成想象到行动的闭环。我们建立了WoWBench，一个新的基准测试，专注于视频中的物理一致性和因果推理，WoW在人类和自主评估中均取得了最先进的性能，展示了强大的物理因果关系、碰撞动力学和物体持久性能力。我们的工作提供了大规模真实世界交互是开发AI物理直觉基石的系统性证据。模型、数据和基准将开源。 

---
# EgoDemoGen: Novel Egocentric Demonstration Generation Enables Viewpoint-Robust Manipulation 

**Title (ZH)**: EgoDemoGen: 新颖的主观示范生成 enables 视点鲁棒操作 

**Authors**: Yuan Xu, Jiabing Yang, Xiaofeng Wang, Yixiang Chen, Zheng Zhu, Bowen Fang, Guan Huang, Xinze Chen, Yun Ye, Qiang Zhang, Peiyan Li, Xiangnan Wu, Kai Wang, Bing Zhan, Shuo Lu, Jing Liu, Nianfeng Liu, Yan Huang, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22578)  

**Abstract**: Imitation learning based policies perform well in robotic manipulation, but they often degrade under *egocentric viewpoint shifts* when trained from a single egocentric viewpoint. To address this issue, we present **EgoDemoGen**, a framework that generates *paired* novel egocentric demonstrations by retargeting actions in the novel egocentric frame and synthesizing the corresponding egocentric observation videos with proposed generative video repair model **EgoViewTransfer**, which is conditioned by a novel-viewpoint reprojected scene video and a robot-only video rendered from the retargeted joint actions. EgoViewTransfer is finetuned from a pretrained video generation model using self-supervised double reprojection strategy. We evaluate EgoDemoGen on both simulation (RoboTwin2.0) and real-world robot. After training with a mixture of EgoDemoGen-generated novel egocentric demonstrations and original standard egocentric demonstrations, policy success rate improves **absolutely** by **+17.0%** for standard egocentric viewpoint and by **+17.7%** for novel egocentric viewpoints in simulation. On real-world robot, the **absolute** improvements are **+18.3%** and **+25.8%**. Moreover, performance continues to improve as the proportion of EgoDemoGen-generated demonstrations increases, with diminishing returns. These results demonstrate that EgoDemoGen provides a practical route to egocentric viewpoint-robust robotic manipulation. 

**Abstract (ZH)**: 基于模仿学习的策略在机器人操作中表现良好，但当从单一主观视角训练时，它们在主观视角转换下往往会性能下降。为解决这一问题，我们提出了一种名为EgoDemoGen的框架，该框架通过重新目标化动作并在新颖主观视角框架下合成相应的主观视角观察视频，生成配对的新颖主观视角示范。所提出的生成视频修复模型EgoViewTransfer依据新颖视角重新投影的场景视频和仅包含机器人的视频进行微调，后者是根据重新目标化关节动作渲染的。EgoViewTransfer基于自监督双重重新投影策略从预训练的视频生成模型中微调。我们在仿真（RoboTwin2.0）和真实机器人上评估了EgoDemoGen。经过混合使用EgoDemoGen生成的新颖主观视角示范和原始标准主观视角示范训练后，标准主观视角下的策略成功率绝对提升17.0%，新颖主观视角下的策略成功率绝对提升17.7%。在真实机器人上，绝对提升分别为18.3%和25.8%。此外，随着EgoDemoGen生成示范的比例增加，性能持续提升，但边际效应递减。这些结果表明，EgoDemoGen为实现主观视角鲁棒的机器人操作提供了一种实用途径。 

---
# MINT-RVAE: Multi-Cues Intention Prediction of Human-Robot Interaction using Human Pose and Emotion Information from RGB-only Camera Data 

**Title (ZH)**: MINT-RVAE：基于RGB相机数据的基于人体姿态和情绪信息的人机交互意图预测模型 

**Authors**: Farida Mohsen, Ali Safa  

**Link**: [PDF](https://arxiv.org/pdf/2509.22573)  

**Abstract**: Efficiently detecting human intent to interact with ubiquitous robots is crucial for effective human-robot interaction (HRI) and collaboration. Over the past decade, deep learning has gained traction in this field, with most existing approaches relying on multimodal inputs, such as RGB combined with depth (RGB-D), to classify time-sequence windows of sensory data as interactive or non-interactive. In contrast, we propose a novel RGB-only pipeline for predicting human interaction intent with frame-level precision, enabling faster robot responses and improved service quality. A key challenge in intent prediction is the class imbalance inherent in real-world HRI datasets, which can hinder the model's training and generalization. To address this, we introduce MINT-RVAE, a synthetic sequence generation method, along with new loss functions and training strategies that enhance generalization on out-of-sample data. Our approach achieves state-of-the-art performance (AUROC: 0.95) outperforming prior works (AUROC: 0.90-0.912), while requiring only RGB input and supporting precise frame onset prediction. Finally, to support future research, we openly release our new dataset with frame-level labeling of human interaction intent. 

**Abstract (ZH)**: 高效检测人类与物联网机器人交互意图对于有效的无人机制导交互（HRI）和协作至关重要。在过去十年中，深度学习在该领域取得了进展，大多数现有方法依赖于多模态输入，如RGB与深度（RGB-D）结合，以分类传感器数据的时间序列窗口为交互或非交互。相比之下，我们提出了一种基于帧级精度预测人类交互意图的纯RGB管道，能够加快机器人响应速度并提高服务质量。意图预测的关键挑战是在真实世界的人机交互（HRI）数据集中固有的类别不平衡问题，这可能会影响模型的训练和泛化能力。为解决这一问题，我们引入了MINT-RVAE合成序列生成方法，以及新的损失函数和训练策略，以增强对未见数据的泛化能力。我们的方法在AUC-ROC方面达到最新技术水平（0.95），超越了先前的工作（0.90-0.912），同时仅需RGB输入并支持精确的帧起始点预测。最后，为了支持未来的研究，我们公开发布了带有帧级人类交互意图标注的新数据集。 

---
# HELIOS: Hierarchical Exploration for Language-grounded Interaction in Open Scenes 

**Title (ZH)**: HELIOS：层级探索语言导向开放场景交互 

**Authors**: Katrina Ashton, Chahyon Ku, Shrey Shah, Wen Jiang, Kostas Daniilidis, Bernadette Bucher  

**Link**: [PDF](https://arxiv.org/pdf/2509.22498)  

**Abstract**: Language-specified mobile manipulation tasks in novel environments simultaneously face challenges interacting with a scene which is only partially observed, grounding semantic information from language instructions to the partially observed scene, and actively updating knowledge of the scene with new observations. To address these challenges, we propose HELIOS, a hierarchical scene representation and associated search objective to perform language specified pick and place mobile manipulation tasks. We construct 2D maps containing the relevant semantic and occupancy information for navigation while simultaneously actively constructing 3D Gaussian representations of task-relevant objects. We fuse observations across this multi-layered representation while explicitly modeling the multi-view consistency of the detections of each object. In order to efficiently search for the target object, we formulate an objective function balancing exploration of unobserved or uncertain regions with exploitation of scene semantic information. We evaluate HELIOS on the OVMM benchmark in the Habitat simulator, a pick and place benchmark in which perception is challenging due to large and complex scenes with comparatively small target objects. HELIOS achieves state-of-the-art results on OVMM. As our approach is zero-shot, HELIOS can also transfer to the real world without requiring additional data, as we illustrate by demonstrating it in a real world office environment on a Spot robot. 

**Abstract (ZH)**: 基于语言指定的移动操作任务在新型环境中的层次化场景表示与搜索目标 

---
# Ontological foundations for contrastive explanatory narration of robot plans 

**Title (ZH)**: 基于本体论基础的对比解释性叙述机器人计划 

**Authors**: Alberto Olivares-Alarcos, Sergi Foix, Júlia Borràs, Gerard Canal, Guillem Alenyà  

**Link**: [PDF](https://arxiv.org/pdf/2509.22493)  

**Abstract**: Mutual understanding of artificial agents' decisions is key to ensuring a trustworthy and successful human-robot interaction. Hence, robots are expected to make reasonable decisions and communicate them to humans when needed. In this article, the focus is on an approach to modeling and reasoning about the comparison of two competing plans, so that robots can later explain the divergent result. First, a novel ontological model is proposed to formalize and reason about the differences between competing plans, enabling the classification of the most appropriate one (e.g., the shortest, the safest, the closest to human preferences, etc.). This work also investigates the limitations of a baseline algorithm for ontology-based explanatory narration. To address these limitations, a novel algorithm is presented, leveraging divergent knowledge between plans and facilitating the construction of contrastive narratives. Through empirical evaluation, it is observed that the explanations excel beyond the baseline method. 

**Abstract (ZH)**: 人工代理决策间的相互理解是确保人机交互可信和成功的关键。因此，机器人在必要时应作出合理决策并进行沟通。本文专注于一种建模和推理两个竞争计划差异的方法，以便机器人后来可以解释不同的结果。首先，提出了一种新颖的本体模型来形式化和推理竞争计划之间的差异，从而实现对最合适的计划（如最短的、最安全的、最符合人类偏好的等）的分类。此外，还研究了基于本体解释性叙述的基础算法的局限性。为了解决这些局限性，提出了一种新的算法，利用计划间的差异性知识，促进对比叙述的构建。通过实证评估，发现这些解释超越了基准方法。 

---
# UnderwaterVLA: Dual-brain Vision-Language-Action architecture for Autonomous Underwater Navigation 

**Title (ZH)**: 水下VLA：自主水下导航的双脑视觉-语言-行动架构 

**Authors**: Zhangyuan Wang, Yunpeng Zhu, Yuqi Yan, Xiaoyuan Tian, Xinhao Shao, Meixuan Li, Weikun Li, Guangsheng Su, Weicheng Cui, Dixia Fan  

**Link**: [PDF](https://arxiv.org/pdf/2509.22441)  

**Abstract**: This paper presents UnderwaterVLA, a novel framework for autonomous underwater navigation that integrates multimodal foundation models with embodied intelligence systems. Underwater operations remain difficult due to hydrodynamic disturbances, limited communication bandwidth, and degraded sensing in turbid waters. To address these challenges, we introduce three innovations. First, a dual-brain architecture decouples high-level mission reasoning from low-level reactive control, enabling robust operation under communication and computational constraints. Second, we apply Vision-Language-Action(VLA) models to underwater robotics for the first time, incorporating structured chain-of-thought reasoning for interpretable decision-making. Third, a hydrodynamics-informed Model Predictive Control(MPC) scheme compensates for fluid effects in real time without costly task-specific training. Experimental results in field tests show that UnderwaterVLA reduces navigation errors in degraded visual conditions while maintaining higher task completion by 19% to 27% over baseline. By minimizing reliance on underwater-specific training data and improving adaptability across environments, UnderwaterVLA provides a scalable and cost-effective path toward the next generation of intelligent AUVs. 

**Abstract (ZH)**: 水下VLA：一种结合多模态基础模型与实体智能系统的自主水下导航新型框架 

---
# RoboView-Bias: Benchmarking Visual Bias in Embodied Agents for Robotic Manipulation 

**Title (ZH)**: RoboView-偏差：机器人操控中体态代理视觉偏差的基准测试 

**Authors**: Enguang Liu, Siyuan Liang, Liming Lu, Xiyu Zeng, Xiaochun Cao, Aishan Liu, Shuchao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22356)  

**Abstract**: The safety and reliability of embodied agents rely on accurate and unbiased visual perception. However, existing benchmarks mainly emphasize generalization and robustness under perturbations, while systematic quantification of visual bias remains scarce. This gap limits a deeper understanding of how perception influences decision-making stability. To address this issue, we propose RoboView-Bias, the first benchmark specifically designed to systematically quantify visual bias in robotic manipulation, following a principle of factor isolation. Leveraging a structured variant-generation framework and a perceptual-fairness validation protocol, we create 2,127 task instances that enable robust measurement of biases induced by individual visual factors and their interactions. Using this benchmark, we systematically evaluate three representative embodied agents across two prevailing paradigms and report three key findings: (i) all agents exhibit significant visual biases, with camera viewpoint being the most critical factor; (ii) agents achieve their highest success rates on highly saturated colors, indicating inherited visual preferences from underlying VLMs; and (iii) visual biases show strong, asymmetric coupling, with viewpoint strongly amplifying color-related bias. Finally, we demonstrate that a mitigation strategy based on a semantic grounding layer substantially reduces visual bias by approximately 54.5\% on MOKA. Our results highlight that systematic analysis of visual bias is a prerequisite for developing safe and reliable general-purpose embodied agents. 

**Abstract (ZH)**: 机器人视觉偏差的系统量化：RoboView-Bias基准 

---
# Beyond Detection -- Orchestrating Human-Robot-Robot Assistance via an Internet of Robotic Things Paradigm 

**Title (ZH)**: 超越检测——通过物联网机器人范式 orchestrating 人类-机器人-机器人协作 

**Authors**: Joseph Hunt, Koyo Fujii, Aly Magassouba, Praminda Caleb-Solly  

**Link**: [PDF](https://arxiv.org/pdf/2509.22296)  

**Abstract**: Hospital patient falls remain a critical and costly challenge worldwide. While conventional fall prevention systems typically rely on post-fall detection or reactive alerts, they also often suffer from high false positive rates and fail to address the underlying patient needs that lead to bed-exit attempts. This paper presents a novel system architecture that leverages the Internet of Robotic Things (IoRT) to orchestrate human-robot-robot interaction for proactive and personalized patient assistance. The system integrates a privacy-preserving thermal sensing model capable of real-time bed-exit prediction, with two coordinated robotic agents that respond dynamically based on predicted intent and patient input. This orchestrated response could not only reduce fall risk but also attend to the patient's underlying motivations for movement, such as thirst, discomfort, or the need for assistance, before a hazardous situation arises. Our contributions with this pilot study are three-fold: (1) a modular IoRT-based framework enabling distributed sensing, prediction, and multi-robot coordination; (2) a demonstration of low-resolution thermal sensing for accurate, privacy-preserving preemptive bed-exit detection; and (3) results from a user study and systematic error analysis that inform the design of situationally aware, multi-agent interactions in hospital settings. The findings highlight how interactive and connected robotic systems can move beyond passive monitoring to deliver timely, meaningful assistance, empowering safer, more responsive care environments. 

**Abstract (ZH)**: 基于物联网机器人技术的主动个性化患者辅助系统架构：减少医院患者跌倒挑战 

---
# Leveraging Large Language Models for Robot-Assisted Learning of Morphological Structures in Preschool Children with Language Vulnerabilities 

**Title (ZH)**: 利用大型语言模型辅助具备语言脆弱性的学龄前儿童学习形态结构 

**Authors**: Stina Sundstedt, Mattias Wingren, Susanne Hägglund, Daniel Ventus  

**Link**: [PDF](https://arxiv.org/pdf/2509.22287)  

**Abstract**: Preschool children with language vulnerabilities -- such as developmental language disorders or immigration related language challenges -- often require support to strengthen their expressive language skills. Based on the principle of implicit learning, speech-language therapists (SLTs) typically embed target morphological structures (e.g., third person -s) into everyday interactions or game-based learning activities. Educators are recommended by SLTs to do the same. This approach demands precise linguistic knowledge and real-time production of various morphological forms (e.g., "Daddy wears these when he drives to work"). The task becomes even more demanding when educators or parent also must keep children engaged and manage turn-taking in a game-based activity. In the TalBot project our multiprofessional team have developed an application in which the Furhat conversational robot plays the word retrieval game "Alias" with children to improve language skills. Our application currently employs a large language model (LLM) to manage gameplay, dialogue, affective responses, and turn-taking. Our next step is to further leverage the capacity of LLMs so the robot can generate and deliver specific morphological targets during the game. We hypothesize that a robot could outperform humans at this task. Novel aspects of this approach are that the robot could ultimately serve as a model and tutor for both children and professionals and that using LLM capabilities in this context would support basic communication needs for children with language vulnerabilities. Our long-term goal is to create a robust LLM-based Robot-Assisted Language Learning intervention capable of teaching a variety of morphological structures across different languages. 

**Abstract (ZH)**: 语言脆弱性学前儿童的支持：基于隐性学习原理的机器人辅助语言训练方法 

---
# MimicDreamer: Aligning Human and Robot Demonstrations for Scalable VLA Training 

**Title (ZH)**: MimicDreamer: 将人类和机器人演示相结合以实现可扩展的联合学习训练 

**Authors**: Haoyun Li, Ivan Zhang, Runqi Ouyang, Xiaofeng Wang, Zheng Zhu, Zhiqin Yang, Zhentao Zhang, Boyuan Wang, Chaojun Ni, Wenkang Qin, Xinze Chen, Yun Ye, Guan Huang, Zhenbo Song, Xingang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22199)  

**Abstract**: Vision Language Action (VLA) models derive their generalization capability from diverse training data, yet collecting embodied robot interaction data remains prohibitively expensive. In contrast, human demonstration videos are far more scalable and cost-efficient to collect, and recent studies confirm their effectiveness in training VLA models. However, a significant domain gap persists between human videos and robot-executed videos, including unstable camera viewpoints, visual discrepancies between human hands and robotic arms, and differences in motion dynamics. To bridge this gap, we propose MimicDreamer, a framework that turns fast, low-cost human demonstrations into robot-usable supervision by jointly aligning vision, viewpoint, and actions to directly support policy training. For visual alignment, we propose H2R Aligner, a video diffusion model that generates high-fidelity robot demonstration videos by transferring motion from human manipulation footage. For viewpoint stabilization, EgoStabilizer is proposed, which canonicalizes egocentric videos via homography and inpaints occlusions and distortions caused by warping. For action alignment, we map human hand trajectories to the robot frame and apply a constrained inverse kinematics solver to produce feasible, low-jitter joint commands with accurate pose tracking. Empirically, VLA models trained purely on our synthesized human-to-robot videos achieve few-shot execution on real robots. Moreover, scaling training with human data significantly boosts performance compared to models trained solely on real robot data; our approach improves the average success rate by 14.7\% across six representative manipulation tasks. 

**Abstract (ZH)**: Vision Language Action (VLA)模型的通用能力来源于多样化的训练数据，然而收集带有机器人互动的数据仍然极为昂贵。相比之下，人类演示视频的收集更为规模性强且成本效益高，近期的研究证实其在训练VLA模型方面的有效性。然而，人类视频与机器人执行视频之间仍然存在显著的领域差距，包括不稳定的摄像机视角、人类手部与机器人手臂之间的视觉差异以及运动动态的差异。为了弥合这一差距，我们提出了MimicDreamer框架，通过联合对齐视觉、视角和动作，将其快速低成本的人类演示转化为可用于机器人训练的支持策略训练的监督数据。在视觉对齐方面，我们提出了H2R对齐器，这是一种视频扩散模型，通过从人类操作片段中转移运动以生成高保真度的机器人演示视频。对于视角稳定，我们提出了EgoStabilizer，它通过透射变换规范化第一人称视角视频，并修补扭曲运动导致的遮挡和失真。对于动作对齐，我们将人类手部轨迹映射到机器人坐标系，并应用约束逆运动学求解器以产生具有准确姿态跟踪的可行、低抖动的关节命令。实验结果显示，仅使用我们合成的从人到机器人的视频训练的VLA模型能够在真实机器人上实现少量样本执行。此外，使用人类数据扩展训练显著提升了性能，与仅基于真实机器人数据训练的模型相比，我们方法在六个代表性操作任务中将平均成功率提高了14.7%。 

---
# Actions as Language: Fine-Tuning VLMs into VLAs Without Catastrophic Forgetting 

**Title (ZH)**: 动作即语言：在不发生灾难性遗忘的情况下Fine-Tuning VLMs为VLAs 

**Authors**: Asher J. Hancock, Xindi Wu, Lihan Zha, Olga Russakovsky, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2509.22195)  

**Abstract**: Fine-tuning vision-language models (VLMs) on robot teleoperation data to create vision-language-action (VLA) models is a promising paradigm for training generalist policies, but it suffers from a fundamental tradeoff: learning to produce actions often diminishes the VLM's foundational reasoning and multimodal understanding, hindering generalization to novel scenarios, instruction following, and semantic understanding. We argue that this catastrophic forgetting is due to a distribution mismatch between the VLM's internet-scale pretraining corpus and the robotics fine-tuning data. Inspired by this observation, we introduce VLM2VLA: a VLA training paradigm that first resolves this mismatch at the data level by representing low-level actions with natural language. This alignment makes it possible to train VLAs solely with Low-Rank Adaptation (LoRA), thereby minimally modifying the VLM backbone and averting catastrophic forgetting. As a result, the VLM can be fine-tuned on robot teleoperation data without fundamentally altering the underlying architecture and without expensive co-training on internet-scale VLM datasets. Through extensive Visual Question Answering (VQA) studies and over 800 real-world robotics experiments, we demonstrate that VLM2VLA preserves the VLM's core capabilities, enabling zero-shot generalization to novel tasks that require open-world semantic reasoning and multilingual instruction following. 

**Abstract (ZH)**: 基于机器人远程操作数据 fine-tuning 视觉-语言模型以创建视觉-语言-行动模型：解决灾难性遗忘的范式 

---
# Action-aware Dynamic Pruning for Efficient Vision-Language-Action Manipulation 

**Title (ZH)**: 基于动作感知的动态剪枝以实现高效的视觉-语言-动作操纵 

**Authors**: Xiaohuan Pei, Yuxing Chen, Siyu Xu, Yunke Wang, Yuheng Shi, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.22093)  

**Abstract**: Robotic manipulation with Vision-Language-Action models requires efficient inference over long-horizon multi-modal context, where attention to dense visual tokens dominates computational cost. Existing methods optimize inference speed by reducing visual redundancy within VLA models, but they overlook the varying redundancy across robotic manipulation stages. We observe that the visual token redundancy is higher in coarse manipulation phase than in fine-grained operations, and is strongly correlated with the action dynamic. Motivated by this observation, we propose \textbf{A}ction-aware \textbf{D}ynamic \textbf{P}runing (\textbf{ADP}), a multi-modal pruning framework that integrates text-driven token selection with action-aware trajectory gating. Our method introduces a gating mechanism that conditions the pruning signal on recent action trajectories, using past motion windows to adaptively adjust token retention ratios in accordance with dynamics, thereby balancing computational efficiency and perceptual precision across different manipulation stages. Extensive experiments on the LIBERO suites and diverse real-world scenarios demonstrate that our method significantly reduces FLOPs and action inference latency (\textit{e.g.} $1.35 \times$ speed up on OpenVLA-OFT) while maintaining competitive success rates (\textit{e.g.} 25.8\% improvements with OpenVLA) compared to baselines, thereby providing a simple plug-in path to efficient robot policies that advances the efficiency and performance frontier of robotic manipulation. Our project website is: \href{this https URL}{this http URL}. 

**Abstract (ZH)**: 具有视觉-语言-动作模型的机器人操作需要高效处理长时跨多模态上下文，其中对密集视觉标记的关注主导了计算成本。现有方法通过减少VLA模型内的视觉冗余来优化推理速度，但忽略了机器人操作各阶段的冗余变化。我们观察到，在粗粒度操作阶段，视觉标记冗余高于精细操作阶段，并且与动作动态密切相关。受此观察启发，我们提出了一种多模态剪枝框架——感知动作动态剪枝（ADP），该框架结合了文本驱动的标记选择和感知动作轨迹门控。我们的方法引入了一种门控机制，该机制根据近期动作轨迹来调整剪枝信号，并使用过去的运动窗口来适应性调整标记保留比例，从而在不同操作阶段平衡计算效率和感知精度。在LIBERO套件和多种现实世界场景上的广泛实验表明，与基线方法相比，我们的方法在保持竞争力的操作成功率（例如，在OpenVLA上提高25.8%）的同时显著减少了FLOPs和动作推理延迟（例如，在OpenVLA-OFT上加速1.35倍），从而提供了一条简单插件路径，以提高机器人操作的效率和性能前沿。我们的项目网站是：\href{this https URL}{this http URL}。 

---
# Effect of Gait Design on Proprioceptive Sensing of Terrain Properties in a Quadrupedal Robot 

**Title (ZH)**: quadruped机器人步态设计对地形本体感受特性影响的研究 

**Authors**: Ethan Fulcher, J. Diego Caporale, Yifeng Zhang, John Ruck, Feifei Qian  

**Link**: [PDF](https://arxiv.org/pdf/2509.22065)  

**Abstract**: In-situ robotic exploration is an important tool for advancing knowledge of geological processes that describe the Earth and other Planetary bodies. To inform and enhance operations for these roving laboratories, it is imperative to understand the terramechanical properties of their environments, especially for traversing on loose, deformable substrates. Recent research suggested that legged robots with direct-drive and low-gear ratio actuators can sensitively detect external forces, and therefore possess the potential to measure terrain properties with their legs during locomotion, providing unprecedented sampling speed and density while accessing terrains previously too risky to sample. This paper explores these ideas by investigating the impact of gait on proprioceptive terrain sensing accuracy, particularly comparing a sensing-oriented gait, Crawl N' Sense, with a locomotion-oriented gait, Trot-Walk. Each gait's ability to measure the strength and texture of deformable substrate is quantified as the robot locomotes over a laboratory transect consisting of a rigid surface, loose sand, and loose sand with synthetic surface crusts. Our results suggest that with both the sensing-oriented crawling gait and locomotion-oriented trot gait, the robot can measure a consistent difference in the strength (in terms of penetration resistance) between the low- and high-resistance substrates; however, the locomotion-oriented trot gait contains larger magnitude and variance in measurements. Furthermore, the slower crawl gait can detect brittle ruptures of the surface crusts with significantly higher accuracy than the faster trot gait. Our results offer new insights that inform legged robot "sensing during locomotion" gait design and planning for scouting the terrain and producing scientific measurements on other worlds to advance our understanding of their geology and formation. 

**Abstract (ZH)**: 基于就地行进的轮腿机器人地形感知研究：步态对 proprioceptive 地形感知精度的影响 

---
# SAGE: Scene Graph-Aware Guidance and Execution for Long-Horizon Manipulation Tasks 

**Title (ZH)**: SAGE: 场景图感知的长期操作任务引导与执行 

**Authors**: Jialiang Li, Wenzheng Wu, Gaojing Zhang, Yifan Han, Wenzhao Lian  

**Link**: [PDF](https://arxiv.org/pdf/2509.21928)  

**Abstract**: Successfully solving long-horizon manipulation tasks remains a fundamental challenge. These tasks involve extended action sequences and complex object interactions, presenting a critical gap between high-level symbolic planning and low-level continuous control. To bridge this gap, two essential capabilities are required: robust long-horizon task planning and effective goal-conditioned manipulation. Existing task planning methods, including traditional and LLM-based approaches, often exhibit limited generalization or sparse semantic reasoning. Meanwhile, image-conditioned control methods struggle to adapt to unseen tasks. To tackle these problems, we propose SAGE, a novel framework for Scene Graph-Aware Guidance and Execution in Long-Horizon Manipulation Tasks. SAGE utilizes semantic scene graphs as a structural representation for scene states. A structural scene graph enables bridging task-level semantic reasoning and pixel-level visuo-motor control. This also facilitates the controllable synthesis of accurate, novel sub-goal images. SAGE consists of two key components: (1) a scene graph-based task planner that uses VLMs and LLMs to parse the environment and reason about physically-grounded scene state transition sequences, and (2) a decoupled structural image editing pipeline that controllably converts each target sub-goal graph into a corresponding image through image inpainting and composition. Extensive experiments have demonstrated that SAGE achieves state-of-the-art performance on distinct long-horizon tasks. 

**Abstract (ZH)**: 长_horizon操作任务的成功解决仍然是一个基本挑战：SAGE——场景图aware指导与执行框架 

---
# Learning Multi-Skill Legged Locomotion Using Conditional Adversarial Motion Priors 

**Title (ZH)**: 基于条件对抗运动先验的学习多技能-legged运动 

**Authors**: Ning Huang, Zhentao Xie, Qinchuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.21810)  

**Abstract**: Despite growing interest in developing legged robots that emulate biological locomotion for agile navigation of complex environments, acquiring a diverse repertoire of skills remains a fundamental challenge in robotics. Existing methods can learn motion behaviors from expert data, but they often fail to acquire multiple locomotion skills through a single policy and lack smooth skill transitions. We propose a multi-skill learning framework based on Conditional Adversarial Motion Priors (CAMP), with the aim of enabling quadruped robots to efficiently acquire a diverse set of locomotion skills from expert demonstrations. Precise skill reconstruction is achieved through a novel skill discriminator and skill-conditioned reward design. The overall framework supports the active control and reuse of multiple skills, providing a practical solution for learning generalizable policies in complex environments. 

**Abstract (ZH)**: 基于条件对抗运动先验的多技能学习框架：使 quadruped 机器人高效习得多样化的运动技能 

---
# The Turkish Ice Cream Robot: Examining Playful Deception in Social Human-Robot Interactions 

**Title (ZH)**: 土耳其冰淇淋机器人：探究社交人机交互中的玩乐欺骗现象 

**Authors**: Hyeonseong Kim, Roy El-Helou, Seungbeen Lee, Sungjoon Choi, Matthew Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.21776)  

**Abstract**: Playful deception, a common feature in human social interactions, remains underexplored in Human-Robot Interaction (HRI). Inspired by the Turkish Ice Cream (TIC) vendor routine, we investigate how bounded, culturally familiar forms of deception influence user trust, enjoyment, and engagement during robotic handovers. We design a robotic manipulator equipped with a custom end-effector and implement five TIC-inspired trick policies that deceptively delay the handover of an ice cream-shaped object. Through a mixed-design user study with 91 participants, we evaluate the effects of playful deception and interaction duration on user experience. Results reveal that TIC-inspired deception significantly enhances enjoyment and engagement, though reduces perceived safety and trust, suggesting a structured trade-off across the multi-dimensional aspects. Our findings demonstrate that playful deception can be a valuable design strategy for interactive robots in entertainment and engagement-focused contexts, while underscoring the importance of deliberate consideration of its complex trade-offs. You can find more information, including demonstration videos, on this https URL . 

**Abstract (ZH)**: 人在机器人交互中未充分探索的 playful deception：基于土耳其冰淇淋摊贩routine的研究 

---
# VLBiMan: Vision-Language Anchored One-Shot Demonstration Enables Generalizable Robotic Bimanual Manipulation 

**Title (ZH)**: VLBiMan: 视听锚定的一次性示范 enables 可泛化的双臂机器人操作 

**Authors**: Huayi Zhou, Kui Jia  

**Link**: [PDF](https://arxiv.org/pdf/2509.21723)  

**Abstract**: Achieving generalizable bimanual manipulation requires systems that can learn efficiently from minimal human input while adapting to real-world uncertainties and diverse embodiments. Existing approaches face a dilemma: imitation policy learning demands extensive demonstrations to cover task variations, while modular methods often lack flexibility in dynamic scenes. We introduce VLBiMan, a framework that derives reusable skills from a single human example through task-aware decomposition, preserving invariant primitives as anchors while dynamically adapting adjustable components via vision-language grounding. This adaptation mechanism resolves scene ambiguities caused by background changes, object repositioning, or visual clutter without policy retraining, leveraging semantic parsing and geometric feasibility constraints. Moreover, the system inherits human-like hybrid control capabilities, enabling mixed synchronous and asynchronous use of both arms. Extensive experiments validate VLBiMan across tool-use and multi-object tasks, demonstrating: (1) a drastic reduction in demonstration requirements compared to imitation baselines, (2) compositional generalization through atomic skill splicing for long-horizon tasks, (3) robustness to novel but semantically similar objects and external disturbances, and (4) strong cross-embodiment transfer, showing that skills learned from human demonstrations can be instantiated on different robotic platforms without retraining. By bridging human priors with vision-language anchored adaptation, our work takes a step toward practical and versatile dual-arm manipulation in unstructured settings. 

**Abstract (ZH)**: 实现可泛化的双臂操作需要能够从最少的人类输入中高效学习并适应实际环境中的不确定性及多样化实体的系统。现有方法面临困境：模仿策略学习需要大量演示来覆盖任务变异性，而模块化方法在动态场景中往往缺乏灵活性。我们引入了VLBiMan框架，该框架通过任务感知分解从单个人类示例中提取可重用技能，保留不变的基础技能作为锚点，同时通过视觉-语言接地动态调整可调节组件。这种适应机制在背景变化、物体重定位或视觉杂乱的情况下解决了场景模糊性，而不需策略重新训练，利用语义解析和几何可行性约束。此外，该系统继承了类似人类的混合控制能力，能够灵活使用双臂进行同步和异步操作。广泛实验验证了VLBiMan在工具有用性和多对象任务中的表现，表明：（1）与模仿基线相比，演示需求大幅减少；（2）通过原子技能拼接实现长期任务的组合泛化；（3）对新型但语义相似的对象和外部干扰具有鲁棒性；（4）强大的跨实体迁移，表明从人类演示学到的技能可以在不同的机器人平台上实例化而无需重新训练。通过将人类先验与视觉-语言锚定的适应相结合，我们的工作朝着在非结构化环境中实现实用且多用途的双臂操作迈出了重要一步。 

---
# Towards Versatile Humanoid Table Tennis: Unified Reinforcement Learning with Prediction Augmentation 

**Title (ZH)**: 面向多功能类人乒乓球：预测增强的统一强化学习 

**Authors**: Muqun Hu, Wenxi Chen, Wenjing Li, Falak Mandali, Zijian He, Renhong Zhang, Praveen Krisna, Katherine Christian, Leo Benaharon, Dizhi Ma, Karthik Ramani, Yan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21690)  

**Abstract**: Humanoid table tennis (TT) demands rapid perception, proactive whole-body motion, and agile footwork under strict timing -- capabilities that remain difficult for unified controllers. We propose a reinforcement learning framework that maps ball-position observations directly to whole-body joint commands for both arm striking and leg locomotion, strengthened by predictive signals and dense, physics-guided rewards. A lightweight learned predictor, fed with recent ball positions, estimates future ball states and augments the policy's observations for proactive decision-making. During training, a physics-based predictor supplies precise future states to construct dense, informative rewards that lead to effective exploration. The resulting policy attains strong performance across varied serve ranges (hit rate $\geq$ 96% and success rate $\geq$ 92%) in simulations. Ablation studies confirm that both the learned predictor and the predictive reward design are critical for end-to-end learning. Deployed zero-shot on a physical Booster T1 humanoid with 23 revolute joints, the policy produces coordinated lateral and forward-backward footwork with accurate, fast returns, suggesting a practical path toward versatile, competitive humanoid TT. 

**Abstract (ZH)**: 人体形乒乓球（TT）需要在严格的时间限制下进行快速感知、全身先导运动和敏捷步法——这些能力对于统一控制器来说仍然是难题。我们提出了一种强化学习框架，该框架直接将球位观察映射到手臂打击和腿部运动的全身关节命令，并通过预测信号和密集的物理引导奖励加以强化。一个轻量级的学习预测器以最近的球位信息为输入，预测未来的球状态，并增强策略的观测以促进积极决策。在训练过程中，基于物理的预测器提供精确的未来状态，构建密集且有信息量的奖励，促进有效的探索。所得到的策略在各种发球范围内的模拟中表现出强大的性能（击球率≥96%，成功率≥92%）。消融研究证实，学习预测器和预测奖励设计对于端到端学习至关重要。该策略在物理Booster T1人型机器人上零样本部署，机器人拥有23个转动关节，能够产生协调的横向和前后步法，并能实现准确快速的回球，表明了一条通向下肢灵活且具有竞争力的人体形乒乓球路径。 

---
# Generating Stable Placements via Physics-guided Diffusion Models 

**Title (ZH)**: 基于物理引导的扩散模型生成稳定放置 

**Authors**: Philippe Nadeau, Miguel Rogel, Ivan Bilić, Ivan Petrović, Jonathan Kelly  

**Link**: [PDF](https://arxiv.org/pdf/2509.21664)  

**Abstract**: Stably placing an object in a multi-object scene is a fundamental challenge in robotic manipulation, as placements must be penetration-free, establish precise surface contact, and result in a force equilibrium. To assess stability, existing methods rely on running a simulation engine or resort to heuristic, appearance-based assessments. In contrast, our approach integrates stability directly into the sampling process of a diffusion model. To this end, we query an offline sampling-based planner to gather multi-modal placement labels and train a diffusion model to generate stable placements. The diffusion model is conditioned on scene and object point clouds, and serves as a geometry-aware prior. We leverage the compositional nature of score-based generative models to combine this learned prior with a stability-aware loss, thereby increasing the likelihood of sampling from regions of high stability. Importantly, this strategy requires no additional re-training or fine-tuning, and can be directly applied to off-the-shelf models. We evaluate our method on four benchmark scenes where stability can be accurately computed. Our physics-guided models achieve placements that are 56% more robust to forceful perturbations while reducing runtime by 47% compared to a state-of-the-art geometric method. 

**Abstract (ZH)**: 在多对象场景中稳定放置物体是机器人操作中的一个基本挑战，因为放置必须无穿插、建立精确的表面接触，并导致力的平衡。为了评估稳定性，现有方法依赖于运行仿真引擎或采用启发式、基于外观的评估。相比之下，我们的方法将稳定性直接集成到扩散模型的采样过程中。为此，我们查询离线的基于采样的规划器以收集多模态放置标签，并训练扩散模型以生成稳定的放置。扩散模型以场景和物体点云为条件，并作为几何感知的先验。我们利用基于评分生成模型的组合特性，将这种学习到的先验与稳定性感知的损失结合起来，从而增加从高稳定性区域采样的可能性。重要的是，这种策略不需要额外的重新训练或微调，并可以直接应用于现成的模型。我们在四个基准场景上评估了我们的方法，其中稳定性可以准确计算。我们的物理导向模型在承受外力扰动时的表现比最先进的几何方法更稳定，同时运行时间减少了47%。 

---
# Language-in-the-Loop Culvert Inspection on the Erie Canal 

**Title (ZH)**: Linguistic-in-the-Loop 沃尔什运河 culvert 检查 

**Authors**: Yashom Dighe, Yash Turkar, Karthik Dantu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21370)  

**Abstract**: Culverts on canals such as the Erie Canal, built originally in 1825, require frequent inspections to ensure safe operation. Human inspection of culverts is challenging due to age, geometry, poor illumination, weather, and lack of easy access. We introduce VISION, an end-to-end, language-in-the-loop autonomy system that couples a web-scale vision-language model (VLM) with constrained viewpoint planning for autonomous inspection of culverts. Brief prompts to the VLM solicit open-vocabulary ROI proposals with rationales and confidences, stereo depth is fused to recover scale, and a planner -- aware of culvert constraints -- commands repositioning moves to capture targeted close-ups. Deployed on a quadruped in a culvert under the Erie Canal, VISION closes the see, decide, move, re-image loop on-board and produces high-resolution images for detailed reporting without domain-specific fine-tuning. In an external evaluation by New York Canal Corporation personnel, initial ROI proposals achieved 61.4\% agreement with subject-matter experts, and final post-re-imaging assessments reached 80\%, indicating that VISION converts tentative hypotheses into grounded, expert-aligned findings. 

**Abstract (ZH)**: Erie运河等渠道上的涵洞，最初建于1825年，需要定期检查以确保安全运行。由于老化、几何形状、光照差、天气和难以接近等因素，人工检查涵洞具有挑战性。我们引入了VISION，这是一个端到端、带有语言循环的自主系统，将大规模网络视觉-语言模型（VLM）与受限视角规划相结合，以实现涵洞的自主检查。通过简短的提示，VLM征集开放词汇的目标区域建议及其理由和置信度，融合了立体深度以恢复尺度，并由计划者根据涵洞约束命令重新定位动作以捕捉目标特写。该系统部署在Erie运河下的四足机器人上，实现了从观察、决策、移动到重新成像的闭环操作，并生成高分辨率图像以供详细报告，无需特定领域的微调。在纽约运河公司人员的外部评估中，初始目标区域建议与主题专家一致率为61.4%，最终重新成像后的评估达到80%，表明VISION将初步假设转化为与专家一致的确定性结果。 

---
# JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation 

**Title (ZH)**: JanusVLN：通过双隐式记忆解藕语义与空间性在视觉语言导航中的应用 

**Authors**: Shuang Zeng, Dekang Qi, Xinyuan Chang, Feng Xiong, Shichao Xie, Xiaolong Wu, Shiyi Liang, Mu Xu, Xing Wei  

**Link**: [PDF](https://arxiv.org/pdf/2509.22548)  

**Abstract**: Vision-and-Language Navigation requires an embodied agent to navigate through unseen environments, guided by natural language instructions and a continuous video stream. Recent advances in VLN have been driven by the powerful semantic understanding of Multimodal Large Language Models. However, these methods typically rely on explicit semantic memory, such as building textual cognitive maps or storing historical visual frames. This type of method suffers from spatial information loss, computational redundancy, and memory bloat, which impede efficient navigation. Inspired by the implicit scene representation in human navigation, analogous to the left brain's semantic understanding and the right brain's spatial cognition, we propose JanusVLN, a novel VLN framework featuring a dual implicit neural memory that models spatial-geometric and visual-semantic memory as separate, compact, and fixed-size neural representations. This framework first extends the MLLM to incorporate 3D prior knowledge from the spatial-geometric encoder, thereby enhancing the spatial reasoning capabilities of models based solely on RGB input. Then, the historical key-value caches from the spatial-geometric and visual-semantic encoders are constructed into a dual implicit memory. By retaining only the KVs of tokens in the initial and sliding window, redundant computation is avoided, enabling efficient incremental updates. Extensive experiments demonstrate that JanusVLN outperforms over 20 recent methods to achieve SOTA performance. For example, the success rate improves by 10.5-35.5 compared to methods using multiple data types as input and by 3.6-10.8 compared to methods using more RGB training data. This indicates that the proposed dual implicit neural memory, as a novel paradigm, explores promising new directions for future VLN research. Ours project page: this https URL. 

**Abstract (ZH)**: 基于视觉-语言导航需要一种通过自然语言指令和连续视频流引导的具身代理在未见环境中导航。最近的基于视觉-语言导航的进步由多模态大型语言模型的强大语义理解推动。然而，这些方法通常依赖显式的语义记忆，如构建文本认知地图或存储历史视觉帧。这类方法遭受空间信息损失、计算冗余和内存膨胀的困扰，阻碍了高效的导航。受到人类导航中隐式场景表示的启发，类似于左脑的语义理解与右脑的空间认知，我们提出JanusVLN，一种新颖的基于双隐式神经记忆的视觉-语言导航框架，将空间-几何和视觉-语义记忆建模为各自的紧凑且固定大小的神经表示。该框架首先扩展了多模态大型语言模型，合并空间-几何编码器的3D先验知识，从而增强仅基于RGB输入的模型的时空推理能力。然后，从空间-几何和视觉-语义编码器构建历史键值缓存，并形成双隐式记忆，仅保留初始窗口和滑动窗口中的键值，避免冗余计算，实现高效的增量更新。大量实验结果表明，JanusVLN在性能上超过了20多种近期方法，取得了SOTA表现。例如，成功率为10.5%-35.5%优于使用多种数据类型的输入方法，为使用更多RGB训练数据的方法提高了3.6%-10.8%。这表明，提出的双隐式神经记忆作为一种新的范式，探索了未来基于视觉-语言导航研究的有希望的新方向。我们的项目页面：this https URL。 

---
# Learning to Ball: Composing Policies for Long-Horizon Basketball Moves 

**Title (ZH)**: 学习运球：组成面向长时程篮球动作的策略 

**Authors**: Pei Xu, Zhen Wu, Ruocheng Wang, Vishnu Sarukkai, Kayvon Fatahalian, Ioannis Karamouzas, Victor Zordan, C. Karen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.22442)  

**Abstract**: Learning a control policy for a multi-phase, long-horizon task, such as basketball maneuvers, remains challenging for reinforcement learning approaches due to the need for seamless policy composition and transitions between skills. A long-horizon task typically consists of distinct subtasks with well-defined goals, separated by transitional subtasks with unclear goals but critical to the success of the entire task. Existing methods like the mixture of experts and skill chaining struggle with tasks where individual policies do not share significant commonly explored states or lack well-defined initial and terminal states between different phases. In this paper, we introduce a novel policy integration framework to enable the composition of drastically different motor skills in multi-phase long-horizon tasks with ill-defined intermediate states. Based on that, we further introduce a high-level soft router to enable seamless and robust transitions between the subtasks. We evaluate our framework on a set of fundamental basketball skills and challenging transitions. Policies trained by our approach can effectively control the simulated character to interact with the ball and accomplish the long-horizon task specified by real-time user commands, without relying on ball trajectory references. 

**Abstract (ZH)**: 一种新型策略集成框架：在具有非明确中间状态的多阶段长期任务中组成截然不同的运动技能及其平滑过渡 

---
# EMMA: Generalizing Real-World Robot Manipulation via Generative Visual Transfer 

**Title (ZH)**: EMMA: 通过生成视觉转移实现通用现实世界机器人操作 

**Authors**: Zhehao Dong, Xiaofeng Wang, Zheng Zhu, Yirui Wang, Yang Wang, Yukun Zhou, Boyuan Wang, Chaojun Ni, Runqi Ouyang, Wenkang Qin, Xinze Chen, Yun Ye, Guan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22407)  

**Abstract**: Vision-language-action (VLA) models increasingly rely on diverse training data to achieve robust generalization. However, collecting large-scale real-world robot manipulation data across varied object appearances and environmental conditions remains prohibitively time-consuming and expensive. To overcome this bottleneck, we propose Embodied Manipulation Media Adaptation (EMMA), a VLA policy enhancement framework that integrates a generative data engine with an effective training pipeline. We introduce DreamTransfer, a diffusion Transformer-based framework for generating multi-view consistent, geometrically grounded embodied manipulation videos. DreamTransfer enables text-controlled visual editing of robot videos, transforming foreground, background, and lighting conditions without compromising 3D structure or geometrical plausibility. Furthermore, we explore hybrid training with real and generated data, and introduce AdaMix, a hard-sample-aware training strategy that dynamically reweights training batches to focus optimization on perceptually or kinematically challenging samples. Extensive experiments show that videos generated by DreamTransfer significantly outperform prior video generation methods in multi-view consistency, geometric fidelity, and text-conditioning accuracy. Crucially, VLAs trained with generated data enable robots to generalize to unseen object categories and novel visual domains using only demonstrations from a single appearance. In real-world robotic manipulation tasks with zero-shot visual domains, our approach achieves over a 200% relative performance gain compared to training on real data alone, and further improves by 13% with AdaMix, demonstrating its effectiveness in boosting policy generalization. 

**Abstract (ZH)**: 视觉-语言-动作（VLA）模型 increasingly 依赖多样化训练数据以实现稳健的泛化。然而，跨不同物体外观和环境条件收集大规模真实机器人操作数据仍然是时间上和经济上极为耗时且昂贵的。为了克服这一瓶颈，我们提出了一种物质交互媒体适应（EMMA）框架，该框架将生成数据引擎与有效的训练管道集成起来，增强VLA策略。我们引入了DreamTransfer，这是一种基于扩散Transformer的框架，用于生成多视图一致、几何上可信的物质交互视频。DreamTransfer允许对机器人视频进行文本控制的视觉编辑，可以改变前景、背景和光照条件而不损害三维结构或几何合理性。此外，我们探索了真实数据与生成数据的混合训练，并引入了AdaMix，这是一种感知或动力学上具有挑战认样样本的自适应训练策略，动态调整训练批次的权重以聚焦于优化难点样本。广泛实验证明，由DreamTransfer生成的视频在多视图一致性、几何保真度和文本条件准确性方面显著优于先前的视频生成方法。最关键的是，使用生成数据训练的VLA能够在仅从单一外观的演示中泛化到未见过的对象类别和新的视觉域。在零样本视觉域的现实世界机器人操作任务中，与仅使用真实数据训练相比，我们的方法实现了超过200%的相对性能提升，并且与AdaMix结合使用时再提升了13%，证明了其提高策略泛化效果的有效性。 

---
# ReLAM: Learning Anticipation Model for Rewarding Visual Robotic Manipulation 

**Title (ZH)**: ReLAM: 学习预判模型以奖励视觉机器人操作 

**Authors**: Nan Tang, Jing-Cheng Pang, Guanlin Li, Chao Qian, Yang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.22402)  

**Abstract**: Reward design remains a critical bottleneck in visual reinforcement learning (RL) for robotic manipulation. In simulated environments, rewards are conventionally designed based on the distance to a target position. However, such precise positional information is often unavailable in real-world visual settings due to sensory and perceptual limitations. In this study, we propose a method that implicitly infers spatial distances through keypoints extracted from images. Building on this, we introduce Reward Learning with Anticipation Model (ReLAM), a novel framework that automatically generates dense, structured rewards from action-free video demonstrations. ReLAM first learns an anticipation model that serves as a planner and proposes intermediate keypoint-based subgoals on the optimal path to the final goal, creating a structured learning curriculum directly aligned with the task's geometric objectives. Based on the anticipated subgoals, a continuous reward signal is provided to train a low-level, goal-conditioned policy under the hierarchical reinforcement learning (HRL) framework with provable sub-optimality bound. Extensive experiments on complex, long-horizon manipulation tasks show that ReLAM significantly accelerates learning and achieves superior performance compared to state-of-the-art methods. 

**Abstract (ZH)**: 视觉强化学习（RL）中用于机器人操作的奖励设计仍然是一个关键瓶颈。在模拟环境中，奖励通常基于到达目标位置的距离进行设计。然而，在现实世界的视觉设置中，由于感觉和知觉的限制，这种精确的位置信息往往不可用。在本研究中，我们提出了一种通过从图像中提取的关键点隐式推断空间距离的方法。在此基础上，我们引入了预见奖励学习模型（ReLAM），这是一种新型框架，可以从无动作视频演示中自动生成密集的结构化奖励。ReLAM 首先学习一个预见模型，作为规划器，在最优路径到最终目标的过程中提出基于关键点的中间子目标，从而直接构建与任务几何目标对齐的结构化学习课程。基于预见的子目标，提供连续的奖励信号，在分层强化学习（HRL）框架下训练低层次的目标条件策略，并具有可证明的次优性界。在复杂的长时程操作任务上的广泛实验表明，ReLAM 显著加速了学习并达到了优于现有方法的性能。 

---
# Human Autonomy and Sense of Agency in Human-Robot Interaction: A Systematic Literature Review 

**Title (ZH)**: 人类自主性和主观操控感在人机交互中的作用：一项系统文献综述 

**Authors**: Felix Glawe, Tim Schmeckel, Philipp Brauner, Martina Ziefle  

**Link**: [PDF](https://arxiv.org/pdf/2509.22271)  

**Abstract**: Human autonomy and sense of agency are increasingly recognised as critical for user well-being, motivation, and the ethical deployment of robots in human-robot interaction (HRI). Given the rapid development of artificial intelligence, robot capabilities and their potential to function as colleagues and companions are growing. This systematic literature review synthesises 22 empirical studies selected from an initial pool of 728 articles published between 2011 and 2024. Articles were retrieved from major scientific databases and identified based on empirical focus and conceptual relevance, namely, how to preserve and promote human autonomy and sense of agency in HRI. Derived through thematic synthesis, five clusters of potentially influential factors are revealed: robot adaptiveness, communication style, anthropomorphism, presence of a robot and individual differences. Measured through psychometric scales or the intentional binding paradigm, perceptions of autonomy and agency varied across industrial, educational, healthcare, care, and hospitality settings. The review underscores the theoretical differences between both concepts, but their yet entangled use in HRI. Despite increasing interest, the current body of empirical evidence remains limited and fragmented, underscoring the necessity for standardised definitions, more robust operationalisations, and further exploratory and qualitative research. By identifying existing gaps and highlighting emerging trends, this review contributes to the development of human-centered, autonomy-supportive robot design strategies that uphold ethical and psychological principles, ultimately supporting well-being in human-robot interaction. 

**Abstract (ZH)**: 人类自主性与代理感在用户福祉、动机及机器人在人机交互中的伦理部署中的重要性日益受到认可。鉴于人工智能的快速发展，机器人的能力和其作为同事和伴侣的潜力正在增长。本系统文献综述综合分析了2011年至2024年间发表的728篇文章中筛选出的22篇实证研究，探讨如何在人机交互中保存和促进人类的自主性与代理感。通过主题综合分析，揭示了五个可能有影响力的因素群：机器人适应性、沟通方式、拟人化、机器人的存在以及个体差异。通过心理测量量表或意图绑定范式评估，人类在工业、教育、健康护理、照护和酒店业等不同场景中的自主性和代理感感知有所不同。综述突出了这两个概念在理论上的差异，但它们在人机交互中仍然纠缠不清地使用。尽管人们对这一领域越来越感兴趣，但目前的实证证据仍然有限且碎片化，强调需要标准化定义、更稳健的操作化以及进一步的探索性和质性研究。通过识别现有差距并强调新兴趋势，本综述贡献了以人为本、支持自主性的机器人设计策略，最终支持人机交互中的福祉，并遵循伦理和心理原则。 

---
# Lightweight Structured Multimodal Reasoning for Clinical Scene Understanding in Robotics 

**Title (ZH)**: 轻量级结构化多模态推理在机器人临床场景理解中的应用 

**Authors**: Saurav Jha, Stefan K. Ehrlich  

**Link**: [PDF](https://arxiv.org/pdf/2509.22014)  

**Abstract**: Healthcare robotics requires robust multimodal perception and reasoning to ensure safety in dynamic clinical environments. Current Vision-Language Models (VLMs) demonstrate strong general-purpose capabilities but remain limited in temporal reasoning, uncertainty estimation, and structured outputs needed for robotic planning. We present a lightweight agentic multimodal framework for video-based scene understanding. Combining the Qwen2.5-VL-3B-Instruct model with a SmolAgent-based orchestration layer, it supports chain-of-thought reasoning, speech-vision fusion, and dynamic tool invocation. The framework generates structured scene graphs and leverages a hybrid retrieval module for interpretable and adaptive reasoning. Evaluations on the Video-MME benchmark and a custom clinical dataset show competitive accuracy and improved robustness compared to state-of-the-art VLMs, demonstrating its potential for applications in robot-assisted surgery, patient monitoring, and decision support. 

**Abstract (ZH)**: healthcare 机器人要求具备 robust 多模态感知与推理能力，以确保在动态临床环境中的安全性。当前的 Vision-Language 模型 (VLMs) 展现出强大的通用能力，但在时间推理、不确定性估计和所需供机器人规划的结构化输出方面仍存在局限性。我们提出了一种轻量级的 agentic 多模态框架，用于基于视频的场景理解。通过将 Qwen2.5-VL-3B-Instruct 模型与基于 SmolAgent 的编排层结合，该框架支持链式思考推理、语音-视觉融合以及动态工具调用。该框架生成结构化的场景图，并利用混合检索模块实现可解释和自适应推理。在 Video-MME 基准和自定义临床数据集上的评估表明，其准确性和鲁棒性优于当前最先进的 VLMs，展示了其在机器人辅助手术、患者监测和决策支持方面的潜在应用。 

---
# DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation 

**Title (ZH)**: DynaNav: 动态特征与层选择的高效视觉导航 

**Authors**: Jiahui Wang, Changhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21930)  

**Abstract**: Visual navigation is essential for robotics and embodied AI. However, existing foundation models, particularly those with transformer decoders, suffer from high computational overhead and lack interpretability, limiting their deployment in resource-tight scenarios. To address this, we propose DynaNav, a Dynamic Visual Navigation framework that adapts feature and layer selection based on scene complexity. It employs a trainable hard feature selector for sparse operations, enhancing efficiency and interpretability. Additionally, we integrate feature selection into an early-exit mechanism, with Bayesian Optimization determining optimal exit thresholds to reduce computational cost. Extensive experiments in real-world-based datasets and simulated environments demonstrate the effectiveness of DynaNav. Compared to ViNT, DynaNav achieves a 2.26x reduction in FLOPs, 42.3% lower inference time, and 32.8% lower memory usage, while improving navigation performance across four public datasets. 

**Abstract (ZH)**: 动态可视化导航框架DynaNav在基于场景复杂性的特征和层选择中实现了高效性和可解释性，并通过集成特征选择到早期退出机制中，利用贝叶斯优化确定最优退出阈值以降低计算成本。广泛的实验证明了DynaNav的有效性。与ViNT相比，DynaNav在FLOPs上减少2.26倍，推理时间降低42.3%，内存使用减少32.8%，同时在四个公开数据集上提高了导航性能。 

---
# Guiding Evolution of Artificial Life Using Vision-Language Models 

**Title (ZH)**: 使用视觉-语言模型引导人工生命演化 

**Authors**: Nikhil Baid, Hannah Erlebach, Paul Hellegouarch, Frederico Wieser  

**Link**: [PDF](https://arxiv.org/pdf/2509.22447)  

**Abstract**: Foundation models (FMs) have recently opened up new frontiers in the field of artificial life (ALife) by providing powerful tools to automate search through ALife simulations. Previous work aligns ALife simulations with natural language target prompts using vision-language models (VLMs). We build on Automated Search for Artificial Life (ASAL) by introducing ASAL++, a method for open-ended-like search guided by multimodal FMs. We use a second FM to propose new evolutionary targets based on a simulation's visual history. This induces an evolutionary trajectory with increasingly complex targets.
We explore two strategies: (1) evolving a simulation to match a single new prompt at each iteration (Evolved Supervised Targets: EST) and (2) evolving a simulation to match the entire sequence of generated prompts (Evolved Temporal Targets: ETT). We test our method empirically in the Lenia substrate using Gemma-3 to propose evolutionary targets, and show that EST promotes greater visual novelty, while ETT fosters more coherent and interpretable evolutionary sequences.
Our results suggest that ASAL++ points towards new directions for FM-driven ALife discovery with open-ended characteristics. 

**Abstract (ZH)**: 基于模型（FMs）在人工生命（ALife）领域的新兴前沿：通过提供自动搜索ALife模拟的强大工具，FMs正在开辟新的方向。我们在此基础上引入ASAL++方法，该方法利用多模态FMs进行类似开放搜索的引导。我们使用第二个FM根据模拟的视觉历史提出新的进化目标，从而引发越来越复杂的进化轨迹。我们探索了两种策略：（1）每次迭代进化模拟以匹配单个新提示（进化监督目标：EST）；（2）进化模拟以匹配整个生成提示序列（进化时间目标：ETT）。我们在Lenia基质中使用Gemma-3实验测试了我们的方法，并表明EST促进了更大的视觉新颖性，而ETT促进了更连贯和可解释的进化序列。研究结果表明，ASAL++指出了具有开放性特征的FM驱动ALife发现的新方向。 

---
# CoBel-World: Harnessing LLM Reasoning to Build a Collaborative Belief World for Optimizing Embodied Multi-Agent Collaboration 

**Title (ZH)**: CoBel-World: 利用大规模语言模型推理构建协作信念世界以优化具身多智能体协作 

**Authors**: Zhimin Wang, Shaokang He, Duo Wu, Jinghe Wang, Linjia Kang, Jing Yu, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21981)  

**Abstract**: Effective real-world multi-agent collaboration requires not only accurate planning but also the ability to reason about collaborators' intents -- a crucial capability for avoiding miscoordination and redundant communication under partial observable environments. Due to their strong planning and reasoning capabilities, large language models (LLMs) have emerged as promising autonomous agents for collaborative task solving. However, existing collaboration frameworks for LLMs overlook their reasoning potential for dynamic intent inference, and thus produce inconsistent plans and redundant communication, reducing collaboration efficiency. To bridge this gap, we propose CoBel-World, a novel framework that equips LLM agents with a collaborative belief world -- an internal representation jointly modeling the physical environment and collaborators' mental states. CoBel-World enables agents to parse open-world task knowledge into structured beliefs via a symbolic belief language, and perform zero-shot Bayesian-style belief updates through LLM reasoning. This allows agents to proactively detect potential miscoordination (e.g., conflicting plans) and communicate adaptively. Evaluated on challenging embodied benchmarks (i.e., TDW-MAT and C-WAH), CoBel-World significantly reduces communication costs by 22-60% and improves task completion efficiency by 4-28% compared to the strongest baseline. Our results show that explicit, intent-aware belief modeling is essential for efficient and human-like collaboration in LLM-based multi-agent systems. 

**Abstract (ZH)**: Effective Real-World 多agent 协作需要准确的规划能力和推理能力——在部分可观测环境中避免误协调和冗余通信的关键能力。由于大型语言模型（LLMs）的强大规划和推理能力，它们已成为协作任务解决的有前途的自主代理。然而，现有的LLM协作框架忽视了它们动态意图推理的潜力，从而产生了不一致的计划和冗余通信，降低了协作效率。为了弥补这一差距，我们提出了 CoBel-World，一种新型框架，为LLM代理配备协作信念世界——一种同时建模物理环境和协作者心理状态的内部表示。CoBel-World 使代理能够通过符号信念语言将开放世界的任务知识解析为结构化的信念，并通过LLM推理进行零样本贝叶斯式信念更新。这使代理能够主动检测潜在的误协调（例如，冲突的计划）并适应性地通信。在具有挑战性的体感基准测试（即TDW-MAT和C-WAH）上，CoBel-World 将通信成本降低了22-60%，提高了4-28%的任务完成效率，相较最强基线。我们的结果表明，显式的、意图感知的信念建模对于基于LLM的多agent系统中的高效和类人协作至关重要。 

---
# DeepTravel: An End-to-End Agentic Reinforcement Learning Framework for Autonomous Travel Planning Agents 

**Title (ZH)**: DeepTravel：自主旅行规划代理的端到端代理 reinforcement 学习框架 

**Authors**: Yansong Ning, Rui Liu, Jun Wang, Kai Chen, Wei Li, Jun Fang, Kan Zheng, Naiqiang Tan, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21842)  

**Abstract**: Travel planning (TP) agent has recently worked as an emerging building block to interact with external tools and resources for travel itinerary generation, ensuring enjoyable user experience. Despite its benefits, existing studies rely on hand craft prompt and fixed agent workflow, hindering more flexible and autonomous TP agent. This paper proposes DeepTravel, an end to end agentic reinforcement learning framework for building autonomous travel planning agent, capable of autonomously planning, executing tools, and reflecting on tool responses to explore, verify, and refine intermediate actions in multi step reasoning. To achieve this, we first construct a robust sandbox environment by caching transportation, accommodation and POI data, facilitating TP agent training without being constrained by real world APIs limitations (e.g., inconsistent outputs). Moreover, we develop a hierarchical reward modeling system, where a trajectory level verifier first checks spatiotemporal feasibility and filters unsatisfied travel itinerary, and then the turn level verifier further validate itinerary detail consistency with tool responses, enabling efficient and precise reward service. Finally, we propose the reply augmented reinforcement learning method that enables TP agent to periodically replay from a failures experience buffer, emerging notable agentic capacity. We deploy trained TP agent on DiDi Enterprise Solutions App and conduct comprehensive online and offline evaluations, demonstrating that DeepTravel enables small size LLMs (e.g., Qwen3 32B) to significantly outperform existing frontier LLMs such as OpenAI o1, o3 and DeepSeek R1 in travel planning tasks. 

**Abstract (ZH)**: 基于深度学习的自主旅行规划代理框架：DeepTravel 

---
# D-Artemis: A Deliberative Cognitive Framework for Mobile GUI Multi-Agents 

**Title (ZH)**: D-Artemis: 一种移动GUI多代理的反思性认知框架 

**Authors**: Hongze Mi, Yibo Feng, Wenjie Lu, Yuqi Wang, Jinyuan Li, Song Cao, He Cui, Tengfei Tian, Xuelin Zhang, Haotian Luo, Di Sun, Naiqiang Tan, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.21799)  

**Abstract**: Graphical User Interface (GUI) agents aim to automate a wide spectrum of human tasks by emulating user interaction. Despite rapid advancements, current approaches are hindered by several critical challenges: data bottleneck in end-to-end training, high cost of delayed error detection, and risk of contradictory guidance. Inspired by the human cognitive loop of Thinking, Alignment, and Reflection, we present D-Artemis -- a novel deliberative framework in this paper. D-Artemis leverages a fine-grained, app-specific tip retrieval mechanism to inform its decision-making process. It also employs a proactive Pre-execution Alignment stage, where Thought-Action Consistency (TAC) Check module and Action Correction Agent (ACA) work in concert to mitigate the risk of execution failures. A post-execution Status Reflection Agent (SRA) completes the cognitive loop, enabling strategic learning from experience. Crucially, D-Artemis enhances the capabilities of general-purpose Multimodal large language models (MLLMs) for GUI tasks without the need for training on complex trajectory datasets, demonstrating strong generalization. D-Artemis establishes new state-of-the-art (SOTA) results across both major benchmarks, achieving a 75.8% success rate on AndroidWorld and 96.8% on ScreenSpot-V2. Extensive ablation studies further demonstrate the significant contribution of each component to the framework. 

**Abstract (ZH)**: 图形用户界面（GUI）代理旨在通过模拟用户交互来自动化广泛的 humano 任务。受人类认知循环（思考、对齐和反思）的启发，我们提出了 D-Artemis —— 一种新颖的反思框架。D-Artemis 利用细粒度的、特定于应用程序的提示检索机制来指导其决策过程。它还采用了一种主动的预执行对齐阶段，其中思路-行动一致性（TAC）检查模块和行动纠正代理（ACA）协同工作，以减轻执行失败的风险。在执行后的状态反思代理（SRA）完成认知循环，使代理能够从经验中进行战略学习。关键的是，D-Artemis 无需针对复杂轨迹数据集进行训练，即可增强通用多模态大语言模型（MLLM）在 GUI 任务中的能力，显示出强大的泛化能力。D-Artemis 在两个主要基准测试中建立了新的最佳结果，在 AndroidWorld 中达到了 75.8% 的成功率，在 ScreenSpot-V2 中达到了 96.8%。广泛的消融研究进一步证明了框架中每个组件的重要贡献。 

---
# Can AI Perceive Physical Danger and Intervene? 

**Title (ZH)**: AI能否感知物理危险并干预？ 

**Authors**: Abhishek Jindal, Dmitry Kalashnikov, Oscar Chang, Divya Garikapati, Anirudha Majumdar, Pierre Sermanet, Vikas Sindhwani  

**Link**: [PDF](https://arxiv.org/pdf/2509.21651)  

**Abstract**: When AI interacts with the physical world -- as a robot or an assistive agent -- new safety challenges emerge beyond those of purely ``digital AI". In such interactions, the potential for physical harm is direct and immediate. How well do state-of-the-art foundation models understand common-sense facts about physical safety, e.g. that a box may be too heavy to lift, or that a hot cup of coffee should not be handed to a child? In this paper, our contributions are three-fold: first, we develop a highly scalable approach to continuous physical safety benchmarking of Embodied AI systems, grounded in real-world injury narratives and operational safety constraints. To probe multi-modal safety understanding, we turn these narratives and constraints into photorealistic images and videos capturing transitions from safe to unsafe states, using advanced generative models. Secondly, we comprehensively analyze the ability of major foundation models to perceive risks, reason about safety, and trigger interventions; this yields multi-faceted insights into their deployment readiness for safety-critical agentic applications. Finally, we develop a post-training paradigm to teach models to explicitly reason about embodiment-specific safety constraints provided through system instructions. The resulting models generate thinking traces that make safety reasoning interpretable and transparent, achieving state of the art performance in constraint satisfaction evaluations. The benchmark will be released at this https URL 

**Abstract (ZH)**: 当AI与物理世界交互——作为机器人或辅助代理时，新的安全挑战在纯粹的“数字AI”之外浮现。在这些交互中，物理伤害的可能性是直接且即时的。当前最先进的基础模型是否理解关于物理安全的常识性事实，例如一个箱子可能太重而无法提起，或者不应该将热咖啡杯递给小孩？在本文中，我们的贡献有三个方面：首先，我们开发了一种高度可扩展的方法，通过基于真实世界的伤害叙事和操作安全约束来进行体感AI系统的持续物理安全基准测试。为了探究多模态安全理解，我们将这些叙事和约束转化为逼真的图像和视频，捕捉从安全状态到不安全状态的过渡，使用高级生成模型。其次，我们全面分析了主要基础模型感知风险、推理安全以及触发干预措施的能力；这为我们提供了关于其在关键安全任务中部署准备情况的多方面洞见。最后，我们开发了一种后训练范式，教导模型根据系统指令具体推理关于体感特定的安全约束。结果模型生成的推理痕迹使安全推理变得可解释和透明，并在约束满足评估中达到了最先进的性能。基准测试将在以下链接发布：this https URL。 

---
# An Ontology for Unified Modeling of Tasks, Actions, Environments, and Capabilities in Personal Service Robotics 

**Title (ZH)**: 统一建模任务、行为、环境和能力的本体论研究 

**Authors**: Margherita Martorana, Francesca Urgese, Ilaria Tiddi, Stefan Schlobach  

**Link**: [PDF](https://arxiv.org/pdf/2509.22434)  

**Abstract**: Personal service robots are increasingly used in domestic settings to assist older adults and people requiring support. Effective operation involves not only physical interaction but also the ability to interpret dynamic environments, understand tasks, and choose appropriate actions based on context. This requires integrating both hardware components (e.g. sensors, actuators) and software systems capable of reasoning about tasks, environments, and robot capabilities. Frameworks such as the Robot Operating System (ROS) provide open-source tools that help connect low-level hardware with higher-level functionalities. However, real-world deployments remain tightly coupled to specific platforms. As a result, solutions are often isolated and hard-coded, limiting interoperability, reusability, and knowledge sharing. Ontologies and knowledge graphs offer a structured way to represent tasks, environments, and robot capabilities. Existing ontologies, such as the Socio-physical Model of Activities (SOMA) and the Descriptive Ontology for Linguistic and Cognitive Engineering (DOLCE), provide models for activities, spatial relationships, and reasoning structures. However, they often focus on specific domains and do not fully capture the connection between environment, action, robot capabilities, and system-level integration. In this work, we propose the Ontology for roBOts and acTions (OntoBOT), which extends existing ontologies to provide a unified representation of tasks, actions, environments, and capabilities. Our contributions are twofold: (1) we unify these aspects into a cohesive ontology to support formal reasoning about task execution, and (2) we demonstrate its generalizability by evaluating competency questions across four embodied agents - TIAGo, HSR, UR3, and Stretch - showing how OntoBOT enables context-aware reasoning, task-oriented execution, and knowledge sharing in service robotics. 

**Abstract (ZH)**: 基于行动的机器人本体（OntoBOT）：统一任务、行动、环境和能力的本体表示及其在服务机器人中的应用 

---
# Context and Diversity Matter: The Emergence of In-Context Learning in World Models 

**Title (ZH)**: 情境和多样性很重要：世界模型中基于上下文的学习的出现 

**Authors**: Fan Wang, Zhiyuan Chen, Yuxuan Zhong, Sunjian Zheng, Pengtao Shao, Bo Yu, Shaoshan Liu, Jianan Wang, Ning Ding, Yang Cao, Yu Kang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22353)  

**Abstract**: The capability of predicting environmental dynamics underpins both biological neural systems and general embodied AI in adapting to their surroundings. Yet prevailing approaches rest on static world models that falter when confronted with novel or rare configurations. We investigate in-context environment learning (ICEL), shifting attention from zero-shot performance to the growth and asymptotic limits of the world model. Our contributions are three-fold: (1) we formalize in-context learning of a world model and identify two core mechanisms: environment recognition and environment learning; (2) we derive error upper-bounds for both mechanisms that expose how the mechanisms emerge; and (3) we empirically confirm that distinct ICL mechanisms exist in the world model, and we further investigate how data distribution and model architecture affect ICL in a manner consistent with theory. These findings demonstrate the potential of self-adapting world models and highlight the key factors behind the emergence of ICEL, most notably the necessity of long context and diverse environments. 

**Abstract (ZH)**: 在上下文中学贯环境的潜力及其关键因素 

---
# Adaptive Policy Backbone via Shared Network 

**Title (ZH)**: 自适应策略骨干网通过共享网络 

**Authors**: Bumgeun Park, Donghwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.22310)  

**Abstract**: Reinforcement learning (RL) has achieved impressive results across domains, yet learning an optimal policy typically requires extensive interaction data, limiting practical deployment. A common remedy is to leverage priors, such as pre-collected datasets or reference policies, but their utility degrades under task mismatch between training and deployment. While prior work has sought to address this mismatch, it has largely been restricted to in-distribution settings. To address this challenge, we propose Adaptive Policy Backbone (APB), a meta-transfer RL method that inserts lightweight linear layers before and after a shared backbone, thereby enabling parameter-efficient fine-tuning (PEFT) while preserving prior knowledge during adaptation. Our results show that APB improves sample efficiency over standard RL and adapts to out-of-distribution (OOD) tasks where existing meta-RL baselines typically fail. 

**Abstract (ZH)**: 强化学习（RL）已在多个领域取得了显著成果，但学习最优策略通常需要大量的交互数据，限制了其实用部署。一种常见的解决方法是利用先验知识，如预先收集的数据集或参考策略，但这些方法在训练和部署之间的任务不匹配情况下效用会下降。虽然已有工作试图解决这一问题，但主要集中在同分布设置中。为应对这一挑战，我们提出了自适应策略主干（APB），这是一种元迁移RL方法，在共享主干前后插入轻量级线性层，从而实现参数高效微调（PEFT）并在适应过程中保留先验知识。我们的结果显示，APB在标准RL方法上提高了样本效率，并能够适应现有元RL基线通常会失败的跨分布（OOD）任务。 

---
# Teaching AI to Feel: A Collaborative, Full-Body Exploration of Emotive Communication 

**Title (ZH)**: 教AI感受：一种协作式的全身探索情感交流 

**Authors**: Esen K. Tütüncü, Lissette Lemus, Kris Pilcher, Holger Sprengel, Jordi Sabater-Mir  

**Link**: [PDF](https://arxiv.org/pdf/2509.22168)  

**Abstract**: Commonaiverse is an interactive installation exploring human emotions through full-body motion tracking and real-time AI feedback. Participants engage in three phases: Teaching, Exploration and the Cosmos Phase, collaboratively expressing and interpreting emotions with the system. The installation integrates MoveNet for precise motion tracking and a multi-recommender AI system to analyze emotional states dynamically, responding with adaptive audiovisual outputs. By shifting from top-down emotion classification to participant-driven, culturally diverse definitions, we highlight new pathways for inclusive, ethical affective computing. We discuss how this collaborative, out-of-the-box approach pushes multimedia research beyond single-user facial analysis toward a more embodied, co-created paradigm of emotional AI. Furthermore, we reflect on how this reimagined framework fosters user agency, reduces bias, and opens avenues for advanced interactive applications. 

**Abstract (ZH)**: Commonaiverse：通过全身动作追踪和实时AI反馈探索人类情绪的互动安装艺术 

---
# Developing Vision-Language-Action Model from Egocentric Videos 

**Title (ZH)**: 基于第一人称视频发展视觉-语言-动作模型 

**Authors**: Tomoya Yoshida, Shuhei Kurita, Taichi Nishimura, Shinsuke Mori  

**Link**: [PDF](https://arxiv.org/pdf/2509.21986)  

**Abstract**: Egocentric videos capture how humans manipulate objects and tools, providing diverse motion cues for learning object manipulation. Unlike the costly, expert-driven manual teleoperation commonly used in training Vision-Language-Action models (VLAs), egocentric videos offer a scalable alternative. However, prior studies that leverage such videos for training robot policies typically rely on auxiliary annotations, such as detailed hand-pose recordings. Consequently, it remains unclear whether VLAs can be trained directly from raw egocentric videos. In this work, we address this challenge by leveraging EgoScaler, a framework that extracts 6DoF object manipulation trajectories from egocentric videos without requiring auxiliary recordings. We apply EgoScaler to four large-scale egocentric video datasets and automatically refine noisy or incomplete trajectories, thereby constructing a new large-scale dataset for VLA pre-training. Our experiments with a state-of-the-art $\pi_0$ architecture in both simulated and real-robot environments yield three key findings: (i) pre-training on our dataset improves task success rates by over 20\% compared to training from scratch, (ii) the performance is competitive with that achieved using real-robot datasets, and (iii) combining our dataset with real-robot data yields further improvements. These results demonstrate that egocentric videos constitute a promising and scalable resource for advancing VLA research. 

**Abstract (ZH)**: 自视点视频捕捉人类操作物体和工具的方式，提供了学习物体操作的多样性运动线索。与训练视觉-语言-动作模型（VLAs）过程中常见的昂贵且由专家驱动的手动远程操作不同，自视点视频提供了可扩展的替代方案。然而，之前利用此类视频训练机器人策略的研究通常依赖于辅助标注，如详细的手部姿态记录。因此，仍然不清楚VLAs是否可以直接从原始自视点视频中训练得到。在本工作中，我们通过利用EgoScaler框架解决了这一挑战，该框架可以从自视点视频中提取6自由度的物体操作轨迹，而无需辅助记录。我们将EgoScaler应用于四个大规模自视点视频数据集，并自动精炼噪音或不完整轨迹，从而构建了一个新的大型数据集用于VLAs的预训练。我们使用最先进的$\pi_0$架构在模拟和实际机器人环境中进行的实验揭示了三个关键发现：（i）在我们的数据集上预训练比从头开始训练的任务成功率提高了超过20%；（ii）性能与使用实际机器人数据集所取得的性能相当；（iii）将我们的数据集与实际机器人数据集结合使用进一步提高了性能。这些结果表明，自视点视频是推动VLAs研究的一种有前景且可扩展的资源。 

---
# DiTraj: training-free trajectory control for video diffusion transformer 

**Title (ZH)**: DiTraj: 无需训练的动力学轨迹控制视频扩散变换器 

**Authors**: Cheng Lei, Jiayu Zhang, Yue Ma, Xinyu Wang, Long Chen, Liang Tang, Yiqiang Yan, Fei Su, Zhicheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.21839)  

**Abstract**: Diffusion Transformers (DiT)-based video generation models with 3D full attention exhibit strong generative capabilities. Trajectory control represents a user-friendly task in the field of controllable video generation. However, existing methods either require substantial training resources or are specifically designed for U-Net, do not take advantage of the superior performance of DiT. To address these issues, we propose DiTraj, a simple but effective training-free framework for trajectory control in text-to-video generation, tailored for DiT. Specifically, first, to inject the object's trajectory, we propose foreground-background separation guidance: we use the Large Language Model (LLM) to convert user-provided prompts into foreground and background prompts, which respectively guide the generation of foreground and background regions in the video. Then, we analyze 3D full attention and explore the tight correlation between inter-token attention scores and position embedding. Based on this, we propose inter-frame Spatial-Temporal Decoupled 3D-RoPE (STD-RoPE). By modifying only foreground tokens' position embedding, STD-RoPE eliminates their cross-frame spatial discrepancies, strengthening cross-frame attention among them and thus enhancing trajectory control. Additionally, we achieve 3D-aware trajectory control by regulating the density of position embedding. Extensive experiments demonstrate that our method outperforms previous methods in both video quality and trajectory controllability. 

**Abstract (ZH)**: 基于DiT的具有3D全注意力的视频生成模型展示了强大的生成能力。针对轨迹控制任务，我们提出DiTraj——一种适用于DiT的简单有效的无需训练框架，以实现文本到视频生成的轨迹控制。 

---
# Guiding Audio Editing with Audio Language Model 

**Title (ZH)**: 指导音频编辑的音频语言模型 

**Authors**: Zitong Lan, Yiduo Hao, Mingmin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.21625)  

**Abstract**: Audio editing plays a central role in VR/AR immersion, virtual conferencing, sound design, and other interactive media. However, recent generative audio editing models depend on template-like instruction formats and are restricted to mono-channel audio. These models fail to deal with declarative audio editing, where the user declares what the desired outcome should be, while leaving the details of editing operations to the system. We introduce SmartDJ, a novel framework for stereo audio editing that combines the reasoning capability of audio language models with the generative power of latent diffusion. Given a high-level instruction, SmartDJ decomposes it into a sequence of atomic edit operations, such as adding, removing, or spatially relocating events. These operations are then executed by a diffusion model trained to manipulate stereo audio. To support this, we design a data synthesis pipeline that produces paired examples of high-level instructions, atomic edit operations, and audios before and after each edit operation. Experiments demonstrate that SmartDJ achieves superior perceptual quality, spatial realism, and semantic alignment compared to prior audio editing methods. Demos are available at this https URL. 

**Abstract (ZH)**: 智能混音师：一种结合音频语言模型推理能力和潜在扩散生成力量的立体声音频编辑框架 

---
# Dynamic Multi-Target Fusion for Efficient Audio-Visual Navigation 

**Title (ZH)**: 高效的音频-视觉多目标动态融合导航 

**Authors**: Yinfeng Yu, Hailong Zhang, Meiling Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21377)  

**Abstract**: Audiovisual embodied navigation enables robots to locate audio sources by dynamically integrating visual observations from onboard sensors with the auditory signals emitted by the target. The core challenge lies in effectively leveraging multimodal cues to guide navigation. While prior works have explored basic fusion of visual and audio data, they often overlook deeper perceptual context. To address this, we propose the Dynamic Multi-Target Fusion for Efficient Audio-Visual Navigation (DMTF-AVN). Our approach uses a multi-target architecture coupled with a refined Transformer mechanism to filter and selectively fuse cross-modal information. Extensive experiments on the Replica and Matterport3D datasets demonstrate that DMTF-AVN achieves state-of-the-art performance, outperforming existing methods in success rate (SR), path efficiency (SPL), and scene adaptation (SNA). Furthermore, the model exhibits strong scalability and generalizability, paving the way for advanced multimodal fusion strategies in robotic navigation. The code and videos are available at
this https URL. 

**Abstract (ZH)**: 视听 embodied 导航使机器人能够通过动态整合搭载传感器的视觉观察与目标发出的听觉信号来定位声源。核心挑战在于有效利用多模态线索来引导导航。尽管先前的工作已经探索了视觉和听觉数据的基本融合，但往往忽视了更深入的感知上下文。为解决这一问题，我们提出了动态多目标融合用于高效视听导航（DMTF-AVN）。我们的方法采用多目标架构并结合精细的Transformer机制来筛选和选择性融合跨模态信息。在Replica和Matterport3D数据集上的 extensive 实验表明，DMTF-AVN 达到了最先进的性能，在成功率达（SR）、路径效率（SPL）和场景适应性（SNA）方面超越了现有方法。此外，模型具有强大的可扩展性和通用性，为机械臂导航中的高级多模态融合策略铺平了道路。代码和视频可在以下链接获取：this https URL。 

---
