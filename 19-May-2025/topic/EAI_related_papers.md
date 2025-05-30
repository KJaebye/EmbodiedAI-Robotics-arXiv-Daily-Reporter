# Bracing for Impact: Robust Humanoid Push Recovery and Locomotion with Reduced Order Models 

**Title (ZH)**: 预判冲击：基于降阶模型的鲁棒人形推倒恢复与移动控制 

**Authors**: Lizhi Yang, Blake Werner, Adrian B.Ghansah, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2505.11495)  

**Abstract**: Push recovery during locomotion will facilitate the deployment of humanoid robots in human-centered environments. In this paper, we present a unified framework for walking control and push recovery for humanoid robots, leveraging the arms for push recovery while dynamically walking. The key innovation is to use the environment, such as walls, to facilitate push recovery by combining Single Rigid Body model predictive control (SRB-MPC) with Hybrid Linear Inverted Pendulum (HLIP) dynamics to enable robust locomotion, push detection, and recovery by utilizing the robot's arms to brace against such walls and dynamically adjusting the desired contact forces and stepping patterns. Extensive simulation results on a humanoid robot demonstrate improved perturbation rejection and tracking performance compared to HLIP alone, with the robot able to recover from pushes up to 100N for 0.2s while walking at commanded speeds up to 0.5m/s. Robustness is further validated in scenarios with angled walls and multi-directional pushes. 

**Abstract (ZH)**: 在人本环境中，步行过程中的推力恢复将促进类人机器人部署。本文提出了一种结合使用双臂进行推力恢复的类人机器人步行控制和推力恢复统一框架，利用单一刚体模型预测控制（SRB-MPC）与混合线性倒 pendulum（HLIP）动力学结合技术，在动态行走过程中利用环境（如墙壁）辅助推力恢复，通过机器人手臂支撑墙壁并动态调整期望接触力和步态模式，实现稳健的行走、推力检测和恢复。仿真结果表明，与仅使用HLIP相比，该方法具有更好的扰动 rejection 和跟踪性能，在行走速度高达0.5m/s的情况下，机器人能够从最大100N的推力中恢复过来。鲁棒性还在斜墙和多方向推力的场景中得到验证。 

---
# SHIELD: Safety on Humanoids via CBFs In Expectation on Learned Dynamics 

**Title (ZH)**: SHIELD: 基于学习的动力学期望值下的CBFs安全性保障用于类人机器人 

**Authors**: Lizhi Yang, Blake Werner, Ryan K. Cosner, David Fridovich-Keil, Preston Culbertson, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2505.11494)  

**Abstract**: Robot learning has produced remarkably effective ``black-box'' controllers for complex tasks such as dynamic locomotion on humanoids. Yet ensuring dynamic safety, i.e., constraint satisfaction, remains challenging for such policies. Reinforcement learning (RL) embeds constraints heuristically through reward engineering, and adding or modifying constraints requires retraining. Model-based approaches, like control barrier functions (CBFs), enable runtime constraint specification with formal guarantees but require accurate dynamics models. This paper presents SHIELD, a layered safety framework that bridges this gap by: (1) training a generative, stochastic dynamics residual model using real-world data from hardware rollouts of the nominal controller, capturing system behavior and uncertainties; and (2) adding a safety layer on top of the nominal (learned locomotion) controller that leverages this model via a stochastic discrete-time CBF formulation enforcing safety constraints in probability. The result is a minimally-invasive safety layer that can be added to the existing autonomy stack to give probabilistic guarantees of safety that balance risk and performance. In hardware experiments on an Unitree G1 humanoid, SHIELD enables safe navigation (obstacle avoidance) through varied indoor and outdoor environments using a nominal (unknown) RL controller and onboard perception. 

**Abstract (ZH)**: SHIELD：一种分层安全框架以实现动态安全的机器人学习控制器 

---
# Learning Multimodal AI Algorithms for Amplifying Limited User Input into High-dimensional Control Space 

**Title (ZH)**: 学习多模态AI算法以放大有限用户输入的高维控制空间 

**Authors**: Ali Rabiee, Sima Ghafoori, MH Farhadi, Robert Beyer, Xiangyu Bai, David J Lin, Sarah Ostadabbas, Reza Abiri  

**Link**: [PDF](https://arxiv.org/pdf/2505.11366)  

**Abstract**: Current invasive assistive technologies are designed to infer high-dimensional motor control signals from severely paralyzed patients. However, they face significant challenges, including public acceptance, limited longevity, and barriers to commercialization. Meanwhile, noninvasive alternatives often rely on artifact-prone signals, require lengthy user training, and struggle to deliver robust high-dimensional control for dexterous tasks. To address these issues, this study introduces a novel human-centered multimodal AI approach as intelligent compensatory mechanisms for lost motor functions that could potentially enable patients with severe paralysis to control high-dimensional assistive devices, such as dexterous robotic arms, using limited and noninvasive inputs. In contrast to the current state-of-the-art (SoTA) noninvasive approaches, our context-aware, multimodal shared-autonomy framework integrates deep reinforcement learning algorithms to blend limited low-dimensional user input with real-time environmental perception, enabling adaptive, dynamic, and intelligent interpretation of human intent for complex dexterous manipulation tasks, such as pick-and-place. The results from our ARAS (Adaptive Reinforcement learning for Amplification of limited inputs in Shared autonomy) trained with synthetic users over 50,000 computer simulation episodes demonstrated the first successful implementation of the proposed closed-loop human-in-the-loop paradigm, outperforming the SoTA shared autonomy algorithms. Following a zero-shot sim-to-real transfer, ARAS was evaluated on 23 human subjects, demonstrating high accuracy in dynamic intent detection and smooth, stable 3D trajectory control for dexterous pick-and-place tasks. ARAS user study achieved a high task success rate of 92.88%, with short completion times comparable to those of SoTA invasive assistive technologies. 

**Abstract (ZH)**: 当前侵入性辅助技术旨在从严重瘫痪患者中推断高维度运动控制信号，但面临公众接受度低、使用寿命有限以及商业化困难等挑战。同时，非侵入性替代方案常常依赖易受干扰的信号、需要长时间用户训练，并且难以提供稳健的高维度控制以执行灵巧任务。为解决这些问题，本研究提出了一种以用户为中心的多模态AI方法，作为失去运动功能的智能补偿机制，有望使严重瘫痪的患者能够使用有限的非侵入性输入控制高维度辅助设备，如灵巧的机械臂。与当前最先进的（SoTA）非侵入性方法不同，我们的上下文感知多模态自助协同时，结合了深度强化学习算法，将有限的低维度用户输入与实时环境感知相结合，从而实现对复杂灵巧操作任务（如拾取和放置）的人类意图的适应性、动态和智能解释。与合成用户训练了50,000个计算机仿真回路后训练的ARAS（适应性强化学习放大有限输入的自助协同时）算法表明，首次成功实现了所提议的闭环人机在网络中的范式，并优于现有的自助协作算法。经过零样本从仿真到现实的迁移，ARAS 在23名人类受试者上进行了评估，显示出在动态意图检测和灵巧拾取放置任务中的平滑、稳定的3D轨迹控制中的高准确性。ARAS用户研究实现了92.88%的任务成功率，完成时间与现有的侵入性辅助技术相当。 

---
# Unveiling the Potential of Vision-Language-Action Models with Open-Ended Multimodal Instructions 

**Title (ZH)**: 探索开放型多模态指令下视觉-语言-动作模型的潜力 

**Authors**: Wei Zhao, Gongsheng Li, Zhefei Gong, Pengxiang Ding, Han Zhao, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11214)  

**Abstract**: Vision-Language-Action (VLA) models have recently become highly prominent in the field of robotics. Leveraging vision-language foundation models trained on large-scale internet data, the VLA model can generate robotic actions directly from visual observations and human instructions through a single end-to-end neural network. Despite their effectiveness, current VLA models usually accept only one form of human prompting, language instructions, which may constrain their applicability in open-ended human-robot interactions. For example, a user might expect the robot to retrieve an object shown in an image, follow an instruction written on the whiteboard, or imitate a behavior demonstrated in a video, rather than relying solely on language-based descriptions. To address this gap, we introduce OE-VLA, which explores the potential of VLA models for open-ended multimodal instructions. Extensive results demonstrate that our OE-VLA not only achieves comparable performance to traditional VLA models with linguistic input but also delivers impressive results across four additional categories of open-ended tasks. The proposed methodology could significantly expand the applications of VLA models across various everyday scenarios and facilitate human-robot interaction. 

**Abstract (ZH)**: Vision-Language-Action (VLA) 模型在机器人领域Recently Became Highly Prominent并提出了OE-VLA以探索VLA模型在开放域多模态指令中的潜力 

---
# Real-Time Verification of Embodied Reasoning for Generative Skill Acquisition 

**Title (ZH)**: 实时验证具身推理的生成技能获取 

**Authors**: Bo Yue, Shuqi Guo, Kaiyu Hu, Chujiao Wang, Benyou Wang, Kui Jia, Guiliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11175)  

**Abstract**: Generative skill acquisition enables embodied agents to actively learn a scalable and evolving repertoire of control skills, crucial for the advancement of large decision models. While prior approaches often rely on supervision signals from generalist agents (e.g., LLMs), their effectiveness in complex 3D environments remains unclear; exhaustive evaluation incurs substantial computational costs, significantly hindering the efficiency of skill learning. Inspired by recent successes in verification models for mathematical reasoning, we propose VERGSA (Verifying Embodied Reasoning in Generative Skill Acquisition), a framework that systematically integrates real-time verification principles into embodied skill learning. VERGSA establishes 1) a seamless extension from verification of mathematical reasoning into embodied learning by dynamically incorporating contextually relevant tasks into prompts and defining success metrics for both subtasks and overall tasks, and 2) an automated, scalable reward labeling scheme that synthesizes dense reward signals by iteratively finalizing the contribution of scene configuration and subtask learning to overall skill acquisition. To the best of our knowledge, this approach constitutes the first comprehensive training dataset for verification-driven generative skill acquisition, eliminating arduous manual reward engineering. Experiments validate the efficacy of our approach: 1) the exemplar task pool improves the average task success rates by 21%, 2) our verification model boosts success rates by 24% for novel tasks and 36% for encountered tasks, and 3) outperforms LLM-as-a-Judge baselines in verification quality. 

**Abstract (ZH)**: 生成技能习得促进具身代理主动学习可扩展且不断演化的控制技能，对于大型决策模型的发展至关重要。受近期数学推理验证模型成功经验的启发，我们提出了VERGSA（Verifying Embodied Reasoning in Generative Skill Acquisition）框架，该框架系统地将实时验证原则整合到具身技能学习中。VERGSA通过动态纳入与上下文相关任务并为子任务和整体任务定义成功指标，实现从数学推理验证到具身学习的无缝扩展，并提出了一种自动化、可扩展的奖励标签方案，通过迭代确定情景配置和子任务学习对总体技能习得的贡献来合成密集的奖励信号。据我们所知，这是首次为驱动验证的生成性技能习得构建全面的训练数据集，消除了繁琐的手动奖励工程。实验验证了该方法的有效性：1）范例任务池将平均任务成功率提高了21%；2）验证模型对于新任务将成功率提高了24%，对于遇到的任务提高了36%；3）在验证质量上优于LLM-as-a-Judge基线。 

---
# Parkour in the Wild: Learning a General and Extensible Agile Locomotion Policy Using Multi-expert Distillation and RL Fine-tuning 

**Title (ZH)**: 在野外的Parkour：基于多专家蒸馏和RL微调的学习通用可扩展敏捷运动政策 

**Authors**: Nikita Rudin, Junzhe He, Joshua Aurand, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2505.11164)  

**Abstract**: Legged robots are well-suited for navigating terrains inaccessible to wheeled robots, making them ideal for applications in search and rescue or space exploration. However, current control methods often struggle to generalize across diverse, unstructured environments. This paper introduces a novel framework for agile locomotion of legged robots by combining multi-expert distillation with reinforcement learning (RL) fine-tuning to achieve robust generalization. Initially, terrain-specific expert policies are trained to develop specialized locomotion skills. These policies are then distilled into a unified foundation policy via the DAgger algorithm. The distilled policy is subsequently fine-tuned using RL on a broader terrain set, including real-world 3D scans. The framework allows further adaptation to new terrains through repeated fine-tuning. The proposed policy leverages depth images as exteroceptive inputs, enabling robust navigation across diverse, unstructured terrains. Experimental results demonstrate significant performance improvements over existing methods in synthesizing multi-terrain skills into a single controller. Deployment on the ANYmal D robot validates the policy's ability to navigate complex environments with agility and robustness, setting a new benchmark for legged robot locomotion. 

**Abstract (ZH)**: 基于多专家提炼与强化学习微调的腿足机器人灵活运动框架 

---
# X2C: A Dataset Featuring Nuanced Facial Expressions for Realistic Humanoid Imitation 

**Title (ZH)**: X2C: 一个展现细腻面部表情的数据集用于 realistic 人形模仿 

**Authors**: Peizhen Li, Longbing Cao, Xiao-Ming Wu, Runze Yang, Xiaohan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11146)  

**Abstract**: The ability to imitate realistic facial expressions is essential for humanoid robots engaged in affective human-robot communication. However, the lack of datasets containing diverse humanoid facial expressions with proper annotations hinders progress in realistic humanoid facial expression imitation. To address these challenges, we introduce X2C (Anything to Control), a dataset featuring nuanced facial expressions for realistic humanoid imitation. With X2C, we contribute: 1) a high-quality, high-diversity, large-scale dataset comprising 100,000 (image, control value) pairs. Each image depicts a humanoid robot displaying a diverse range of facial expressions, annotated with 30 control values representing the ground-truth expression configuration; 2) X2CNet, a novel human-to-humanoid facial expression imitation framework that learns the correspondence between nuanced humanoid expressions and their underlying control values from X2C. It enables facial expression imitation in the wild for different human performers, providing a baseline for the imitation task, showcasing the potential value of our dataset; 3) real-world demonstrations on a physical humanoid robot, highlighting its capability to advance realistic humanoid facial expression imitation. Code and Data: this https URL 

**Abstract (ZH)**: 模仿逼真面部表情的能力对于参与情感人机通信的人形机器人至关重要。然而，缺乏包含多样面部表情且标注恰当的数据集阻碍了逼真人形面部表情模仿的进步。为应对这些挑战，我们引入了X2C（Anything to Control），一个用于真实人形仿真的细腻面部表情数据集。通过X2C，我们贡献了：1) 一个高质量、高多样性和大规模的数据集，包含100,000个（图像，控制值）对。每个图像展示了一个展示多样化面部表情的人形机器人，并用30个控制值标注其真实表情配置；2) X2CNet，一种新颖的人类到人形面部表情模仿框架，它从X2C中学习细腻人形表情与其底层控制值之间的对应关系，使其能够在不同的表演者中进行真实环境下的面部表情模仿，并为模仿任务提供基线，展示了我们数据集的潜在价值；3) 在实际人形机器人上的真实世界演示，突显了其在推进真实人形面部表情模仿方面的能力。代码和数据：this https URL。 

---
# PARSEC: Preference Adaptation for Robotic Object Rearrangement from Scene Context 

**Title (ZH)**: PARSEC: 基于场景上下文的物体重新排列偏好适应 

**Authors**: Kartik Ramachandruni, Sonia Chernova  

**Link**: [PDF](https://arxiv.org/pdf/2505.11108)  

**Abstract**: Object rearrangement is a key task for household robots requiring personalization without explicit instructions, meaningful object placement in environments occupied with objects, and generalization to unseen objects and new environments. To facilitate research addressing these challenges, we introduce PARSEC, an object rearrangement benchmark for learning user organizational preferences from observed scene context to place objects in a partially arranged environment. PARSEC is built upon a novel dataset of 110K rearrangement examples crowdsourced from 72 users, featuring 93 object categories and 15 environments. We also propose ContextSortLM, an LLM-based rearrangement model that places objects in partially arranged environments by adapting to user preferences from prior and current scene context while accounting for multiple valid placements. We evaluate ContextSortLM and existing personalized rearrangement approaches on the PARSEC benchmark and complement these findings with a crowdsourced evaluation of 108 online raters ranking model predictions based on alignment with user preferences. Our results indicate that personalized rearrangement models leveraging multiple scene context sources perform better than models relying on a single context source. Moreover, ContextSortLM outperforms other models in placing objects to replicate the target user's arrangement and ranks among the top two in all three environment categories, as rated by online evaluators. Importantly, our evaluation highlights challenges associated with modeling environment semantics across different environment categories and provides recommendations for future work. 

**Abstract (ZH)**: 基于场景上下文学习用户组织偏好的物体重排基准PARSEC 

---
# DexGarmentLab: Dexterous Garment Manipulation Environment with Generalizable Policy 

**Title (ZH)**: DexGarmentLab: 拟人化服装操作环境与可泛化的策略 

**Authors**: Yuran Wang, Ruihai Wu, Yue Chen, Jiarui Wang, Jiaqi Liang, Ziyu Zhu, Haoran Geng, Jitendra Malik, Pieter Abbeel, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.11032)  

**Abstract**: Garment manipulation is a critical challenge due to the diversity in garment categories, geometries, and deformations. Despite this, humans can effortlessly handle garments, thanks to the dexterity of our hands. However, existing research in the field has struggled to replicate this level of dexterity, primarily hindered by the lack of realistic simulations of dexterous garment manipulation. Therefore, we propose DexGarmentLab, the first environment specifically designed for dexterous (especially bimanual) garment manipulation, which features large-scale high-quality 3D assets for 15 task scenarios, and refines simulation techniques tailored for garment modeling to reduce the sim-to-real gap. Previous data collection typically relies on teleoperation or training expert reinforcement learning (RL) policies, which are labor-intensive and inefficient. In this paper, we leverage garment structural correspondence to automatically generate a dataset with diverse trajectories using only a single expert demonstration, significantly reducing manual intervention. However, even extensive demonstrations cannot cover the infinite states of garments, which necessitates the exploration of new algorithms. To improve generalization across diverse garment shapes and deformations, we propose a Hierarchical gArment-manipuLation pOlicy (HALO). It first identifies transferable affordance points to accurately locate the manipulation area, then generates generalizable trajectories to complete the task. Through extensive experiments and detailed analysis of our method and baseline, we demonstrate that HALO consistently outperforms existing methods, successfully generalizing to previously unseen instances even with significant variations in shape and deformation where others fail. Our project page is available at: this https URL. 

**Abstract (ZH)**: 服装操作是由于服装类别、几何形状和变形的多样性的关键挑战。然而，人类能够凭借灵巧的手部动作轻松应对这些挑战。尽管如此，现有研究在复制这种灵巧性方面仍面临困难，主要是由于缺乏现实的灵巧服装操作模拟。因此，我们提出了DexGarmentLab，这是首个专门设计用于灵巧（尤其是双手灵巧）服装操作的环境，它包含了15种任务场景的大规模高质量3D资产，并通过定制化的模拟技术提高服装建模精度，缩小模拟与现实之间的差距。以往的数据收集通常依赖于遥操作或训练专家强化学习（RL）策略，这既耗时又低效。在本文中，我们利用服装结构对应关系，仅通过单次专家示范即可自动生成多样轨迹的数据集，大大减少了手动干预。然而，即使进行广泛的示范也无法覆盖服装的所有状态，这需要探索新的算法。为了提高在不同服装形状和变形中的泛化能力，我们提出了一种层次化服装操作策略（HALO）。首先，它识别可转移的功能点以准确定位操作区域，然后生成可泛化的轨迹以完成任务。通过广泛的实验和对我们的方法和基线的详细分析，我们证明了HALO在各种形状和变形变化情况下始终优于现有方法，能够成功泛化到未见过的实例。项目主页：this https URL。 

---
# GROQLoco: Generalist and RObot-agnostic Quadruped Locomotion Control using Offline Datasets 

**Title (ZH)**: GROQLoco: 通用且机器人无关的四足运动控制方法基于离线数据集 

**Authors**: Narayanan PP, Sarvesh Prasanth Venkatesan, Srinivas Kantha Reddy, Shishir Kolathaya  

**Link**: [PDF](https://arxiv.org/pdf/2505.10973)  

**Abstract**: Recent advancements in large-scale offline training have demonstrated the potential of generalist policy learning for complex robotic tasks. However, applying these principles to legged locomotion remains a challenge due to continuous dynamics and the need for real-time adaptation across diverse terrains and robot morphologies. In this work, we propose GROQLoco, a scalable, attention-based framework that learns a single generalist locomotion policy across multiple quadruped robots and terrains, relying solely on offline datasets. Our approach leverages expert demonstrations from two distinct locomotion behaviors - stair traversal (non-periodic gaits) and flat terrain traversal (periodic gaits) - collected across multiple quadruped robots, to train a generalist model that enables behavior fusion for both behaviors. Crucially, our framework operates directly on proprioceptive data from all robots without incorporating any robot-specific encodings. The policy is directly deployable on an Intel i7 nuc, producing low-latency control outputs without any test-time optimization. Our extensive experiments demonstrate strong zero-shot transfer across highly diverse quadruped robots and terrains, including hardware deployment on the Unitree Go1, a commercially available 12kg robot. Notably, we evaluate challenging cross-robot training setups where different locomotion skills are unevenly distributed across robots, yet observe successful transfer of both flat walking and stair traversal behaviors to all robots at test time. We also show preliminary walking on Stoch 5, a 70kg quadruped, on flat and outdoor terrains without requiring any fine tuning. These results highlight the potential for robust generalist locomotion across diverse robots and terrains. 

**Abstract (ZH)**: Recent advancements in large-scale offline training have demonstrated the potential of generalist policy learning for complex robotic tasks. However, applying these principles to legged locomotion remains a challenge due to continuous dynamics and the need for real-time adaptation across diverse terrains and robot morphologies. In this work, we propose GROQLoco, a scalable, attention-based framework that learns a single generalist locomotion policy across multiple quadruped robots and terrains, relying solely on offline datasets. 

---
# Unleashing Humanoid Reaching Potential via Real-world-Ready Skill Space 

**Title (ZH)**: 面向现实世界的类人达 manipulability 通过技能空间释放潜力 

**Authors**: Zhikai Zhang, Chao Chen, Han Xue, Jilong Wang, Sikai Liang, Yun Liu, Zongzhang Zhang, He Wang, Li Yi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10918)  

**Abstract**: Humans possess a large reachable space in the 3D world, enabling interaction with objects at varying heights and distances. However, realizing such large-space reaching on humanoids is a complex whole-body control problem and requires the robot to master diverse skills simultaneously-including base positioning and reorientation, height and body posture adjustments, and end-effector pose control. Learning from scratch often leads to optimization difficulty and poor sim2real transferability. To address this challenge, we propose Real-world-Ready Skill Space (R2S2). Our approach begins with a carefully designed skill library consisting of real-world-ready primitive skills. We ensure optimal performance and robust sim2real transfer through individual skill tuning and sim2real evaluation. These skills are then ensembled into a unified latent space, serving as a structured prior that helps task execution in an efficient and sim2real transferable manner. A high-level planner, trained to sample skills from this space, enables the robot to accomplish real-world goal-reaching tasks. We demonstrate zero-shot sim2real transfer and validate R2S2 in multiple challenging goal-reaching scenarios. 

**Abstract (ZH)**: 面向现实世界的技能空间（R2S2） 

---
# ReWiND: Language-Guided Rewards Teach Robot Policies without New Demonstrations 

**Title (ZH)**: ReWiND：语言引导的奖励教机器人策略无需新演示 

**Authors**: Jiahui Zhang, Yusen Luo, Abrar Anwar, Sumedh Anand Sontakke, Joseph J Lim, Jesse Thomason, Erdem Biyik, Jesse Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10911)  

**Abstract**: We introduce ReWiND, a framework for learning robot manipulation tasks solely from language instructions without per-task demonstrations. Standard reinforcement learning (RL) and imitation learning methods require expert supervision through human-designed reward functions or demonstrations for every new task. In contrast, ReWiND starts from a small demonstration dataset to learn: (1) a data-efficient, language-conditioned reward function that labels the dataset with rewards, and (2) a language-conditioned policy pre-trained with offline RL using these rewards. Given an unseen task variation, ReWiND fine-tunes the pre-trained policy using the learned reward function, requiring minimal online interaction. We show that ReWiND's reward model generalizes effectively to unseen tasks, outperforming baselines by up to 2.4x in reward generalization and policy alignment metrics. Finally, we demonstrate that ReWiND enables sample-efficient adaptation to new tasks, beating baselines by 2x in simulation and improving real-world pretrained bimanual policies by 5x, taking a step towards scalable, real-world robot learning. See website at this https URL. 

**Abstract (ZH)**: 我们介绍ReWiND框架，该框架能仅从语言指令学习机器人操作任务，而无需每项任务的具体示范。标准强化学习（RL）和模仿学习方法需要通过人工设计的奖励函数或示范来获取专家监督，以完成每个新任务。相比之下，ReWiND 从一个小的示范数据集开始学习：(1) 一个数据效率高、受语言条件指导的奖励函数，该函数可以为数据集打上奖励标签，(2) 一个受语言条件指导的策略，该策略在使用这些奖励进行离线强化学习预训练后进行微调。面对未见过的任务变体，ReWiND 使用学习到的奖励函数微调预训练策略，所需的在线交互最少。我们证明，ReWiND 的奖励模型能有效泛化到未见过的任务，其在奖励泛化和策略对齐度量上比基线高出2.4倍。最后，我们展示了ReWiND 能实现样本高效的新任务适应，其在模拟中优于基线2倍，在实际预训练双臂策略的表现上提升了5倍，朝着可扩展的、面向实际应用的机器人学习迈出了一步。详见网站：https://www.example.com 

---
# Estimating Deformable-Rigid Contact Interactions for a Deformable Tool via Learning and Model-Based Optimization 

**Title (ZH)**: 基于学习和模型优化的可变形工具可变形-刚性接触交互估计 

**Authors**: Mark Van der Merwe, Miquel Oller, Dmitry Berenson, Nima Fazeli  

**Link**: [PDF](https://arxiv.org/pdf/2505.10884)  

**Abstract**: Dexterous manipulation requires careful reasoning over extrinsic contacts. The prevalence of deforming tools in human environments, the use of deformable sensors, and the increasing number of soft robots yields a need for approaches that enable dexterous manipulation through contact reasoning where not all contacts are well characterized by classical rigid body contact models. Here, we consider the case of a deforming tool dexterously manipulating a rigid object. We propose a hybrid learning and first-principles approach to the modeling of simultaneous motion and force transfer of tools and objects. The learned module is responsible for jointly estimating the rigid object's motion and the deformable tool's imparted contact forces. We then propose a Contact Quadratic Program to recover forces between the environment and object subject to quasi-static equilibrium and Coulomb friction. The results is a system capable of modeling both intrinsic and extrinsic motions, contacts, and forces during dexterous deformable manipulation. We train our method in simulation and show that our method outperforms baselines under varying block geometries and physical properties, during pushing and pivoting manipulations, and demonstrate transfer to real world interactions. Video results can be found at this https URL. 

**Abstract (ZH)**: 灵巧操作需要细致的接触推理。变形工具在人类环境中的普遍性、使用变形传感器以及软机器人数量的增加，产生了通过接触推理实现灵巧操作的需求，其中并非所有接触都能用经典的刚体接触模型充分描述。在此，我们考虑一种灵巧操作刚体对象的变形工具的情况。我们提出了一种结合学习和第一原理的方法来建模工具和物体的同时运动和力传递。学习模块负责联合估计刚体对象的运动和变形工具施加的接触力。然后，我们提出了一种接触二次规划方法，以恢复环境与对象之间的力，条件是满足准静态平衡和库仑摩擦。该系统能够建模灵巧变形操作中的内在和外在运动、接触和力。我们在仿真中训练该方法，并在不同几何结构和物理属性的块件以及推拉操作下展示了其性能优于基线方法，并展示了其在现实世界交互中的应用潜力。相关视频结果可在此处查看：this https URL。 

---
# REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning? 

**Title (ZH)**: REI-Bench: 机器人物体理解含糊的人类任务指示能力研究？ 

**Authors**: Chenxi Jiang, Chuhao Zhou, Jianfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10872)  

**Abstract**: Robot task planning decomposes human instructions into executable action sequences that enable robots to complete a series of complex tasks. Although recent large language model (LLM)-based task planners achieve amazing performance, they assume that human instructions are clear and straightforward. However, real-world users are not experts, and their instructions to robots often contain significant vagueness. Linguists suggest that such vagueness frequently arises from referring expressions (REs), whose meanings depend heavily on dialogue context and environment. This vagueness is even more prevalent among the elderly and children, who robots should serve more. This paper studies how such vagueness in REs within human instructions affects LLM-based robot task planning and how to overcome this issue. To this end, we propose the first robot task planning benchmark with vague REs (REI-Bench), where we discover that the vagueness of REs can severely degrade robot planning performance, leading to success rate drops of up to 77.9%. We also observe that most failure cases stem from missing objects in planners. To mitigate the REs issue, we propose a simple yet effective approach: task-oriented context cognition, which generates clear instructions for robots, achieving state-of-the-art performance compared to aware prompt and chains of thought. This work contributes to the research community of human-robot interaction (HRI) by making robot task planning more practical, particularly for non-expert users, e.g., the elderly and children. 

**Abstract (ZH)**: 基于含模糊指代表达式的机器人任务规划基准：理解与克服指令歧义以提升机器人任务规划性能 

---
# Infinigen-Sim: Procedural Generation of Articulated Simulation Assets 

**Title (ZH)**: Infinigen-Sim：articulated simulation资产的程序生成 

**Authors**: Abhishek Joshi, Beining Han, Jack Nugent, Yiming Zuo, Jonathan Liu, Hongyu Wen, Stamatis Alexandropoulos, Tao Sun, Alexander Raistrick, Gaowen Liu, Yi Shao, Jia Deng  

**Link**: [PDF](https://arxiv.org/pdf/2505.10755)  

**Abstract**: We introduce Infinigen-Sim, a toolkit which enables users to create diverse and realistic articulated object procedural generators. These tools are composed of high-level utilities for use creating articulated assets in Blender, as well as an export pipeline to integrate the resulting assets into common robotics simulators. We demonstrate our system by creating procedural generators for 5 common articulated object categories. Experiments show that assets sampled from these generators are useful for movable object segmentation, training generalizable reinforcement learning policies, and sim-to-real transfer of imitation learning policies. 

**Abstract (ZH)**: Infinigen-Sim：一种用于创建多样化和现实主义 articulated 物体过程生成器的工具包 

---
# TartanGround: A Large-Scale Dataset for Ground Robot Perception and Navigation 

**Title (ZH)**: TartanGround：大规模机器人地面感知与导航数据集 

**Authors**: Manthan Patel, Fan Yang, Yuheng Qiu, Cesar Cadena, Sebastian Scherer, Marco Hutter, Wenshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10696)  

**Abstract**: We present TartanGround, a large-scale, multi-modal dataset to advance the perception and autonomy of ground robots operating in diverse environments. This dataset, collected in various photorealistic simulation environments includes multiple RGB stereo cameras for 360-degree coverage, along with depth, optical flow, stereo disparity, LiDAR point clouds, ground truth poses, semantic segmented images, and occupancy maps with semantic labels. Data is collected using an integrated automatic pipeline, which generates trajectories mimicking the motion patterns of various ground robot platforms, including wheeled and legged robots. We collect 910 trajectories across 70 environments, resulting in 1.5 million samples. Evaluations on occupancy prediction and SLAM tasks reveal that state-of-the-art methods trained on existing datasets struggle to generalize across diverse scenes. TartanGround can serve as a testbed for training and evaluation of a broad range of learning-based tasks, including occupancy prediction, SLAM, neural scene representation, perception-based navigation, and more, enabling advancements in robotic perception and autonomy towards achieving robust models generalizable to more diverse scenarios. The dataset and codebase for data collection will be made publicly available upon acceptance. Webpage: this https URL 

**Abstract (ZH)**: 我们介绍TartanGround：一个大规模多模态数据集，旨在推进地面机器人在多样化环境中感知与自主性的进步。 

---
# Dynam3D: Dynamic Layered 3D Tokens Empower VLM for Vision-and-Language Navigation 

**Title (ZH)**: Dynam3D: 动态分层3D令牌助力视觉-语言导航大模型 

**Authors**: Zihan Wang, Seungjun Lee, Gim Hee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.11383)  

**Abstract**: Vision-and-Language Navigation (VLN) is a core task where embodied agents leverage their spatial mobility to navigate in 3D environments toward designated destinations based on natural language instructions. Recently, video-language large models (Video-VLMs) with strong generalization capabilities and rich commonsense knowledge have shown remarkable performance when applied to VLN tasks. However, these models still encounter the following challenges when applied to real-world 3D navigation: 1) Insufficient understanding of 3D geometry and spatial semantics; 2) Limited capacity for large-scale exploration and long-term environmental memory; 3) Poor adaptability to dynamic and changing this http URL address these limitations, we propose Dynam3D, a dynamic layered 3D representation model that leverages language-aligned, generalizable, and hierarchical 3D representations as visual input to train 3D-VLM in navigation action prediction. Given posed RGB-D images, our Dynam3D projects 2D CLIP features into 3D space and constructs multi-level 3D patch-instance-zone representations for 3D geometric and semantic understanding with a dynamic and layer-wise update strategy. Our Dynam3D is capable of online encoding and localization of 3D instances, and dynamically updates them in changing environments to provide large-scale exploration and long-term memory capabilities for navigation. By leveraging large-scale 3D-language pretraining and task-specific adaptation, our Dynam3D sets new state-of-the-art performance on VLN benchmarks including R2R-CE, REVERIE-CE and NavRAG-CE under monocular settings. Furthermore, experiments for pre-exploration, lifelong memory, and real-world robot validate the effectiveness of practical deployment. 

**Abstract (ZH)**: 动态层次三维表示模型：面向真实世界3D导航的视语音导航（VLN） 

---
# Multi-Modal Multi-Task (M3T) Federated Foundation Models for Embodied AI: Potentials and Challenges for Edge Integration 

**Title (ZH)**: 面向身体化人工智能的多模态多任务（M3T）联邦基础模型：边缘集成的潜力与挑战 

**Authors**: Kasra Borazjani, Payam Abdisarabshali, Fardis Nadimi, Naji Khosravan, Minghui Liwang, Xianbin Wang, Yiguang Hong, Seyyedali Hosseinalipour  

**Link**: [PDF](https://arxiv.org/pdf/2505.11191)  

**Abstract**: As embodied AI systems become increasingly multi-modal, personalized, and interactive, they must learn effectively from diverse sensory inputs, adapt continually to user preferences, and operate safely under resource and privacy constraints. These challenges expose a pressing need for machine learning models capable of swift, context-aware adaptation while balancing model generalization and personalization. Here, two methods emerge as suitable candidates, each offering parts of these capabilities: Foundation Models (FMs) provide a pathway toward generalization across tasks and modalities, whereas Federated Learning (FL) offers the infrastructure for distributed, privacy-preserving model updates and user-level model personalization. However, when used in isolation, each of these approaches falls short of meeting the complex and diverse capability requirements of real-world embodied environments. In this vision paper, we introduce Federated Foundation Models (FFMs) for embodied AI, a new paradigm that unifies the strengths of multi-modal multi-task (M3T) FMs with the privacy-preserving distributed nature of FL, enabling intelligent systems at the wireless edge. We collect critical deployment dimensions of FFMs in embodied AI ecosystems under a unified framework, which we name "EMBODY": Embodiment heterogeneity, Modality richness and imbalance, Bandwidth and compute constraints, On-device continual learning, Distributed control and autonomy, and Yielding safety, privacy, and personalization. For each, we identify concrete challenges and envision actionable research directions. We also present an evaluation framework for deploying FFMs in embodied AI systems, along with the associated trade-offs. 

**Abstract (ZH)**: 面向物联网边缘的联合基础模型：统一多模态多任务基础模型与联邦学习的优势 

---
# Reinforcement Learning for AMR Charging Decisions: The Impact of Reward and Action Space Design 

**Title (ZH)**: 基于强化学习的AMR充电决策：奖励和动作空间设计的影响 

**Authors**: Janik Bischoff, Alexandru Rinciog, Anne Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2505.11136)  

**Abstract**: We propose a novel reinforcement learning (RL) design to optimize the charging strategy for autonomous mobile robots in large-scale block stacking warehouses. RL design involves a wide array of choices that can mostly only be evaluated through lengthy experimentation. Our study focuses on how different reward and action space configurations, ranging from flexible setups to more guided, domain-informed design configurations, affect the agent performance. Using heuristic charging strategies as a baseline, we demonstrate the superiority of flexible, RL-based approaches in terms of service times. Furthermore, our findings highlight a trade-off: While more open-ended designs are able to discover well-performing strategies on their own, they may require longer convergence times and are less stable, whereas guided configurations lead to a more stable learning process but display a more limited generalization potential. Our contributions are threefold. First, we extend SLAPStack, an open-source, RL-compatible simulation-framework to accommodate charging strategies. Second, we introduce a novel RL design for tackling the charging strategy problem. Finally, we introduce several novel adaptive baseline heuristics and reproducibly evaluate the design using a Proximal Policy Optimization agent and varying different design configurations, with a focus on reward. 

**Abstract (ZH)**: 我们提出了一种新颖的强化学习（RL）设计，以优化自主移动机器人在大规模块堆叠仓库中的充电策略。我们的研究重点在于不同奖励和动作空间配置，从灵活设置到更引导性的、领域导向的设计配置，对代理性能的影响。以启发式充电策略为基准，我们展示了基于RL的方法在服务时间方面的优越性。此外，我们的研究结果揭示了一种权衡：虽然更为开放的设计能够自主发现高性能策略，但可能需要更长的收敛时间且稳定性较差，而引导性配置则促进了更稳定的训练过程，但其泛化能力更为有限。我们的贡献包括三个部分。首先，我们将SLAPStack扩展为一种兼容RL的开源仿真框架，以包含充电策略。其次，我们引入了一种新的RL设计以解决充电策略问题。最后，我们引入了若干新的自适应基准启发式方法，并通过使用Proximal Policy Optimization代理和不同的设计配置进行可重复评估，重点在于奖励。 

---
# Embodied AI in Machine Learning -- is it Really Embodied? 

**Title (ZH)**: 机器学习中的具身AI——它真是具身的吗？ 

**Authors**: Matej Hoffmann, Shubhan Parag Patni  

**Link**: [PDF](https://arxiv.org/pdf/2505.10705)  

**Abstract**: Embodied Artificial Intelligence (Embodied AI) is gaining momentum in the machine learning communities with the goal of leveraging current progress in AI (deep learning, transformers, large language and visual-language models) to empower robots. In this chapter we put this work in the context of "Good Old-Fashioned Artificial Intelligence" (GOFAI) (Haugeland, 1989) and the behavior-based or embodied alternatives (R. A. Brooks 1991; Pfeifer and Scheier 2001). We claim that the AI-powered robots are only weakly embodied and inherit some of the problems of GOFAI. Moreover, we review and critically discuss the possibility of cross-embodiment learning (Padalkar et al. 2024). We identify fundamental roadblocks and propose directions on how to make progress. 

**Abstract (ZH)**: 基于体征的人工智能：从GOFAI到跨体征学习的研究进展 

---
# Decision Making in Urban Traffic: A Game Theoretic Approach for Autonomous Vehicles Adhering to Traffic Rules 

**Title (ZH)**: 城市交通中的决策制定：一种自主车辆遵守交通规则的游戏理论方法 

**Authors**: Keqi Shu, Minghao Ning, Ahmad Alghooneh, Shen Li, Mohammad Pirani, Amir Khajepour  

**Link**: [PDF](https://arxiv.org/pdf/2505.10690)  

**Abstract**: One of the primary challenges in urban autonomous vehicle decision-making and planning lies in effectively managing intricate interactions with diverse traffic participants characterized by unpredictable movement patterns. Additionally, interpreting and adhering to traffic regulations within rapidly evolving traffic scenarios pose significant hurdles. This paper proposed a rule-based autonomous vehicle decision-making and planning framework which extracts right-of-way from traffic rules to generate behavioural parameters, integrating them to effectively adhere to and navigate through traffic regulations. The framework considers the strong interaction between traffic participants mathematically by formulating the decision-making and planning problem into a differential game. By finding the Nash equilibrium of the problem, the autonomous vehicle is able to find optimal decisions. The proposed framework was tested under simulation as well as full-size vehicle platform, the results show that the ego vehicle is able to safely interact with surrounding traffic participants while adhering to traffic rules. 

**Abstract (ZH)**: 城市自动驾驶车辆决策与规划中的主要挑战在于有效地管理多样且不可预测的交通参与者之间的复杂交互。此外，在快速变化的交通场景中解释和遵守交通规则也带来了重大障碍。本文提出了一种基于规则的自动驾驶车辆决策与规划框架，从交通规则中提取优先通行权，生成行为参数，并将这些参数整合以有效遵守和导航通过交通规则。该框架通过将决策与规划问题形式化为微分博弈来考虑交通参与者之间强烈的交互作用。通过找到问题的纳什均衡，自动驾驶车辆能够找到最优决策。所提出框架在仿真和全尺寸车辆平台上进行了测试，结果显示自动驾驶车辆能够在遵守交通规则的同时安全地与周围交通参与者互动。 

---
# Accelerating Visual-Policy Learning through Parallel Differentiable Simulation 

**Title (ZH)**: 通过并行可微模拟加速视觉策略学习 

**Authors**: Haoxiang You, Yilang Liu, Ian Abraham  

**Link**: [PDF](https://arxiv.org/pdf/2505.10646)  

**Abstract**: In this work, we propose a computationally efficient algorithm for visual policy learning that leverages differentiable simulation and first-order analytical policy gradients. Our approach decouple the rendering process from the computation graph, enabling seamless integration with existing differentiable simulation ecosystems without the need for specialized differentiable rendering software. This decoupling not only reduces computational and memory overhead but also effectively attenuates the policy gradient norm, leading to more stable and smoother optimization. We evaluate our method on standard visual control benchmarks using modern GPU-accelerated simulation. Experiments show that our approach significantly reduces wall-clock training time and consistently outperforms all baseline methods in terms of final returns. Notably, on complex tasks such as humanoid locomotion, our method achieves a $4\times$ improvement in final return, and successfully learns a humanoid running policy within 4 hours on a single GPU. 

**Abstract (ZH)**: 本研究提出了一种利用可微模拟和一阶分析性策略梯度的计算高效视觉策略学习算法。该方法将渲染过程与计算图解耦，使其能够无缝集成到现有的可微模拟生态系统中，无需专门的可微渲染软件。这种解耦不仅减少了计算和内存开销，还有效地减弱了策略梯度范数，从而使优化更加稳定和平滑。我们在使用现代GPU加速模拟的标准视觉控制基准上评估了该方法。实验表明，该方法显著减少了 wall-clock 训练时间，并在最终回报上始终优于所有基线方法。特别是在复杂任务如人形机器人行走中，该方法在单个GPU上4小时内成功学习了一个人形机器人跑步策略，并实现了最终回报4倍的提升。 

---
# Developing and Integrating Trust Modeling into Multi-Objective Reinforcement Learning for Intelligent Agricultural Management 

**Title (ZH)**: 开发并集成信任模型到多目标强化学习中的智能农业管理 

**Authors**: Zhaoan Wang, Wonseok Jang, Bowen Ruan, Jun Wang, Shaoping Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.10803)  

**Abstract**: Precision agriculture, enhanced by artificial intelligence (AI), offers promising tools such as remote sensing, intelligent irrigation, fertilization management, and crop simulation to improve agricultural efficiency and sustainability. Reinforcement learning (RL), in particular, has outperformed traditional methods in optimizing yields and resource management. However, widespread AI adoption is limited by gaps between algorithmic recommendations and farmers' practical experience, local knowledge, and traditional practices. To address this, our study emphasizes Human-AI Interaction (HAII), focusing on transparency, usability, and trust in RL-based farm management. We employ a well-established trust framework - comprising ability, benevolence, and integrity - to develop a novel mathematical model quantifying farmers' confidence in AI-based fertilization strategies. Surveys conducted with farmers for this research reveal critical misalignments, which are integrated into our trust model and incorporated into a multi-objective RL framework. Unlike prior methods, our approach embeds trust directly into policy optimization, ensuring AI recommendations are technically robust, economically feasible, context-aware, and socially acceptable. By aligning technical performance with human-centered trust, this research supports broader AI adoption in agriculture. 

**Abstract (ZH)**: 人工智能增强的精确农业：基于强化学习的信任人类-人工智能交互研究 

---
# Qualia Optimization 

**Title (ZH)**: 质体优化 

**Authors**: Philip S. Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2505.10779)  

**Abstract**: This report explores the speculative question: what if current or future AI systems have qualia, such as pain or pleasure? It does so by assuming that AI systems might someday possess qualia -- and that the quality of these subjective experiences should be considered alongside performance metrics. Concrete mathematical problem settings, inspired by reinforcement learning formulations and theories from philosophy of mind, are then proposed and initial approaches and properties are presented. These properties enable refinement of the problem setting, culminating with the proposal of methods that promote reinforcement. 

**Abstract (ZH)**: 本报告探讨了假设性问题：当前或未来的AI系统是否具有主观体验（如疼痛或快乐），并通过假设AI系统 someday 可能具备主观体验，进而考虑这些主观体验的质量与性能指标并重，提出具体的数学问题设定，借鉴强化学习的表述形式和心灵哲学理论，介绍初步方法和性质，这些性质使得问题设定得以细化，最终提出促进强化的方法。 

---
# ChestyBot: Detecting and Disrupting Chinese Communist Party Influence Stratagems 

**Title (ZH)**: ChestyBot: 识别和遏制中国共产党影响力策略 

**Authors**: Matthew Stoffolano, Ayush Rout, Justin M. Pelletier  

**Link**: [PDF](https://arxiv.org/pdf/2505.10746)  

**Abstract**: Foreign information operations conducted by Russian and Chinese actors exploit the United States' permissive information environment. These campaigns threaten democratic institutions and the broader Westphalian model. Yet, existing detection and mitigation strategies often fail to identify active information campaigns in real time. This paper introduces ChestyBot, a pragmatics-based language model that detects unlabeled foreign malign influence tweets with up to 98.34% accuracy. The model supports a novel framework to disrupt foreign influence operations in their formative stages. 

**Abstract (ZH)**: 俄罗斯和中国行为体开展的外国信息操作利用了美国宽松的信息环境，威胁着民主制度和更广泛的威斯特伐利亚模型。现有检测和缓解策略往往无法实时识别活跃的信息campaign。本文介绍了一种基于语用学的语言模型ChestyBot，该模型以高达98.34%的准确率检测未标记的外国恶意影响推文，并支持一种在形成阶段削弱外国影响操作的新框架。 

---
# GarmentPile: Point-Level Visual Affordance Guided Retrieval and Adaptation for Cluttered Garments Manipulation 

**Title (ZH)**: 服装堆叠：点级视觉潜能引导的检索与适应以应对杂乱服装操作 

**Authors**: Ruihai Wu, Ziyu Zhu, Yuran Wang, Yue Chen, Jiarui Wang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2503.09243)  

**Abstract**: Cluttered garments manipulation poses significant challenges due to the complex, deformable nature of garments and intricate garment relations. Unlike single-garment manipulation, cluttered scenarios require managing complex garment entanglements and interactions, while maintaining garment cleanliness and manipulation stability. To address these demands, we propose to learn point-level affordance, the dense representation modeling the complex space and multi-modal manipulation candidates, while being aware of garment geometry, structure, and inter-object relations. Additionally, as it is difficult to directly retrieve a garment in some extremely entangled clutters, we introduce an adaptation module, guided by learned affordance, to reorganize highly-entangled garments into states plausible for manipulation. Our framework demonstrates effectiveness over environments featuring diverse garment types and pile configurations in both simulation and the real world. Project page: this https URL. 

**Abstract (ZH)**: 杂乱衣物操作由于衣物的复杂可变形性质和复杂的衣物关系而面临显著挑战。与单件衣物操作不同，杂乱场景需要管理复杂的衣物纠缠和相互作用，同时保持衣物的清洁和操作稳定性。为应对这些需求，我们提出学习点级功能，即密集表示复杂空间和多模态操作候选方式，并考虑到衣物的几何形状、结构及其与其他物体的关系。此外，由于在某些极其纠缠的杂乱环境中难以直接检索衣物，我们引入了一个由学习到的功能引导的适应模块，将高度纠缠的衣物重新整理为易于操作的状态。我们的框架在包含多种衣物类型和堆积配置的模拟和真实环境中均证明了其有效性。项目页面: [这个链接](this https URL)。 

---
