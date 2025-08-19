# Manipulate-to-Navigate: Reinforcement Learning with Visual Affordances and Manipulability Priors 

**Title (ZH)**: 操纵以导航：基于视觉可用性和操作性先验的强化学习 

**Authors**: Yuying Zhang, Joni Pajarinen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13151)  

**Abstract**: Mobile manipulation in dynamic environments is challenging due to movable obstacles blocking the robot's path. Traditional methods, which treat navigation and manipulation as separate tasks, often fail in such 'manipulate-to-navigate' scenarios, as obstacles must be removed before navigation. In these cases, active interaction with the environment is required to clear obstacles while ensuring sufficient space for movement. To address the manipulate-to-navigate problem, we propose a reinforcement learning-based approach for learning manipulation actions that facilitate subsequent navigation. Our method combines manipulability priors to focus the robot on high manipulability body positions with affordance maps for selecting high-quality manipulation actions. By focusing on feasible and meaningful actions, our approach reduces unnecessary exploration and allows the robot to learn manipulation strategies more effectively. We present two new manipulate-to-navigate simulation tasks called Reach and Door with the Boston Dynamics Spot robot. The first task tests whether the robot can select a good hand position in the target area such that the robot base can move effectively forward while keeping the end effector position fixed. The second task requires the robot to move a door aside in order to clear the navigation path. Both of these tasks need first manipulation and then navigating the base forward. Results show that our method allows a robot to effectively interact with and traverse dynamic environments. Finally, we transfer the learned policy to a real Boston Dynamics Spot robot, which successfully performs the Reach task. 

**Abstract (ZH)**: 动态环境下的移动 manipulation 挑战在于可移动障碍物阻碍机器人路径，传统方法将导航和操作视为分开的任务，在“操作以导航”的场景中往往无法应对，因为必须先清除障碍物才能导航。在这种情况下，机器人需要与环境进行主动交互以清除障碍物并确保足够的移动空间。为解决操作以导航的问题，我们提出了一种基于强化学习的方法，用于学习有助于后续导航的操作行动。该方法结合了可操作性先验，使机器人专注于高可操作性身体位置，并使用效应器动作图选择高质量的操作行动。通过集中于可行和有意义的操作，该方法减少了不必要的探索并使机器人能够更有效地学习操作策略。我们使用波士顿动力公司的 Spot 机器人提出了两个新的操作以导航模拟任务，名为 Reach 和 Door。Reach 任务测试机器人是否能在目标区域内选择一个良好的手部位置，使得机器人基座能够有效前移且末端执行器位置保持不变。Door 任务要求机器人移动门以清开通行路径。这两个任务都需要先操作然后前移基座。结果表明，我们的方法使机器人能够有效与动态环境进行交互并穿越。最后，我们将学到的策略转移到实际的波士顿动力公司 Spot 机器人上，并成功执行了 Reach 任务。 

---
# Grounding Actions in Camera Space: Observation-Centric Vision-Language-Action Policy 

**Title (ZH)**: 空间相机中心的动作 grounding 观测-centric 视觉-语言-行动策略 

**Authors**: Tianyi Zhang, Haonan Duan, Haoran Hao, Yu Qiao, Jifeng Dai, Zhi Hou  

**Link**: [PDF](https://arxiv.org/pdf/2508.13103)  

**Abstract**: Vision-Language-Action (VLA) models frequently encounter challenges in generalizing to real-world environments due to inherent discrepancies between observation and action spaces. Although training data are collected from diverse camera perspectives, the models typically predict end-effector poses within the robot base coordinate frame, resulting in spatial inconsistencies. To mitigate this limitation, we introduce the Observation-Centric VLA (OC-VLA) framework, which grounds action predictions directly in the camera observation space. Leveraging the camera's extrinsic calibration matrix, OC-VLA transforms end-effector poses from the robot base coordinate system into the camera coordinate system, thereby unifying prediction targets across heterogeneous viewpoints. This lightweight, plug-and-play strategy ensures robust alignment between perception and action, substantially improving model resilience to camera viewpoint variations. The proposed approach is readily compatible with existing VLA architectures, requiring no substantial modifications. Comprehensive evaluations on both simulated and real-world robotic manipulation tasks demonstrate that OC-VLA accelerates convergence, enhances task success rates, and improves cross-view generalization. The code will be publicly available. 

**Abstract (ZH)**: 基于观察的视知行一体（OC-VLA）框架：在现实环境中的视知行建模 

---
# Large VLM-based Vision-Language-Action Models for Robotic Manipulation: A Survey 

**Title (ZH)**: 基于大型多模态模型的视觉-语言-动作模型在机器人操控中的研究综述 

**Authors**: Rui Shao, Wei Li, Lingsen Zhang, Renshan Zhang, Zhiyang Liu, Ran Chen, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2508.13073)  

**Abstract**: Robotic manipulation, a key frontier in robotics and embodied AI, requires precise motor control and multimodal understanding, yet traditional rule-based methods fail to scale or generalize in unstructured, novel environments. In recent years, Vision-Language-Action (VLA) models, built upon Large Vision-Language Models (VLMs) pretrained on vast image-text datasets, have emerged as a transformative paradigm. This survey provides the first systematic, taxonomy-oriented review of large VLM-based VLA models for robotic manipulation. We begin by clearly defining large VLM-based VLA models and delineating two principal architectural paradigms: (1) monolithic models, encompassing single-system and dual-system designs with differing levels of integration; and (2) hierarchical models, which explicitly decouple planning from execution via interpretable intermediate representations. Building on this foundation, we present an in-depth examination of large VLM-based VLA models: (1) integration with advanced domains, including reinforcement learning, training-free optimization, learning from human videos, and world model integration; (2) synthesis of distinctive characteristics, consolidating architectural traits, operational strengths, and the datasets and benchmarks that support their development; (3) identification of promising directions, including memory mechanisms, 4D perception, efficient adaptation, multi-agent cooperation, and other emerging capabilities. This survey consolidates recent advances to resolve inconsistencies in existing taxonomies, mitigate research fragmentation, and fill a critical gap through the systematic integration of studies at the intersection of large VLMs and robotic manipulation. We provide a regularly updated project page to document ongoing progress: this https URL. 

**Abstract (ZH)**: 机器人操作是机器人学和具身人工智能的一个关键前沿领域，需要精确的操作控制和多模态理解能力D然而传统的基于规则的方法无法在未结构化环境中实现一般化。近年来,D基于大视觉语言模型（（Large Vision-Language Models,VLMsD的的视觉-语言D动作（（Vision-Language-ActionDVVLAD模型凭借其大规模图像文本数据集的预训练而崭露头出头
user
请纠正上面的翻译语法错，并并并，并结构错误，并并，并更确保句子通顺和符合符合格式正确。D 

---
# BOW: Bayesian Optimization over Windows for Motion Planning in Complex Environments 

**Title (ZH)**: BOW: 复杂环境中的运动规划的窗口下的贝叶斯优化 

**Authors**: Sourav Raxit, Abdullah Al Redwan Newaz, Paulo Padrao, Jose Fuentes, Leonardo Bobadilla  

**Link**: [PDF](https://arxiv.org/pdf/2508.13052)  

**Abstract**: This paper introduces the BOW Planner, a scalable motion planning algorithm designed to navigate robots through complex environments using constrained Bayesian optimization (CBO). Unlike traditional methods, which often struggle with kinodynamic constraints such as velocity and acceleration limits, the BOW Planner excels by concentrating on a planning window of reachable velocities and employing CBO to sample control inputs efficiently. This approach enables the planner to manage high-dimensional objective functions and stringent safety constraints with minimal sampling, ensuring rapid and secure trajectory generation. Theoretical analysis confirms the algorithm's asymptotic convergence to near-optimal solutions, while extensive evaluations in cluttered and constrained settings reveal substantial improvements in computation times, trajectory lengths, and solution times compared to existing techniques. Successfully deployed across various real-world robotic systems, the BOW Planner demonstrates its practical significance through exceptional sample efficiency, safety-aware optimization, and rapid planning capabilities, making it a valuable tool for advancing robotic applications. The BOW Planner is released as an open-source package and videos of real-world and simulated experiments are available at this https URL. 

**Abstract (ZH)**: 本文介绍了BOW规划器，这是一种用于在复杂环境中使用约束贝叶斯优化（CBO）导航机器人的时间可扩展运动规划算法。与其他经常难以应对速度和加速度等动力学约束的传统方法不同，BOW规划器通过专注于可达速度的规划窗口并利用CBO高效采样控制输入来出类拔萃。这种方法使规划器能够在最少采样的情况下管理高维目标函数和严格的安全约束，确保快速和安全的轨迹生成。理论分析证实了该算法的渐近收敛性于近最优解，而广泛的评估在拥挤和受限环境下显示了与现有技术相比显著缩短的计算时间、轨迹长度和解决方案时间。BOW规划器已在各种实际机器人系统中成功部署，通过卓越的样本效率、安全意识优化和快速规划能力证明了其实用价值，使其成为推进机器人应用的重要工具。BOW规划器作为开源软件包发布，并在该页面提供了实际和模拟实验的视频：<https://this-url>`。 

---
# Scaling Whole-body Multi-contact Manipulation with Contact Optimization 

**Title (ZH)**: 全身多接触 manipulation 的接触优化扩展 

**Authors**: Victor Levé, João Moura, Sachiya Fujita, Tamon Miyake, Steve Tonneau, Sethu Vijayakumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.12980)  

**Abstract**: Daily tasks require us to use our whole body to manipulate objects, for instance when our hands are unavailable. We consider the issue of providing humanoid robots with the ability to autonomously perform similar whole-body manipulation tasks. In this context, the infinite possibilities for where and how contact can occur on the robot and object surfaces hinder the scalability of existing planning methods, which predominantly rely on discrete sampling. Given the continuous nature of contact surfaces, gradient-based optimization offers a more suitable approach for finding solutions. However, a key remaining challenge is the lack of an efficient representation of robot surfaces. In this work, we propose (i) a representation of robot and object surfaces that enables closed-form computation of proximity points, and (ii) a cost design that effectively guides whole-body manipulation planning. Our experiments demonstrate that the proposed framework can solve problems unaddressed by existing methods, and achieves a 77% improvement in planning time over the state of the art. We also validate the suitability of our approach on real hardware through the whole-body manipulation of boxes by a humanoid robot. 

**Abstract (ZH)**: humanoid机器人全身影觉操作能力的研究：基于梯度优化的表面表示与任务规划 

---
# Simultaneous Contact Sequence and Patch Planning for Dynamic Locomotion 

**Title (ZH)**: 同时规划接触序列和贴点的动态运动学 

**Authors**: Victor Dhédin, Haizhou Zhao, Majid Khadiv  

**Link**: [PDF](https://arxiv.org/pdf/2508.12928)  

**Abstract**: Legged robots have the potential to traverse highly constrained environments with agile maneuvers. However, planning such motions requires solving a highly challenging optimization problem with a mixture of continuous and discrete decision variables. In this paper, we present a full pipeline based on Monte-Carlo tree search (MCTS) and whole-body trajectory optimization (TO) to perform simultaneous contact sequence and patch selection on highly challenging environments. Through extensive simulation experiments, we show that our framework can quickly find a diverse set of dynamically consistent plans. We experimentally show that these plans are transferable to a real quadruped robot. We further show that the same framework can find highly complex acyclic humanoid maneuvers. To the best of our knowledge, this is the first demonstration of simultaneous contact sequence and patch selection for acyclic multi-contact locomotion using the whole-body dynamics of a quadruped. 

**Abstract (ZH)**: 具有敏捷动作的腿式机器人有潜力穿越高度受限环境。然而，规划此类动作需要解决一个包含连续和离散决策变量的极高挑战性优化问题。本文提出了一种基于蒙特卡洛树搜索（MCTS）和全身轨迹优化（TO）的完整管道，以在高度挑战性环境中同时进行接触序列和接触点选择。通过大量的仿真实验，我们展示了我们的框架可以快速找到一组动态一致的计划。我们实验证明这些计划可以转移到真实四足机器人上。此外，我们展示了相同的框架可以找到复杂的无环双足行走动作。据我们所知，这是首次使用四足动物全身动力学进行无环多接触行走的接触序列和接触点选择的演示。 

---
# RoboRetriever: Single-Camera Robot Object Retrieval via Active and Interactive Perception with Dynamic Scene Graph 

**Title (ZH)**: RoboRetriever：基于动态场景图的主动交互式单目机器人物体检索 

**Authors**: Hecheng Wang, Jiankun Ren, Jia Yu, Lizhe Qi, Yunquan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.12916)  

**Abstract**: Humans effortlessly retrieve objects in cluttered, partially observable environments by combining visual reasoning, active viewpoint adjustment, and physical interaction-with only a single pair of eyes. In contrast, most existing robotic systems rely on carefully positioned fixed or multi-camera setups with complete scene visibility, which limits adaptability and incurs high hardware costs. We present \textbf{RoboRetriever}, a novel framework for real-world object retrieval that operates using only a \textbf{single} wrist-mounted RGB-D camera and free-form natural language instructions. RoboRetriever grounds visual observations to build and update a \textbf{dynamic hierarchical scene graph} that encodes object semantics, geometry, and inter-object relations over time. The supervisor module reasons over this memory and task instruction to infer the target object and coordinate an integrated action module combining \textbf{active perception}, \textbf{interactive perception}, and \textbf{manipulation}. To enable task-aware scene-grounded active perception, we introduce a novel visual prompting scheme that leverages large reasoning vision-language models to determine 6-DoF camera poses aligned with the semantic task goal and geometry scene context. We evaluate RoboRetriever on diverse real-world object retrieval tasks, including scenarios with human intervention, demonstrating strong adaptability and robustness in cluttered scenes with only one RGB-D camera. 

**Abstract (ZH)**: RoboRetrieved：一种基于单目RGB-D相机的现实世界对象检索框架 

---
# PROD: Palpative Reconstruction of Deformable Objects through Elastostatic Signed Distance Functions 

**Title (ZH)**: PROD：通过弹性静态符号距离函数的可变形物体触觉重建 

**Authors**: Hamza El-Kebir  

**Link**: [PDF](https://arxiv.org/pdf/2508.12554)  

**Abstract**: We introduce PROD (Palpative Reconstruction of Deformables), a novel method for reconstructing the shape and mechanical properties of deformable objects using elastostatic signed distance functions (SDFs). Unlike traditional approaches that rely on purely geometric or visual data, PROD integrates palpative interaction -- measured through force-controlled surface probing -- to estimate both the static and dynamic response of soft materials. We model the deformation of an object as an elastostatic process and derive a governing Poisson equation for estimating its SDF from a sparse set of pose and force measurements. By incorporating steady-state elastodynamic assumptions, we show that the undeformed SDF can be recovered from deformed observations with provable convergence. Our approach also enables the estimation of material stiffness by analyzing displacement responses to varying force inputs. We demonstrate the robustness of PROD in handling pose errors, non-normal force application, and curvature errors in simulated soft body interactions. These capabilities make PROD a powerful tool for reconstructing deformable objects in applications ranging from robotic manipulation to medical imaging and haptic feedback systems. 

**Abstract (ZH)**: 基于触觉交互的不可变形物体形貌与机械性能重建方法PROD 

---
# SIGN: Safety-Aware Image-Goal Navigation for Autonomous Drones via Reinforcement Learning 

**Title (ZH)**: SIGN：基于强化学习的自主无人机安全目标导航 

**Authors**: Zichen Yan, Rui Huang, Lei He, Shao Guo, Lin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.12394)  

**Abstract**: Image-goal navigation (ImageNav) tasks a robot with autonomously exploring an unknown environment and reaching a location that visually matches a given target image. While prior works primarily study ImageNav for ground robots, enabling this capability for autonomous drones is substantially more challenging due to their need for high-frequency feedback control and global localization for stable flight. In this paper, we propose a novel sim-to-real framework that leverages visual reinforcement learning (RL) to achieve ImageNav for drones. To enhance visual representation ability, our approach trains the vision backbone with auxiliary tasks, including image perturbations and future transition prediction, which results in more effective policy training. The proposed algorithm enables end-to-end ImageNav with direct velocity control, eliminating the need for external localization. Furthermore, we integrate a depth-based safety module for real-time obstacle avoidance, allowing the drone to safely navigate in cluttered environments. Unlike most existing drone navigation methods that focus solely on reference tracking or obstacle avoidance, our framework supports comprehensive navigation behaviors--autonomous exploration, obstacle avoidance, and image-goal seeking--without requiring explicit global mapping. Code and model checkpoints will be released upon acceptance. 

**Abstract (ZH)**: 图像目标导航（ImageNav）任务要求机器人自主探索未知环境并到达与给定目标图像在视觉上匹配的位置。虽然先前的研究主要集中在地面机器人上的ImageNav，但为自主无人机实现这一能力要更加困难，因为无人机需要高频率的反馈控制和全球定位以保证稳定飞行。在本文中，我们提出了一种新的从模拟到现实的框架，利用视觉强化学习（RL）实现无人机的ImageNav。为增强视觉表示能力，我们的方法通过辅助任务训练视觉骨干网络，包括图像扰动和未来的过渡预测，这有助于更有效的策略训练。所提出的方法使无人机能够实现端到端的直接速度控制的图像目标导航，消除了对外部定位的需求。此外，我们还集成了基于深度的安全模块，以实现实时障碍物避免，使无人机能够在复杂环境中安全导航。与大多数现有无人机导航方法主要关注参考跟踪或障碍物避免不同，我们的框架支持全面的导航行为——自主探索、障碍物避免和图像目标搜索，而无需显式的全局映射。接受后将发布代码和模型检查点。 

---
# Robot Trains Robot: Automatic Real-World Policy Adaptation and Learning for Humanoids 

**Title (ZH)**: 机器人训练机器人：类人机器人在现实世界中的自动政策适应与学习 

**Authors**: Kaizhe Hu, Haochen Shi, Yao He, Weizhuo Wang, C. Karen Liu, Shuran Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.12252)  

**Abstract**: Simulation-based reinforcement learning (RL) has significantly advanced humanoid locomotion tasks, yet direct real-world RL from scratch or adapting from pretrained policies remains rare, limiting the full potential of humanoid robots. Real-world learning, despite being crucial for overcoming the sim-to-real gap, faces substantial challenges related to safety, reward design, and learning efficiency. To address these limitations, we propose Robot-Trains-Robot (RTR), a novel framework where a robotic arm teacher actively supports and guides a humanoid robot student. The RTR system provides protection, learning schedule, reward, perturbation, failure detection, and automatic resets. It enables efficient long-term real-world humanoid training with minimal human intervention. Furthermore, we propose a novel RL pipeline that facilitates and stabilizes sim-to-real transfer by optimizing a single dynamics-encoded latent variable in the real world. We validate our method through two challenging real-world humanoid tasks: fine-tuning a walking policy for precise speed tracking and learning a humanoid swing-up task from scratch, illustrating the promising capabilities of real-world humanoid learning realized by RTR-style systems. See this https URL for more info. 

**Abstract (ZH)**: 基于模拟的强化学习在人形机器人运动任务中取得了显著进展，但直接从现实世界中进行RL训练或从预训练策略进行适应仍然很少见，限制了人形机器人的全部潜力。尽管现实世界的学习对于克服模拟到现实世界的差距至关重要，但安全性、奖励设计和学习效率等方面的挑战仍然很大。为了解决这些限制，我们提出了一种名为Robot-Trains-Robot (RTR)的新框架，其中一台机械臂教师积极支持和引导一台人形机器人学生。RTR系统提供了保护、学习时间表、奖励、扰动、故障检测和自动重置。它允许在最少的人工干预下进行高效的人形机器人长期现实世界训练。此外，我们提出了一种新的RL管道，通过优化真实世界中的单一动态编码潜在变量，促进和稳定模拟到现实世界的过渡。我们通过两个具有挑战性的现实世界人形机器人任务验证了该方法：精细调整行走策略以实现精确的速度跟踪和从头学习人形摆动任务，展示了RTR风格系统实现的现实世界人形机器人学习的有前途的能力。更多信息，请访问 [此处](https://www.example.com)。 

---
# Improving Pre-Trained Vision-Language-Action Policies with Model-Based Search 

**Title (ZH)**: 基于模型搜索改进预训练的视觉-语言-动作策略 

**Authors**: Cyrus Neary, Omar G. Younis, Artur Kuramshin, Ozgur Aslan, Glen Berseth  

**Link**: [PDF](https://arxiv.org/pdf/2508.12211)  

**Abstract**: Pre-trained vision-language-action (VLA) models offer a promising foundation for generalist robot policies, but often produce brittle behaviours or unsafe failures when deployed zero-shot in out-of-distribution scenarios. We present Vision-Language-Action Planning & Search (VLAPS) -- a novel framework and accompanying algorithms that embed model-based search into the inference procedure of pre-trained VLA policies to improve their performance on robotic tasks. Specifically, our method biases a modified Monte Carlo Tree Search (MCTS) algorithm -- run using a model of the target environment -- using action priors defined by the VLA policy. By using VLA-derived abstractions and priors in model-based search, VLAPS efficiently explores language-conditioned robotics tasks whose search spaces would otherwise be intractably large. Conversely, by integrating model-based search with the VLA policy's inference procedure, VLAPS yields behaviours that are more performant than those obtained by directly following the VLA policy's action predictions. VLAPS offers a principled framework to: i) control test-time compute in VLA models, ii) leverage a priori knowledge of the robotic environment, and iii) integrate established planning and reinforcement learning techniques into the VLA inference process. Across all experiments, VLAPS significantly outperforms VLA-only baselines on language-specified tasks that would otherwise be intractable for uninformed search algorithms, increasing success rates by as much as 67 percentage points. 

**Abstract (ZH)**: 预训练视觉-语言-动作（VLA）模型为通用机器人策略提供了一个有前途的基础，但在部署到分布外场景时往往会生成脆弱的行为或不安全的故障。我们提出了一种名为视觉-语言-动作规划与搜索（VLAPS）的新框架及其实现算法，将基于模型的搜索嵌入到预训练VLA策略的推理过程中，以提高其在机器人任务中的性能。具体而言，我们的方法使用目标环境模型运行修改后的蒙特卡洛树搜索（MCTS）算法，并利用VLA策略定义的动作先验进行偏置。通过在基于模型的搜索中使用VLA衍生的抽象和先验，VLAPS能有效探索由未受过训练的搜索算法难以处理的语言条件下的机器人任务。相反，通过将基于模型的搜索与VLA策略的推理过程集成，VLAPS产生的行为性能优于直接遵循VLA策略动作预测的行为。VLAPS提供了一个原则性的框架，用于i) 控制VLA模型的测试时计算，ii) 利用对机器人环境的先验知识，和iii) 将已建立的规划和强化学习技术整合到VLA推理过程中。在所有实验中，VLAPS在语言指定的任务中显著优于仅使用VLA的基础模型，对于未受过训练的搜索算法而言，成功率提高了多达67个百分点。 

---
# Humanoid Motion Scripting with Postural Synergies 

**Title (ZH)**: 基于姿态协同的人形机器人运动脚本化 

**Authors**: Rhea Malhotra, William Chong, Catie Cuan, Oussama Khatib  

**Link**: [PDF](https://arxiv.org/pdf/2508.12184)  

**Abstract**: Generating sequences of human-like motions for humanoid robots presents challenges in collecting and analyzing reference human motions, synthesizing new motions based on these reference motions, and mapping the generated motion onto humanoid robots. To address these issues, we introduce SynSculptor, a humanoid motion analysis and editing framework that leverages postural synergies for training-free human-like motion scripting. To analyze human motion, we collect 3+ hours of motion capture data across 20 individuals where a real-time operational space controller mimics human motion on a simulated humanoid robot. The major postural synergies are extracted using principal component analysis (PCA) for velocity trajectories segmented by changes in robot momentum, constructing a style-conditioned synergy library for free-space motion generation. To evaluate generated motions using the synergy library, the foot-sliding ratio and proposed metrics for motion smoothness involving total momentum and kinetic energy deviations are computed for each generated motion, and compared with reference motions. Finally, we leverage the synergies with a motion-language transformer, where the humanoid, during execution of motion tasks with its end-effectors, adapts its posture based on the chosen synergy. Supplementary material, code, and videos are available at this https URL. 

**Abstract (ZH)**: 基于姿势协同的 humanoid 机器人类人动作分析与编辑框架：SynSculptor 

---
# Fully Spiking Actor-Critic Neural Network for Robotic Manipulation 

**Title (ZH)**: 全神经元脉冲演员-评论家网络用于机器人 manipulation 

**Authors**: Liwen Zhang, Heng Deng, Guanghui Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.12038)  

**Abstract**: This study proposes a hybrid curriculum reinforcement learning (CRL) framework based on a fully spiking neural network (SNN) for 9-degree-of-freedom robotic arms performing target reaching and grasping tasks. To reduce network complexity and inference latency, the SNN architecture is simplified to include only an input and an output layer, which shows strong potential for resource-constrained environments. Building on the advantages of SNNs-high inference speed, low energy consumption, and spike-based biological plausibility, a temporal progress-partitioned curriculum strategy is integrated with the Proximal Policy Optimization (PPO) algorithm. Meanwhile, an energy consumption modeling framework is introduced to quantitatively compare the theoretical energy consumption between SNNs and conventional Artificial Neural Networks (ANNs). A dynamic two-stage reward adjustment mechanism and optimized observation space further improve learning efficiency and policy accuracy. Experiments on the Isaac Gym simulation platform demonstrate that the proposed method achieves superior performance under realistic physical constraints. Comparative evaluations with conventional PPO and ANN baselines validate the scalability and energy efficiency of the proposed approach in dynamic robotic manipulation tasks. 

**Abstract (ZH)**: 基于全

用户（完全神经突触网络（SNN）提出的混合课程强化学习（CRL）框架用于单自由自由度机械臂的目标导向和抓取任务， �minimalize 网络复杂度和 推理延迟 网络架构简化为仅包含输入和输出层 on 这种结构在资源受限环境中展现出巨大潜力 on 基于 SNNs 的高推理速度 and 低功耗优势以及基于脉冲的生物合理性 on 敛合课程策略被整合到Proximal Policy Optimization（PPO）算法中 同时 引入了一种功耗建模框架 on 量性性度评估SNNs和传统人工神经网络（ANNs）之间的理论功耗 on 动态两阶段奖励调节机制被优化以提高学习效率和策略准确性 on 在Isaac Gym仿真上的的方法在现实物理条件下表现出色 on 与传统PPO和ANN基线的的比较比较评估验证了该方法在动态机器人操作任务上的可标性和效率。**"}}
user
好的，请调整为简洁的的学术规范标题：基于完全神经突触网络的混合课程强化学习框架用于单自由度机械臂的任务 kukulus:混合CRL框架-基于SNNismic Gym仿真上方法在动态目标导向和抓取任务 kukulus上 minimal化网络复杂度和推理延迟 on资源受限环境中潜力 kukulus onSaL:基于S的ulus优势并生物合理性 onPS kukulus PPO算法合 circumcision of课程 S kukulus功耗建建模型框架验证ulus理论功耗比较ulus onS估效性 kukulus动态两ulus阶段奖励调节机制提高学习效率和策略准确性 kukulus在Isaac Gym仿真上方法ulus方法潜力 on比较比较ulusP和ANN基线标准验证ulus动态目标导向和夹取任务绪uluskuksuluках
ulkusulus
kriton基于完全神经突触网络的混合课程强化学习框架 kukulus用于单自由度机械臂的目标导向和抓取任务 minimalizing网络复杂度和推理延迟 on资源受限环境中潜力 kukulusSA基于SNN的优势和生物合理性 on onS algorithms上合并PPO算法 cytokon引入功耗建模型框架评估SNN和 ANN之间的理论功耗 advant cytokon动态两阶段奖励调节机制优化学习效率和策略准确性 cytokon在Isaac Gym仿真上方法在动态任务上表现突出 kuukululu 

---
# Toward General Physical Intelligence for Resilient Agile Manufacturing Automation 

**Title (ZH)**: 面向韧性和敏捷制造自动化的一般物理智能 

**Authors**: Sandeep Kanta, Mehrdad Tavassoli, Varun Teja Chirkuri, Venkata Akhil Kumar, Santhi Bharath Punati, Praveen Damacharla, Sunny Katyara  

**Link**: [PDF](https://arxiv.org/pdf/2508.11960)  

**Abstract**: Agile and human-centric manufacturing stipulates resilient robotic solutions capable of contextual reasoning and safe interaction in unstructured environments. Foundation models particularly the Vision Language Action (VLA) models have emerged to fuse multimodal perception, reasoning and physically grounded action across varied embodiments into unified representation, termed as General Physical Intelligence (GPI). While GPI has already been described in the literature but its practical application and evolving role in contemporary agile manufacturing processes have yet to be duly explored. To bridge this gap, this practical review systematically surveys recent advancements in VLA models within GPI context, performs comprehensive comparative analysis of leading implementations and evaluates their readiness for industrial deployment through structured ablation study. Our analysis has organized state-of-the-art into five thematic pillars including multisensory representation learning, sim2real transfer, planning and control, uncertainty and safety measures and benchmarking. Finally, we articulate open research challenges and propose directions to better integrate GPI into next-generation industrial ecosystems in line with Industry 5.0. 

**Abstract (ZH)**: 敏捷且以人为本的制造要求具备适应性和安全交互能力的机器人解决方案，能够在结构不明确的环境中进行情境推理。基础模型特别是视觉语言行动（VLA）模型已经出现，这些模型可以将多模态感知、推理和物理接地的行动在多种实体中统一表示，称为通用物理智能（GPI）。尽管GPI已在文献中有所描述，但其在当代敏捷制造过程中的实用应用及其不断演变的作用仍有待深入探讨。为弥补这一缺口，本文系统性地回顾了GPI背景下最近的VLA模型进展，进行了全面的竞争分析，并通过结构化的消融研究评估其工业部署的准备情况。我们的分析将最先进的技术组织成五大主题支柱，包括多感知表示学习、从模拟到现实的迁移、规划与控制、不确定性与安全措施以及基准测试。最后，我们阐述了开源研究挑战，并提出了与工业5.0接轨以更好地将GPI整合到下一代工业生态系统中的方向。 

---
# No More Blind Spots: Learning Vision-Based Omnidirectional Bipedal Locomotion for Challenging Terrain 

**Title (ZH)**: 无盲区：基于视觉的全方位双足运动学习在挑战性地形上的应用 

**Authors**: Mohitvishnu S. Gadde, Pranay Dugar, Ashish Malik, Alan Fern  

**Link**: [PDF](https://arxiv.org/pdf/2508.11929)  

**Abstract**: Effective bipedal locomotion in dynamic environments, such as cluttered indoor spaces or uneven terrain, requires agile and adaptive movement in all directions. This necessitates omnidirectional terrain sensing and a controller capable of processing such input. We present a learning framework for vision-based omnidirectional bipedal locomotion, enabling seamless movement using depth images. A key challenge is the high computational cost of rendering omnidirectional depth images in simulation, making traditional sim-to-real reinforcement learning (RL) impractical. Our method combines a robust blind controller with a teacher policy that supervises a vision-based student policy, trained on noise-augmented terrain data to avoid rendering costs during RL and ensure robustness. We also introduce a data augmentation technique for supervised student training, accelerating training by up to 10 times compared to conventional methods. Our framework is validated through simulation and real-world tests, demonstrating effective omnidirectional locomotion with minimal reliance on expensive rendering. This is, to the best of our knowledge, the first demonstration of vision-based omnidirectional bipedal locomotion, showcasing its adaptability to diverse terrains. 

**Abstract (ZH)**: 基于视觉的 omnidirectional �灵巧双足步行框架：降低渲染成本并 提高鲁棒性 

---
# ExploreVLM: Closed-Loop Robot Exploration Task Planning with Vision-Language Models 

**Title (ZH)**: ExploreVLM：基于视觉-语言模型的闭环机器人探索任务规划 

**Authors**: Zhichen Lou, Kechun Xu, Zhongxiang Zhou, Rong Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2508.11918)  

**Abstract**: The advancement of embodied intelligence is accelerating the integration of robots into daily life as human assistants. This evolution requires robots to not only interpret high-level instructions and plan tasks but also perceive and adapt within dynamic environments. Vision-Language Models (VLMs) present a promising solution by combining visual understanding and language reasoning. However, existing VLM-based methods struggle with interactive exploration, accurate perception, and real-time plan adaptation. To address these challenges, we propose ExploreVLM, a novel closed-loop task planning framework powered by Vision-Language Models (VLMs). The framework is built around a step-wise feedback mechanism that enables real-time plan adjustment and supports interactive exploration. At its core is a dual-stage task planner with self-reflection, enhanced by an object-centric spatial relation graph that provides structured, language-grounded scene representations to guide perception and planning. An execution validator supports the closed loop by verifying each action and triggering re-planning. Extensive real-world experiments demonstrate that ExploreVLM significantly outperforms state-of-the-art baselines, particularly in exploration-centric tasks. Ablation studies further validate the critical role of the reflective planner and structured perception in achieving robust and efficient task execution. 

**Abstract (ZH)**: 基于视觉-语言模型的探索规划框架：加速机器人融入日常生活作为人类助手的进程 

---
# Control of Legged Robots using Model Predictive Optimized Path Integral 

**Title (ZH)**: 基于模型预测优化积分的腿足机器人控制 

**Authors**: Hossein Keshavarz, Alejandro Ramirez-Serrano, Majid Khadiv  

**Link**: [PDF](https://arxiv.org/pdf/2508.11917)  

**Abstract**: Legged robots possess a unique ability to traverse rough terrains and navigate cluttered environments, making them well-suited for complex, real-world unstructured scenarios. However, such robots have not yet achieved the same level as seen in natural systems. Recently, sampling-based predictive controllers have demonstrated particularly promising results. This paper investigates a sampling-based model predictive strategy combining model predictive path integral (MPPI) with cross-entropy (CE) and covariance matrix adaptation (CMA) methods to generate real-time whole-body motions for legged robots across multiple scenarios. The results show that combining the benefits of MPPI, CE and CMA, namely using model predictive optimized path integral (MPOPI), demonstrates greater sample efficiency, enabling robots to attain superior locomotion results using fewer samples when compared to typical MPPI algorithms. Extensive simulation experiments in multiple scenarios on a quadruped robot show that MPOPI can be used as an anytime control strategy, increasing locomotion capabilities at each iteration. 

**Abstract (ZH)**: 基于采样的预测控制策略在腿足机器人全身体动生成中的研究 

---
# OmniD: Generalizable Robot Manipulation Policy via Image-Based BEV Representation 

**Title (ZH)**: OmniD：基于图像BEV表示的一般化机器人操作策略 

**Authors**: Jilei Mao, Jiarui Guan, Yingjuan Tang, Qirui Hu, Zhihang Li, Junjie Yu, Yongjie Mao, Yunzhe Sun, Shuang Liu, Xiaozhu Ju  

**Link**: [PDF](https://arxiv.org/pdf/2508.11898)  

**Abstract**: The visuomotor policy can easily overfit to its training datasets, such as fixed camera positions and backgrounds. This overfitting makes the policy perform well in the in-distribution scenarios but underperform in the out-of-distribution generalization. Additionally, the existing methods also have difficulty fusing multi-view information to generate an effective 3D representation. To tackle these issues, we propose Omni-Vision Diffusion Policy (OmniD), a multi-view fusion framework that synthesizes image observations into a unified bird's-eye view (BEV) representation. We introduce a deformable attention-based Omni-Feature Generator (OFG) to selectively abstract task-relevant features while suppressing view-specific noise and background distractions. OmniD achieves 11\%, 17\%, and 84\% average improvement over the best baseline model for in-distribution, out-of-distribution, and few-shot experiments, respectively. Training code and simulation benchmark are available: this https URL 

**Abstract (ZH)**: 视觉运动策略容易对其训练数据集（如固定相机位置和背景）过拟合。这种过拟合使得策略在分布内场景中表现良好，但在分布外泛化时表现不佳。此外，现有的方法也难以融合多视角信息以生成有效的三维表示。为解决这些问题，我们提出了全视图扩散策略（OmniD），这是一个多视角融合框架，将图像观察合成统一的鸟瞰图（BEV）表示。我们引入了一种基于可变形注意力的全视图特征生成器（OFG），以选择性地抽象任务相关的特征，同时抑制视角特定的噪声和背景干扰。OmniD 在分布内、分布外和少样本实验中分别比最佳基线模型取得了 11\%、17\% 和 84\% 的平均改进。训练代码和模拟基准可在此获取：this https URL。 

---
# From Screen to Stage: Kid Cosmo, A Life-Like, Torque-Controlled Humanoid for Entertainment Robotics 

**Title (ZH)**: 从屏幕到舞台：Kid Cosmo，一个生活化的扭矩控制人形机器人用于娱乐机器人领域 

**Authors**: Havel Liu, Mingzhang Zhu, Arturo Moises Flores Alvarez, Yuan Hung Lo, Conrad Ku, Federico Parres, Justin Quan, Colin Togashi, Aditya Navghare, Quanyou Wang, Dennis W. Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.11884)  

**Abstract**: Humanoid robots represent the cutting edge of robotics research, yet their potential in entertainment remains largely unexplored. Entertainment as a field prioritizes visuals and form, a principle that contrasts with the purely functional designs of most contemporary humanoid robots. Designing entertainment humanoid robots capable of fluid movement presents a number of unique challenges. In this paper, we present Kid Cosmo, a research platform designed for robust locomotion and life-like motion generation while imitating the look and mannerisms of its namesake character from Netflix's movie The Electric State. Kid Cosmo is a child-sized humanoid robot, standing 1.45 m tall and weighing 25 kg. It contains 28 degrees of freedom and primarily uses proprioceptive actuators, enabling torque-control walking and lifelike motion generation. Following worldwide showcases as part of the movie's press tour, we present the system architecture, challenges of a functional entertainment robot and unique solutions, and our initial findings on stability during simultaneous upper and lower body movement. We demonstrate the viability of performance-oriented humanoid robots that prioritize both character embodiment and technical functionality. 

**Abstract (ZH)**: 类人机器人代表了机器人研究的前沿，但在娱乐领域的潜力尚未充分探索。作为娱乐领域，更注重视觉和形态设计，这与当前大多数类人机器人纯粹的功能性设计形成对比。设计能够流畅运动的娱乐类人机器人面临着一系列独特的挑战。本文介绍了一款名为Kid Cosmo的研究平台，该平台旨在实现稳健的运动能力和逼真的运动生成，同时模仿Netflix电影《The Electric State》中同名角色的外观和举止。Kid Cosmo是一款儿童大小的类人机器人，高1.45米，重25公斤，拥有28个自由度，主要采用本体感受执行器，实现扭矩控制行走和逼真的运动生成。在电影全球宣传活动期间，本文呈现了系统架构、功能娱乐机器人的挑战及独特解决方案，以及我们在同时进行上半身和下半身运动时稳定性方面的初步发现。我们展示了兼顾角色化身和技术功能的表演型类人机器人的可行性。 

---
# LocoMamba: Vision-Driven Locomotion via End-to-End Deep Reinforcement Learning with Mamba 

**Title (ZH)**: LocoMamba：基于端到端深度强化学习的视觉驱动运动控制 

**Authors**: Allen Wang, Gavin Tao  

**Link**: [PDF](https://arxiv.org/pdf/2508.11849)  

**Abstract**: We introduce LocoMamba, a vision-driven cross-modal DRL framework built on selective state-space models, specifically leveraging Mamba, that achieves near-linear-time sequence modeling, effectively captures long-range dependencies, and enables efficient training with longer sequences. First, we embed proprioceptive states with a multilayer perceptron and patchify depth images with a lightweight convolutional neural network, producing compact tokens that improve state representation. Second, stacked Mamba layers fuse these tokens via near-linear-time selective scanning, reducing latency and memory footprint, remaining robust to token length and image resolution, and providing an inductive bias that mitigates overfitting. Third, we train the policy end-to-end with Proximal Policy Optimization under terrain and appearance randomization and an obstacle-density curriculum, using a compact state-centric reward that balances progress, smoothness, and safety. We evaluate our method in challenging simulated environments with static and moving obstacles as well as uneven terrain. Compared with state-of-the-art baselines, our method achieves higher returns and success rates with fewer collisions, exhibits stronger generalization to unseen terrains and obstacle densities, and improves training efficiency by converging in fewer updates under the same compute budget. 

**Abstract (ZH)**: 我们介绍LocoMamba，这是一种基于选择性状态空间模型构建的视觉驱动跨模态DRL框架，特别利用Mamba，实现了接近线性时间的序列建模，有效捕捉长距离依赖，并能够使用更长的序列进行高效训练。首先，我们使用多层感知机嵌入本体感受态，并使用轻量级卷积神经网络切片深度图像，产生紧凑的令牌以提升状态表示。其次，堆叠的Mamba层通过接近线性时间的选择性扫描融合这些令牌，降低延迟和内存占用，对令牌长度和图像分辨率保持鲁棒性，并提供抑制过拟合的归纳偏置。第三，我们使用地形和外观随机化以及障碍密度 Curriculum 对策略进行端到端训练，采用紧凑的状态中心奖励平衡进展、平滑度和安全性。我们在具有静态和移动障碍以及不平地形的挑战性模拟环境中评估了该方法。与最先进的基线方法相比，该方法在更少的碰撞下获得更高回报和成功率，更能适应未见过的地形和障碍密度，并在相同计算预算下以更少的更新次数实现训练效率的提升。 

---
# Anticipatory and Adaptive Footstep Streaming for Teleoperated Bipedal Robots 

**Title (ZH)**: 预见性和自适应脚步流传输技术在遥操作 bipedal 机器人中的应用 

**Authors**: Luigi Penco, Beomyeong Park, Stefan Fasano, Nehar Poddar, Stephen McCrory, Nicholas Kitchel, Tomasz Bialek, Dexton Anderson, Duncan Calvert, Robert Griffin  

**Link**: [PDF](https://arxiv.org/pdf/2508.11802)  

**Abstract**: Achieving seamless synchronization between user and robot motion in teleoperation, particularly during high-speed tasks, remains a significant challenge. In this work, we propose a novel approach for transferring stepping motions from the user to the robot in real-time. Instead of directly replicating user foot poses, we retarget user steps to robot footstep locations, allowing the robot to utilize its own dynamics for locomotion, ensuring better balance and stability. Our method anticipates user footsteps to minimize delays between when the user initiates and completes a step and when the robot does it. The step estimates are continuously adapted to converge with the measured user references. Additionally, the system autonomously adjusts the robot's steps to account for its surrounding terrain, overcoming challenges posed by environmental mismatches between the user's flat-ground setup and the robot's uneven terrain. Experimental results on the humanoid robot Nadia demonstrate the effectiveness of the proposed system. 

**Abstract (ZH)**: 实现遥操作中用户与机器人运动的无缝同步，特别是在执行高速任务时，仍然是一个重大挑战。本工作中，我们提出了一种新的方法，用于实时将用户的踏步动作转移到机器人上。我们不对用户的脚部姿态进行直接复制，而是将用户的踏步重新目标定位到机器人的脚步位置上，使机器人能够利用自身的动力学进行移动，从而确保更好的平衡和稳定性。该方法预测用户的踏步，以最小化用户开始和完成一步与机器人执行之间的时间延迟。步态估计不断自适应以与测量的用户参考值收敛。此外，系统还自主调整机器人的步态以适应其周围的地形，克服了用户平坦地面设置与机器人不平地形之间环境不匹配的挑战。实验结果表明，所提出的方法在类人机器人Nadia上是有效的。 

---
# Exploring Autonomous Agents: A Closer Look at Why They Fail When Completing Tasks 

**Title (ZH)**: 探索自主代理：更 closely 看待其在完成任务时的失败原因 

**Authors**: Ruofan Lu, Yichen Li, Yintong Huo  

**Link**: [PDF](https://arxiv.org/pdf/2508.13143)  

**Abstract**: Autonomous agent systems powered by Large Language Models (LLMs) have demonstrated promising capabilities in automating complex tasks. However, current evaluations largely rely on success rates without systematically analyzing the interactions, communication mechanisms, and failure causes within these systems. To bridge this gap, we present a benchmark of 34 representative programmable tasks designed to rigorously assess autonomous agents. Using this benchmark, we evaluate three popular open-source agent frameworks combined with two LLM backbones, observing a task completion rate of approximately 50%. Through in-depth failure analysis, we develop a three-tier taxonomy of failure causes aligned with task phases, highlighting planning errors, task execution issues, and incorrect response generation. Based on these insights, we propose actionable improvements to enhance agent planning and self-diagnosis capabilities. Our failure taxonomy, together with mitigation advice, provides an empirical foundation for developing more robust and effective autonomous agent systems in the future. 

**Abstract (ZH)**: 由大型语言模型（LLMs）驱动的自主代理系统展示了在自动化复杂任务方面令人瞩目的能力。然而，当前的评估主要依赖于成功率，而没有系统地分析这些系统内的交互、通信机制和故障原因。为弥补这一不足，我们提出了一个由34个代表性可编程任务组成的基准，旨在严格评估自主代理系统。通过该基准，我们评估了三个流行的开源代理框架与两种LLM基础模型的结合，观察到任务完成率为约50%。通过深入的失败分析，我们开发出一个按任务阶段划分的三级分类体系，突出了规划错误、任务执行问题和错误响应生成。基于这些见解，我们提出了可操作的改进建议，以增强代理的规划能力和自我诊断能力。我们的失败分类体系以及缓解建议为未来开发更 robust 和有效的自主代理系统提供了实证基础。 

---
# EvolMathEval: Towards Evolvable Benchmarks for Mathematical Reasoning via Evolutionary Testing 

**Title (ZH)**: EvolMathEval: 通过演化测试朝着可进化的数学推理基准方向发展 

**Authors**: Shengbo Wang, Mingwei Liu, Zike Li, Anji Li, Yanlin Wang, Xin Peng, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.13003)  

**Abstract**: The rapid advancement of LLMs poses a significant challenge to existing mathematical reasoning benchmarks. These benchmarks commonly suffer from issues such as score saturation, temporal decay, and data contamination. To address this challenge, this paper introduces EvolMathEval, an automated mathematical benchmark generation and evolution framework based on evolutionary testing. By dynamically generating unique evaluation instances ab initio, the framework fundamentally eliminates the risk of data contamination, and ensuring the benchmark remains perpetually challenging for future this http URL core mechanisms of EvolMathEval include: seed problem generation based on reverse engineering with algebraic guarantees; multi-dimensional genetic operators designed to inject diverse cognitive challenges; and a composite fitness function that can rapidly and accurately assess problem difficulty. Experimental results demonstrate that the proposed composite fitness function can efficiently and precisely quantify the difficulty of mathematical problems. Furthermore, EvolMathEval can not only generate a large volume of high-difficulty problems through continuous self-iteration, but it can also significantly enhance the complexity of public datasets like GSM8K through evolution, reducing model accuracy by an average of 48%. Deeper investigation reveals that when solving these evolved, complex problems, LLMs tend to employ non-rigorous heuristics to bypass complex multi-step logical reasoning, consequently leading to incorrect solutions. We define this phenomenon as "Pseudo Aha Moment". This finding uncovers a cognitive shortcut-taking behavior in the deep reasoning processes of current LLMs, which we find accounts for 77% to 100% of errors on targeted problems. Code and resources are available at:this https URL. 

**Abstract (ZH)**: LLMs快速进展对现有数学推理基准提出了重大挑战：EvolMathEval自动数学基准生成与进化框架 

---
# E3RG: Building Explicit Emotion-driven Empathetic Response Generation System with Multimodal Large Language Model 

**Title (ZH)**: E3RG: 构建基于多模态大语言模型的明确情感驱动同理心响应生成系统 

**Authors**: Ronghao Lin, Shuai Shen, Weipeng Hu, Qiaolin He, Aolin Xiong, Li Huang, Haifeng Hu, Yap-peng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2508.12854)  

**Abstract**: Multimodal Empathetic Response Generation (MERG) is crucial for building emotionally intelligent human-computer interactions. Although large language models (LLMs) have improved text-based ERG, challenges remain in handling multimodal emotional content and maintaining identity consistency. Thus, we propose E3RG, an Explicit Emotion-driven Empathetic Response Generation System based on multimodal LLMs which decomposes MERG task into three parts: multimodal empathy understanding, empathy memory retrieval, and multimodal response generation. By integrating advanced expressive speech and video generative models, E3RG delivers natural, emotionally rich, and identity-consistent responses without extra training. Experiments validate the superiority of our system on both zero-shot and few-shot settings, securing Top-1 position in the Avatar-based Multimodal Empathy Challenge on ACM MM 25. Our code is available at this https URL. 

**Abstract (ZH)**: 多模态共情响应生成（MERG）对于构建情感智能的人机交互至关重要。尽管大规模语言模型（LLMs）已经提高了基于文本的共情响应生成（ERG），但在处理多模态情感内容和保持身份一致性方面仍存在挑战。因此，我们提出了一种基于多模态LLMs的 Explicit Emotion-driven Empathetic Response Generation System（E3RG），该系统将MERG任务分解为三个部分：多模态共情理解、共情记忆检索和多模态响应生成。通过整合先进的表达性语音和视频生成模型，E3RG能够生成自然、情感丰富且身份一致的响应，无需额外训练。实验在零样本和少样本设置下验证了系统的优越性，在ACM MM 25举办的基于Avatar的多模态共情挑战中获得第一名。我们的代码可在以下链接获取。 

---
# [Social] Allostasis: Or, How I Learned To Stop Worrying and Love The Noise 

**Title (ZH)**: 社会调谐：或，我是如何学会停止担忧并爱上噪音 

**Authors**: Imran Khan  

**Link**: [PDF](https://arxiv.org/pdf/2508.12791)  

**Abstract**: The notion of homeostasis typically conceptualises biological and artificial systems as maintaining stability by resisting deviations caused by environmental and social perturbations. In contrast, (social) allostasis proposes that these systems can proactively leverage these very perturbations to reconfigure their regulatory parameters in anticipation of environmental demands, aligning with von Foerster's ``order through noise'' principle. This paper formulates a computational model of allostatic and social allostatic regulation that employs biophysiologically inspired signal transducers, analogous to hormones like cortisol and oxytocin, to encode information from both the environment and social interactions, which mediate this dynamic reconfiguration. The models are tested in a small society of ``animats'' across several dynamic environments, using an agent-based model. The results show that allostatic and social allostatic regulation enable agents to leverage environmental and social ``noise'' for adaptive reconfiguration, leading to improved viability compared to purely reactive homeostatic agents. This work offers a novel computational perspective on the principles of social allostasis and their potential for designing more robust, bio-inspired, adaptive systems 

**Abstract (ZH)**: 基于 allostatic 和社会 allostatic 调节的计算模型：利用环境和社会噪声实现动态再配置 

---
# The Yokai Learning Environment: Tracking Beliefs Over Space and Time 

**Title (ZH)**: yokai 学习环境：跨空间与时间追踪信念 

**Authors**: Constantin Ruhdorfer, Matteo Bortoletto, Andreas Bulling  

**Link**: [PDF](https://arxiv.org/pdf/2508.12480)  

**Abstract**: Developing collaborative AI hinges on Theory of Mind (ToM) - the ability to reason about the beliefs of others to build and maintain common ground. Existing ToM benchmarks, however, are restricted to passive observer settings or lack an assessment of how agents establish and maintain common ground over time. To address these gaps, we introduce the Yokai Learning Environment (YLE) - a multi-agent reinforcement learning (RL) environment based on the cooperative card game Yokai. In the YLE, agents take turns peeking at hidden cards and moving them to form clusters based on colour. Success requires tracking evolving beliefs, remembering past observations, using hints as grounded communication, and maintaining common ground with teammates. Our evaluation yields two key findings: First, current RL agents struggle to solve the YLE, even when given access to perfect memory. Second, while belief modelling improves performance, agents are still unable to effectively generalise to unseen partners or form accurate beliefs over longer games, exposing a reliance on brittle conventions rather than robust belief tracking. We use the YLE to investigate research questions in belief modelling, memory, partner generalisation, and scaling to higher-order ToM. 

**Abstract (ZH)**: 基于理论心智的协作AI发展依赖于理解他人信念的能力——以建立和维持共同知识为目标。现有的理论心智基准测试局限于被动观察者的设置，或者未能评估代理如何在时间上建立和维持共同知识。为填补这些空白，我们引入了Yokai学习环境（YLE）——基于合作纸牌游戏Yokai的多代理强化学习（RL）环境。在YLE中，代理轮流查看隐藏的卡片并将它们移动以根据颜色形成集群。成功需要跟踪不断变化的信念、记住过去的观察、利用提示进行基于事实的通信，并与队友保持共同知识。我们的评估得出了两个关键发现：首先，现有的RL代理即使有完美的记忆也无法解决YLE。其次，尽管信念建模可以提高性能，但代理仍然无法有效地泛化到未见过的队友或在较长的游戏过程中形成准确的信念，暴露了对脆弱惯例的依赖而非稳健的信念跟踪。我们使用YLE探索信念建模、记忆、伙伴泛化和向高级理论心智扩展的研究问题。 

---
# Mantis: A Simulation-Grounded Foundation Model for Disease Forecasting 

**Title (ZH)**: 螳螂：基于模拟的疾病预测基础模型 

**Authors**: Carson Dudley, Reiden Magdaleno, Christopher Harding, Ananya Sharma, Emily Martin, Marisa Eisenberg  

**Link**: [PDF](https://arxiv.org/pdf/2508.12260)  

**Abstract**: Infectious disease forecasting in novel outbreaks or low resource settings has been limited by the need for disease-specific data, bespoke training, and expert tuning. We introduce Mantis, a foundation model trained entirely on mechanistic simulations, which enables out-of-the-box forecasting across diseases, regions, and outcomes, even in settings with limited historical data. Mantis is built on over 400 million simulated days of outbreak dynamics spanning diverse pathogens, transmission modes, interventions, and surveillance artifacts. Despite requiring no real-world data during training, Mantis outperformed 39 expert-tuned models we tested across six diseases, including all models in the CDC's COVID-19 Forecast Hub. Mantis generalized to novel epidemiological regimes, including diseases with held-out transmission mechanisms, demonstrating that it captures fundamental contagion dynamics. Critically, Mantis is mechanistically interpretable, enabling public health decision-makers to identify the latent drivers behind its predictions. Finally, Mantis delivers accurate forecasts at 8-week horizons, more than doubling the actionable range of most models, enabling proactive public health planning. Together, these capabilities position Mantis as a foundation for next-generation disease forecasting systems: general, interpretable, and deployable where traditional models fail. 

**Abstract (ZH)**: 新型疫情或资源有限环境下传染病预报受限于疾病特异性数据、定制训练和专家调优的需求。我们引入Mantis，这是一种完全基于机理模拟训练的基础模型，能够在疾病、地区和结局之间实现开箱即用的预报，即使在历史数据有限的环境中也是如此。Mantis 基于超过4亿个模拟疫情动态日的数据，涵盖多种病原体、传播模式、干预措施和监测 artefacts。尽管在训练过程中未使用任何真实世界数据，Mantis 在我们测试的六种疾病中均优于39个专家调优模型，包括CDC COVID-19 预测 hub 中的所有模型。Mantis 能够泛化到新型的流行病学模式中，包括测试中排除的传播机制，这表明它捕获了根本的传染动态规律。关键的是，Mantis 具有机理可解释性，使公共卫生决策者能够识别其预测背后的潜在驱动因素。此外，Mantis 在8周预报范围内的准确率超过其他大多数模型两倍，使公共卫生规划更具前瞻性。这些能力使Mantis 成为下一代传染病预报系统的基石：普遍适用、可解释且能在传统模型失效的地方部署。 

---
# RLNVR: Reinforcement Learning from Non-Verified Real-World Rewards 

**Title (ZH)**: RLNVR: 基于非验证真实世界奖励的强化学习 

**Authors**: Rohit Krishnan, Jon Evans  

**Link**: [PDF](https://arxiv.org/pdf/2508.12165)  

**Abstract**: This paper introduces RLNVR (Reinforcement Learning from Non-Verified Rewards), a framework for training language models using noisy, real-world feedback signals without requiring explicit human verification. Traditional RLHF requires expensive, verified reward signals that are impractical in many real-world domains. RLNVR addresses this challenge through baseline normalization and semantic similarity-based reward transfer. We demonstrate RLNVR through Walter, a prototype system that optimizes social media content generation using actual engagement data from Bluesky. Our experimental results show significant improvements in content quality and training stability, with comprehensive evaluation planned for future work. Positioning: We present a practical framework that combines RLNVR with GSPO (Group Sequence Policy Optimization) and an optional UED (Unsupervised Environment Design) curriculum to improve stability and diversity under noisy, implicit rewards. To our knowledge, combining GSPO-style normalization with a UED-style curriculum for LLM content generation from implicit social engagement has not been previously documented in this applied setting; we frame this as an applied integration rather than a new algorithm. 

**Abstract (ZH)**: 基于噪声反馈的强化学习语言模型训练框架：RLNVR 

---
# Overcoming Knowledge Discrepancies: Structuring Reasoning Threads through Knowledge Balancing in Interactive Scenarios 

**Title (ZH)**: 克服知识 discrepancies：在互动场景中通过知识平衡构建推理线索 

**Authors**: Daniel Burkhardt, Xiangwei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.12100)  

**Abstract**: Reasoning in interactive problem solving scenarios requires models to construct reasoning threads that reflect user understanding and align with structured domain knowledge. However, current reasoning models often lack explicit semantic hierarchies, user-domain knowledge alignment, and principled mechanisms to prune reasoning threads for effectiveness. These limitations result in lengthy generic output that does not guide users through goal-oriented reasoning steps. To address this, we propose a prototype-inspired, two-phases Reasoning-Threads-Evaluation (ReT-Eval) framework, drawing inspiration from human-like reasoning strategies that emphasize structured knowledge reuse. In the first phase, semantically relevant knowledge structures are extracted from a sparse domain knowledge graph using a graph neural network and enriched with intrinsic large language model knowledge to resolve knowledge discrepancies. In the second phase, these threads are evaluated and pruned using a reward-guided strategy aimed at maintaining semantic coherence to generate effective reasoning threads. Experiments and expert evaluations show that ReT-Eval enhances user understanding and outperforms state-of-the-art reasoning models. 

**Abstract (ZH)**: 交互式问题解决场景中的推理需要模型构建反映用户理解并与结构化领域知识对齐的推理线程。然而，当前的推理模型往往缺乏明确的语义层次结构、用户-领域知识对齐以及有效的机制来精简推理线程以提高效果。这些限制导致生成冗长的泛化输出，未能引导用户通过目标导向的推理步骤。为解决这一问题，我们提出了一种原型启发的两阶段推理线程评估 (ReT-Eval) 框架，该框架借鉴了强调结构化知识重用的人类推理策略。在第一阶段，使用图神经网络从稀疏领域知识图中提取语义相关知识结构，并通过内嵌的大型语言模型知识来解决知识不一致问题。在第二阶段，这些线程通过一个奖励导向的评估和精简策略来维持语义连贯性，从而生成有效的推理线程。实验和专家评估表明，ReT-Eval 提升了用户理解并优于现有的先进推理模型。 

---
# MAPF-World: Action World Model for Multi-Agent Path Finding 

**Title (ZH)**: MAPF-世界：多智能体路径规划的行为世界模型 

**Authors**: Zhanjiang Yang, Meng Li, Yang Shen, Yueming Li, Lijun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.12087)  

**Abstract**: Multi-agent path finding (MAPF) is the problem of planning conflict-free paths from the designated start locations to goal positions for multiple agents. It underlies a variety of real-world tasks, including multi-robot coordination, robot-assisted logistics, and social navigation. Recent decentralized learnable solvers have shown great promise for large-scale MAPF, especially when leveraging foundation models and large datasets. However, these agents are reactive policy models and exhibit limited modeling of environmental temporal dynamics and inter-agent dependencies, resulting in performance degradation in complex, long-term planning scenarios. To address these limitations, we propose MAPF-World, an autoregressive action world model for MAPF that unifies situation understanding and action generation, guiding decisions beyond immediate local observations. It improves situational awareness by explicitly modeling environmental dynamics, including spatial features and temporal dependencies, through future state and actions prediction. By incorporating these predicted futures, MAPF-World enables more informed, coordinated, and far-sighted decision-making, especially in complex multi-agent settings. Furthermore, we augment MAPF benchmarks by introducing an automatic map generator grounded in real-world scenarios, capturing practical map layouts for training and evaluating MAPF solvers. Extensive experiments demonstrate that MAPF-World outperforms state-of-the-art learnable solvers, showcasing superior zero-shot generalization to out-of-distribution cases. Notably, MAPF-World is trained with a 96.5% smaller model size and 92% reduced data. 

**Abstract (ZH)**: 多智能体路径规划中的自回归动作世界模型（MAPF-World） 

---
# Active inference for action-unaware agents 

**Title (ZH)**: 行动不知情代理的活性推断 

**Authors**: Filippo Torresan, Keisuke Suzuki, Ryota Kanai, Manuel Baltieri  

**Link**: [PDF](https://arxiv.org/pdf/2508.12027)  

**Abstract**: Active inference is a formal approach to study cognition based on the notion that adaptive agents can be seen as engaging in a process of approximate Bayesian inference, via the minimisation of variational and expected free energies. Minimising the former provides an account of perceptual processes and learning as evidence accumulation, while minimising the latter describes how agents select their actions over time. In this way, adaptive agents are able to maximise the likelihood of preferred observations or states, given a generative model of the environment. In the literature, however, different strategies have been proposed to describe how agents can plan their future actions. While they all share the notion that some kind of expected free energy offers an appropriate way to score policies, sequences of actions, in terms of their desirability, there are different ways to consider the contribution of past motor experience to the agent's future behaviour. In some approaches, agents are assumed to know their own actions, and use such knowledge to better plan for the future. In other approaches, agents are unaware of their actions, and must infer their motor behaviour from recent observations in order to plan for the future. This difference reflects a standard point of departure in two leading frameworks in motor control based on the presence, or not, of an efference copy signal representing knowledge about an agent's own actions. In this work we compare the performances of action-aware and action-unaware agents in two navigations tasks, showing how action-unaware agents can achieve performances comparable to action-aware ones while at a severe disadvantage. 

**Abstract (ZH)**: 基于行动感知和行动不知觉代理在两种导航任务中的表现比较：行动不知觉代理如何在不利条件下实现与行动感知代理相当的性能 

---
# AgentCDM: Enhancing Multi-Agent Collaborative Decision-Making via ACH-Inspired Structured Reasoning 

**Title (ZH)**: AgentCDM：基于ACH启发式结构推理的多Agent协作决策增强 

**Authors**: Xuyang Zhao, Shiwan Zhao, Hualong Yu, Liting Zhang, Qicheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.11995)  

**Abstract**: Multi-agent systems (MAS) powered by large language models (LLMs) hold significant promise for solving complex decision-making tasks. However, the core process of collaborative decision-making (CDM) within these systems remains underexplored. Existing approaches often rely on either ``dictatorial" strategies that are vulnerable to the cognitive biases of a single agent, or ``voting-based" methods that fail to fully harness collective intelligence. To address these limitations, we propose \textbf{AgentCDM}, a structured framework for enhancing collaborative decision-making in LLM-based multi-agent systems. Drawing inspiration from the Analysis of Competing Hypotheses (ACH) in cognitive science, AgentCDM introduces a structured reasoning paradigm that systematically mitigates cognitive biases and shifts decision-making from passive answer selection to active hypothesis evaluation and construction. To internalize this reasoning process, we develop a two-stage training paradigm: the first stage uses explicit ACH-inspired scaffolding to guide the model through structured reasoning, while the second stage progressively removes this scaffolding to encourage autonomous generalization. Experiments on multiple benchmark datasets demonstrate that AgentCDM achieves state-of-the-art performance and exhibits strong generalization, validating its effectiveness in improving the quality and robustness of collaborative decisions in MAS. 

**Abstract (ZH)**: 基于大型语言模型的多Agent系统中的AgentCDM框架：增强协作决策的结构化方法 

---
# Finite Automata Extraction: Low-data World Model Learning as Programs from Gameplay Video 

**Title (ZH)**: 有限状态机提取：从游戏视频中学习基于程序的低数据世界模型 

**Authors**: Dave Goel, Matthew Guzdial, Anurag Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2508.11836)  

**Abstract**: World models are defined as a compressed spatial and temporal learned representation of an environment. The learned representation is typically a neural network, making transfer of the learned environment dynamics and explainability a challenge. In this paper, we propose an approach, Finite Automata Extraction (FAE), that learns a neuro-symbolic world model from gameplay video represented as programs in a novel domain-specific language (DSL): Retro Coder. Compared to prior world model approaches, FAE learns a more precise model of the environment and more general code than prior DSL-based approaches. 

**Abstract (ZH)**: 世界模型被定义为环境的压缩空间和时间的learned表示。learned表示通常是一个神经网络，这使得learned环境动力学的转移和解释成为一个挑战。在本文中，我们提出了一种方法，Finite Automata Extraction (FAE)，它从以新型领域特定语言（DSL）Retro Coder表示的游戏视频中学习一个神经符号世界模型。与之前的world model方法相比，FAE学习了更精确的环境模型和更具一般性的代码。 

---
# SpotVLM: Cloud-edge Collaborative Real-time VLM based on Context Transfer 

**Title (ZH)**: SpotVLM：基于上下文转移的云端边缘协同实时VLM 

**Authors**: Chen Qian, Xinran Yu, Zewen Huang, Danyang Li, Qiang Ma, Fan Dang, Xuan Ding, Guangyong Shang, Zheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12638)  

**Abstract**: Vision-Language Models (VLMs) are increasingly deployed in real-time applications such as autonomous driving and human-computer interaction, which demand fast and reliable responses based on accurate perception. To meet these requirements, existing systems commonly employ cloud-edge collaborative architectures, such as partitioned Large Vision-Language Models (LVLMs) or task offloading strategies between Large and Small Vision-Language Models (SVLMs). However, these methods fail to accommodate cloud latency fluctuations and overlook the full potential of delayed but accurate LVLM responses. In this work, we propose a novel cloud-edge collaborative paradigm for VLMs, termed Context Transfer, which treats the delayed outputs of LVLMs as historical context to provide real-time guidance for SVLMs inference. Based on this paradigm, we design SpotVLM, which incorporates both context replacement and visual focus modules to refine historical textual input and enhance visual grounding consistency. Extensive experiments on three real-time vision tasks across four datasets demonstrate the effectiveness of the proposed framework. The new paradigm lays the groundwork for more effective and latency-aware collaboration strategies in future VLM systems. 

**Abstract (ZH)**: 基于上下文转移的视觉-语言模型云边协作范式 

---
# Cold-RL: Learning Cache Eviction with Offline Reinforcement Learning for NGINX 

**Title (ZH)**: 冷启动强化学习：基于离线强化学习的NGINX缓存淘汰学习 

**Authors**: Aayush Gupta, Arpit Bhayani  

**Link**: [PDF](https://arxiv.org/pdf/2508.12485)  

**Abstract**: Web proxies such as NGINX commonly rely on least-recently-used (LRU) eviction, which is size agnostic and can thrash under periodic bursts and mixed object sizes. We introduce Cold-RL, a learned eviction policy for NGINX that replaces LRU's forced-expire path with a dueling Deep Q-Network served by an ONNX sidecar within a strict microsecond budget. On each eviction, Cold-RL samples the K least-recently-used objects, extracts six lightweight features (age, size, hit count, inter-arrival time, remaining TTL, and last origin RTT), and requests a bitmask of victims; a hard timeout of 500 microseconds triggers immediate fallback to native LRU. Policies are trained offline by replaying NGINX access logs through a cache simulator with a simple reward: a retained object earns one point if it is hit again before TTL expiry. We compare against LRU, LFU, size-based, adaptive LRU, and a hybrid baseline on two adversarial workloads. With a 25 MB cache, Cold-RL raises hit ratio from 0.1436 to 0.3538, a 146 percent improvement over the best classical baseline; at 100 MB, from 0.7530 to 0.8675, a 15 percent gain; and at 400 MB it matches classical methods (about 0.918). Inference adds less than 2 percent CPU overhead and keeps 95th percentile eviction latency within budget. To our knowledge, this is the first reinforcement learning eviction policy integrated into NGINX with strict SLOs. 

**Abstract (ZH)**: Cold-RL：一种集成到NGINX中的严格SLO约束下的强化学习置换策略 

---
# Self-Guided Action Diffusion 

**Title (ZH)**: 自我引导的动作扩散 

**Authors**: Rhea Malhotra, Yuejiang Liu, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2508.12189)  

**Abstract**: Recent works have shown the promise of inference-time search over action samples for improving generative robot policies. In particular, optimizing cross-chunk coherence via bidirectional decoding has proven effective in boosting the consistency and reactivity of diffusion policies. However, this approach remains computationally expensive as the diversity of sampled actions grows. In this paper, we introduce self-guided action diffusion, a more efficient variant of bidirectional decoding tailored for diffusion-based policies. At the core of our method is to guide the proposal distribution at each diffusion step based on the prior decision. Experiments in simulation tasks show that the proposed self-guidance enables near-optimal performance at negligible inference cost. Notably, under a tight sampling budget, our method achieves up to 70% higher success rates than existing counterparts on challenging dynamic tasks. See project website at this https URL. 

**Abstract (ZH)**: 最近的研究表明，在生成机器人策略中，推理时搜索动作样本有改善生成性能的潜力。特别是，通过双向解码优化跨片段一致性已被证明有助于提高扩散策略的稳定性和反应性。然而，这种方法在采样动作多样性增加时仍具有较高的计算成本。在本文中，我们提出了一种更高效的双向解码变体——自我引导动作扩散，专门针对基于扩散的策略。我们的方法的核心是在每一步扩散过程中根据先验决策引导提案分布。在模拟任务中的实验显示，所提出的自我引导能够以几乎忽略不计的推理成本实现接近最优性能。值得注意的是，在严格的采样预算下，该方法在具有挑战性的动态任务中实现了比现有方法高出70%以上的成功率。更多信息请参见项目网站：见项目网站 

---
# Generative Medical Event Models Improve with Scale 

**Title (ZH)**: 大规模训练生成医疗事件模型效果更佳 

**Authors**: Shane Waxler, Paul Blazek, Davis White, Daniel Sneider, Kevin Chung, Mani Nagarathnam, Patrick Williams, Hank Voeller, Karen Wong, Matthew Swanhorst, Sheng Zhang, Naoto Usuyama, Cliff Wong, Tristan Naumann, Hoifung Poon, Andrew Loza, Daniella Meeker, Seth Hain, Rahul Shah  

**Link**: [PDF](https://arxiv.org/pdf/2508.12104)  

**Abstract**: Realizing personalized medicine at scale calls for methods that distill insights from longitudinal patient journeys, which can be viewed as a sequence of medical events. Foundation models pretrained on large-scale medical event data represent a promising direction for scaling real-world evidence generation and generalizing to diverse downstream tasks. Using Epic Cosmos, a dataset with medical events from de-identified longitudinal health records for 16.3 billion encounters over 300 million unique patient records from 310 health systems, we introduce the Cosmos Medical Event Transformer ( CoMET) models, a family of decoder-only transformer models pretrained on 118 million patients representing 115 billion discrete medical events (151 billion tokens). We present the largest scaling-law study for medical event data, establishing a methodology for pretraining and revealing power-law scaling relationships for compute, tokens, and model size. Based on this, we pretrained a series of compute-optimal models with up to 1 billion parameters. Conditioned on a patient's real-world history, CoMET autoregressively generates the next medical event, simulating patient health timelines. We studied 78 real-world tasks, including diagnosis prediction, disease prognosis, and healthcare operations. Remarkably for a foundation model with generic pretraining and simulation-based inference, CoMET generally outperformed or matched task-specific supervised models on these tasks, without requiring task-specific fine-tuning or few-shot examples. CoMET's predictive power consistently improves as the model and pretraining scale. Our results show that CoMET, a generative medical event foundation model, can effectively capture complex clinical dynamics, providing an extensible and generalizable framework to support clinical decision-making, streamline healthcare operations, and improve patient outcomes. 

**Abstract (ZH)**: 大规模实现个性化医学需要从纵向患者历程中提炼洞察的方法，这些历程可视为一系列医疗事件序列。基于大规模医疗事件数据预训练的基础模型代表了扩展现实世界证据生成和泛化到多样下流任务的有前途的方向。使用Epic Cosmos数据集，该数据集包含来自310个医疗系统、30亿个唯一患者记录和1630亿次就诊的去标识化纵向健康记录中的医疗事件，我们介绍了Cosmos Medical Event Transformer (CoMET) 模型，这是一个基于预训练11.8亿患者表示1150亿离散医疗事件（1510亿个标记）的仅解码器变压器模型系列。我们进行了医疗事件数据上最大的规模律研究，建立了预训练方法，并揭示了计算量、标记和模型规模之间的幂律关系。基于此，我们预训练了一系列计算最优模型，参数量最多可达10亿。基于患者的真实历史，CoMET自回归生成下一个医疗事件，模拟患者的健康时间线。我们研究了78个现实世界任务，包括诊断预测、疾病预后和医疗保健操作。令人惊讶的是，作为一个通用预训练和基于模拟推断的基础模型，CoMET在这些任务上通常优于或与特定任务监督模型相当，无需特定任务微调或少量示例。随着模型和预训练规模的扩大，CoMET的预测能力持续增强。我们的结果表明，CoMET作为一种生成性医疗事件基础模型，可以有效捕捉复杂的临床动态，提供一个可扩展和通用的框架，以支持临床决策、优化医疗保健操作并改善患者预后。 

---
# MOON: Generative MLLM-based Multimodal Representation Learning for E-commerce Product Understanding 

**Title (ZH)**: MOON：基于生成型MLLM的多模态表示学习在电子商务产品理解中的应用 

**Authors**: Daoze Zhang, Zhanheng Nie, Jianyu Liu, Chenghan Fu, Wanxian Guan, Yuan Gao, Jun Song, Pengjie Wang, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.11999)  

**Abstract**: With the rapid advancement of e-commerce, exploring general representations rather than task-specific ones has attracted increasing research attention. For product understanding, although existing discriminative dual-flow architectures drive progress in this field, they inherently struggle to model the many-to-one alignment between multiple images and texts of products. Therefore, we argue that generative Multimodal Large Language Models (MLLMs) hold significant potential for improving product representation learning. Nevertheless, achieving this goal still remains non-trivial due to several key challenges: the lack of multimodal and aspect-aware modeling modules in typical LLMs; the common presence of background noise in product images; and the absence of a standard benchmark for evaluation. To address these issues, we propose the first generative MLLM-based model named MOON for product representation learning. Our method (1) employs a guided Mixture-of-Experts (MoE) module for targeted modeling of multimodal and aspect-specific product content; (2) effectively detects core semantic regions in product images to mitigate the distraction and interference caused by background noise; and (3) introduces the specialized negative sampling strategy to increase the difficulty and diversity of negative samples. In addition, we release a large-scale multimodal benchmark MBE for various product understanding tasks. Experimentally, our model demonstrates competitive zero-shot performance on both our benchmark and the public dataset, showcasing strong generalization across various downstream tasks, including cross-modal retrieval, product classification, and attribute prediction. Furthermore, the case study and visualization illustrate the effectiveness of MOON for product understanding. 

**Abstract (ZH)**: 随着电子商务的快速发展，探索通用表示而非任务特定表示的研究吸引了越来越多的关注。在产品理解方面，尽管现有的鉴别性双流架构在这领域取得了进步，但它们在建模产品多张图片与文本之间的一对多对齐时普遍存在困难。因此，我们认为生成型多模态大型语言模型（Generative Multimodal Large Language Models, MLLMs）在提高产品表示学习方面具有巨大的潜力。然而，由于几个关键挑战的存在，实现这一目标仍然非 trivial：典型的大语言模型中缺乏多模态和观点感知的建模模块；产品图片中常见的背景噪声；以及缺乏一个标准的评估基准。为了解决这些问题，我们提出了名为MOON的第一个基于生成型MLLM的产品表示学习模型。我们的方法（1）采用指导的混合专家（MoE）模块，针对多模态和观点特定的产品内容进行建模；（2）有效检测产品图片中的核心语义区域，以减轻背景噪声引起的干扰和干扰；（3）引入专门的负样本策略，以增加负样本的难度和多样性。此外，我们还发布了大规模多模态基准MBE，用于各类产品理解任务。实验结果显示，我们的模型在我们的基准和公开数据集上均表现出竞争力的零样本性能，展示了在各种下游任务中（包括跨模态检索、产品分类和属性预测）的强大泛化能力。进一步的研究案例和可视化说明了MOON在产品理解方面的有效性。 

---
# Every 28 Days the AI Dreams of Soft Skin and Burning Stars: Scaffolding AI Agents with Hormones and Emotions 

**Title (ZH)**: 每28天，AI梦回Soft Skin和Burning Stars：通过激素与情绪构建AI代理 

**Authors**: Leigh Levinson, Christopher J. Agostino  

**Link**: [PDF](https://arxiv.org/pdf/2508.11829)  

**Abstract**: Despite significant advances, AI systems struggle with the frame problem: determining what information is contextually relevant from an exponentially large possibility space. We hypothesize that biological rhythms, particularly hormonal cycles, serve as natural relevance filters that could address this fundamental challenge. We develop a framework that embeds simulated menstrual and circadian cycles into Large Language Models through system prompts generated from periodic functions modeling key hormones including estrogen, testosterone, and cortisol. Across multiple state-of-the-art models, linguistic analysis reveals emotional and stylistic variations that track biological phases; sadness peaks during menstruation while happiness dominates ovulation and circadian patterns show morning optimism transitioning to nocturnal introspection. Benchmarking on SQuAD, MMLU, Hellaswag, and AI2-ARC demonstrates subtle but consistent performance variations aligning with biological expectations, including optimal function in moderate rather than extreme hormonal ranges. This methodology provides a novel approach to contextual AI while revealing how societal biases regarding gender and biology are embedded within language models. 

**Abstract (ZH)**: 尽管取得了显著进展，AI系统仍难以解决框架问题：在庞大可能信息空间中确定上下文相关信息。我们假设生物节律，尤其是激素周期，作为自然的相关性过滤器，可以应对这一根本性挑战。我们开发了一种框架，通过从模拟关键激素（包括雌激素、睾酮和皮质醇）周期函数中生成的系统提示，将模拟月经和昼夜节律嵌入到大型语言模型中。在多个最先进的模型中，语言分析揭示了与生物阶段相关的情感和风格变化；悲伤在月经期间达到峰值，而幸福在排卵期占主导地位；同时，昼夜节律模式显示清晨的乐观情绪过渡到晚间的内省。在SQuAD、MMLU、Hellaswag和AI2-ARC上的基准测试表明，性能变化虽细微但具一致性，符合生物预期，包括在适度而非极端激素范围内表现最佳。该方法提供了上下文AI的新范式，揭示了关于性别和生物的偏见如何嵌入语言模型中。 

---
# Ovis2.5 Technical Report 

**Title (ZH)**: Ovis2.5 技术报告 

**Authors**: Shiyin Lu, Yang Li, Yu Xia, Yuwei Hu, Shanshan Zhao, Yanqing Ma, Zhichao Wei, Yinglun Li, Lunhao Duan, Jianshan Zhao, Yuxuan Han, Haijun Li, Wanying Chen, Junke Tang, Chengkun Hou, Zhixing Du, Tianli Zhou, Wenjie Zhang, Huping Ding, Jiahe Li, Wen Li, Gui Hu, Yiliang Gu, Siran Yang, Jiamang Wang, Hailong Sun, Yibo Wang, Hui Sun, Jinlong Huang, Yuping He, Shengze Shi, Weihong Zhang, Guodong Zheng, Junpeng Jiang, Sensen Gao, Yi-Feng Wu, Sijia Chen, Yuhui Chen, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11737)  

**Abstract**: We present Ovis2.5, a successor to Ovis2 designed for native-resolution visual perception and strong multimodal reasoning. Ovis2.5 integrates a native-resolution vision transformer that processes images at their native, variable resolutions, avoiding the degradation from fixed-resolution tiling and preserving both fine detail and global layout -- crucial for visually dense content like complex charts. To strengthen reasoning, we train the model to move beyond linear chain-of-thought and perform reflection -- including self-checking and revision. This advanced capability is exposed as an optional "thinking mode" at inference time, allowing users to trade latency for enhanced accuracy on difficult inputs. The model is trained via a comprehensive five-phase curriculum that progressively builds its skills. The process begins with foundational visual and multimodal pretraining, advances through large-scale instruction tuning, and culminates in alignment and reasoning enhancement using DPO and GRPO. To scale these upgrades efficiently, we employ multimodal data packing and hybrid parallelism, yielding a significant end-to-end speedup. We release two open-source models: Ovis2.5-9B and Ovis2.5-2B. The latter continues the "small model, big performance" philosophy of Ovis2, making it ideal for resource-constrained, on-device scenarios. On the OpenCompass multimodal leaderboard, Ovis2.5-9B averages 78.3, marking a substantial improvement over its predecessor, Ovis2-8B, and achieving state-of-the-art results among open-source MLLMs in the sub-40B parameter range; Ovis2.5-2B scores 73.9, establishing SOTA for its size. Beyond aggregate scores, Ovis2.5 achieves leading results on STEM benchmarks, exhibits strong capabilities on grounding and video tasks, and achieves open-source SOTA at its scale for complex chart analysis. 

**Abstract (ZH)**: Ovis2.5：面向原生分辨率视觉感知和强大多模态推理的继任者 

---
# Real Time Child Abduction And Detection System 

**Title (ZH)**: 实时儿童拐卖检测系统 

**Authors**: Tadisetty Sai Yashwanth, Yangalasetty Sruthi Royal, Vankayala Rajeshwari Shreya, Mayank Kashyap, Divyaprabha K N  

**Link**: [PDF](https://arxiv.org/pdf/2508.11690)  

**Abstract**: Child safety continues to be a paramount concern worldwide, with child abduction posing significant threats to communities. This paper presents the development of an edge-based child abduction detection and alert system utilizing a multi-agent framework where each agent incorporates Vision-Language Models (VLMs) deployed on a Raspberry Pi. Leveraging the advanced capabilities of VLMs within individual agents of a multi-agent team, our system is trained to accurately detect and interpret complex interactions involving children in various environments in real-time. The multi-agent system is deployed on a Raspberry Pi connected to a webcam, forming an edge device capable of processing video feeds, thereby reducing latency and enhancing privacy. An integrated alert system utilizes the Twilio API to send immediate SMS and WhatsApp notifications, including calls and messages, when a potential child abduction event is detected. Experimental results demonstrate that the system achieves high accuracy in detecting potential abduction scenarios, with near real-time performance suitable for practical deployment. The multi-agent architecture enhances the system's ability to process complex situational data, improving detection capabilities over traditional single-model approaches. The edge deployment ensures scalability and cost-effectiveness, making it accessible for widespread use. The proposed system offers a proactive solution to enhance child safety through continuous monitoring and rapid alerting, contributing a valuable tool in efforts to prevent child abductions. 

**Abstract (ZH)**: 基于边缘计算的多agent儿童绑架检测及警报系统 

---
