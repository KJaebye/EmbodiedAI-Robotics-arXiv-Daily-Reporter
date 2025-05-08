# Hierarchical Task Decomposition for Execution Monitoring and Error Recovery: Understanding the Rationale Behind Task Demonstrations 

**Title (ZH)**: 分层任务分解以实现执行监控和错误恢复：理解任务示范背后的原理 

**Authors**: Christoph Willibald, Dongheui Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.04565)  

**Abstract**: Multi-step manipulation tasks where robots interact with their environment and must apply process forces based on the perceived situation remain challenging to learn and prone to execution errors. Accurately simulating these tasks is also difficult. Hence, it is crucial for robust task performance to learn how to coordinate end-effector pose and applied force, monitor execution, and react to deviations. To address these challenges, we propose a learning approach that directly infers both low- and high-level task representations from user demonstrations on the real system. We developed an unsupervised task segmentation algorithm that combines intention recognition and feature clustering to infer the skills of a task. We leverage the inferred characteristic features of each skill in a novel unsupervised anomaly detection approach to identify deviations from the intended task execution. Together, these components form a comprehensive framework capable of incrementally learning task decisions and new behaviors as new situations arise. Compared to state-of-the-art learning techniques, our approach significantly reduces the required amount of training data and computational complexity while efficiently learning complex in-contact behaviors and recovery strategies. Our proposed task segmentation and anomaly detection approaches outperform state-of-the-art methods on force-based tasks evaluated on two different robotic systems. 

**Abstract (ZH)**: 基于多步操作的机器人环境交互及力控制任务的学习与执行：一种从实际演示中直接推断低级和高级任务表示的方法 

---
# RGB-Event Fusion with Self-Attention for Collision Prediction 

**Title (ZH)**: RGB-事件融合注意力机制在碰撞预测中的应用 

**Authors**: Pietro Bonazzi, Christian Vogt, Michael Jost, Haotong Qin, Lyes Khacef, Federico Paredes-Valles, Michele Magno  

**Link**: [PDF](https://arxiv.org/pdf/2505.04258)  

**Abstract**: Ensuring robust and real-time obstacle avoidance is critical for the safe operation of autonomous robots in dynamic, real-world environments. This paper proposes a neural network framework for predicting the time and collision position of an unmanned aerial vehicle with a dynamic object, using RGB and event-based vision sensors. The proposed architecture consists of two separate encoder branches, one for each modality, followed by fusion by self-attention to improve prediction accuracy. To facilitate benchmarking, we leverage the ABCD [8] dataset collected that enables detailed comparisons of single-modality and fusion-based approaches. At the same prediction throughput of 50Hz, the experimental results show that the fusion-based model offers an improvement in prediction accuracy over single-modality approaches of 1% on average and 10% for distances beyond 0.5m, but comes at the cost of +71% in memory and + 105% in FLOPs. Notably, the event-based model outperforms the RGB model by 4% for position and 26% for time error at a similar computational cost, making it a competitive alternative. Additionally, we evaluate quantized versions of the event-based models, applying 1- to 8-bit quantization to assess the trade-offs between predictive performance and computational efficiency. These findings highlight the trade-offs of multi-modal perception using RGB and event-based cameras in robotic applications. 

**Abstract (ZH)**: 确保自主机器人在动态真实环境中的鲁棒且实时的障碍物避让至关重要。本文提出一种神经网络框架，利用RGB和事件驱动视觉传感器预测无人驾驶航空器与动态物体的碰撞时间及位置。提出的架构包括两个独立的编码分支，分别用于每种模态，之后通过自注意力机制进行融合以提高预测准确性。为了便于基准测试，我们利用了ABCД [8] 数据集，该数据集支持单模态和融合方法的详细对比。在相同的预测通量50Hz下，实验结果表明，与单模态方法相比，基于融合的方法平均提高了1%的预测准确性，在距离超过0.5m时提高了10%，但代价是内存消耗增加71%，FLOPs增加105%。值得注意的是，在相似计算成本下，事件驱动模型的位置误差和时间误差分别优于RGB模型4%和26%，使其成为一个有竞争力的替代方案。此外，我们还评估了事件驱动模型的量化版本，应用1至8位量化以评估预测性能与计算效率之间的权衡。这些发现突显了在机器人应用中使用RGB和事件驱动摄像头进行多模态感知的权衡。 

---
# Multi-Agent Reinforcement Learning-based Cooperative Autonomous Driving in Smart Intersections 

**Title (ZH)**: 基于多agent增强学习的智能交叉口协同自主驾驶 

**Authors**: Taoyuan Yu, Kui Wang, Zongdian Li, Tao Yu, Kei Sakaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04231)  

**Abstract**: Unsignalized intersections pose significant safety and efficiency challenges due to complex traffic flows. This paper proposes a novel roadside unit (RSU)-centric cooperative driving system leveraging global perception and vehicle-to-infrastructure (V2I) communication. The core of the system is an RSU-based decision-making module using a two-stage hybrid reinforcement learning (RL) framework. At first, policies are pre-trained offline using conservative Q-learning (CQL) combined with behavior cloning (BC) on collected dataset. Subsequently, these policies are fine-tuned in the simulation using multi-agent proximal policy optimization (MAPPO), aligned with a self-attention mechanism to effectively solve inter-agent dependencies. RSUs perform real-time inference based on the trained models to realize vehicle control via V2I communications. Extensive experiments in CARLA environment demonstrate high effectiveness of the proposed system, by: \textit{(i)} achieving failure rates below 0.03\% in coordinating three connected and autonomous vehicles (CAVs) through complex intersection scenarios, significantly outperforming the traditional Autoware control method, and \textit{(ii)} exhibiting strong robustness across varying numbers of controlled agents and shows promising generalization capabilities on other maps. 

**Abstract (ZH)**: 无需信号控制交叉口由于复杂的交通流而提出显著的安全和效率挑战。本文提出了一种新型以路侧单元(RSU)-为中心的协同驾驶系统，利用全局感知和车辆到基础设施(V2I)通信。该系统的核心是一个基于RSU的决策模块，采用两阶段混合强化学习(RL)框架。首先，使用保守Q学习(CQL)结合行为克隆(BC)在收集的数据集上进行离线策略预训练。随后，这些策略在仿真中使用多智能体近端策略优化(MAPPO)进行微调，并采用自注意力机制有效解决智能体间的依赖性。RSU基于训练模型进行实时推理，通过V2I通信实现车辆控制。在CARLA环境中进行的大量实验表明，该系统具有高度有效性：(i) 在复杂交叉口场景中协调三辆连接和自动驾驶车辆(CAVs)时，失败率低于0.03%，明显优于传统的Autoware控制方法；(ii) 具有较强的鲁棒性，并在其他地图上展示了良好的泛化能力。 

---
# Beyond Task Performance: Human Experience in Human-Robot Collaboration 

**Title (ZH)**: 超越任务绩效：人机协作中的人类体验 

**Authors**: Sean Kille, Jan Heinrich Robens, Philipp Dahlinger, Alejandra Rodriguez-Velasquez, Simon Rothfuß, Balint Varga, Andreas Lindenmann, Gerhard Neumann, Sven Matthiesen, Andrea Kiesel, Sören Hohmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.04182)  

**Abstract**: Human interaction experience plays a crucial role in the effectiveness of human-machine collaboration, especially as interactions in future systems progress towards tighter physical and functional integration. While automation design has been shown to impact task performance, its influence on human experi- ence metrics such as flow, sense of agency (SoA), and embodiment remains underexplored. This study investigates how variations in automation design affect these psychological experience mea- sures and examines correlations between subjective experience and physiological indicators. A user study was conducted in a simulated wood workshop, where participants collaborated with a lightweight robot under four automation levels. The results of the study indicate that medium automation levels enhance flow, SoA and embodiment, striking a balance between support and user autonomy. In contrast, higher automation, despite optimizing task performance, diminishes perceived flow and agency. Furthermore, we observed that grip force might be considered as a real-time proxy of SoA, while correlations with heart rate variability were inconclusive. The findings underscore the necessity for automation strategies that integrate human- centric metrics, aiming to optimize both performance and user experience in collaborative robotic systems 

**Abstract (ZH)**: 人类交互体验对未来人机协作效果至关重要，尤其是在未来系统中物理和功能集成程度日益紧密的情况下。虽然自动化设计已显示出对任务性能的影响，但其对流畅感、控制感和实体感等体验指标的影响尚未充分探索。本研究探讨了自动化设计变化如何影响这些心理体验指标，并考察了主观体验与生理指标之间的关联。在模拟的木材车间中，参与者在四种自动化水平下与轻型机器人合作。研究结果表明，中等自动化水平能够提升流畅感、控制感和实体感，实现支持与用户自主性的平衡。相比之下，较高水平的自动化虽然优化了任务性能，但降低了感知到的流畅感和控制感。此外，研究还发现握力可能被视为控制感的实时代理指标，但心率变异性与之的相关性尚不明确。研究结果强调了在协作机器人系统中结合人类中心指标的必要性，旨在同时优化性能和用户体验。 

---
# NAMO-LLM: Efficient Navigation Among Movable Obstacles with Large Language Model Guidance 

**Title (ZH)**: NAMO-LLM: 大语言模型指导下的可移动障碍物导航高效方法 

**Authors**: Yuqing Zhang, Yiannis Kantaros  

**Link**: [PDF](https://arxiv.org/pdf/2505.04141)  

**Abstract**: Several planners have been proposed to compute robot paths that reach desired goal regions while avoiding obstacles. However, these methods fail when all pathways to the goal are blocked. In such cases, the robot must reason about how to reconfigure the environment to access task-relevant regions - a problem known as Navigation Among Movable Objects (NAMO). While various solutions to this problem have been developed, they often struggle to scale to highly cluttered environments. To address this, we propose NAMO-LLM, a sampling-based planner that searches over robot and obstacle configurations to compute feasible plans specifying which obstacles to move, where, and in what order. Its key novelty is a non-uniform sampling strategy guided by Large Language Models (LLMs) biasing the tree construction toward directions more likely to yield a solution. We show that NAMO-LLM is probabilistically complete and demonstrate through experiments that it efficiently scales to cluttered environments, outperforming related works in both runtime and plan quality. 

**Abstract (ZH)**: 基于采样的导航 among 可移动物体规划器 NAMO-LLM 

---
# OpenHelix: A Short Survey, Empirical Analysis, and Open-Source Dual-System VLA Model for Robotic Manipulation 

**Title (ZH)**: OpenHelix: 一种针对机器人 manipulation 的简要综述、实证分析及开源双系统VLA模型 

**Authors**: Can Cui, Pengxiang Ding, Wenxuan Song, Shuanghao Bai, Xinyang Tong, Zirui Ge, Runze Suo, Wanqi Zhou, Yang Liu, Bofang Jia, Han Zhao, Siteng Huang, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03912)  

**Abstract**: Dual-system VLA (Vision-Language-Action) architectures have become a hot topic in embodied intelligence research, but there is a lack of sufficient open-source work for further performance analysis and optimization. To address this problem, this paper will summarize and compare the structural designs of existing dual-system architectures, and conduct systematic empirical evaluations on the core design elements of existing dual-system architectures. Ultimately, it will provide a low-cost open-source model for further exploration. Of course, this project will continue to update with more experimental conclusions and open-source models with improved performance for everyone to choose from. Project page: this https URL. 

**Abstract (ZH)**: 基于视觉-语言-行动的双系统架构在体态智能研究中已成为一个热点话题，但缺乏足够的开源工作供进一步的性能分析与优化。为解决这一问题，本文将总结和比较现有双系统架构的结构设计，并系统地评估现有双系统架构的核心设计要素。最终，将提供一个低成本的开源模型，以供进一步探索。当然，该项目将不断更新，提供更多实验结论和性能改进的开源模型供众人选择。项目页面：this https URL。 

---
# Towards Cognitive Collaborative Robots: Semantic-Level Integration and Explainable Control for Human-Centric Cooperation 

**Title (ZH)**: 面向认知协作机器人：以人类为中心的合作中的语义级集成与可解释控制 

**Authors**: Jaehong Oh  

**Link**: [PDF](https://arxiv.org/pdf/2505.03815)  

**Abstract**: This is a preprint of a review article that has not yet undergone peer review. The content is intended for early dissemination and academic discussion. The final version may differ upon formal publication. As the Fourth Industrial Revolution reshapes industrial paradigms, human-robot collaboration (HRC) has transitioned from a desirable capability to an operational necessity. In response, collaborative robots (Cobots) are evolving beyond repetitive tasks toward adaptive, semantically informed interaction with humans and environments. This paper surveys five foundational pillars enabling this transformation: semantic-level perception, cognitive action planning, explainable learning and control, safety-aware motion design, and multimodal human intention recognition. We examine the role of semantic mapping in transforming spatial data into meaningful context, and explore cognitive planning frameworks that leverage this context for goal-driven decision-making. Additionally, we analyze explainable reinforcement learning methods, including policy distillation and attention mechanisms, which enhance interpretability and trust. Safety is addressed through force-adaptive control and risk-aware trajectory planning, while seamless human interaction is supported via gaze and gesture-based intent recognition. Despite these advancements, challenges such as perception-action disjunction, real-time explainability limitations, and incomplete human trust persist. To address these, we propose a unified Cognitive Synergy Architecture, integrating all modules into a cohesive framework for truly human-centric cobot collaboration. 

**Abstract (ZH)**: 第四次工业革命重塑工业范式背景下的人机协作：语义感知、认知行动规划、可解释学习与控制、安全aware运动设计及多模态人类意图识别的综述 

---
# Merging and Disentangling Views in Visual Reinforcement Learning for Robotic Manipulation 

**Title (ZH)**: 视觉强化学习中机器人操纵的视角合并与解缠 

**Authors**: Abdulaziz Almuzairee, Rohan Patil, Dwait Bhatt, Henrik I. Christensen  

**Link**: [PDF](https://arxiv.org/pdf/2505.04619)  

**Abstract**: Vision is well-known for its use in manipulation, especially using visual servoing. To make it robust, multiple cameras are needed to expand the field of view. That is computationally challenging. Merging multiple views and using Q-learning allows the design of more effective representations and optimization of sample efficiency. Such a solution might be expensive to deploy. To mitigate this, we introduce a Merge And Disentanglement (MAD) algorithm that efficiently merges views to increase sample efficiency while augmenting with single-view features to allow lightweight deployment and ensure robust policies. We demonstrate the efficiency and robustness of our approach using Meta-World and ManiSkill3. For project website and code, see this https URL 

**Abstract (ZH)**: 视觉在操作中的应用尤为显著，尤其是通过视觉伺服技术。为了使其更加 robust，需要使用多个摄像头来扩展视野，这在计算上极具挑战性。通过合并多视角并结合Q-learning可以使设计更有效的表示，并优化样本效率。这样的解决方案可能部署成本较高。为缓解这一问题，我们提出了一个Merge And Disentanglement (MAD) 算法，该算法高效地合并视图以提高样本效率，并通过添加单一视角特征来实现轻量级部署，从而确保鲁棒的策略。我们使用Meta-World和ManiSkill3来展示我们方法的效率和稳健性。更多项目信息和代码请参见此链接：https://github.com/your-repository。 

---
# Trajectory Entropy Reinforcement Learning for Predictable and Robust Control 

**Title (ZH)**: 轨迹熵强化学习：可预测性和鲁棒性控制 

**Authors**: Bang You, Chenxu Wang, Huaping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04193)  

**Abstract**: Simplicity is a critical inductive bias for designing data-driven controllers, especially when robustness is important. Despite the impressive results of deep reinforcement learning in complex control tasks, it is prone to capturing intricate and spurious correlations between observations and actions, leading to failure under slight perturbations to the environment. To tackle this problem, in this work we introduce a novel inductive bias towards simple policies in reinforcement learning. The simplicity inductive bias is introduced by minimizing the entropy of entire action trajectories, corresponding to the number of bits required to describe information in action trajectories after the agent observes state trajectories. Our reinforcement learning agent, Trajectory Entropy Reinforcement Learning, is optimized to minimize the trajectory entropy while maximizing rewards. We show that the trajectory entropy can be effectively estimated by learning a variational parameterized action prediction model, and use the prediction model to construct an information-regularized reward function. Furthermore, we construct a practical algorithm that enables the joint optimization of models, including the policy and the prediction model. Experimental evaluations on several high-dimensional locomotion tasks show that our learned policies produce more cyclical and consistent action trajectories, and achieve superior performance, and robustness to noise and dynamic changes than the state-of-the-art. 

**Abstract (ZH)**: 简洁性是对数据驱动控制器进行设计时的关键归纳偏置，尤其是当需要鲁棒性时。尽管深度强化学习在复杂控制任务中取得了令人印象深刻的成果，但它容易捕捉到观测与动作之间的复杂和虚假相关性，从而在环境出现轻微扰动时导致失败。为解决这一问题，本研究引入了一种新的强化学习归纳偏置，即倾向于简单的策略。这种简单性偏置通过最小化整个动作轨迹的熵来引入，对应的熵反映了智能体在观察状态轨迹后描述动作轨迹所需的信息量。我们的强化学习智能体，轨迹熵强化学习，旨在在最大化奖励的同时最小化轨迹熵。我们展示了轨迹熵可以通过学习参数化动作预测模型来有效估计，并利用预测模型构造一个信息正则化的奖励函数。此外，我们构建了一个实用算法，使模型（包括策略和预测模型）的联合优化成为可能。在多个高维运动任务上的实验评估表明，我们学到的策略产生了更加周期性和一致的动作轨迹，并在噪声和动态变化方面表现出优越的性能和鲁棒性。 

---
# PARC: Physics-based Augmentation with Reinforcement Learning for Character Controllers 

**Title (ZH)**: PARC：基于物理的增强学习字符控制器辅助方法 

**Authors**: Michael Xu, Yi Shi, KangKang Yin, Xue Bin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04002)  

**Abstract**: Humans excel in navigating diverse, complex environments with agile motor skills, exemplified by parkour practitioners performing dynamic maneuvers, such as climbing up walls and jumping across gaps. Reproducing these agile movements with simulated characters remains challenging, in part due to the scarcity of motion capture data for agile terrain traversal behaviors and the high cost of acquiring such data. In this work, we introduce PARC (Physics-based Augmentation with Reinforcement Learning for Character Controllers), a framework that leverages machine learning and physics-based simulation to iteratively augment motion datasets and expand the capabilities of terrain traversal controllers. PARC begins by training a motion generator on a small dataset consisting of core terrain traversal skills. The motion generator is then used to produce synthetic data for traversing new terrains. However, these generated motions often exhibit artifacts, such as incorrect contacts or discontinuities. To correct these artifacts, we train a physics-based tracking controller to imitate the motions in simulation. The corrected motions are then added to the dataset, which is used to continue training the motion generator in the next iteration. PARC's iterative process jointly expands the capabilities of the motion generator and tracker, creating agile and versatile models for interacting with complex environments. PARC provides an effective approach to develop controllers for agile terrain traversal, which bridges the gap between the scarcity of motion data and the need for versatile character controllers. 

**Abstract (ZH)**: 基于物理增强与强化学习的地形导航角色控制器框架 

---
# Mastering Multi-Drone Volleyball through Hierarchical Co-Self-Play Reinforcement Learning 

**Title (ZH)**: 通过分层协同自博弈强化学习掌握多无人机排球技能 

**Authors**: Ruize Zhang, Sirui Xiang, Zelai Xu, Feng Gao, Shilong Ji, Wenhao Tang, Wenbo Ding, Chao Yu, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04317)  

**Abstract**: In this paper, we tackle the problem of learning to play 3v3 multi-drone volleyball, a new embodied competitive task that requires both high-level strategic coordination and low-level agile control. The task is turn-based, multi-agent, and physically grounded, posing significant challenges due to its long-horizon dependencies, tight inter-agent coupling, and the underactuated dynamics of quadrotors. To address this, we propose Hierarchical Co-Self-Play (HCSP), a hierarchical reinforcement learning framework that separates centralized high-level strategic decision-making from decentralized low-level motion control. We design a three-stage population-based training pipeline to enable both strategy and skill to emerge from scratch without expert demonstrations: (I) training diverse low-level skills, (II) learning high-level strategy via self-play with fixed low-level controllers, and (III) joint fine-tuning through co-self-play. Experiments show that HCSP achieves superior performance, outperforming non-hierarchical self-play and rule-based hierarchical baselines with an average 82.9\% win rate and a 71.5\% win rate against the two-stage variant. Moreover, co-self-play leads to emergent team behaviors such as role switching and coordinated formations, demonstrating the effectiveness of our hierarchical design and training scheme. 

**Abstract (ZH)**: 基于层次协作自博弈的三对三多旋翼排球学习方法 

---
# Flow Models for Unbounded and Geometry-Aware Distributional Reinforcement Learning 

**Title (ZH)**: 流模型在无界和几何感知分布强化学习中的应用 

**Authors**: Simo Alami C., Rim Kaddah, Jesse Read, Marie-Paule Cani  

**Link**: [PDF](https://arxiv.org/pdf/2505.04310)  

**Abstract**: We introduce a new architecture for Distributional Reinforcement Learning (DistRL) that models return distributions using normalizing flows. This approach enables flexible, unbounded support for return distributions, in contrast to categorical approaches like C51 that rely on fixed or bounded representations. It also offers richer modeling capacity to capture multi-modality, skewness, and tail behavior than quantile based approaches. Our method is significantly more parameter-efficient than categorical approaches. Standard metrics used to train existing models like KL divergence or Wasserstein distance either are scale insensitive or have biased sample gradients, especially when return supports do not overlap. To address this, we propose a novel surrogate for the Cramèr distance, that is geometry-aware and computable directly from the return distribution's PDF, avoiding the costly CDF computation. We test our model on the ATARI-5 sub-benchmark and show that our approach outperforms PDF based models while remaining competitive with quantile based methods. 

**Abstract (ZH)**: 我们提出了一种新的分布强化学习（DistRL）架构，使用归一化流来建模回报分布。该方法提供了灵活且无界的回报分布支持，相比之下，如C51等基于分类的方法依赖于固定或有界的表现形式。此外，该方法具有更强的建模能力，能够捕捉多模态、偏斜度和尾部行为，优于基于分位数的方法。与分类方法相比，我们的方法参数效率更高。用于训练现有模型的标准度量标准，如KL散度或Wasserstein距离，要么对尺度不敏感，要么在回报支持不重叠的情况下有偏的样本梯度。为了解决这个问题，我们提出了一种新的Cramèr距离的替代方法，该方法具有几何感知性并且可以直接从回报分布的PDF中计算，避免了昂贵的CDF计算。我们在ATARI-5子基准上测试了我们的模型，并证明了我们的方法在保持与基于分位数方法竞争力的同时优于基于PDF的方法。 

---
# Frog Soup: Zero-Shot, In-Context, and Sample-Efficient Frogger Agents 

**Title (ZH)**: 青蛙汤：零样本、上下文相关且样本高效的青蛙 Agents 

**Authors**: Xiang Li, Yiyang Hao, Doug Fulop  

**Link**: [PDF](https://arxiv.org/pdf/2505.03947)  

**Abstract**: One of the primary aspirations in reinforcement learning research is developing general-purpose agents capable of rapidly adapting to and mastering novel tasks. While RL gaming agents have mastered many Atari games, they remain slow and costly to train for each game. In this work, we demonstrate that latest reasoning LLMs with out-of-domain RL post-training can play a challenging Atari game called Frogger under a zero-shot setting. We then investigate the effect of in-context learning and the amount of reasoning effort on LLM performance. Lastly, we demonstrate a way to bootstrap traditional RL method with LLM demonstrations, which significantly improves their performance and sample efficiency. Our implementation is open sourced at this https URL. 

**Abstract (ZH)**: 强化学习研究中的一项主要目标是开发能够快速适应并掌握新型任务的一般用途代理。尽管RL游戏代理已经掌握了许多Atari游戏，但它们在每个游戏中进行训练仍显得缓慢且成本较高。在本文中，我们展示了最新推理大语言模型在域外RL训练后，在零样本设置下能够玩一个名为Frogger的具有挑战性的Atari游戏。然后，我们调查了上下文学习和推理努力对大语言模型性能的影响。最后，我们展示了如何通过大语言模型的示范来提升传统RL方法，这显著提高了其性能和样本效率。我们的实现已在以下链接开源：this https URL。 

---
# "I Can See Forever!": Evaluating Real-time VideoLLMs for Assisting Individuals with Visual Impairments 

**Title (ZH)**: “我可以看到永远！”: 评估实时视频LLM辅助视觉障碍个体的技术 

**Authors**: Ziyi Zhang, Zhen Sun, Zongmin Zhang, Zifan Peng, Yuemeng Zhao, Zichun Wang, Zeren Luo, Ruiting Zuo, Xinlei He  

**Link**: [PDF](https://arxiv.org/pdf/2505.04488)  

**Abstract**: The visually impaired population, especially the severely visually impaired, is currently large in scale, and daily activities pose significant challenges for them. Although many studies use large language and vision-language models to assist the blind, most focus on static content and fail to meet real-time perception needs in dynamic and complex environments, such as daily activities. To provide them with more effective intelligent assistance, it is imperative to incorporate advanced visual understanding technologies. Although real-time vision and speech interaction VideoLLMs demonstrate strong real-time visual understanding, no prior work has systematically evaluated their effectiveness in assisting visually impaired individuals. In this work, we conduct the first such evaluation. First, we construct a benchmark dataset (VisAssistDaily), covering three categories of assistive tasks for visually impaired individuals: Basic Skills, Home Life Tasks, and Social Life Tasks. The results show that GPT-4o achieves the highest task success rate. Next, we conduct a user study to evaluate the models in both closed-world and open-world scenarios, further exploring the practical challenges of applying VideoLLMs in assistive contexts. One key issue we identify is the difficulty current models face in perceiving potential hazards in dynamic environments. To address this, we build an environment-awareness dataset named SafeVid and introduce a polling mechanism that enables the model to proactively detect environmental risks. We hope this work provides valuable insights and inspiration for future research in this field. 

**Abstract (ZH)**: 视觉受损人群，尤其是重度视觉受损人群，目前规模庞大，日常活动对其构成重大挑战。尽管许多研究利用大规模语言模型和多模态模型来协助视障人士，大多数研究重点在于静态内容，未能满足在动态和复杂环境中（如日常活动）的实时感知需求。为了提供更多有效的智能辅助，有必要融入先进的视觉理解技术。虽然实时视觉和语音交互VideoLLMs展现了强大的实时视觉理解能力，但此前没有任何研究系统性地评估其在辅助视障人士中的有效性。在本工作中，我们首次进行了此类评价。首先，我们构建了一个基准数据集（VisAssistDaily），涵盖三种视障辅助任务类别：基本技能、家庭生活任务和社会生活任务。结果显示，GPT-4o 达到最高的任务成功率。随后，我们进行了一项用户研究，评估模型在闭世界和开放世界场景中的表现，进一步探讨将VideoLLMs应用于辅助情境的实际挑战。一个我们识别的关键问题是当前模型在动态环境中感知潜在风险的困难。为此，我们构建了一个环境感知数据集SafeVid，并引入了一种投票机制，使模型能够主动检测环境风险。我们希望本工作为未来该领域的研究提供有价值的见解和灵感。 

---
# Object-Shot Enhanced Grounding Network for Egocentric Video 

**Title (ZH)**: 基于对象短语增强的自我中心视频目标 grounding 网络 

**Authors**: Yisen Feng, Haoyu Zhang, Meng Liu, Weili Guan, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2505.04270)  

**Abstract**: Egocentric video grounding is a crucial task for embodied intelligence applications, distinct from exocentric video moment localization. Existing methods primarily focus on the distributional differences between egocentric and exocentric videos but often neglect key characteristics of egocentric videos and the fine-grained information emphasized by question-type queries. To address these limitations, we propose OSGNet, an Object-Shot enhanced Grounding Network for egocentric video. Specifically, we extract object information from videos to enrich video representation, particularly for objects highlighted in the textual query but not directly captured in the video features. Additionally, we analyze the frequent shot movements inherent to egocentric videos, leveraging these features to extract the wearer's attention information, which enhances the model's ability to perform modality alignment. Experiments conducted on three datasets demonstrate that OSGNet achieves state-of-the-art performance, validating the effectiveness of our approach. Our code can be found at this https URL. 

**Abstract (ZH)**: 自我中心视频锚定是体现智能应用中的一个关键任务，与外视角视频moment定位不同。现有方法主要关注自我中心视频和外视角视频之间的分布差异，但往往忽视了自我中心视频的关键特征以及文本查询强调的细粒度信息。为了解决这些局限性，我们提出了一种基于对象-shot的锚定网络OSGNet，用于自我中心视频。具体而言，我们从视频中提取对象信息以丰富视频表示，特别是对于文本查询中强调但在视频特征中未直接捕捉到的对象。此外，我们分析了自我中心视频中固有的频繁镜头运动，利用这些特征来提取佩戴者注意力信息，从而增强模型的模态对齐能力。在三个数据集上的实验结果显示，OSGNet 达到了最先进的性能，验证了我们方法的有效性。我们的代码可以通过以下链接获取：this https URL。 

---
# An Active Inference Model of Covert and Overt Visual Attention 

**Title (ZH)**: 隐蔽与外显视觉注意的主动推断模型 

**Authors**: Tin Mišić, Karlo Koledić, Fabio Bonsignorio, Ivan Petrović, Ivan Marković  

**Link**: [PDF](https://arxiv.org/pdf/2505.03856)  

**Abstract**: The ability to selectively attend to relevant stimuli while filtering out distractions is essential for agents that process complex, high-dimensional sensory input. This paper introduces a model of covert and overt visual attention through the framework of active inference, utilizing dynamic optimization of sensory precisions to minimize free-energy. The model determines visual sensory precisions based on both current environmental beliefs and sensory input, influencing attentional allocation in both covert and overt modalities. To test the effectiveness of the model, we analyze its behavior in the Posner cueing task and a simple target focus task using two-dimensional(2D) visual data. Reaction times are measured to investigate the interplay between exogenous and endogenous attention, as well as valid and invalid cueing. The results show that exogenous and valid cues generally lead to faster reaction times compared to endogenous and invalid cues. Furthermore, the model exhibits behavior similar to inhibition of return, where previously attended locations become suppressed after a specific cue-target onset asynchrony interval. Lastly, we investigate different aspects of overt attention and show that involuntary, reflexive saccades occur faster than intentional ones, but at the expense of adaptability. 

**Abstract (ZH)**: 处理复杂高维度感官输入的代理需要具备选择性关注相关刺激并过滤干扰的能力。本文通过主动推断框架引入了一种视觉注意模型，利用传感器精度的动态优化以最小化自由能。该模型根据当前环境信念和感官输入来确定视觉感官精度，影响知觉和外显模态中的注意分配。为了测试该模型的有效性，我们使用二维(2D)视觉数据分析了其在Posner指导任务和简单目标聚焦任务中的行为，并测量反应时间来探究外源性与内源性注意以及有效和无效引导之间的相互作用。结果显示，外源性与有效引导通常会导致更快的反应时间，而内源性与无效引导则不然。此外，模型表现出类似抑制回返的行为，即在特定的刺激-目标出现间隔后，先前注意的区域会受到抑制。最后，我们探讨了外显注意的不同方面，并发现不自主的反射性眨眼比有意的眨眼更快发生，但牺牲了适应性。 

---
# Facilitating Video Story Interaction with Multi-Agent Collaborative System 

**Title (ZH)**: 促进基于多Agent协作系统的视频故事互动 

**Authors**: Yiwen Zhang, Jianing Hao, Zhan Wang, Hongling Sheng, Wei Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.03807)  

**Abstract**: Video story interaction enables viewers to engage with and explore narrative content for personalized experiences. However, existing methods are limited to user selection, specially designed narratives, and lack customization. To address this, we propose an interactive system based on user intent. Our system uses a Vision Language Model (VLM) to enable machines to understand video stories, combining Retrieval-Augmented Generation (RAG) and a Multi-Agent System (MAS) to create evolving characters and scene experiences. It includes three stages: 1) Video story processing, utilizing VLM and prior knowledge to simulate human understanding of stories across three modalities. 2) Multi-space chat, creating growth-oriented characters through MAS interactions based on user queries and story stages. 3) Scene customization, expanding and visualizing various story scenes mentioned in dialogue. Applied to the Harry Potter series, our study shows the system effectively portrays emergent character social behavior and growth, enhancing the interactive experience in the video story world. 

**Abstract (ZH)**: 视频故事交互使得观众能够参与并探索叙事内容以获得个性化体验。然而，现有方法局限于用户选择、特别设计的叙事，并缺乏个性化定制。为了解决这一问题，我们提出了一种基于用户意图的交互系统。该系统利用视觉语言模型（VLM）使机器理解视频故事，并结合检索增强生成（RAG）和多代理系统（MAS）来创建不断发展的角色和场景体验。它包括三个阶段：1）视频故事处理，利用VLM和先验知识在三个模态中模拟人类对故事的理解。2）多空间聊天，通过MAS交互根据用户查询和故事阶段生成增长导向的角色。3）场景定制，扩展和可视化对话中提到的各种故事场景。应用于哈利·波特系列，我们的研究表明该系统有效地展示了角色社会行为的涌现性和增长性，增强了视频故事世界中的互动体验。 

---
# Towards Efficient Online Tuning of VLM Agents via Counterfactual Soft Reinforcement Learning 

**Title (ZH)**: 通过反事实软强化学习实现高效的在线VLM代理调优 

**Authors**: Lang Feng, Weihao Tan, Zhiyi Lyu, Longtao Zheng, Haiyang Xu, Ming Yan, Fei Huang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2505.03792)  

**Abstract**: Online fine-tuning vision-language model (VLM) agents with reinforcement learning (RL) has shown promise for equipping agents with multi-step, goal-oriented capabilities in dynamic environments. However, their open-ended textual action space and non-end-to-end nature of action generation present significant challenges to effective online exploration in RL, e.g., explosion of the exploration space. We propose a novel online fine-tuning method, Counterfactual Soft Reinforcement Learning (CoSo), better suited to the textual output space of VLM agents. Compared to prior methods that assign uniform uncertainty to all tokens, CoSo leverages counterfactual reasoning to dynamically assess the causal influence of individual tokens on post-processed actions. By prioritizing the exploration of action-critical tokens while reducing the impact of semantically redundant or low-impact tokens, CoSo enables a more targeted and efficient online rollout process. We provide theoretical analysis proving CoSo's convergence and policy improvement guarantees, and extensive empirical evaluations supporting CoSo's effectiveness. Our results across a diverse set of agent tasks, including Android device control, card gaming, and embodied AI, highlight its remarkable ability to enhance exploration efficiency and deliver consistent performance gains. The code is available at this https URL. 

**Abstract (ZH)**: 基于反事实软强化学习的在线微调视觉语言模型智能体方法在动态环境中的多步目标导向能力研究 

---
