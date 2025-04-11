# Sim-to-Real Transfer in Reinforcement Learning for Maneuver Control of a Variable-Pitch MAV 

**Title (ZH)**: Sim-to-Real Transfer in Reinforcement Learning for Maneuver Control of a Variable-Pitch MAV 

**Authors**: Zhikun Wang, Shiyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07694)  

**Abstract**: Reinforcement learning (RL) algorithms can enable high-maneuverability in unmanned aerial vehicles (MAVs), but transferring them from simulation to real-world use is challenging. Variable-pitch propeller (VPP) MAVs offer greater agility, yet their complex dynamics complicate the sim-to-real transfer. This paper introduces a novel RL framework to overcome these challenges, enabling VPP MAVs to perform advanced aerial maneuvers in real-world settings. Our approach includes real-to-sim transfer techniques-such as system identification, domain randomization, and curriculum learning to create robust training simulations and a sim-to-real transfer strategy combining a cascade control system with a fast-response low-level controller for reliable deployment. Results demonstrate the effectiveness of this framework in achieving zero-shot deployment, enabling MAVs to perform complex maneuvers such as flips and wall-backtracking. 

**Abstract (ZH)**: 基于强化学习的可变桨距旋翼无人机从仿真到现实世界的高效操控方法 

---
# Localization Meets Uncertainty: Uncertainty-Aware Multi-Modal Localization 

**Title (ZH)**: 定位遇见不确定性：不确定性感知多模态定位 

**Authors**: Hye-Min Won, Jieun Lee, Jiyong Oh  

**Link**: [PDF](https://arxiv.org/pdf/2504.07677)  

**Abstract**: Reliable localization is critical for robot navigation in complex indoor environments. In this paper, we propose an uncertainty-aware localization method that enhances the reliability of localization outputs without modifying the prediction model itself. This study introduces a percentile-based rejection strategy that filters out unreliable 3-DoF pose predictions based on aleatoric and epistemic uncertainties the network estimates. We apply this approach to a multi-modal end-to-end localization that fuses RGB images and 2D LiDAR data, and we evaluate it across three real-world datasets collected using a commercialized serving robot. Experimental results show that applying stricter uncertainty thresholds consistently improves pose accuracy. Specifically, the mean position error is reduced by 41.0%, 56.7%, and 69.4%, and the mean orientation error by 55.6%, 65.7%, and 73.3%, when applying 90%, 80%, and 70% thresholds, respectively. Furthermore, the rejection strategy effectively removes extreme outliers, resulting in better alignment with ground truth trajectories. To the best of our knowledge, this is the first study to quantitatively demonstrate the benefits of percentile-based uncertainty rejection in multi-modal end-to-end localization tasks. Our approach provides a practical means to enhance the reliability and accuracy of localization systems in real-world deployments. 

**Abstract (ZH)**: 可靠的位置定位对于机器人在复杂室内环境中的导航至关重要。本文提出了一种不确定性感知的位置定位方法，该方法在不修改预测模型本身的情况下，增强位置定位输出的可靠性。本研究引入了一种基于百分位数的拒绝策略，该策略根据网络估计的 aleatoric 和 epistemic 不确定性，过滤掉不可靠的 3-DoF 姿态预测。我们将此方法应用于融合 RGB 图像和 2D LiDAR 数据的端到端多模态定位中，并在使用商用服务机器人收集的三个现实世界数据集中进行了评估。实验结果显示，应用更严格的不确定性阈值可以一致地提高姿态准确性。具体而言，当应用 90%、80% 和 70% 的阈值时，位置误差的均值分别减少了 41.0%、56.7% 和 69.4%，姿态误差的均值分别减少了 55.6%、65.7% 和 73.3%。此外，拒绝策略有效去除极端异常值，从而更好地与真实轨迹对齐。据我们所知，这是首次定量证明多模态端到端定位任务中基于百分位数不确定性拒绝策略益处的研究。我们的方法为在实际部署中提升定位系统的可靠性和准确性提供了一种实用手段。 

---
# Learning Long Short-Term Intention within Human Daily Behaviors 

**Title (ZH)**: 学习人类日常行为中的长期短期意图 

**Authors**: Zhe Sun, Rujie Wu, Xiaodong Yang, Hongzhao Xie, Haiyan Jiang, Junda Bi, Zhenliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07597)  

**Abstract**: In the domain of autonomous household robots, it is of utmost importance for robots to understand human behaviors and provide appropriate services. This requires the robots to possess the capability to analyze complex human behaviors and predict the true intentions of humans. Traditionally, humans are perceived as flawless, with their decisions acting as the standards that robots should strive to align with. However, this raises a pertinent question: What if humans make mistakes? In this research, we present a unique task, termed "long short-term intention prediction". This task requires robots can predict the long-term intention of humans, which aligns with human values, and the short term intention of humans, which reflects the immediate action intention. Meanwhile, the robots need to detect the potential non-consistency between the short-term and long-term intentions, and provide necessary warnings and suggestions. To facilitate this task, we propose a long short-term intention model to represent the complex intention states, and build a dataset to train this intention model. Then we propose a two-stage method to integrate the intention model for robots: i) predicting human intentions of both value-based long-term intentions and action-based short-term intentions; and 2) analyzing the consistency between the long-term and short-term intentions. Experimental results indicate that the proposed long short-term intention model can assist robots in comprehending human behavioral patterns over both long-term and short-term durations, which helps determine the consistency between long-term and short-term intentions of humans. 

**Abstract (ZH)**: 自主家庭机器人领域中的人类行为理解和短期长期意图预测研究 

---
# Drive in Corridors: Enhancing the Safety of End-to-end Autonomous Driving via Corridor Learning and Planning 

**Title (ZH)**: 基于走廊学习与规划的端到端自动驾驶安全性提升 

**Authors**: Zhiwei Zhang, Ruichen Yang, Ke Wu, Zijun Xu, Jingchu Liu, Lisen Mu, Zhongxue Gan, Wenchao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2504.07507)  

**Abstract**: Safety remains one of the most critical challenges in autonomous driving systems. In recent years, the end-to-end driving has shown great promise in advancing vehicle autonomy in a scalable manner. However, existing approaches often face safety risks due to the lack of explicit behavior constraints. To address this issue, we uncover a new paradigm by introducing the corridor as the intermediate representation. Widely adopted in robotics planning, the corridors represents spatio-temporal obstacle-free zones for the vehicle to traverse. To ensure accurate corridor prediction in diverse traffic scenarios, we develop a comprehensive learning pipeline including data annotation, architecture refinement and loss formulation. The predicted corridor is further integrated as the constraint in a trajectory optimization process. By extending the differentiability of the optimization, we enable the optimized trajectory to be seamlessly trained within the end-to-end learning framework, improving both safety and interpretability. Experimental results on the nuScenes dataset demonstrate state-of-the-art performance of our approach, showing a 66.7% reduction in collisions with agents and a 46.5% reduction with curbs, significantly enhancing the safety of end-to-end driving. Additionally, incorporating the corridor contributes to higher success rates in closed-loop evaluations. 

**Abstract (ZH)**: 自主驾驶系统中安全性仍然是最关键的挑战之一。近年来，端到端驾驶展示了在规模化提升车辆自主性方面的巨大潜力。然而，现有方法往往由于缺乏显式的行为约束而面临安全风险。为解决这一问题，我们通过引入走廊作为中间表示，提出了一种新的范式。走廊在机器人规划中广泛采用，代表了车辆可以穿越的空间-时间无障碍区域。为了确保在不同交通场景下准确预测走廊，我们开发了一个全面的学习管道，包括数据标注、架构优化和损失函数设计。预测的走廊进一步被作为约束整合到轨迹优化过程中。通过扩展优化的可微性，我们使优化的轨迹能够在端到端学习框架中无缝训练，从而同时提高安全性和可解释性。实验结果表明，我们的方法在nuScenes数据集上的性能达到最新水平，碰撞代理减少66.7%，撞缘减少46.5%，显著提升了端到端驾驶的安全性。此外，将走廊纳入还提高了闭环评估的成功率。 

---
# Bridging Deep Reinforcement Learning and Motion Planning for Model-Free Navigation in Cluttered Environments 

**Title (ZH)**: 深度强化学习与运动规划在受阻环境下的模型自由导航桥梁构建 

**Authors**: Licheng Luo, Mingyu Cai  

**Link**: [PDF](https://arxiv.org/pdf/2504.07283)  

**Abstract**: Deep Reinforcement Learning (DRL) has emerged as a powerful model-free paradigm for learning optimal policies. However, in real-world navigation tasks, DRL methods often suffer from insufficient exploration, particularly in cluttered environments with sparse rewards or complex dynamics under system disturbances. To address this challenge, we bridge general graph-based motion planning with DRL, enabling agents to explore cluttered spaces more effectively and achieve desired navigation performance. Specifically, we design a dense reward function grounded in a graph structure that spans the entire state space. This graph provides rich guidance, steering the agent toward optimal strategies. We validate our approach in challenging environments, demonstrating substantial improvements in exploration efficiency and task success rates. The project website is available at: this https URL 

**Abstract (ZH)**: 深度 reinforcement learning (DRL) 已经成为一种强大的无模型范式，用于学习最优策略。然而，在现实世界的导航任务中，DRL 方法往往由于探索不足而在复杂环境或系统干扰下的稀疏奖励或复杂动力学中表现出不佳。为了应对这一挑战，我们将通用图基运动规划与 DRL 结合起来，使代理能够在更复杂的环境中更有效地探索，并实现期望的导航性能。具体而言，我们设计了一个基于图结构的密集奖励函数，该图结构覆盖整个状态空间。这个图提供了丰富的指导，引导代理采取最优策略。我们在具有挑战性的环境中验证了这种方法，展示了在探索效率和任务成功率方面显著的改善。项目网站可访问：this https URL。 

---
# Expectations, Explanations, and Embodiment: Attempts at Robot Failure Recovery 

**Title (ZH)**: 期望、解释与 embodient: 机器人故障恢复的尝试 

**Authors**: Elmira Yadollahi, Fethiye Irmak Dogan, Yujing Zhang, Beatriz Nogueira, Tiago Guerreiro, Shelly Levy Tzedek, Iolanda Leite  

**Link**: [PDF](https://arxiv.org/pdf/2504.07266)  

**Abstract**: Expectations critically shape how people form judgments about robots, influencing whether they view failures as minor technical glitches or deal-breaking flaws. This work explores how high and low expectations, induced through brief video priming, affect user perceptions of robot failures and the utility of explanations in HRI. We conducted two online studies ($N=600$ total participants); each replicated two robots with different embodiments, Furhat and Pepper. In our first study, grounded in expectation theory, participants were divided into two groups, one primed with positive and the other with negative expectations regarding the robot's performance, establishing distinct expectation frameworks. This validation study aimed to verify whether the videos could reliably establish low and high-expectation profiles. In the second study, participants were primed using the validated videos and then viewed a new scenario in which the robot failed at a task. Half viewed a version where the robot explained its failure, while the other half received no explanation. We found that explanations significantly improved user perceptions of Furhat, especially when participants were primed to have lower expectations. Explanations boosted satisfaction and enhanced the robot's perceived expressiveness, indicating that effectively communicating the cause of errors can help repair user trust. By contrast, Pepper's explanations produced minimal impact on user attitudes, suggesting that a robot's embodiment and style of interaction could determine whether explanations can successfully offset negative impressions. Together, these findings underscore the need to consider users' expectations when tailoring explanation strategies in HRI. When expectations are initially low, a cogent explanation can make the difference between dismissing a failure and appreciating the robot's transparency and effort to communicate. 

**Abstract (ZH)**: 期望对机器人判断形成过程至关重要，影响人们对机器人失败的解读，是视为微不足道的技术问题还是致命缺陷。本研究探讨了通过短暂视频引导建立的高期待和低期待如何影响用户对机器人失败的感知及其在人机交互中的解释效用。我们进行了两项在线研究（总共600名参与者），每项研究使用了不同的机器人实体，Furhat和Pepper。在第一项研究中，基于期望理论，参与者被分为两组，一组被引导形成对机器人性能的正面期待，另一组形成负面期待，从而建立不同的期望框架。这项验证性研究旨在验证引导视频能否可靠地建立低期待和高期待的用户画像。在第二项研究中，使用了经过验证的视频引导参与者，并观察机器人在完成任务时出现故障的新场景。一半参与者看到了机器人解释故障的版本，另一半没有收到解释。研究发现，当参与者被引导形成较低的期待时，解释显著改善了他们对Furhat的感知，尤其是在满足度和增强机器人表达能力方面。相比之下，对Pepper的解释几乎没有影响，表明机器人的载体形式和交互方式可能决定了解释能否成功抵消负面印象。这些发现强调了在人机交互中考虑用户期待，制定合适的解释策略的重要性。当初期期待较低时，一个合理的解释可以将失败视为机器人透明性和沟通努力的表现而被理解。 

---
# A Pointcloud Registration Framework for Relocalization in Subterranean Environments 

**Title (ZH)**: 地下环境重定位的点云配准框架 

**Authors**: David Akhihiero, Jason N. Gross  

**Link**: [PDF](https://arxiv.org/pdf/2504.07231)  

**Abstract**: Relocalization, the process of re-establishing a robot's position within an environment, is crucial for ensuring accurate navigation and task execution when external positioning information, such as GPS, is unavailable or has been lost. Subterranean environments present significant challenges for relocalization due to limited external positioning information, poor lighting that affects camera localization, irregular and often non-distinct surfaces, and dust, which can introduce noise and occlusion in sensor data. In this work, we propose a robust, computationally friendly framework for relocalization through point cloud registration utilizing a prior point cloud map. The framework employs Intrinsic Shape Signatures (ISS) to select feature points in both the target and prior point clouds. The Fast Point Feature Histogram (FPFH) algorithm is utilized to create descriptors for these feature points, and matching these descriptors yields correspondences between the point clouds. A 3D transformation is estimated using the matched points, which initializes a Normal Distribution Transform (NDT) registration. The transformation result from NDT is further refined using the Iterative Closest Point (ICP) registration algorithm. This framework enhances registration accuracy even in challenging conditions, such as dust interference and significant initial transformations between the target and source, making it suitable for autonomous robots operating in underground mines and tunnels. This framework was validated with experiments in simulated and real-world mine datasets, demonstrating its potential for improving relocalization. 

**Abstract (ZH)**: 基于点云配准的鲁棒重本地化方法：利用先验点云地图在地下环境中重获机器人位置 

---
# Fast Adaptation with Behavioral Foundation Models 

**Title (ZH)**: 快速适应：基于行为的基础模型 

**Authors**: Harshit Sikchi, Andrea Tirinzoni, Ahmed Touati, Yingchen Xu, Anssi Kanervisto, Scott Niekum, Amy Zhang, Alessandro Lazaric, Matteo Pirotta  

**Link**: [PDF](https://arxiv.org/pdf/2504.07896)  

**Abstract**: Unsupervised zero-shot reinforcement learning (RL) has emerged as a powerful paradigm for pretraining behavioral foundation models (BFMs), enabling agents to solve a wide range of downstream tasks specified via reward functions in a zero-shot fashion, i.e., without additional test-time learning or planning. This is achieved by learning self-supervised task embeddings alongside corresponding near-optimal behaviors and incorporating an inference procedure to directly retrieve the latent task embedding and associated policy for any given reward function. Despite promising results, zero-shot policies are often suboptimal due to errors induced by the unsupervised training process, the embedding, and the inference procedure. In this paper, we focus on devising fast adaptation strategies to improve the zero-shot performance of BFMs in a few steps of online interaction with the environment while avoiding any performance drop during the adaptation process. Notably, we demonstrate that existing BFMs learn a set of skills containing more performant policies than those identified by their inference procedure, making them well-suited for fast adaptation. Motivated by this observation, we propose both actor-critic and actor-only fast adaptation strategies that search in the low-dimensional task-embedding space of the pre-trained BFM to rapidly improve the performance of its zero-shot policies on any downstream task. Notably, our approach mitigates the initial "unlearning" phase commonly observed when fine-tuning pre-trained RL models. We evaluate our fast adaptation strategies on top of four state-of-the-art zero-shot RL methods in multiple navigation and locomotion domains. Our results show that they achieve 10-40% improvement over their zero-shot performance in a few tens of episodes, outperforming existing baselines. 

**Abstract (ZH)**: 无监督零样本强化学习（RL）已成为预训练行为基础模型（BFMs）的一种强大范式，使智能体能够通过奖励函数指定的一系列下游任务以零样本方式求解，即无需额外的测试时学习或规划。这通过在学习自我监督的任务嵌入的同时学习相应的近最优行为，并结合一种推断过程来直接检索任何给定奖励函数的潜在任务嵌入及其关联策略来实现。尽管取得了有希望的结果，零样本策略往往由于无监督训练过程、嵌入和推断过程引起的误差而不够优化。在本文中，我们专注于设计快速适应策略，以在与环境进行少量在线交互的几步内提高BFMs的零样本性能，并且在适应过程中避免性能下降。值得注意的是，我们展示了现有的BFMs学习了一组包含比其推断过程识别的更高效的策略的技能，使它们非常适合快速适应。受到这一观察的启发，我们提出了一种演员-批评家式和仅演员的快速适应策略，这些策略在预训练BFM的低维任务嵌入空间中搜索，以快速提高其实现零样本策略在任何下游任务上的性能。值得注意的是，我们的方法减轻了在微调预训练RL模型时通常观察到的初始“遗忘”阶段。我们在四个最先进的零样本RL方法上多个导航和运动学域中评估了我们的快速适应策略。结果显示，它们在几轮测试内将零样本性能提高了10-40%，并优于现有基线。 

---
# Better Decisions through the Right Causal World Model 

**Title (ZH)**: 通过合适的因果世界模型做出更好决策 

**Authors**: Elisabeth Dillies, Quentin Delfosse, Jannis Blüml, Raban Emunds, Florian Peter Busch, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2504.07257)  

**Abstract**: Reinforcement learning (RL) agents have shown remarkable performances in various environments, where they can discover effective policies directly from sensory inputs. However, these agents often exploit spurious correlations in the training data, resulting in brittle behaviours that fail to generalize to new or slightly modified environments. To address this, we introduce the Causal Object-centric Model Extraction Tool (COMET), a novel algorithm designed to learn the exact interpretable causal world models (CWMs). COMET first extracts object-centric state descriptions from observations and identifies the environment's internal states related to the depicted objects' properties. Using symbolic regression, it models object-centric transitions and derives causal relationships governing object dynamics. COMET further incorporates large language models (LLMs) for semantic inference, annotating causal variables to enhance interpretability.
By leveraging these capabilities, COMET constructs CWMs that align with the true causal structure of the environment, enabling agents to focus on task-relevant features. The extracted CWMs mitigate the danger of shortcuts, permitting the development of RL systems capable of better planning and decision-making across dynamic scenarios. Our results, validated in Atari environments such as Pong and Freeway, demonstrate the accuracy and robustness of COMET, highlighting its potential to bridge the gap between object-centric reasoning and causal inference in reinforcement learning. 

**Abstract (ZH)**: 因果对象中心模型提取工具（COMET）：一种学习可解释因果世界模型的新算法 

---
# SF2T: Self-supervised Fragment Finetuning of Video-LLMs for Fine-Grained Understanding 

**Title (ZH)**: SF2T: 自监督视频LLMs的片段微调以实现细粒度理解 

**Authors**: Yangliu Hu, Zikai Song, Na Feng, Yawei Luo, Junqing Yu, Yi-Ping Phoebe Chen, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07745)  

**Abstract**: Video-based Large Language Models (Video-LLMs) have witnessed substantial advancements in recent years, propelled by the advancement in multi-modal LLMs. Although these models have demonstrated proficiency in providing the overall description of videos, they struggle with fine-grained understanding, particularly in aspects such as visual dynamics and video details inquiries. To tackle these shortcomings, we find that fine-tuning Video-LLMs on self-supervised fragment tasks, greatly improve their fine-grained video understanding abilities. Hence we propose two key contributions:(1) Self-Supervised Fragment Fine-Tuning (SF$^2$T), a novel effortless fine-tuning method, employs the rich inherent characteristics of videos for training, while unlocking more fine-grained understanding ability of Video-LLMs. Moreover, it relieves researchers from labor-intensive annotations and smartly circumvents the limitations of natural language, which often fails to capture the complex spatiotemporal variations in videos; (2) A novel benchmark dataset, namely FineVidBench, for rigorously assessing Video-LLMs' performance at both the scene and fragment levels, offering a comprehensive evaluation of their capabilities. We assessed multiple models and validated the effectiveness of SF$^2$T on them. Experimental results reveal that our approach improves their ability to capture and interpret spatiotemporal details. 

**Abstract (ZH)**: 基于视频的大型语言模型（Video-LLMs）在近年来取得了显著进展，得益于多模态LLMs的发展。尽管这些模型在提供视频的整体描述方面表现出色，但在视觉动态和视频细节查询等方面仍存在细致理解的不足。为解决这些问题，我们发现对Video-LLMs进行自我监督片段微调（SF$^2$T）能够显著提高其细致的视频理解能力。因此，我们提出了两项关键贡献：（1）自我监督片段微调（SF$^2$T），这是一种新颖的简便微调方法，利用视频的丰富固有特性进行训练，同时增强Video-LLMs的细致理解能力，同时缓解了劳动密集型标注问题，并巧妙地规避了自然语言在捕捉视频复杂的空间时间变化方面的局限性；（2）一种新的基准数据集FineVidBench，用于严格评估Video-LLMs在场景和片段层面的性能，提供对其能力的全面评估。我们评估了多个模型并验证了SF$^2$T的有效性。实验结果表明，我们的方法提高了其捕捉和解释空间时间细节的能力。 

---
