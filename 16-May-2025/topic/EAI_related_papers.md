# Real-Time Out-of-Distribution Failure Prevention via Multi-Modal Reasoning 

**Title (ZH)**: 基于多模态推理的实时 Out-of-Distribution 失效预防 

**Authors**: Milan Ganai, Rohan Sinha, Christopher Agia, Daniel Morton, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2505.10547)  

**Abstract**: Foundation models can provide robust high-level reasoning on appropriate safety interventions in hazardous scenarios beyond a robot's training data, i.e. out-of-distribution (OOD) failures. However, due to the high inference latency of Large Vision and Language Models, current methods rely on manually defined intervention policies to enact fallbacks, thereby lacking the ability to plan generalizable, semantically safe motions. To overcome these challenges we present FORTRESS, a framework that generates and reasons about semantically safe fallback strategies in real time to prevent OOD failures. At a low frequency in nominal operations, FORTRESS uses multi-modal reasoners to identify goals and anticipate failure modes. When a runtime monitor triggers a fallback response, FORTRESS rapidly synthesizes plans to fallback goals while inferring and avoiding semantically unsafe regions in real time. By bridging open-world, multi-modal reasoning with dynamics-aware planning, we eliminate the need for hard-coded fallbacks and human safety interventions. FORTRESS outperforms on-the-fly prompting of slow reasoning models in safety classification accuracy on synthetic benchmarks and real-world ANYmal robot data, and further improves system safety and planning success in simulation and on quadrotor hardware for urban navigation. 

**Abstract (ZH)**: 基于基础模型的方法可以提供鲁棒的高层推理，在机器人训练数据之外的危险场景中确定适当的安全干预措施，即超出分布（OOD）故障。然而，由于大型视觉和语言模型的推断延迟较高，当前方法依赖于手动定义的干预策略来实施回退，这限制了其规划可泛化的语义安全动作的能力。为克服这些挑战，我们提出了一种FORTRESS框架，该框架可以实时生成和推理语义安全的回退策略，以防止OOD故障。在正常操作的低频率下，FORTRESS使用多模态推理器来识别目标并预见故障模式。当运行时监控触发回退响应时，FORTRESS可以迅速合成回退目标的计划，同时实时推断和避开语义不安全区域。通过将开放世界、多模态推理与动力学感知规划相结合，FORTRESS消除了硬编码回退和人工安全干预的需要。FORTRESS在合成基准和实际的ANYmal机器人数据中的安全分类准确性上优于即时提示的缓慢推理模型，并进一步提高了模拟和四旋翼硬件中城市导航系统的安全性和规划成功率。 

---
# Knowledge capture, adaptation and composition (KCAC): A framework for cross-task curriculum learning in robotic manipulation 

**Title (ZH)**: 知识捕获、适应与组合（KCAC）：跨任务机器人 manipulation 课程学习的框架 

**Authors**: Xinrui Wang, Yan Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.10522)  

**Abstract**: Reinforcement learning (RL) has demonstrated remarkable potential in robotic manipulation but faces challenges in sample inefficiency and lack of interpretability, limiting its applicability in real world scenarios. Enabling the agent to gain a deeper understanding and adapt more efficiently to diverse working scenarios is crucial, and strategic knowledge utilization is a key factor in this process. This paper proposes a Knowledge Capture, Adaptation, and Composition (KCAC) framework to systematically integrate knowledge transfer into RL through cross-task curriculum learning. KCAC is evaluated using a two block stacking task in the CausalWorld benchmark, a complex robotic manipulation environment. To our knowledge, existing RL approaches fail to solve this task effectively, reflecting deficiencies in knowledge capture. In this work, we redesign the benchmark reward function by removing rigid constraints and strict ordering, allowing the agent to maximize total rewards concurrently and enabling flexible task completion. Furthermore, we define two self-designed sub-tasks and implement a structured cross-task curriculum to facilitate efficient learning. As a result, our KCAC approach achieves a 40 percent reduction in training time while improving task success rates by 10 percent compared to traditional RL methods. Through extensive evaluation, we identify key curriculum design parameters subtask selection, transition timing, and learning rate that optimize learning efficiency and provide conceptual guidance for curriculum based RL frameworks. This work offers valuable insights into curriculum design in RL and robotic learning. 

**Abstract (ZH)**: 基于知识捕捉、适应与整合的增强学习框架：因果世界基准中的模块堆叠任务 

---
# IN-RIL: Interleaved Reinforcement and Imitation Learning for Policy Fine-Tuning 

**Title (ZH)**: 交错强化学习与模仿学习方法fine-tuning策略 

**Authors**: Dechen Gao, Hang Wang, Hanchu Zhou, Nejib Ammar, Shatadal Mishra, Ahmadreza Moradipari, Iman Soltani, Junshan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10442)  

**Abstract**: Imitation learning (IL) and reinforcement learning (RL) each offer distinct advantages for robotics policy learning: IL provides stable learning from demonstrations, and RL promotes generalization through exploration. While existing robot learning approaches using IL-based pre-training followed by RL-based fine-tuning are promising, this two-step learning paradigm often suffers from instability and poor sample efficiency during the RL fine-tuning phase. In this work, we introduce IN-RIL, INterleaved Reinforcement learning and Imitation Learning, for policy fine-tuning, which periodically injects IL updates after multiple RL updates and hence can benefit from the stability of IL and the guidance of expert data for more efficient exploration throughout the entire fine-tuning process. Since IL and RL involve different optimization objectives, we develop gradient separation mechanisms to prevent destructive interference during \ABBR fine-tuning, by separating possibly conflicting gradient updates in orthogonal subspaces. Furthermore, we conduct rigorous analysis, and our findings shed light on why interleaving IL with RL stabilizes learning and improves sample-efficiency. Extensive experiments on 14 robot manipulation and locomotion tasks across 3 benchmarks, including FurnitureBench, OpenAI Gym, and Robomimic, demonstrate that \ABBR can significantly improve sample efficiency and mitigate performance collapse during online finetuning in both long- and short-horizon tasks with either sparse or dense rewards. IN-RIL, as a general plug-in compatible with various state-of-the-art RL algorithms, can significantly improve RL fine-tuning, e.g., from 12\% to 88\% with 6.3x improvement in the success rate on Robomimic Transport. Project page: this https URL. 

**Abstract (ZH)**: 交替强化学习与模仿学习（IN-RIL）：一种用于策略微调的混合学习方法 

---
# Internal State Estimation in Groups via Active Information Gathering 

**Title (ZH)**: 群体中的内部状态估计通过主动信息收集 

**Authors**: Xuebo Ji, Zherong Pan, Xifeng Gao, Lei Yang, Xinxin Du, Kaiyun Li, Yongjin Liu, Wenping Wang, Changhe Tu, Jia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.10415)  

**Abstract**: Accurately estimating human internal states, such as personality traits or behavioral patterns, is critical for enhancing the effectiveness of human-robot interaction, particularly in group settings. These insights are key in applications ranging from social navigation to autism diagnosis. However, prior methods are limited by scalability and passive observation, making real-time estimation in complex, multi-human settings difficult. In this work, we propose a practical method for active human personality estimation in groups, with a focus on applications related to Autism Spectrum Disorder (ASD). Our method combines a personality-conditioned behavior model, based on the Eysenck 3-Factor theory, with an active robot information gathering policy that triggers human behaviors through a receding-horizon planner. The robot's belief about human personality is then updated via Bayesian inference. We demonstrate the effectiveness of our approach through simulations, user studies with typical adults, and preliminary experiments involving participants with ASD. Our results show that our method can scale to tens of humans and reduce personality prediction error by 29.2% and uncertainty by 79.9% in simulation. User studies with typical adults confirm the method's ability to generalize across complex personality distributions. Additionally, we explore its application in autism-related scenarios, demonstrating that the method can identify the difference between neurotypical and autistic behavior, highlighting its potential for diagnosing ASD. The results suggest that our framework could serve as a foundation for future ASD-specific interventions. 

**Abstract (ZH)**: 准确估计人类内心状态，如个性特征或行为模式，对于增强人机交互的有效性，尤其是在群体设置中，至关重要。这些见解在从社会导航到自闭症诊断的应用中至关重要。然而，先前的方法由于可扩展性和被动观察的限制，使得在复杂多人类环境中实现实时估计变得困难。在本工作中，我们提出了一种实用的方法，用于群体中的人格主动估计，重点关注与自闭症谱系障碍（ASD）相关应用。我们的方法结合了基于耶森三因素理论的个性条件行为模型，以及通过回溯规划器触发人类行为的主动机器人信息收集策略。机器人的关于人类人格的信念通过贝叶斯推理进行更新。我们通过仿真、典型成人用户的实验以及自闭症谱系障碍参与者的初步实验，证明了该方法的有效性。结果显示，我们的方法可以扩展到数十人，并在仿真中将人格预测误差降低了29.2%，不确定性降低了79.9%。典型成人用户的实验确认了该方法在复杂人格分布中的泛化能力。此外，我们探讨了该方法在自闭症相关场景中的应用，证明该方法可以识别正常人和自闭症患者的行为差异，突显了其在诊断自闭症谱系障碍方面的潜力。研究结果表明，我们的框架可以为未来的自闭症特定干预措施提供基础。 

---
# NVSPolicy: Adaptive Novel-View Synthesis for Generalizable Language-Conditioned Policy Learning 

**Title (ZH)**: NVSPolicy：自适应新颖视角合成的通用语言条件政策学习 

**Authors**: Le Shi, Yifei Shi, Xin Xu, Tenglong Liu, Junhua Xi, Chengyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.10359)  

**Abstract**: Recent advances in deep generative models demonstrate unprecedented zero-shot generalization capabilities, offering great potential for robot manipulation in unstructured environments. Given a partial observation of a scene, deep generative models could generate the unseen regions and therefore provide more context, which enhances the capability of robots to generalize across unseen environments. However, due to the visual artifacts in generated images and inefficient integration of multi-modal features in policy learning, this direction remains an open challenge. We introduce NVSPolicy, a generalizable language-conditioned policy learning method that couples an adaptive novel-view synthesis module with a hierarchical policy network. Given an input image, NVSPolicy dynamically selects an informative viewpoint and synthesizes an adaptive novel-view image to enrich the visual context. To mitigate the impact of the imperfect synthesized images, we adopt a cycle-consistent VAE mechanism that disentangles the visual features into the semantic feature and the remaining feature. The two features are then fed into the hierarchical policy network respectively: the semantic feature informs the high-level meta-skill selection, and the remaining feature guides low-level action estimation. Moreover, we propose several practical mechanisms to make the proposed method efficient. Extensive experiments on CALVIN demonstrate the state-of-the-art performance of our method. Specifically, it achieves an average success rate of 90.4\% across all tasks, greatly outperforming the recent methods. Ablation studies confirm the significance of our adaptive novel-view synthesis paradigm. In addition, we evaluate NVSPolicy on a real-world robotic platform to demonstrate its practical applicability. 

**Abstract (ZH)**: Recent advances in深生成模型展示了前所未有的零样本泛化能力，为机器人在非结构化环境中的操作提供了巨大潜力。给定场景的部分观测，深度生成模型可以生成未观测到的区域，从而提供更多的上下文信息，增强机器人在未见过的环境中泛化的能力。然而，由于生成图像中的视觉伪影以及多模态特征在策略学习中的低效整合，这一方向仍是一项开放性挑战。我们提出了NVSPolicy，这是一种通用的语言条件策略学习方法，结合了自适应新颖视图合成模块和分层策略网络。给定输入图像，NVSPolicy动态选择一个信息丰富的视角并合成自适应新颖视图图像以丰富视觉上下文。为减轻合成图像不完美的影响，我们采用了循环一致的VAE机制，将视觉特征分解为语义特征和剩余特征。这两个特征分别输入到分层策略网络中：语义特征指导高层元技能的选择，而剩余特征指导低层面动作的估计。此外，我们还提出了一些实用机制以提高所提方法的效率。在CALVIN上的广泛实验表明，我们的方法实现了最先进的性能。具体而言，该方法在所有任务中的平均成功率达到了90.4%，显著优于最近的方法。消融研究证实了我们自适应新颖视图合成框架的重要性。此外，我们在实际的机器人平台上评估了NVSPolicy，以证明其实际应用性。 

---
# SRT-H: A Hierarchical Framework for Autonomous Surgery via Language Conditioned Imitation Learning 

**Title (ZH)**: SRT-H：一种基于语言条件化imitation learning的分级自主手术框架 

**Authors**: Ji Woong Kim, Juo-Tung Chen, Pascal Hansen, Lucy X. Shi, Antony Goldenberg, Samuel Schmidgall, Paul Maria Scheikl, Anton Deguet, Brandon M. White, De Ru Tsai, Richard Cha, Jeffrey Jopling, Chelsea Finn, Axel Krieger  

**Link**: [PDF](https://arxiv.org/pdf/2505.10251)  

**Abstract**: Research on autonomous robotic surgery has largely focused on simple task automation in controlled environments. However, real-world surgical applications require dexterous manipulation over extended time scales while demanding generalization across diverse variations in human tissue. These challenges remain difficult to address using existing logic-based or conventional end-to-end learning strategies. To bridge this gap, we propose a hierarchical framework for dexterous, long-horizon surgical tasks. Our method employs a high-level policy for task planning and a low-level policy for generating task-space controls for the surgical robot. The high-level planner plans tasks using language, producing task-specific or corrective instructions that guide the robot at a coarse level. Leveraging language as a planning modality offers an intuitive and generalizable interface, mirroring how experienced surgeons instruct traineers during procedures. We validate our framework in ex-vivo experiments on a complex minimally invasive procedure, cholecystectomy, and conduct ablative studies to assess key design choices. Our approach achieves a 100% success rate across n=8 different ex-vivo gallbladders, operating fully autonomously without human intervention. The hierarchical approach greatly improves the policy's ability to recover from suboptimal states that are inevitable in the highly dynamic environment of realistic surgical applications. This work represents the first demonstration of step-level autonomy, marking a critical milestone toward autonomous surgical systems for clinical studies. By advancing generalizable autonomy in surgical robotics, our approach brings the field closer to real-world deployment. 

**Abstract (ZH)**: 基于语言的层次化框架用于长时间尺度的灵巧手术任务研究 

---
# Force-Driven Validation for Collaborative Robotics in Automated Avionics Testing 

**Title (ZH)**: 基于力驱动验证的协作机器人在自动化航空电子测试中的应用 

**Authors**: Pietro Dardano, Paolo Rocco, David Frisini  

**Link**: [PDF](https://arxiv.org/pdf/2505.10224)  

**Abstract**: ARTO is a project combining collaborative robots (cobots) and Artificial Intelligence (AI) to automate functional test procedures for civilian and military aircraft certification. This paper proposes a Deep Learning (DL) and eXplainable AI (XAI) approach, equipping ARTO with interaction analysis capabilities to verify and validate the operations on cockpit components. During these interactions, forces, torques, and end effector poses are recorded and preprocessed to filter disturbances caused by low performance force controllers and embedded Force Torque Sensors (FTS). Convolutional Neural Networks (CNNs) then classify the cobot actions as Success or Fail, while also identifying and reporting the causes of failure. To improve interpretability, Grad CAM, an XAI technique for visual explanations, is integrated to provide insights into the models decision making process. This approach enhances the reliability and trustworthiness of the automated testing system, facilitating the diagnosis and rectification of errors that may arise during testing. 

**Abstract (ZH)**: ARTO是一个结合协作机器人（cobot）和人工智能（AI）的项目，旨在自动化民用和军事航空器认证的功能测试程序。本文提出了一种深度学习（DL）和可解释人工智能（XAI）的方法，使ARTO具备交互分析能力，以验证和验证对驾驶舱组件操作的正确性。在这些交互过程中，记录并预处理力、扭矩和末端执行器姿态，以过滤由低性能力控制器和嵌入式力矩传感器（FTS）引起的干扰。卷积神经网络（CNN）对cobot的动作进行分类，区分成功和失败，并识别和报告失败的原因。为了提高可解释性，整合了Grad CAM这一XAI技术，用于视觉解释，提供对模型决策过程的洞察。该方法增强了自动化测试系统的可靠性和可信度，有助于诊断和纠正测试中可能出现的错误。 

---
# Towards Safe Robot Foundation Models Using Inductive Biases 

**Title (ZH)**: 基于归纳偏置实现安全的机器人基础模型 

**Authors**: Maximilian Tölle, Theo Gruner, Daniel Palenicek, Tim Schneider, Jonas Günster, Joe Watson, Davide Tateo, Puze Liu, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2505.10219)  

**Abstract**: Safety is a critical requirement for the real-world deployment of robotic systems. Unfortunately, while current robot foundation models show promising generalization capabilities across a wide variety of tasks, they fail to address safety, an important aspect for ensuring long-term operation. Current robot foundation models assume that safe behavior should emerge by learning from a sufficiently large dataset of demonstrations. However, this approach has two clear major drawbacks. Firstly, there are no formal safety guarantees for a behavior cloning policy trained using supervised learning. Secondly, without explicit knowledge of any safety constraints, the policy may require an unreasonable number of additional demonstrations to even approximate the desired constrained behavior. To solve these key issues, we show how we can instead combine robot foundation models with geometric inductive biases using ATACOM, a safety layer placed after the foundation policy that ensures safe state transitions by enforcing action constraints. With this approach, we can ensure formal safety guarantees for generalist policies without providing extensive demonstrations of safe behavior, and without requiring any specific fine-tuning for safety. Our experiments show that our approach can be beneficial both for classical manipulation tasks, where we avoid unwanted collisions with irrelevant objects, and for dynamic tasks, such as the robot air hockey environment, where we can generate fast trajectories respecting complex tasks and joint space constraints. 

**Abstract (ZH)**: 机器人系统中安全性是实际部署的关键要求。尽管当前的机器人基础模型在广泛的任务中展现了有希望的泛化能力，但它们未能解决安全性问题，这是确保长期运行的重要方面。当前的机器人基础模型假设安全行为可以通过从足够大的示例数据集中学习而自然地产生。然而，这种做法有两个明显的重大缺点。首先，通过监督学习训练的行为克隆策略无法提供任何形式的安全保证。其次，在没有明确的安全约束知识的情况下，策略可能需要不合理的大量额外示例来近似所需的约束行为。为解决这些关键问题，我们展示了如何通过ATACOM安全层将机器人基础模型与几何归纳偏置相结合，在基础策略之后放置一个确保安全状态转换的安全层，通过强制执行动作约束来实现。借助此方法，我们可以在不提供大量安全行为示例的情况下，确保通用策略的形式安全保证，并且不需要任何特定的安全微调。我们的实验表明，该方法在传统操作任务中（如避免不相关物体的不必要的碰撞）和动态任务（如机器人桌上冰壶环境）中均可带来益处，可以生成尊重复杂任务和关节空间约束的快速轨迹。 

---
# EmbodiedMAE: A Unified 3D Multi-Modal Representation for Robot Manipulation 

**Title (ZH)**: EmbodiedMAE：一种统一的机器人 manipulation 三维多模态表示方法 

**Authors**: Zibin Dong, Fei Ni, Yifu Yuan, Yinchuan Li, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2505.10105)  

**Abstract**: We present EmbodiedMAE, a unified 3D multi-modal representation for robot manipulation. Current approaches suffer from significant domain gaps between training datasets and robot manipulation tasks, while also lacking model architectures that can effectively incorporate 3D information. To overcome these limitations, we enhance the DROID dataset with high-quality depth maps and point clouds, constructing DROID-3D as a valuable supplement for 3D embodied vision research. Then we develop EmbodiedMAE, a multi-modal masked autoencoder that simultaneously learns representations across RGB, depth, and point cloud modalities through stochastic masking and cross-modal fusion. Trained on DROID-3D, EmbodiedMAE consistently outperforms state-of-the-art vision foundation models (VFMs) in both training efficiency and final performance across 70 simulation tasks and 20 real-world robot manipulation tasks on two robot platforms. The model exhibits strong scaling behavior with size and promotes effective policy learning from 3D inputs. Experimental results establish EmbodiedMAE as a reliable unified 3D multi-modal VFM for embodied AI systems, particularly in precise tabletop manipulation settings where spatial perception is critical. 

**Abstract (ZH)**: 基于感知的统一3D多模态表示方法EmbodiedMAE在机器人操作中的应用 

---
# FlowDreamer: A RGB-D World Model with Flow-based Motion Representations for Robot Manipulation 

**Title (ZH)**: FlowDreamer：一种基于流的运动表示的RGB-D世界模型在机器人操作中的应用 

**Authors**: Jun Guo, Xiaojian Ma, Yikai Wang, Min Yang, Huaping Liu, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10075)  

**Abstract**: This paper investigates training better visual world models for robot manipulation, i.e., models that can predict future visual observations by conditioning on past frames and robot actions. Specifically, we consider world models that operate on RGB-D frames (RGB-D world models). As opposed to canonical approaches that handle dynamics prediction mostly implicitly and reconcile it with visual rendering in a single model, we introduce FlowDreamer, which adopts 3D scene flow as explicit motion representations. FlowDreamer first predicts 3D scene flow from past frame and action conditions with a U-Net, and then a diffusion model will predict the future frame utilizing the scene flow. FlowDreamer is trained end-to-end despite its modularized nature. We conduct experiments on 4 different benchmarks, covering both video prediction and visual planning tasks. The results demonstrate that FlowDreamer achieves better performance compared to other baseline RGB-D world models by 7% on semantic similarity, 11% on pixel quality, and 6% on success rate in various robot manipulation domains. 

**Abstract (ZH)**: 本文研究了更好的视觉世界模型在机器人操作中的训练，即能够在过去帧和机器人动作的条件下预测未来视觉观察的模型。具体而言，我们考虑基于RGB-D帧的视觉世界模型（RGB-D视觉世界模型）。与大多数经典方法主要通过隐式处理动力学预测并将其与视觉渲染统一在一个模型中不同，我们提出了FlowDreamer，它采用三维场景流作为显式的运动表示。FlowDreamer 首先使用U-Net预测三维场景流，并利用场景流预测未来帧，然后通过扩散模型完成预测。尽管FlowDreamer 具有模块化结构，但它是端到端训练的。我们在4个不同的基准上进行了实验，涵盖了视频预测和视觉规划任务。实验结果表明，FlowDreamer 在语义相似性、像素质量以及不同机器人操作领域的成功率上分别优于其他基线RGB-D视觉世界模型7%、11%和6%。 

---
# APEX: Action Priors Enable Efficient Exploration for Skill Imitation on Articulated Robots 

**Title (ZH)**: APEX: 动作先验使有装配限制的机器人技能模仿探索更高效 

**Authors**: Shivam Sood, Laukik B Nakhwa, Yuhong Cao, Sun Ge, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2505.10022)  

**Abstract**: Learning by imitation provides an effective way for robots to develop well-regulated complex behaviors and directly benefit from natural demonstrations. State-of-the-art imitation learning (IL) approaches typically leverage Adversarial Motion Priors (AMP), which, despite their impressive results, suffer from two key limitations. They are prone to mode collapse, which often leads to overfitting to the simulation environment and thus increased sim-to-real gap, and they struggle to learn diverse behaviors effectively. To overcome these limitations, we introduce APEX (Action Priors enable Efficient eXploration): a simple yet versatile imitation learning framework that integrates demonstrations directly into reinforcement learning (RL), maintaining high exploration while grounding behavior with expert-informed priors. We achieve this through a combination of decaying action priors, which initially bias exploration toward expert demonstrations but gradually allow the policy to explore independently. This is complemented by a multi-critic RL framework that effectively balances stylistic consistency with task performance. Our approach achieves sample-efficient imitation learning and enables the acquisition of diverse skills within a single policy. APEX generalizes to varying velocities and preserves reference-like styles across complex tasks such as navigating rough terrain and climbing stairs, utilizing only flat-terrain kinematic motion data as a prior. We validate our framework through extensive hardware experiments on the Unitree Go2 quadruped. There, APEX yields diverse and agile locomotion gaits, inherent gait transitions, and the highest reported speed for the platform to the best of our knowledge (peak velocity of ~3.3 m/s on hardware). Our results establish APEX as a compelling alternative to existing IL methods, offering better efficiency, adaptability, and real-world performance. 

**Abstract (ZH)**: 通过模仿学习使机器人能够发展出受良好调控的复杂行为并直接从中受益于自然示范提供了有效的方法。最先进的模仿学习（IL）方法通常利用对抗运动先验（AMP），尽管它们取得了令人印象深刻的成果，但仍面临两个关键限制。它们容易发生模式崩溃，这通常导致过度拟合模拟环境，从而增加了模拟到现实的差距，并且难以有效学习多种行为。为克服这些限制，我们提出了APEX（动作先验促进高效探索）：一种简单但功能强大的模仿学习框架，将示范直接集成到强化学习（RL）中，同时保持高探索性并用专家指导的先验知识为基础行为。我们通过衰减动作先验实现这一点，这些先验最初偏向于专家示范，但逐渐允许策略独立探索。这与多批评家RL框架相辅相成，该框架有效地平衡了风格一致性与任务性能。我们的方法实现了高效的模仿学习，并能够在一个策略中获得多种技能。APEX能够泛化到不同的速度，并在复杂的任务如穿越崎岖地形和上下楼梯中保留参考样式的风格，仅使用平坦地形的运动学运动数据作为先验。我们通过在Unitree Go2四足机器人的广泛硬件实验验证了该框架。在那里，APEX产生了多样且灵活的运动模式，内在的步态转换，并且据我们所知，该平台最高的工作效率（峰值速度约为3.3 m/s）。我们的研究成果确立了APEX作为一种比现有IL方法更具吸引力的替代方案的地位，提供了更好的效率、适应性和实际性能。 

---
# Learning Diverse Natural Behaviors for Enhancing the Agility of Quadrupedal Robots 

**Title (ZH)**: 增强四足机器人敏捷性的多样化自然行为学习 

**Authors**: Huiqiao Fu, Haoyu Dong, Wentao Xu, Zhehao Zhou, Guizhou Deng, Kaiqiang Tang, Daoyi Dong, Chunlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.09979)  

**Abstract**: Achieving animal-like agility is a longstanding goal in quadrupedal robotics. While recent studies have successfully demonstrated imitation of specific behaviors, enabling robots to replicate a broader range of natural behaviors in real-world environments remains an open challenge. Here we propose an integrated controller comprising a Basic Behavior Controller (BBC) and a Task-Specific Controller (TSC) which can effectively learn diverse natural quadrupedal behaviors in an enhanced simulator and efficiently transfer them to the real world. Specifically, the BBC is trained using a novel semi-supervised generative adversarial imitation learning algorithm to extract diverse behavioral styles from raw motion capture data of real dogs, enabling smooth behavior transitions by adjusting discrete and continuous latent variable inputs. The TSC, trained via privileged learning with depth images as input, coordinates the BBC to efficiently perform various tasks. Additionally, we employ evolutionary adversarial simulator identification to optimize the simulator, aligning it closely with reality. After training, the robot exhibits diverse natural behaviors, successfully completing the quadrupedal agility challenge at an average speed of 1.1 m/s and achieving a peak speed of 3.2 m/s during hurdling. This work represents a substantial step toward animal-like agility in quadrupedal robots, opening avenues for their deployment in increasingly complex real-world environments. 

**Abstract (ZH)**: 实现类似动物的敏捷性一直是 quadruped 机器人领域一个长期的目标。虽然近期的研究成功地展示了特定行为的模仿，但在真实环境中超范围复制自然行为依然是一个开放的挑战。我们提出了一种综合控制器，包括基本行为控制器（BBC）和任务特定控制器（TSC），它可以有效地在增强的模拟器中学习多种自然的 quadruped 行为，并高效地将其转移到真实世界。具体而言，BBC 使用一种新的半监督生成对抗模仿学习算法进行训练，从真实的狗的原始运动捕捉数据中提取多样的行为风格，通过调整离散和连续的潜在变量输入实现平滑的行为过渡。TSC 通过特权学习训练，并以深度图像作为输入，协调 BBC 高效地执行各种任务。此外，我们采用了进化对抗模拟器识别来优化模拟器，使其与现实更为贴近。经过训练后的机器人表现出多样化的自然行为，在 quadruped 敏捷性挑战中以平均每秒 1.1 米的速度成功完成任务，并在跳跃过程中达到每秒 3.2 米的最高速度。这项工作代表了向 quadruped 机器人实现类似动物的敏捷性迈出的重要一步，为它们在日益复杂的现实环境中的部署提供了可能。 

---
# Diffusion-SAFE: Shared Autonomy Framework with Diffusion for Safe Human-to-Robot Driving Handover 

**Title (ZH)**: 扩散-SAFE：包含扩散的共享自主框架以实现安全的人机驾驶权交接 

**Authors**: Yunxin Fan, Monroe Kennedy III  

**Link**: [PDF](https://arxiv.org/pdf/2505.09889)  

**Abstract**: Safe handover in shared autonomy for vehicle control is well-established in modern vehicles. However, avoiding accidents often requires action several seconds in advance. This necessitates understanding human driver behavior and an expert control strategy for seamless intervention when a collision or unsafe state is predicted. We propose Diffusion-SAFE, a closed-loop shared autonomy framework leveraging diffusion models to: (1) predict human driving behavior for detection of potential risks, (2) generate safe expert trajectories, and (3) enable smooth handovers by blending human and expert policies over a short time horizon. Unlike prior works which use engineered score functions to rate driving performance, our approach enables both performance evaluation and optimal action sequence generation from demonstrations. By adjusting the forward and reverse processes of the diffusion-based copilot, our method ensures a gradual transition of control authority, by mimicking the drivers' behavior before intervention, which mitigates abrupt takeovers, leading to smooth transitions. We evaluated Diffusion-SAFE in both simulation (CarRacing-v0) and real-world (ROS-based race car), measuring human-driving similarity, safety, and computational efficiency. Results demonstrate a 98.5\% successful handover rate, highlighting the framework's effectiveness in progressively correcting human actions and continuously sampling optimal robot actions. 

**Abstract (ZH)**: 基于扩散模型的Safe手递在共享自主车辆控制中的安全递归研究 

---
# Neural Inertial Odometry from Lie Events 

**Title (ZH)**: 基于李事件的神经惯性里程计 

**Authors**: Royina Karegoudra Jayanth, Yinshuang Xu, Evangelos Chatzipantazis, Kostas Daniilidis, Daniel Gehrig  

**Link**: [PDF](https://arxiv.org/pdf/2505.09780)  

**Abstract**: Neural displacement priors (NDP) can reduce the drift in inertial odometry and provide uncertainty estimates that can be readily fused with off-the-shelf filters. However, they fail to generalize to different IMU sampling rates and trajectory profiles, which limits their robustness in diverse settings. To address this challenge, we replace the traditional NDP inputs comprising raw IMU data with Lie events that are robust to input rate changes and have favorable invariances when observed under different trajectory profiles. Unlike raw IMU data sampled at fixed rates, Lie events are sampled whenever the norm of the IMU pre-integration change, mapped to the Lie algebra of the SE(3) group, exceeds a threshold. Inspired by event-based vision, we generalize the notion of level-crossing on 1D signals to level-crossings on the Lie algebra and generalize binary polarities to normalized Lie polarities within this algebra. We show that training NDPs on Lie events incorporating these polarities reduces the trajectory error of off-the-shelf downstream inertial odometry methods by up to 21% with only minimal preprocessing. We conjecture that many more sensors than IMUs or cameras can benefit from an event-based sampling paradigm and that this work makes an important first step in this direction. 

**Abstract (ZH)**: 基于李事件的神经位移先验（NDP）可以减少惯性 odometry 的漂移，并提供可以与商用滤波器轻松融合的不确定性估计。然而，它们无法在不同的 IMU 采样率和轨迹特征下泛化，这限制了它们在多样环境中的鲁棒性。为了解决这一挑战，我们用对输入率变化具有鲁棒性的李事件替代传统的 NDP 输入，这些李事件在 SE(3) 群的李代数中 IMU 预积分变化的范数超过阈值时进行采样。受事件驱动视觉的启发，我们将 1D 信号上的阈值穿越推广到李代数上的阈值穿越，并在该代数中将二进制极性推广为归一化的李极性。实验表明，通过结合这些极性对李事件进行训练，可以将商用的时间下沉惯性 odometry 方法的轨迹误差最多减少 21%，且仅需少量预处理。我们认为，除了 IMU 或相机之外，还有许多其他传感器可以从基于事件的采样范式中受益，而这项工作是朝着这个方向迈出的重要一步。 

---
# Neural Associative Skill Memories for safer robotics and modelling human sensorimotor repertoires 

**Title (ZH)**: 基于神经关联技能记忆的安全机器人和模拟人类感觉运动 repertoire 方法 

**Authors**: Pranav Mahajan, Mufeng Tang, T. Ed Li, Ioannis Havoutis, Ben Seymour  

**Link**: [PDF](https://arxiv.org/pdf/2505.09760)  

**Abstract**: Modern robots face challenges shared by humans, where machines must learn multiple sensorimotor skills and express them adaptively. Equipping robots with a human-like memory of how it feels to do multiple stereotypical movements can make robots more aware of normal operational states and help develop self-preserving safer robots. Associative Skill Memories (ASMs) aim to address this by linking movement primitives to sensory feedback, but existing implementations rely on hard-coded libraries of individual skills. A key unresolved problem is how a single neural network can learn a repertoire of skills while enabling fault detection and context-aware execution. Here we introduce Neural Associative Skill Memories (ASMs), a framework that utilises self-supervised predictive coding for temporal prediction to unify skill learning and expression, using biologically plausible learning rules. Unlike traditional ASMs which require explicit skill selection, Neural ASMs implicitly recognize and express skills through contextual inference, enabling fault detection across learned behaviours without an explicit skill selection mechanism. Compared to recurrent neural networks trained via backpropagation through time, our model achieves comparable qualitative performance in skill memory expression while using local learning rules and predicts a biologically relevant speed-accuracy trade-off during skill memory expression. This work advances the field of neurorobotics by demonstrating how predictive coding principles can model adaptive robot control and human motor preparation. By unifying fault detection, reactive control, skill memorisation and expression into a single energy-based architecture, Neural ASMs contribute to safer robotics and provide a computational lens to study biological sensorimotor learning. 

**Abstract (ZH)**: 基于预测编码的神经关联技能记忆：统一技能学习与表达以实现自适应机器人控制和生物传感器运动学习 

---
# Unfettered Forceful Skill Acquisition with Physical Reasoning and Coordinate Frame Labeling 

**Title (ZH)**: 无约束的物理推理与坐标系标签驱动的技能习得 

**Authors**: William Xie, Max Conway, Yutong Zhang, Nikolaus Correll  

**Link**: [PDF](https://arxiv.org/pdf/2505.09731)  

**Abstract**: Vision language models (VLMs) exhibit vast knowledge of the physical world, including intuition of physical and spatial properties, affordances, and motion. With fine-tuning, VLMs can also natively produce robot trajectories. We demonstrate that eliciting wrenches, not trajectories, allows VLMs to explicitly reason about forces and leads to zero-shot generalization in a series of manipulation tasks without pretraining. We achieve this by overlaying a consistent visual representation of relevant coordinate frames on robot-attached camera images to augment our query. First, we show how this addition enables a versatile motion control framework evaluated across four tasks (opening and closing a lid, pushing a cup or chair) spanning prismatic and rotational motion, an order of force and position magnitude, different camera perspectives, annotation schemes, and two robot platforms over 220 experiments, resulting in 51% success across the four tasks. Then, we demonstrate that the proposed framework enables VLMs to continually reason about interaction feedback to recover from task failure or incompletion, with and without human supervision. Finally, we observe that prompting schemes with visual annotation and embodied reasoning can bypass VLM safeguards. We characterize prompt component contribution to harmful behavior elicitation and discuss its implications for developing embodied reasoning. Our code, videos, and data are available at: this https URL. 

**Abstract (ZH)**: 视觉语言模型通过在机器人附着的相机图像上叠加一致的视觉表示相关坐标框架，激发力而不是轨迹，展示了在一系列 manipulation 任务中零样本泛化的潜力，无需预训练。我们的代码、视频和数据可在以下网址获取：this https URL。 

---
# EnerVerse-AC: Envisioning Embodied Environments with Action Condition 

**Title (ZH)**: EnerVerse-AC: 融入动作条件的体现环境憧憬 

**Authors**: Yuxin Jiang, Shengcong Chen, Siyuan Huang, Liliang Chen, Pengfei Zhou, Yue Liao, Xindong He, Chiming Liu, Hongsheng Li, Maoqing Yao, Guanghui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.09723)  

**Abstract**: Robotic imitation learning has advanced from solving static tasks to addressing dynamic interaction scenarios, but testing and evaluation remain costly and challenging due to the need for real-time interaction with dynamic environments. We propose EnerVerse-AC (EVAC), an action-conditional world model that generates future visual observations based on an agent's predicted actions, enabling realistic and controllable robotic inference. Building on prior architectures, EVAC introduces a multi-level action-conditioning mechanism and ray map encoding for dynamic multi-view image generation while expanding training data with diverse failure trajectories to improve generalization. As both a data engine and evaluator, EVAC augments human-collected trajectories into diverse datasets and generates realistic, action-conditioned video observations for policy testing, eliminating the need for physical robots or complex simulations. This approach significantly reduces costs while maintaining high fidelity in robotic manipulation evaluation. Extensive experiments validate the effectiveness of our method. Code, checkpoints, and datasets can be found at <this https URL. 

**Abstract (ZH)**: 基于动作条件的世界模型EnerVerse-AC (EVAC):实现真实且可控的机器人推理 

---
# ManipBench: Benchmarking Vision-Language Models for Low-Level Robot Manipulation 

**Title (ZH)**: ManipBench：用于低级机器人操作的视觉-语言模型基准测试 

**Authors**: Enyu Zhao, Vedant Raval, Hejia Zhang, Jiageng Mao, Zeyu Shangguan, Stefanos Nikolaidis, Yue Wang, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2505.09698)  

**Abstract**: Vision-Language Models (VLMs) have revolutionized artificial intelligence and robotics due to their commonsense reasoning capabilities. In robotic manipulation, VLMs are used primarily as high-level planners, but recent work has also studied their lower-level reasoning ability, which refers to making decisions about precise robot movements. However, the community currently lacks a clear and common benchmark that can evaluate how well VLMs can aid low-level reasoning in robotics. Consequently, we propose a novel benchmark, ManipBench, to evaluate the low-level robot manipulation reasoning capabilities of VLMs across various dimensions, including how well they understand object-object interactions and deformable object manipulation. We extensively test 33 representative VLMs across 10 model families on our benchmark, including variants to test different model sizes. Our evaluation shows that the performance of VLMs significantly varies across tasks, and there is a strong correlation between this performance and trends in our real-world manipulation tasks. It also shows that there remains a significant gap between these models and human-level understanding. See our website at: this https URL. 

**Abstract (ZH)**: Vision-Language模型（VLMs）由于其常识推理能力，已彻底改变了人工智能和机器人技术。在机器人操作中，VLMs主要用作高级规划者，但最近的研究也探讨了它们在较低层次上的推理能力，即关于精确机器人运动的决策。然而，当前社区缺乏一个清晰且普遍接受的基准来评估VLMs如何在机器人中辅助低层次推理。因此，我们提出了一种新的基准ManipBench，以从多个维度评估VLMs在机器人低层次操作推理能力，包括它们理解对象间交互和变形物体操作的能力。我们对该基准进行了广泛测试，测试了包括不同模型规模变体在内的33个代表性VLM，覆盖了10个模型家族。我们的评估显示，VLMs在不同任务上的性能差异显著，其性能与我们在真实世界操作任务中的趋势之间存在强烈关联。此外，也显示了这些模型与人类理解之间仍存在显著差距。请访问我们的网站：this https URL。 

---
# EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models 

**Title (ZH)**: EWMBench: 评估体态世界模型中的场景、运动和语义质量 

**Authors**: Hu Yue, Siyuan Huang, Yue Liao, Shengcong Chen, Pengfei Zhou, Liliang Chen, Maoqing Yao, Guanghui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.09694)  

**Abstract**: Recent advances in creative AI have enabled the synthesis of high-fidelity images and videos conditioned on language instructions. Building on these developments, text-to-video diffusion models have evolved into embodied world models (EWMs) capable of generating physically plausible scenes from language commands, effectively bridging vision and action in embodied AI applications. This work addresses the critical challenge of evaluating EWMs beyond general perceptual metrics to ensure the generation of physically grounded and action-consistent behaviors. We propose the Embodied World Model Benchmark (EWMBench), a dedicated framework designed to evaluate EWMs based on three key aspects: visual scene consistency, motion correctness, and semantic alignment. Our approach leverages a meticulously curated dataset encompassing diverse scenes and motion patterns, alongside a comprehensive multi-dimensional evaluation toolkit, to assess and compare candidate models. The proposed benchmark not only identifies the limitations of existing video generation models in meeting the unique requirements of embodied tasks but also provides valuable insights to guide future advancements in the field. The dataset and evaluation tools are publicly available at this https URL. 

**Abstract (ZH)**: Recent Advances in Creative AI Have Enabled the Synthesis of High-Fidelity Images and Videos Conditioned on Language Instructions: Building Embodied World Models (EWMs) for Physically Plausible Scene Generation from Language Commands 

---
# Inferring Driving Maps by Deep Learning-based Trail Map Extraction 

**Title (ZH)**: 基于深度学习的轨迹地图提取驱动地图推断 

**Authors**: Michael Hubbertz, Pascal Colling, Qi Han, Tobias Meisen  

**Link**: [PDF](https://arxiv.org/pdf/2505.10258)  

**Abstract**: High-definition (HD) maps offer extensive and accurate environmental information about the driving scene, making them a crucial and essential element for planning within autonomous driving systems. To avoid extensive efforts from manual labeling, methods for automating the map creation have emerged. Recent trends have moved from offline mapping to online mapping, ensuring availability and actuality of the utilized maps. While the performance has increased in recent years, online mapping still faces challenges regarding temporal consistency, sensor occlusion, runtime, and generalization. We propose a novel offline mapping approach that integrates trails - informal routes used by drivers - into the map creation process. Our method aggregates trail data from the ego vehicle and other traffic participants to construct a comprehensive global map using transformer-based deep learning models. Unlike traditional offline mapping, our approach enables continuous updates while remaining sensor-agnostic, facilitating efficient data transfer. Our method demonstrates superior performance compared to state-of-the-art online mapping approaches, achieving improved generalization to previously unseen environments and sensor configurations. We validate our approach on two benchmark datasets, highlighting its robustness and applicability in autonomous driving systems. 

**Abstract (ZH)**: 高分辨率（HD）地图提供了 Driving 场景的广泛而准确的环境信息，是自主驾驶系统规划中至关重要的元素。为避免手动标注的大量努力，出现了自动化地图创建的方法。近年来的趋势从离线制图转向在线制图，确保使用的地图的可用性和时效性。尽管近年来性能有所提高，但在线制图仍然面临着时间一致性、传感器遮挡、运行时间和泛化等挑战。我们提出了一种新的离线制图方法，将驾驶员使用的随机路线（trails）整合到地图创建过程中。我们的方法使用基于变压器的深度学习模型聚合来自 ego 车辆和其他交通参与者的轨迹数据，构建全面的全局地图。与传统离线制图不同，我们的方法能够持续更新，同时保持传感器无感知，便于高效数据传输。我们的方法在最先进的在线制图方法中表现出优越性能，实现了对以前未见过的环境和传感器配置的更好泛化。我们通过两个基准数据集验证了该方法，强调了其在自主驾驶系统中的鲁棒性和适用性。 

---
# Risk-Aware Safe Reinforcement Learning for Control of Stochastic Linear Systems 

**Title (ZH)**: 风险意识的安全强化学习在随机线性系统控制中的应用 

**Authors**: Babak Esmaeili, Nariman Niknejad, Hamidreza Modares  

**Link**: [PDF](https://arxiv.org/pdf/2505.09734)  

**Abstract**: This paper presents a risk-aware safe reinforcement learning (RL) control design for stochastic discrete-time linear systems. Rather than using a safety certifier to myopically intervene with the RL controller, a risk-informed safe controller is also learned besides the RL controller, and the RL and safe controllers are combined together. Several advantages come along with this approach: 1) High-confidence safety can be certified without relying on a high-fidelity system model and using limited data available, 2) Myopic interventions and convergence to an undesired equilibrium can be avoided by deciding on the contribution of two stabilizing controllers, and 3) highly efficient and computationally tractable solutions can be provided by optimizing over a scalar decision variable and linear programming polyhedral sets. To learn safe controllers with a large invariant set, piecewise affine controllers are learned instead of linear controllers. To this end, the closed-loop system is first represented using collected data, a decision variable, and noise. The effect of the decision variable on the variance of the safe violation of the closed-loop system is formalized. The decision variable is then designed such that the probability of safety violation for the learned closed-loop system is minimized. It is shown that this control-oriented approach reduces the data requirements and can also reduce the variance of safety violations. Finally, to integrate the safe and RL controllers, a new data-driven interpolation technique is introduced. This method aims to maintain the RL agent's optimal implementation while ensuring its safety within environments characterized by noise. The study concludes with a simulation example that serves to validate the theoretical results. 

**Abstract (ZH)**: 基于风险意识的鲁棒强化学习控制设计：适用于随机离散时间线性系统的安全控制器学习 

---
# AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenge 

**Title (ZH)**: AI智能体 vs. 代理型AI：一种概念分类、应用与挑战 

**Authors**: Ranjan Sapkota, Konstantinos I. Roumeliotis, Manoj Karkee  

**Link**: [PDF](https://arxiv.org/pdf/2505.10468)  

**Abstract**: This study critically distinguishes between AI Agents and Agentic AI, offering a structured conceptual taxonomy, application mapping, and challenge analysis to clarify their divergent design philosophies and capabilities. We begin by outlining the search strategy and foundational definitions, characterizing AI Agents as modular systems driven by Large Language Models (LLMs) and Large Image Models (LIMs) for narrow, task-specific automation. Generative AI is positioned as a precursor, with AI Agents advancing through tool integration, prompt engineering, and reasoning enhancements. In contrast, Agentic AI systems represent a paradigmatic shift marked by multi-agent collaboration, dynamic task decomposition, persistent memory, and orchestrated autonomy. Through a sequential evaluation of architectural evolution, operational mechanisms, interaction styles, and autonomy levels, we present a comparative analysis across both paradigms. Application domains such as customer support, scheduling, and data summarization are contrasted with Agentic AI deployments in research automation, robotic coordination, and medical decision support. We further examine unique challenges in each paradigm including hallucination, brittleness, emergent behavior, and coordination failure and propose targeted solutions such as ReAct loops, RAG, orchestration layers, and causal modeling. This work aims to provide a definitive roadmap for developing robust, scalable, and explainable AI agent and Agentic AI-driven systems. >AI Agents, Agent-driven, Vision-Language-Models, Agentic AI Decision Support System, Agentic-AI Applications 

**Abstract (ZH)**: AI代理与自主AI区分研究：结构化概念分类、应用映射与挑战分析 

---
# Demystifying AI Agents: The Final Generation of Intelligence 

**Title (ZH)**: 揭开AI代理的面纱：智能的最终一代 

**Authors**: Kevin J McNamara, Rhea Pritham Marpu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09932)  

**Abstract**: The trajectory of artificial intelligence (AI) has been one of relentless acceleration, evolving from rudimentary rule-based systems to sophisticated, autonomous agents capable of complex reasoning and interaction. This whitepaper chronicles this remarkable journey, charting the key technological milestones--advancements in prompting, training methodologies, hardware capabilities, and architectural innovations--that have converged to create the AI agents of today. We argue that these agents, exemplified by systems like OpenAI's ChatGPT with plugins and xAI's Grok, represent a culminating phase in AI development, potentially constituting the "final generation" of intelligence as we currently conceive it. We explore the capabilities and underlying technologies of these agents, grounded in practical examples, while also examining the profound societal implications and the unprecedented pace of progress that suggests intelligence is now doubling approximately every six months. The paper concludes by underscoring the critical need for wisdom and foresight in navigating the opportunities and challenges presented by this powerful new era of intelligence. 

**Abstract (ZH)**: 人工智能的发展轨迹：从基础规则系统到自主复杂推理代理的持续加速 

---
# Fine-tuning Diffusion Policies with Backpropagation Through Diffusion Timesteps 

**Title (ZH)**: 使用反向传播通过扩散时间步 fine-tuning 扩散策略 

**Authors**: Ningyuan Yang, Jiaxuan Gao, Feng Gao, Yi Wu, Chao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10482)  

**Abstract**: Diffusion policies, widely adopted in decision-making scenarios such as robotics, gaming and autonomous driving, are capable of learning diverse skills from demonstration data due to their high representation power. However, the sub-optimal and limited coverage of demonstration data could lead to diffusion policies that generate sub-optimal trajectories and even catastrophic failures. While reinforcement learning (RL)-based fine-tuning has emerged as a promising solution to address these limitations, existing approaches struggle to effectively adapt Proximal Policy Optimization (PPO) to diffusion models. This challenge stems from the computational intractability of action likelihood estimation during the denoising process, which leads to complicated optimization objectives. In our experiments starting from randomly initialized policies, we find that online tuning of Diffusion Policies demonstrates much lower sample efficiency compared to directly applying PPO on MLP policies (MLP+PPO). To address these challenges, we introduce NCDPO, a novel framework that reformulates Diffusion Policy as a noise-conditioned deterministic policy. By treating each denoising step as a differentiable transformation conditioned on pre-sampled noise, NCDPO enables tractable likelihood evaluation and gradient backpropagation through all diffusion timesteps. Our experiments demonstrate that NCDPO achieves sample efficiency comparable to MLP+PPO when training from scratch, outperforming existing methods in both sample efficiency and final performance across diverse benchmarks, including continuous robot control and multi-agent game scenarios. Furthermore, our experimental results show that our method is robust to the number denoising timesteps in the Diffusion Policy. 

**Abstract (ZH)**: 基于去噪的扩散策略优化：噪声条件下的确定性策略框架 

---
# Vision language models have difficulty recognizing virtual objects 

**Title (ZH)**: 视觉语言模型在识别虚拟物体方面存在困难。 

**Authors**: Tyler Tran, Sangeet Khemlani, J.G. Trafton  

**Link**: [PDF](https://arxiv.org/pdf/2505.10453)  

**Abstract**: Vision language models (VLMs) are AI systems paired with both language and vision encoders to process multimodal input. They are capable of performing complex semantic tasks such as automatic captioning, but it remains an open question about how well they comprehend the visuospatial properties of scenes depicted in the images they process. We argue that descriptions of virtual objects -- objects that are not visually represented in an image -- can help test scene comprehension in these AI systems. For example, an image that depicts a person standing under a tree can be paired with the following prompt: imagine that a kite is stuck in the tree. VLMs that comprehend the scene should update their representations and reason sensibly about the spatial relations between all three objects. We describe systematic evaluations of state-of-the-art VLMs and show that their ability to process virtual objects is inadequate. 

**Abstract (ZH)**: 视觉语言模型（VLMs）是结合了语言和视觉编码器的AI系统，用于处理多模态输入。它们能够执行复杂的语义任务，如自动配图，但关于它们是否能理解处理图像中描绘场景的 visuospatial 特性仍是一个开放问题。我们认为，对虚拟物体的描述——这些物体在图像中未被视觉表示——可以帮助测试这些AI系统的场景理解能力。例如，一幅描绘一个人站在树下的图像可以配上以下提示：想象一个风筝卡在了树上。理解场景的VLM应该更新其表示，并合理地考虑这三个物体之间的空间关系。我们描述了对最先进的VLMs进行系统的评估，并展示了它们处理虚拟物体的能力是不足的。 

---
# Efficient Adaptation of Reinforcement Learning Agents to Sudden Environmental Change 

**Title (ZH)**: 高效的强化学习代理对突发环境变化的适应性调整 

**Authors**: Jonathan Clifford Balloch  

**Link**: [PDF](https://arxiv.org/pdf/2505.10330)  

**Abstract**: Real-world autonomous decision-making systems, from robots to recommendation engines, must operate in environments that change over time. While deep reinforcement learning (RL) has shown an impressive ability to learn optimal policies in stationary environments, most methods are data intensive and assume a world that does not change between training and test time. As a result, conventional RL methods struggle to adapt when conditions change. This poses a fundamental challenge: how can RL agents efficiently adapt their behavior when encountering novel environmental changes during deployment without catastrophically forgetting useful prior knowledge? This dissertation demonstrates that efficient online adaptation requires two key capabilities: (1) prioritized exploration and sampling strategies that help identify and learn from relevant experiences, and (2) selective preservation of prior knowledge through structured representations that can be updated without disruption to reusable components. 

**Abstract (ZH)**: 实世界中的自主决策系统，从机器人到推荐引擎，必须在随时间变化的环境中运行。尽管深度强化学习（RL）在stationary环境中学习最优策略展现了显著的能力，但大多数方法需要大量的数据，并假设训练和测试期间的世界不会发生变化。因此，传统的RL方法在条件变化时难以适应。这提出了一个基本挑战：如何使RL代理在部署过程中遇到新的环境变化时高效地适应其行为，同时避免灾难性遗忘有用的先验知识？本论文证明了高效的在线适应需要两种关键能力：（1）优先探索和采样策略，有助于识别和学习相关经验，以及（2）通过结构化的表示选择性保留先验知识，这些表示可以在不中断可重用组件的情况下进行更新。 

---
# KAITIAN: A Unified Communication Framework for Enabling Efficient Collaboration Across Heterogeneous Accelerators in Embodied AI Systems 

**Title (ZH)**: KAITIAN：跨异构加速器实现实体人工智能系统中高效协作的统一通信框架 

**Authors**: Jieke Lin, Wanyu Wang, Longxiang Yin, Yinhe Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.10183)  

**Abstract**: Embodied Artificial Intelligence (AI) systems, such as autonomous robots and intelligent vehicles, are increasingly reliant on diverse heterogeneous accelerators (e.g., GPGPUs, NPUs, FPGAs) to meet stringent real-time processing and energy-efficiency demands. However, the proliferation of vendor-specific proprietary communication libraries creates significant interoperability barriers, hindering seamless collaboration between different accelerator types and leading to suboptimal resource utilization and performance bottlenecks in distributed AI workloads. This paper introduces KAITIAN, a novel distributed communication framework designed to bridge this gap. KAITIAN provides a unified abstraction layer that intelligently integrates vendor-optimized communication libraries for intra-group efficiency with general-purpose communication protocols for inter-group interoperability. Crucially, it incorporates a load-adaptive scheduling mechanism that dynamically balances computational tasks across heterogeneous devices based on their real-time performance characteristics. Implemented as an extension to PyTorch and rigorously evaluated on a testbed featuring NVIDIA GPUs and Cambricon MLUs, KAITIAN demonstrates significant improvements in resource utilization and scalability for distributed training tasks. Experimental results show that KAITIAN can accelerate training time by up to 42% compared to baseline homogeneous systems, while incurring minimal communication overhead (2.8--4.3%) and maintaining model accuracy. KAITIAN paves the way for more flexible and powerful heterogeneous computing in complex embodied AI applications. 

**Abstract (ZH)**: 嵌入式人工智能系统的实体化：KAITIAN——一种新型分布式通信框架 

---
# AI and Generative AI Transforming Disaster Management: A Survey of Damage Assessment and Response Techniques 

**Title (ZH)**: AI和生成式AItransforming灾难管理：损伤评估与响应技术综述 

**Authors**: Aman Raj, Lakshit Arora, Sanjay Surendranath Girija, Shashank Kapoor, Dipen Pradhan, Ankit Shetgaonkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.08202)  

**Abstract**: Natural disasters, including earthquakes, wildfires and cyclones, bear a huge risk on human lives as well as infrastructure assets. An effective response to disaster depends on the ability to rapidly and efficiently assess the intensity of damage. Artificial Intelligence (AI) and Generative Artificial Intelligence (GenAI) presents a breakthrough solution, capable of combining knowledge from multiple types and sources of data, simulating realistic scenarios of disaster, and identifying emerging trends at a speed previously unimaginable. In this paper, we present a comprehensive review on the prospects of AI and GenAI in damage assessment for various natural disasters, highlighting both its strengths and limitations. We talk about its application to multimodal data such as text, image, video, and audio, and also cover major issues of data privacy, security, and ethical use of the technology during crises. The paper also recognizes the threat of Generative AI misuse, in the form of dissemination of misinformation and for adversarial attacks. Finally, we outline avenues of future research, emphasizing the need for secure, reliable, and ethical Generative AI systems for disaster management in general. We believe that this work represents the first comprehensive survey of Gen-AI techniques being used in the field of Disaster Assessment and Response. 

**Abstract (ZH)**: 自然灾害，包括地震、野火和飓风，对人类生命和基础设施资产构成巨大风险。有效的灾害响应依赖于快速高效地评估破坏程度的能力。人工智能（AI）和生成式人工智能（GenAI）提供了一种突破性的解决方案，能够结合多种类型和来源的数据知识，模拟灾难的 realistic 场景，并以过去无法想象的速度识别新兴趋势。在本文中，我们对 AI 和 GenAI 在各类自然灾害损失评估中的前景进行了全面综述，突出其优势和局限性。我们讨论了其在多模态数据（如文本、图像、视频和音频）中的应用，并涵盖了危机期间数据隐私、安全和伦理使用技术的主要问题。本文还指出了生成式 AI 可能被误用的威胁，包括错误信息的传播和对抗性攻击。最后，我们概述了未来研究的方向，强调了在灾害管理中需要安全、可靠和伦理的生成式 AI 系统的重要性。我们认为，这项工作代表了第一个全面综述在灾害评估和响应领域使用 Gen-AI 技术的研究综述。 

---
