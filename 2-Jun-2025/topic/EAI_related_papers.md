# DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation 

**Title (ZH)**: DexMachina：双臂灵巧操作的功能重定标 

**Authors**: Zhao Mandi, Yifan Hou, Dieter Fox, Yashraj Narang, Ajay Mandlekar, Shuran Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.24853)  

**Abstract**: We study the problem of functional retargeting: learning dexterous manipulation policies to track object states from human hand-object demonstrations. We focus on long-horizon, bimanual tasks with articulated objects, which is challenging due to large action space, spatiotemporal discontinuities, and embodiment gap between human and robot hands. We propose DexMachina, a novel curriculum-based algorithm: the key idea is to use virtual object controllers with decaying strength: an object is first driven automatically towards its target states, such that the policy can gradually learn to take over under motion and contact guidance. We release a simulation benchmark with a diverse set of tasks and dexterous hands, and show that DexMachina significantly outperforms baseline methods. Our algorithm and benchmark enable a functional comparison for hardware designs, and we present key findings informed by quantitative and qualitative results. With the recent surge in dexterous hand development, we hope this work will provide a useful platform for identifying desirable hardware capabilities and lower the barrier for contributing to future research. Videos and more at this https URL 

**Abstract (ZH)**: 功能重定向问题研究：学习跟踪物体状态的灵巧操作策略，基于人类手-物体演示。我们关注长期任务和灵巧物体的双臂操作，由于动作空间庞大、时空连续性中断以及人手与机器人手的实体差距，这一任务具有挑战性。我们提出DexMachina，一种新颖的基于课程的学习算法：关键思想是使用衰减强度的虚拟物体控制器：首先自动驱动物体向目标状态移动，从而使策略能够在运动和接触引导下逐渐学习接管。我们发布了一个包含多种任务和灵巧手的模拟基准，并展示了DexMachina显著优于基线方法。我们的算法和基准为硬件设计的功能比较提供了可能，我们基于定量和定性结果介绍了关键发现。随着灵巧手开发的兴起，我们希望这项工作能为识别 desirable 硬件能力提供一个有用的平台，并降低未来研究的参与门槛。更多信息请访问此链接。 

---
# Bi-Manual Joint Camera Calibration and Scene Representation 

**Title (ZH)**: 双手关节相机标定与场景表示 

**Authors**: Haozhan Tang, Tianyi Zhang, Matthew Johnson-Roberson, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2505.24819)  

**Abstract**: Robot manipulation, especially bimanual manipulation, often requires setting up multiple cameras on multiple robot manipulators. Before robot manipulators can generate motion or even build representations of their environments, the cameras rigidly mounted to the robot need to be calibrated. Camera calibration is a cumbersome process involving collecting a set of images, with each capturing a pre-determined marker. In this work, we introduce the Bi-Manual Joint Calibration and Representation Framework (Bi-JCR). Bi-JCR enables multiple robot manipulators, each with cameras mounted, to circumvent taking images of calibration markers. By leveraging 3D foundation models for dense, marker-free multi-view correspondence, Bi-JCR jointly estimates: (i) the extrinsic transformation from each camera to its end-effector, (ii) the inter-arm relative poses between manipulators, and (iii) a unified, scale-consistent 3D representation of the shared workspace, all from the same captured RGB image sets. The representation, jointly constructed from images captured by cameras on both manipulators, lives in a common coordinate frame and supports collision checking and semantic segmentation to facilitate downstream bimanual coordination tasks. We empirically evaluate the robustness of Bi-JCR on a variety of tabletop environments, and demonstrate its applicability on a variety of downstream tasks. 

**Abstract (ZH)**: 双臂联合校准与表示框架（Bi-JCR）：无需标记的多视角稠密对应与共享工作空间的联合估计 

---
# DiG-Net: Enhancing Quality of Life through Hyper-Range Dynamic Gesture Recognition in Assistive Robotics 

**Title (ZH)**: DiG-Net: 通过超范围动态手势识别提升辅助机器人中的生活质量 

**Authors**: Eran Bamani Beeri, Eden Nissinman, Avishai Sintov  

**Link**: [PDF](https://arxiv.org/pdf/2505.24786)  

**Abstract**: Dynamic hand gestures play a pivotal role in assistive human-robot interaction (HRI), facilitating intuitive, non-verbal communication, particularly for individuals with mobility constraints or those operating robots remotely. Current gesture recognition methods are mostly limited to short-range interactions, reducing their utility in scenarios demanding robust assistive communication from afar. In this paper, we introduce a novel approach designed specifically for assistive robotics, enabling dynamic gesture recognition at extended distances of up to 30 meters, thereby significantly improving accessibility and quality of life. Our proposed Distance-aware Gesture Network (DiG-Net) effectively combines Depth-Conditioned Deformable Alignment (DADA) blocks with Spatio-Temporal Graph modules, enabling robust processing and classification of gesture sequences captured under challenging conditions, including significant physical attenuation, reduced resolution, and dynamic gesture variations commonly experienced in real-world assistive environments. We further introduce the Radiometric Spatio-Temporal Depth Attenuation Loss (RSTDAL), shown to enhance learning and strengthen model robustness across varying distances. Our model demonstrates significant performance improvement over state-of-the-art gesture recognition frameworks, achieving a recognition accuracy of 97.3% on a diverse dataset with challenging hyper-range gestures. By effectively interpreting gestures from considerable distances, DiG-Net significantly enhances the usability of assistive robots in home healthcare, industrial safety, and remote assistance scenarios, enabling seamless and intuitive interactions for users regardless of physical limitations 

**Abstract (ZH)**: 动态手势在辅助人机交互（HRI）中发挥关键作用，促进直观的非语言交流，特别适用于行动受限的个体或远程操作机器人的人。当前的手势识别方法大多局限于短距离交互，限制了其在远距离强辅助通信场景中的应用。本文提出了一种专门为辅助机器人设计的新方法，能够在30米远距离下实现动态手势识别，从而显著提高无障碍性和生活质量。我们提出的Distance-aware Gesture Network (DiG-Net) 有效地结合了Depth-Conditioned Deformable Alignment (DADA) 块与Spatio-Temporal Graph 模块，能够在包括显著物理衰减、降低分辨率和动态手势变化等挑战性条件下，实现手势序列的稳健处理和分类。我们还引入了Radiometric Spatio-Temporal Depth Attenuation Loss (RSTDAL)，证明该损失函数可提高学习效果并增强模型在不同距离下的鲁棒性。我们的模型在多种具有挑战性的超远距离手势数据集上显著优于现有最先进的手势识别框架，准确率达到97.3%。通过有效地从远距离解释手势，DiG-Net 显著提升了辅助机器人在家用医疗保健、工业安全和远程协助场景中的可用性，使用户无论有无身体限制都能实现无缝和直观的交互。 

---
# Reactive Aerobatic Flight via Reinforcement Learning 

**Title (ZH)**: 基于强化学习的反应式 aerobatic 飞行 

**Authors**: Zhichao Han, Xijie Huang, Zhuxiu Xu, Jiarui Zhang, Yuze Wu, Mingyang Wang, Tianyue Wu, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.24396)  

**Abstract**: Quadrotors have demonstrated remarkable versatility, yet their full aerobatic potential remains largely untapped due to inherent underactuation and the complexity of aggressive maneuvers. Traditional approaches, separating trajectory optimization and tracking control, suffer from tracking inaccuracies, computational latency, and sensitivity to initial conditions, limiting their effectiveness in dynamic, high-agility scenarios. Inspired by recent breakthroughs in data-driven methods, we propose a reinforcement learning-based framework that directly maps drone states and aerobatic intentions to control commands, eliminating modular separation to enable quadrotors to perform end-to-end policy optimization for extreme aerobatic maneuvers. To ensure efficient and stable training, we introduce an automated curriculum learning strategy that dynamically adjusts aerobatic task difficulty. Enabled by domain randomization for robust zero-shot sim-to-real transfer, our approach is validated in demanding real-world experiments, including the first demonstration of a drone autonomously performing continuous inverted flight while reactively navigating a moving gate, showcasing unprecedented agility. 

**Abstract (ZH)**: 基于强化学习的无人机端到端极端 aerobatic 调优框架：动态难度自适应训练与实应用场景验证 

---
# Imitation Learning-Based Path Generation for the Complex Assembly of Deformable Objects 

**Title (ZH)**: 基于模仿学习的柔体对象复杂装配路径生成 

**Authors**: Yitaek Kim, Christoffer Sloth  

**Link**: [PDF](https://arxiv.org/pdf/2505.24339)  

**Abstract**: This paper investigates how learning can be used to ease the design of high-quality paths for the assembly of deformable objects. Object dynamics plays an important role when manipulating deformable objects; thus, detailed models are often used when conducting motion planning for deformable objects. We propose to use human demonstrations and learning to enable motion planning of deformable objects with only simple dynamical models of the objects. In particular, we use the offline collision-free path planning, to generate a large number of reference paths based on a simple model of the deformable object. Subsequently, we execute the collision-free paths on a robot with a compliant control such that a human can slightly modify the path to complete the task successfully. Finally, based on the virtual path data sets and the human corrected ones, we use behavior cloning (BC) to create a dexterous policy that follows one reference path to finish a given task. 

**Abstract (ZH)**: 本文探讨了如何通过学习简化对变形物体装配的高质量路径设计。在操作变形物体时，物体动力学起着重要作用；因此，在进行变形物体运动规划时通常会使用详细的模型。我们提出利用人类示范和学习的方法，仅使用简单物体动力模型即可实现变形物体的运动规划。具体而言，我们使用离线无碰撞路径规划，基于简单变形物体模型生成大量参考路径。随后，在具有顺应控制的机器人上执行无碰撞路径，以便人类可以稍微调整路径以成功完成任务。最后，基于虚拟路径数据集和人类修正后的数据集，我们使用行为克隆（BC）创建一个灵巧策略，该策略跟随一个参考路径以完成给定任务。 

---
# SignBot: Learning Human-to-Humanoid Sign Language Interaction 

**Title (ZH)**: SignBot: 学习人类与类人型手语互动 

**Authors**: Guanren Qiao, Sixu Lin, Ronglai Zuo Zhizheng Wu, Kui Jia, Guiliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24266)  

**Abstract**: Sign language is a natural and visual form of language that uses movements and expressions to convey meaning, serving as a crucial means of communication for individuals who are deaf or hard-of-hearing (DHH). However, the number of people proficient in sign language remains limited, highlighting the need for technological advancements to bridge communication gaps and foster interactions with minorities. Based on recent advancements in embodied humanoid robots, we propose SignBot, a novel framework for human-robot sign language interaction. SignBot integrates a cerebellum-inspired motion control component and a cerebral-oriented module for comprehension and interaction. Specifically, SignBot consists of: 1) Motion Retargeting, which converts human sign language datasets into robot-compatible kinematics; 2) Motion Control, which leverages a learning-based paradigm to develop a robust humanoid control policy for tracking sign language gestures; and 3) Generative Interaction, which incorporates translator, responser, and generator of sign language, thereby enabling natural and effective communication between robots and humans. Simulation and real-world experimental results demonstrate that SignBot can effectively facilitate human-robot interaction and perform sign language motions with diverse robots and datasets. SignBot represents a significant advancement in automatic sign language interaction on embodied humanoid robot platforms, providing a promising solution to improve communication accessibility for the DHH community. 

**Abstract (ZH)**: 基于类脑启发的动作控制组件和以大脑为导向的理解与交互模块，我们提出了SignBot，一种新的人机手语互动框架。SignBot 包含：1）动作重定位，将人类手语数据集转换为机器人兼容的动力学；2）动作控制，利用基于学习的方法开发一种稳健的人形控制策略以追踪手语手势；3）生成性交互，集成了手语翻译、响应器和生成器，从而实现机器人与人类之间自然有效的沟通。Simulation 和实地实验结果表明，SignBot 能够有效促进人机互动并在多种机器人和数据集上执行手语动作。SignBot 表现了在人形机器人平台上实现自动手语互动的重要进展，为改善聋人和听力障碍者社区的沟通无障碍提供了一个有前景的解决方案。 

---
# Learning Gentle Humanoid Locomotion and End-Effector Stabilization Control 

**Title (ZH)**: 学习柔和的人形运动和末端执行器稳定控制 

**Authors**: Yitang Li, Yuanhang Zhang, Wenli Xiao, Chaoyi Pan, Haoyang Weng, Guanqi He, Tairan He, Guanya Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.24198)  

**Abstract**: Can your humanoid walk up and hand you a full cup of beer, without spilling a drop? While humanoids are increasingly featured in flashy demos like dancing, delivering packages, traversing rough terrain, fine-grained control during locomotion remains a significant challenge. In particular, stabilizing a filled end-effector (EE) while walking is far from solved, due to a fundamental mismatch in task dynamics: locomotion demands slow-timescale, robust control, whereas EE stabilization requires rapid, high-precision corrections. To address this, we propose SoFTA, a Slow-Fast TwoAgent framework that decouples upper-body and lower-body control into separate agents operating at different frequencies and with distinct rewards. This temporal and objective separation mitigates policy interference and enables coordinated whole-body behavior. SoFTA executes upper-body actions at 100 Hz for precise EE control and lower-body actions at 50 Hz for robust gait. It reduces EE acceleration by 2-5x relative to baselines and performs much closer to human-level stability, enabling delicate tasks such as carrying nearly full cups, capturing steady video during locomotion, and disturbance rejection with EE stability. 

**Abstract (ZH)**: Can Your Humanoid Walk Up and Hand You a Full Cup of Beer Without Spilling a Drop? SoFTA: A Slow-Fast Two-Agent Framework	for Stabilizing Filled End-Effector While Walking 

---
# Humanoid Loco-Manipulations Pattern Generation and Stabilization Control 

**Title (ZH)**: 类人机器人 manipulate 运动模式生成与 stabilization 控制 

**Authors**: Masaki Murooka, Kevin Chappellet, Arnaud Tanguy, Mehdi Benallegue, Iori Kumagai, Mitsuharu Morisawa, Fumio Kanehiro, Abderrahmane Kheddar  

**Link**: [PDF](https://arxiv.org/pdf/2505.24116)  

**Abstract**: In order for a humanoid robot to perform loco-manipulation such as moving an object while walking, it is necessary to account for sustained or alternating external forces other than ground-feet reaction, resulting from humanoid-object contact interactions. In this letter, we propose a bipedal control strategy for humanoid loco-manipulation that can cope with such external forces. First, the basic formulas of the bipedal dynamics, i.e., linear inverted pendulum mode and divergent component of motion, are derived, taking into account the effects of external manipulation forces. Then, we propose a pattern generator to plan center of mass trajectories consistent with the reference trajectory of the manipulation forces, and a stabilizer to compensate for the error between desired and actual manipulation forces. The effectiveness of our controller is assessed both in simulation and loco-manipulation experiments with real humanoid robots. 

**Abstract (ZH)**: humanoid机器人进行搬运操作的双足控制策略：应对外部 manipulative 力 

---
# Towards Tangible Immersion for Cobot Programming-by-Demonstration: Visual, Tactile and Haptic Interfaces for Mixed-Reality Cobot Automation in Semiconductor Manufacturing 

**Title (ZH)**: 面向协作机器人编程示范的实体沉浸感：半导体制造中增强现实协作机器人自动化中的视觉、触觉和 haptic 接口 

**Authors**: David I. Gonzalez-Aguirre, Javier Felip Leon, Javier Felix-Rendon, Roderico Garcia-Leal, Julio C. Zamora Esquivel  

**Link**: [PDF](https://arxiv.org/pdf/2505.24096)  

**Abstract**: Sensor-based reactive and hybrid approaches have proven a promising line of study to address imperfect knowledge in grasping and manipulation. However the reactive approaches are usually tightly coupled to a particular embodiment making transfer of knowledge difficult. This paper proposes a paradigm for modeling and execution of reactive manipulation actions, which makes knowledge transfer to different embodiments possible while retaining the reactive capabilities of the embodiments. The proposed approach extends the idea of control primitives coordinated by a state machine by introducing an embodiment independent layer of abstraction. Abstract manipulation primitives constitute a vocabulary of atomic, embodiment independent actions, which can be coordinated using state machines to describe complex actions. To obtain embodiment specific models, the abstract state machines are automatically translated to embodiment specific models, such that full capabilities of each platform can be utilized. The strength of the manipulation primitives paradigm is demonstrated by developing a set of corresponding embodiment specific primitives for object transport, including a complex reactive grasping primitive. The robustness of the approach is experimentally studied in emptying of a box filled with several unknown objects. The embodiment independence is studied by performing a manipulation task on two different platforms using the same abstract description. 

**Abstract (ZH)**: 基于传感器的反应性和混合方法已被证明是解决抓取和操作中不完善知识问题的一个有前景的研究方向。然而，反应性方法通常与特定的实现紧密耦合，使得知识迁移变得困难。本文提出了一种范式，用于建模和执行反应性操作，使得能够在保持各实现反应性能力的同时，实现知识在不同实现间的迁移。提出的方法通过引入一个与实现无关的抽象层，扩展了由状态机协调的控制原始概念的思想。抽象的操作原始概念构成了一种原子的、实现无关的动作词汇表，可以用状态机协调这些动作来描述复杂动作。为了获取特定于实现的模型，将抽象的状态机自动翻译为特定于实现的模型，以便充分利用每种平台的全部功能。通过为物体运输开发一系列特定于实现的操作原始概念，包括一个复杂的反应性抓取原始概念，展示了操作原始概念范式的强大力量。该方法的鲁棒性通过在一个装有多个未知物体的盒子中执行清空任务进行了实验证明。操作的实现独立性通过使用相同的抽象描述在两种不同平台上执行一项操作任务进行了研究。 

---
# DiffCoTune: Differentiable Co-Tuning for Cross-domain Robot Control 

**Title (ZH)**: 差分共调：跨域机器人控制的可微共调方法 

**Authors**: Lokesh Krishna, Sheng Cheng, Junheng Li, Naira Hovakimyan, Quan Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.24068)  

**Abstract**: The deployment of robot controllers is hindered by modeling discrepancies due to necessary simplifications for computational tractability or inaccuracies in data-generating simulators. Such discrepancies typically require ad-hoc tuning to meet the desired performance, thereby ensuring successful transfer to a target domain. We propose a framework for automated, gradient-based tuning to enhance performance in the deployment domain by leveraging differentiable simulators. Our method collects rollouts in an iterative manner to co-tune the simulator and controller parameters, enabling systematic transfer within a few trials in the deployment domain. Specifically, we formulate multi-step objectives for tuning and employ alternating optimization to effectively adapt the controller to the deployment domain. The scalability of our framework is demonstrated by co-tuning model-based and learning-based controllers of arbitrary complexity for tasks ranging from low-dimensional cart-pole stabilization to high-dimensional quadruped and biped tracking, showing performance improvements across different deployment domains. 

**Abstract (ZH)**: 基于可微模拟器的自动梯度导向调优框架以增强实际部署领域的性能 

---
# Towards a Generalizable Bimanual Foundation Policy via Flow-based Video Prediction 

**Title (ZH)**: 基于流驱动视频预测的可泛化双臂基础策略 

**Authors**: Chenyou Fan, Fangzheng Yan, Chenjia Bai, Jiepeng Wang, Chi Zhang, Zhen Wang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.24156)  

**Abstract**: Learning a generalizable bimanual manipulation policy is extremely challenging for embodied agents due to the large action space and the need for coordinated arm movements. Existing approaches rely on Vision-Language-Action (VLA) models to acquire bimanual policies. However, transferring knowledge from single-arm datasets or pre-trained VLA models often fails to generalize effectively, primarily due to the scarcity of bimanual data and the fundamental differences between single-arm and bimanual manipulation. In this paper, we propose a novel bimanual foundation policy by fine-tuning the leading text-to-video models to predict robot trajectories and training a lightweight diffusion policy for action generation. Given the lack of embodied knowledge in text-to-video models, we introduce a two-stage paradigm that fine-tunes independent text-to-flow and flow-to-video models derived from a pre-trained text-to-video model. Specifically, optical flow serves as an intermediate variable, providing a concise representation of subtle movements between images. The text-to-flow model predicts optical flow to concretize the intent of language instructions, and the flow-to-video model leverages this flow for fine-grained video prediction. Our method mitigates the ambiguity of language in single-stage text-to-video prediction and significantly reduces the robot-data requirement by avoiding direct use of low-level actions. In experiments, we collect high-quality manipulation data for real dual-arm robot, and the results of simulation and real-world experiments demonstrate the effectiveness of our method. 

**Abstract (ZH)**: 基于视觉-语言-动作模型的双臂操作策略学习由于动作空间庞大且需要协调的臂部运动而对体现智能体来说极具挑战性。现有方法依赖于视觉-语言-动作（VLA）模型来获取双臂操作策略。然而，从单臂数据集或预训练的VLA模型迁移知识往往难以有效泛化，主要原因在于双臂数据稀缺以及单臂和双臂操作之间的根本差异。本文提出了一种新颖的双臂基础策略，通过微调领先的文本到视频模型来预测机器人轨迹，并训练一个轻量级的扩散策略以生成动作。鉴于文本到视频模型中缺乏体现知识，我们引入了一种两阶段框架，分别微调独立的文本到流和流到视频模型，这些模型源自预训练的文本到视频模型。具体而言，光学流作为中间变量，提供了一种简洁表示图像间微妙动作的方式。文本到流模型预测光学流以具体化语言指令的意图，而流到视频模型则利用这些流动信息进行精细的视频预测。我们的方法减轻了一步文本到视频预测中语言的模糊性，并通过避免直接使用低级动作显著减少了对机器人数据的要求。在实验中，我们收集了高质量的双臂操作数据用于实际的双臂机器人，并且模拟实验和真实世界的实验结果均证明了本方法的有效性。 

---
# ProofNet++: A Neuro-Symbolic System for Formal Proof Verification with Self-Correction 

**Title (ZH)**: ProofNet++：一种具备自校正能力的神经符号系统形式证明验证 

**Authors**: Murari Ambati  

**Link**: [PDF](https://arxiv.org/pdf/2505.24230)  

**Abstract**: We propose ProofNet++, a neuro-symbolic framework that enhances automated theorem proving by combining large language models (LLMs) with formal proof verification and self-correction mechanisms. Current LLM-based systems suffer from hallucinated logical steps and unverifiable reasoning. ProofNet++ mitigates these limitations by integrating symbolic proof tree supervision, a reinforcement learning loop using verifiers as reward functions, and an iterative self-correction module. Our experiments on miniF2F, Lean's mathlib, and HOL Light show that ProofNet++ significantly improves proof accuracy, correctness, and formal verifiability over prior models. We provide theoretical analysis of the convergence and stability of the verifier-guided RL framework and release our datasets and codebase for future research. 

**Abstract (ZH)**: ProofNet++：一种结合大型语言模型、形式证明验证和自纠错机制的神经符号框架 

---
# InterMT: Multi-Turn Interleaved Preference Alignment with Human Feedback 

**Title (ZH)**: 多轮交错偏好对齐与人类反馈 

**Authors**: Boyuan Chen, Donghai Hong, Jiaming Ji, Jiacheng Zheng, Bowen Dong, Jiayi Zhou, Kaile Wang, Juntao Dai, Xuyao Wang, Wenqi Chen, Qirui Zheng, Wenxin Li, Sirui Han, Yike Guo, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23950)  

**Abstract**: As multimodal large models (MLLMs) continue to advance across challenging tasks, a key question emerges: What essential capabilities are still missing? A critical aspect of human learning is continuous interaction with the environment -- not limited to language, but also involving multimodal understanding and generation. To move closer to human-level intelligence, models must similarly support multi-turn, multimodal interaction. In particular, they should comprehend interleaved multimodal contexts and respond coherently in ongoing exchanges. In this work, we present an initial exploration through the InterMT -- the first preference dataset for multi-turn multimodal interaction, grounded in real human feedback. In this exploration, we particularly emphasize the importance of human oversight, introducing expert annotations to guide the process, motivated by the fact that current MLLMs lack such complex interactive capabilities. InterMT captures human preferences at both global and local levels into nine sub-dimensions, consists of 15.6k prompts, 52.6k multi-turn dialogue instances, and 32.4k human-labeled preference pairs. To compensate for the lack of capability for multi-modal understanding and generation, we introduce an agentic workflow that leverages tool-augmented MLLMs to construct multi-turn QA instances. To further this goal, we introduce InterMT-Bench to assess the ability of MLLMs in assisting judges with multi-turn, multimodal tasks. We demonstrate the utility of \InterMT through applications such as judge moderation and further reveal the multi-turn scaling law of judge model. We hope the open-source of our data can help facilitate further research on aligning current MLLMs to the next step. Our project website can be found at this https URL . 

**Abstract (ZH)**: 多模态大型模型在挑战性任务中不断进步，一个关键问题随之浮现：尚缺失哪些基本能力？人类学习的一个关键方面是对环境的持续互动——不仅限于语言，还包括多模态的理解和生成。为了接近人类级别的智能，模型必须同样支持多轮次、多模态的互动。特别是，它们应该理解交错的多模态上下文并在持续的交流中作出连贯的响应。在本文中，我们通过InterMT进行了初步探索——这是第一个多轮次多模态互动偏好数据集，基于真实人的反馈构建。在这一探索中，我们特别强调了人类监督的重要性，引入了专家注释来引导这一过程，原因是当前的多模态大型模型缺乏这样的复杂互动能力。InterMT在全局和局部两个层次上捕捉人类偏好，包含九个子维度，共有15600个提示，52600个多轮对话实例，以及32400个多轮对话实例的人类标注偏好配对。为了弥补多模态理解和生成能力的缺失，我们引入了一种自主的工作流程，利用工具增强的多模态大型模型构建多轮次问答实例。为了进一步实现这个目标，我们引入了InterMT-Bench来评估多模态大型模型在协助法官完成多轮次、多模态任务方面的能力。我们展示了InterMT在裁判调解等应用中的实用性，并进一步揭示了裁判模型的多轮次扩展规律。我们希望开源数据能帮助推动进一步研究，使其更好地与下一阶段的多模态大型模型对齐。更多内容，请访问此网址：this https URL。 

---
# Mastering Massive Multi-Task Reinforcement Learning via Mixture-of-Expert Decision Transformer 

**Title (ZH)**: 通过专家混合决策变换器掌握大规模多任务强化学习 

**Authors**: Yilun Kong, Guozheng Ma, Qi Zhao, Haoyu Wang, Li Shen, Xueqian Wang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.24378)  

**Abstract**: Despite recent advancements in offline multi-task reinforcement learning (MTRL) have harnessed the powerful capabilities of the Transformer architecture, most approaches focus on a limited number of tasks, with scaling to extremely massive tasks remaining a formidable challenge. In this paper, we first revisit the key impact of task numbers on current MTRL method, and further reveal that naively expanding the parameters proves insufficient to counteract the performance degradation as the number of tasks escalates. Building upon these insights, we propose M3DT, a novel mixture-of-experts (MoE) framework that tackles task scalability by further unlocking the model's parameter scalability. Specifically, we enhance both the architecture and the optimization of the agent, where we strengthen the Decision Transformer (DT) backbone with MoE to reduce task load on parameter subsets, and introduce a three-stage training mechanism to facilitate efficient training with optimal performance. Experimental results show that, by increasing the number of experts, M3DT not only consistently enhances its performance as model expansion on the fixed task numbers, but also exhibits remarkable task scalability, successfully extending to 160 tasks with superior performance. 

**Abstract (ZH)**: 尽管近期离线多任务强化学习（MTRL）的进步充分利用了Transformer架构的强大能力，大多数方法仍专注于少量任务，而将规模扩展到极大量任务仍然是一个艰巨的挑战。在本文中，我们首先重新审视当前MTRL方法中任务数量的关键影响，并进一步揭示简单扩展参数不能有效抵消任务数量增加引起的性能下降。基于这些见解，我们提出了M3DT，这是一种新颖的混合专家（MoE）框架，通过进一步解锁模型的参数规模来应对任务的可扩展性挑战。具体而言，我们增强了智能体的架构和优化，通过将决策Transformer（DT）骨干与MoE结合以减少参数子集上的任务负载，并引入三阶段训练机制以实现高效的训练和最优性能。实验结果表明，通过增加专家数量，M3DT不仅在固定任务数量上的模型扩展中一致提高了性能，而且还表现出显著的任务可扩展性，成功扩展到160个任务并保持了优异的性能。 

---
# Don't Just Follow MLLM Plans: Robust and Efficient Planning for Open-world Agents 

**Title (ZH)**: 别只是跟随MLLM计划：开放世界代理的稳健高效规划 

**Authors**: Seungjoon Lee, Suhwan Kim, Minhyeon Oh, Youngsik Yoon, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2505.24157)  

**Abstract**: Developing autonomous agents capable of mastering complex, multi-step tasks in unpredictable, interactive environments presents a significant challenge. While Large Language Models (LLMs) offer promise for planning, existing approaches often rely on problematic internal knowledge or make unrealistic environmental assumptions. Although recent work explores learning planning knowledge, they still retain limitations due to partial reliance on external knowledge or impractical setups. Indeed, prior research has largely overlooked developing agents capable of acquiring planning knowledge from scratch, directly in realistic settings. While realizing this capability is necessary, it presents significant challenges, primarily achieving robustness given the substantial risk of incorporating LLMs' inaccurate knowledge. Moreover, efficiency is crucial for practicality as learning can demand prohibitive exploration. In response, we introduce Robust and Efficient Planning for Open-world Agents (REPOA), a novel framework designed to tackle these issues. REPOA features three key components: adaptive dependency learning and fine-grained failure-aware operation memory to enhance robustness to knowledge inaccuracies, and difficulty-based exploration to improve learning efficiency. Our evaluation in two established open-world testbeds demonstrates REPOA's robust and efficient planning, showcasing its capability to successfully obtain challenging late-game items that were beyond the reach of prior approaches. 

**Abstract (ZH)**: 开发能够在不确定的交互环境中掌握复杂多步任务的自主代理面临重大挑战。虽然大型语言模型（LLMs）在规划方面展现出潜力，但现有方法往往依赖于有问题的内部知识或不切实际的环境假设。尽管近期工作探索了学习规划知识，但它们仍然受限于对外部知识的部分依赖或不实际的设置。实际上，早期研究大多忽略了开发能够在现实场景中从零开始学习规划知识的代理。虽然实现这一能力是必要的，但这也带来了重大挑战，主要在于如何在大量引入LLMs不准确的知识风险下实现鲁棒性。此外，提高学习效率对于实际应用至关重要，因为学习可能需要大量不可行的探索。为此，我们提出了一种名为Robust and Efficient Planning for Open-world Agents（REPOA）的新型框架，以应对上述问题。REPOA 包含三个关键组件：自适应依赖学习和细粒度失败感知操作记忆，以增强对知识不准确性的鲁棒性，以及基于难度的探索，以提高学习效率。在两个已建立的开放世界测试平台上的评估展示了REPOA 的鲁棒和高效规划能力，证明了其能够成功获取前人方法难以企及的后期游戏资源。 

---
# S4-Driver: Scalable Self-Supervised Driving Multimodal Large Language Modelwith Spatio-Temporal Visual Representation 

**Title (ZH)**: S4-Driver：面向时空视觉表示的可扩展自监督驾驶多模态大语言模型 

**Authors**: Yichen Xie, Runsheng Xu, Tong He, Jyh-Jing Hwang, Katie Luo, Jingwei Ji, Hubert Lin, Letian Chen, Yiren Lu, Zhaoqi Leng, Dragomir Anguelov, Mingxing Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.24139)  

**Abstract**: The latest advancements in multi-modal large language models (MLLMs) have spurred a strong renewed interest in end-to-end motion planning approaches for autonomous driving. Many end-to-end approaches rely on human annotations to learn intermediate perception and prediction tasks, while purely self-supervised approaches--which directly learn from sensor inputs to generate planning trajectories without human annotations often underperform the state of the art. We observe a key gap in the input representation space: end-to-end approaches built on MLLMs are often pretrained with reasoning tasks in 2D image space rather than the native 3D space in which autonomous vehicles plan. To this end, we propose S4-Driver, a scalable self-supervised motion planning algorithm with spatio-temporal visual representation, based on the popular PaLI multimodal large language model. S4-Driver uses a novel sparse volume strategy to seamlessly transform the strong visual representation of MLLMs from perspective view to 3D space without the need to finetune the vision encoder. This representation aggregates multi-view and multi-frame visual inputs and enables better prediction of planning trajectories in 3D space. To validate our method, we run experiments on both nuScenes and Waymo Open Motion Dataset (with in-house camera data). Results show that S4-Driver performs favorably against existing supervised multi-task approaches while requiring no human annotations. It also demonstrates great scalability when pretrained on large volumes of unannotated driving logs. 

**Abstract (ZH)**: 多模态大型语言模型最新进展重新激发了自动驾驶端到端运动规划方法的研究兴趣 

---
# ADG: Ambient Diffusion-Guided Dataset Recovery for Corruption-Robust Offline Reinforcement Learning 

**Title (ZH)**: ADG: 时空扩散引导的数据集恢复方法以实现鲁棒离线强化学习 

**Authors**: Zeyuan Liu, Zhihe Yang, Jiawei Xu, Rui Yang, Jiafei Lyu, Baoxiang Wang, Yunjian Xu, Xiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.23871)  

**Abstract**: Real-world datasets collected from sensors or human inputs are prone to noise and errors, posing significant challenges for applying offline reinforcement learning (RL). While existing methods have made progress in addressing corrupted actions and rewards, they remain insufficient for handling corruption in high-dimensional state spaces and for cases where multiple elements in the dataset are corrupted simultaneously. Diffusion models, known for their strong denoising capabilities, offer a promising direction for this problem-but their tendency to overfit noisy samples limits their direct applicability. To overcome this, we propose Ambient Diffusion-Guided Dataset Recovery (ADG), a novel approach that pioneers the use of diffusion models to tackle data corruption in offline RL. First, we introduce Ambient Denoising Diffusion Probabilistic Models (DDPM) from approximated distributions, which enable learning on partially corrupted datasets with theoretical guarantees. Second, we use the noise-prediction property of Ambient DDPM to distinguish between clean and corrupted data, and then use the clean subset to train a standard DDPM. Third, we employ the trained standard DDPM to refine the previously identified corrupted data, enhancing data quality for subsequent offline RL training. A notable strength of ADG is its versatility-it can be seamlessly integrated with any offline RL algorithm. Experiments on a range of benchmarks, including MuJoCo, Kitchen, and Adroit, demonstrate that ADG effectively mitigates the impact of corrupted data and improves the robustness of offline RL under various noise settings, achieving state-of-the-art results. 

**Abstract (ZH)**: 基于去噪扩散模型的 offline 强化学习数据恢复方法 

---
# DATD3: Depthwise Attention Twin Delayed Deep Deterministic Policy Gradient For Model Free Reinforcement Learning Under Output Feedback Control 

**Title (ZH)**: DATD3:深度可分离注意机制延迟双深确定性策略梯度在输出反馈控制下的模型无关强化学习 

**Authors**: Wuhao Wang, Zhiyong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.23857)  

**Abstract**: Reinforcement learning in real-world applications often involves output-feedback settings, where the agent receives only partial state information. To address this challenge, we propose the Output-Feedback Markov Decision Process (OPMDP), which extends the standard MDP formulation to accommodate decision-making based on observation histories. Building on this framework, we introduce Depthwise Attention Twin Delayed Deep Deterministic Policy Gradient (DATD3), a novel actor-critic algorithm that employs depthwise separable convolution and multi-head attention to encode historical observations. DATD3 maintains policy expressiveness while avoiding the instability of recurrent models. Extensive experiments on continuous control tasks demonstrate that DATD3 outperforms existing memory-based and recurrent baselines under both partial and full observability. 

**Abstract (ZH)**: 输出反馈马尔可夫决策过程（OPMDP）在现实世界应用中的强化学习通常涉及部分状态信息反馈场景。为应对这一挑战，我们提出了输出反馈马尔可夫决策过程（OPMDP），将其标准MDP公式扩展以适应基于观测历史的决策。基于这一框架，我们引入了一种新颖的演员-评论家算法——深度可分离卷积与多头注意力深度确定性策略梯度（DATD3），该算法利用深度可分离卷积和多头注意力来编码历史观测。DATD3保持了策略的表达性，同时避免了循环模型的不稳定性。在连续控制任务上的广泛实验表明，DATD3在部分可观测性和完全可观测性情况下均优于现有的基于记忆和循环的基线方法。 

---
