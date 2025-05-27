# EgoZero: Robot Learning from Smart Glasses 

**Title (ZH)**: EgoZero: 机器人从智能眼镜学习 

**Authors**: Vincent Liu, Ademi Adeniji, Haotian Zhan, Raunaq Bhirangi, Pieter Abbeel, Lerrel Pinto  

**Link**: [PDF](https://arxiv.org/pdf/2505.20290)  

**Abstract**: Despite recent progress in general purpose robotics, robot policies still lag far behind basic human capabilities in the real world. Humans interact constantly with the physical world, yet this rich data resource remains largely untapped in robot learning. We propose EgoZero, a minimal system that learns robust manipulation policies from human demonstrations captured with Project Aria smart glasses, $\textbf{and zero robot data}$. EgoZero enables: (1) extraction of complete, robot-executable actions from in-the-wild, egocentric, human demonstrations, (2) compression of human visual observations into morphology-agnostic state representations, and (3) closed-loop policy learning that generalizes morphologically, spatially, and semantically. We deploy EgoZero policies on a gripper Franka Panda robot and demonstrate zero-shot transfer with 70% success rate over 7 manipulation tasks and only 20 minutes of data collection per task. Our results suggest that in-the-wild human data can serve as a scalable foundation for real-world robot learning - paving the way toward a future of abundant, diverse, and naturalistic training data for robots. Code and videos are available at this https URL. 

**Abstract (ZH)**: 尽管通用机器人技术取得了进步，但机器人策略在实际世界中仍远远落后于基本的人类能力。人类不断与物理世界互动，但这些丰富的数据资源在机器人学习中尚未充分利用。我们提出了EgoZero，一个最小化的系统，通过使用Project Aria智能眼镜捕获的人类演示，结合零机器人数据学习 robust 操作策略。EgoZero能够实现：(1) 从野生环境中的第一人称人类演示中提取完整可由机器人执行的动作，(2) 将人类视觉观察压缩为形态无关的状态表示，以及(3) 具有形态、空间和语义泛化的闭环策略学习。我们在夹爪机器人Franka Panda上部署了EgoZero策略，并在7个操作任务中展示了零样本迁移，成功率达到70%，每任务仅需20分钟的数据收集。我们的结果表明，野生人类数据可以作为现实世界机器人学习的可扩展基础——铺就了一条通往机器人丰富、多样和自然训练数据的道路。代码和视频可在以下链接获取。 

---
# Uncertainty-Aware Safety-Critical Decision and Control for Autonomous Vehicles at Unsignalized Intersections 

**Title (ZH)**: 考虑不确定性的安全critical决策与控制：无信号交叉口自主车辆应用 

**Authors**: Ran Yu, Zhuoren Li, Lu Xiong, Wei Han, Bo Leng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19939)  

**Abstract**: Reinforcement learning (RL) has demonstrated potential in autonomous driving (AD) decision tasks. However, applying RL to urban AD, particularly in intersection scenarios, still faces significant challenges. The lack of safety constraints makes RL vulnerable to risks. Additionally, cognitive limitations and environmental randomness can lead to unreliable decisions in safety-critical scenarios. Therefore, it is essential to quantify confidence in RL decisions to improve safety. This paper proposes an Uncertainty-aware Safety-Critical Decision and Control (USDC) framework, which generates a risk-averse policy by constructing a risk-aware ensemble distributional RL, while estimating uncertainty to quantify the policy's reliability. Subsequently, a high-order control barrier function (HOCBF) is employed as a safety filter to minimize intervention policy while dynamically enhancing constraints based on uncertainty. The ensemble critics evaluate both HOCBF and RL policies, embedding uncertainty to achieve dynamic switching between safe and flexible strategies, thereby balancing safety and efficiency. Simulation tests on unsignalized intersections in multiple tasks indicate that USDC can improve safety while maintaining traffic efficiency compared to baselines. 

**Abstract (ZH)**: 不确定性和安全性aware的决策与控制框架（USDC）：风险规避策略生成及动态安全过滤 

---
# Integrating emotional intelligence, memory architecture, and gestures to achieve empathetic humanoid robot interaction in an educational setting 

**Title (ZH)**: 融合情感智能、记忆架构与手势实现教育环境中同理心 humanoid 机器人交互 

**Authors**: Fuze Sun, Lingyu Li, Shixiangyue Meng, Xiaoming Teng, Terry Payne, Paul Craig  

**Link**: [PDF](https://arxiv.org/pdf/2505.19803)  

**Abstract**: This study investigates the integration of individual human traits into an empathetically adaptive educational robot tutor system designed to improve student engagement and learning outcomes with corresponding Engagement Vector measurement. While prior research in the field of Human-Robot Interaction (HRI) has examined the integration of the traits, such as emotional intelligence, memory-driven personalization, and non-verbal communication, by themselves, they have thus-far neglected to consider their synchronized integration into a cohesive, operational education framework. To address this gap, we customize a Multi-Modal Large Language Model (LLaMa 3.2 from Meta) deployed with modules for human-like traits (emotion, memory and gestures) into an AI-Agent framework. This constitutes to the robot's intelligent core mimicing the human emotional system, memory architecture and gesture control to allow the robot to behave more empathetically while recognizing and responding appropriately to the student's emotional state. It can also recall the student's past learning record and adapt its style of interaction accordingly. This allows the robot tutor to react to the student in a more sympathetic manner by delivering personalized verbal feedback synchronized with relevant gestures. Our study investigates the extent of this effect through the introduction of Engagement Vector Model which can be a surveyor's pole for judging the quality of HRI experience. Quantitative and qualitative results demonstrate that such an empathetic responsive approach significantly improves student engagement and learning outcomes compared with a baseline humanoid robot without these human-like traits. This indicates that robot tutors with empathetic capabilities can create a more supportive, interactive learning experience that ultimately leads to better outcomes for the student. 

**Abstract (ZH)**: 本研究探讨将个体人类特质整合进一种具有同理适应性教育机器教练系统的应用，该系统旨在通过相应的 Engagement Vector 测量来提高学生参与度和学习成果。 

---
# TeViR: Text-to-Video Reward with Diffusion Models for Efficient Reinforcement Learning 

**Title (ZH)**: TeViR: 基于扩散模型的文本到视频奖励高效强化学习 

**Authors**: Yuhui Chen, Haoran Li, Zhennan Jiang, Haowei Wen, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19769)  

**Abstract**: Developing scalable and generalizable reward engineering for reinforcement learning (RL) is crucial for creating general-purpose agents, especially in the challenging domain of robotic manipulation. While recent advances in reward engineering with Vision-Language Models (VLMs) have shown promise, their sparse reward nature significantly limits sample efficiency. This paper introduces TeViR, a novel method that leverages a pre-trained text-to-video diffusion model to generate dense rewards by comparing the predicted image sequence with current observations. Experimental results across 11 complex robotic tasks demonstrate that TeViR outperforms traditional methods leveraging sparse rewards and other state-of-the-art (SOTA) methods, achieving better sample efficiency and performance without ground truth environmental rewards. TeViR's ability to efficiently guide agents in complex environments highlights its potential to advance reinforcement learning applications in robotic manipulation. 

**Abstract (ZH)**: 基于扩散模型的文本到视频奖励生成方法在强化学习中的应用：促进可扩展和通用的奖励工程，特别是在机器人操作的挑战性领域中的应用 

---
# RFTF: Reinforcement Fine-tuning for Embodied Agents with Temporal Feedback 

**Title (ZH)**: RFTF: 基于时间反馈的强化细调用于体现代理 

**Authors**: Junyang Shu, Zhiwei Lin, Yongtao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19767)  

**Abstract**: Vision-Language-Action (VLA) models have demonstrated significant potential in the field of embodied intelligence, enabling agents to follow human instructions to complete complex tasks in physical environments. Existing embodied agents are often trained through behavior cloning, which requires expensive data and computational resources and is constrained by human demonstrations. To address this issue, many researchers explore the application of reinforcement fine-tuning to embodied agents. However, typical reinforcement fine-tuning methods for embodied agents usually rely on sparse, outcome-based rewards, which struggle to provide fine-grained feedback for specific actions within an episode, thus limiting the model's manipulation capabilities and generalization performance. In this paper, we propose RFTF, a novel reinforcement fine-tuning method that leverages a value model to generate dense rewards in embodied scenarios. Specifically, our value model is trained using temporal information, eliminating the need for costly robot action labels. In addition, RFTF incorporates a range of techniques, such as GAE and sample balance to enhance the effectiveness of the fine-tuning process. By addressing the sparse reward problem in reinforcement fine-tuning, our method significantly improves the performance of embodied agents, delivering superior generalization and adaptation capabilities across diverse embodied tasks. Experimental results show that embodied agents fine-tuned with RFTF achieve new state-of-the-art performance on the challenging CALVIN ABC-D with an average success length of 4.296. Moreover, RFTF enables rapid adaptation to new environments. After fine-tuning in the D environment of CALVIN for a few episodes, RFTF achieved an average success length of 4.301 in this new environment. 

**Abstract (ZH)**: 基于视觉-语言-动作的强化细调方法(RFTF)在体态智能中的应用 

---
# Extremum Flow Matching for Offline Goal Conditioned Reinforcement Learning 

**Title (ZH)**: 离线目标条件强化学习中的极值流匹配 

**Authors**: Quentin Rouxel, Clemente Donoso, Fei Chen, Serena Ivaldi, Jean-Baptiste Mouret  

**Link**: [PDF](https://arxiv.org/pdf/2505.19717)  

**Abstract**: Imitation learning is a promising approach for enabling generalist capabilities in humanoid robots, but its scaling is fundamentally constrained by the scarcity of high-quality expert demonstrations. This limitation can be mitigated by leveraging suboptimal, open-ended play data, often easier to collect and offering greater diversity. This work builds upon recent advances in generative modeling, specifically Flow Matching, an alternative to Diffusion models. We introduce a method for estimating the extremum of the learned distribution by leveraging the unique properties of Flow Matching, namely, deterministic transport and support for arbitrary source distributions. We apply this method to develop several goal-conditioned imitation and reinforcement learning algorithms based on Flow Matching, where policies are conditioned on both current and goal observations. We explore and compare different architectural configurations by combining core components, such as critic, planner, actor, or world model, in various ways. We evaluated our agents on the OGBench benchmark and analyzed how different demonstration behaviors during data collection affect performance in a 2D non-prehensile pushing task. Furthermore, we validated our approach on real hardware by deploying it on the Talos humanoid robot to perform complex manipulation tasks based on high-dimensional image observations, featuring a sequence of pick-and-place and articulated object manipulation in a realistic kitchen environment. Experimental videos and code are available at: this https URL 

**Abstract (ZH)**: 模仿学习是一种有望赋予类人机器人一般性能力的方法，但由于高质量专家演示数据稀缺，其扩展受到根本约束。可以通过利用次优的开放性玩耍数据来缓解这一限制，这类数据通常更容易收集并且提供了更大的多样性。本工作基于生成建模领域的近期进展，特别是Flow Matching，这是一种与扩散模型不同的生成模型。我们提出了一种方法，通过利用Flow Matching的独特属性，即确定性的传输和任意源分布的支持，来估计学习分布的极值。我们应用这种方法开发了几种基于Flow Matching的目标条件模仿和强化学习算法，其中策略同时基于当前和目标观察进行条件约束。我们通过以不同方式结合核心组件，如评论者、规划者、演员或世界模型，探索和比较不同的架构配置。我们在OGBench基准上评估了我们的代理，并分析了数据收集过程中不同演示行为对二维非拾取推任务性能的影响。此外，我们在真实的硬件上验证了我们的方法，将其部署在Talos类人机器人上，进行基于高维图像观察的复杂操作任务，包括在真实厨房环境中的拾取-放置和动作化物体操作序列。实验视频和代码可在以下链接获取：this https URL 

---
# Whole-body Multi-contact Motion Control for Humanoid Robots Based on Distributed Tactile Sensors 

**Title (ZH)**: 基于分布式触觉传感器的人形机器人全身多点接触运动控制 

**Authors**: Masaki Murooka, Kensuke Fukumitsu, Marwan Hamze, Mitsuharu Morisawa, Hiroshi Kaminaga, Fumio Kanehiro, Eiichi Yoshida  

**Link**: [PDF](https://arxiv.org/pdf/2505.19580)  

**Abstract**: To enable humanoid robots to work robustly in confined environments, multi-contact motion that makes contacts not only at extremities, such as hands and feet, but also at intermediate areas of the limbs, such as knees and elbows, is essential. We develop a method to realize such whole-body multi-contact motion involving contacts at intermediate areas by a humanoid robot. Deformable sheet-shaped distributed tactile sensors are mounted on the surface of the robot's limbs to measure the contact force without significantly changing the robot body shape. The multi-contact motion controller developed earlier, which is dedicated to contact at extremities, is extended to handle contact at intermediate areas, and the robot motion is stabilized by feedback control using not only force/torque sensors but also distributed tactile sensors. Through verification on dynamics simulations, we show that the developed tactile feedback improves the stability of whole-body multi-contact motion against disturbances and environmental errors. Furthermore, the life-sized humanoid RHP Kaleido demonstrates whole-body multi-contact motions, such as stepping forward while supporting the body with forearm contact and balancing in a sitting posture with thigh contacts. 

**Abstract (ZH)**: 使人形机器人能够在受限环境中 robust 地工作，需要实现多点接触运动，不仅在手和脚等末端部位，还需在肢体的中间部位，如膝盖和肘部，进行接触。我们开发了一种方法，通过在人形机器人肢体表面安装可变形的片状分布式触觉传感器，实现包括中间部位接触的全身多点接触运动。此前专为末端接触开发的多点接触运动控制器被扩展以处理中间部位的接触，并通过力/力矩传感器和分布式触觉传感器的反馈控制来稳定机器人运动。通过动力学仿真验证，我们展示了开发的触觉反馈可提高全身多点接触运动在干扰和环境误差下的稳定性。此外，全尺寸人形机器人 RHP Kaleido 展示了全身多点接触运动，如起立时通过前臂接触支撑身体，以及通过大腿接触实现坐姿平衡。 

---
# Situationally-Aware Dynamics Learning 

**Title (ZH)**: 情境感知动态学习 

**Authors**: Alejandro Murillo-Gonzalez, Lantao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19574)  

**Abstract**: Autonomous robots operating in complex, unstructured environments face significant challenges due to latent, unobserved factors that obscure their understanding of both their internal state and the external world. Addressing this challenge would enable robots to develop a more profound grasp of their operational context. To tackle this, we propose a novel framework for online learning of hidden state representations, with which the robots can adapt in real-time to uncertain and dynamic conditions that would otherwise be ambiguous and result in suboptimal or erroneous behaviors. Our approach is formalized as a Generalized Hidden Parameter Markov Decision Process, which explicitly models the influence of unobserved parameters on both transition dynamics and reward structures. Our core innovation lies in learning online the joint distribution of state transitions, which serves as an expressive representation of latent ego- and environmental-factors. This probabilistic approach supports the identification and adaptation to different operational situations, improving robustness and safety. Through a multivariate extension of Bayesian Online Changepoint Detection, our method segments changes in the underlying data generating process governing the robot's dynamics. The robot's transition model is then informed with a symbolic representation of the current situation derived from the joint distribution of latest state transitions, enabling adaptive and context-aware decision-making. To showcase the real-world effectiveness, we validate our approach in the challenging task of unstructured terrain navigation, where unmodeled and unmeasured terrain characteristics can significantly impact the robot's motion. Extensive experiments in both simulation and real world reveal significant improvements in data efficiency, policy performance, and the emergence of safer, adaptive navigation strategies. 

**Abstract (ZH)**: 自主机器人在复杂、未结构化环境中的操作面临显著挑战，由于潜在、未观察到的因素模糊了其对其内部状态和外部世界的理解。解决这一挑战将使机器人能够更深刻地理解其操作环境。为此，我们提出了一种新颖的在线学习隐藏状态表示框架，使机器人能够实时适应原本模糊且动态的条件，从而避免次优或错误行为。我们的方法形式化为广义隐藏参数马尔可夫决策过程，明确建模未观察到参数对状态转移动力学和奖励结构的影响。我们核心的创新在于在线学习状态转移的联合分布，作为潜在自我和环境因素的表达表示。这种概率方法支持识别和适应不同操作场景，提高鲁棒性和安全性。通过多变量贝叶斯在线变化点检测的扩展，我们的方法将数据生成过程的变化划分为多个段。然后，机器人的转移模型借助最新状态转移联合分布的符号表示来更新，使机器人能够进行适应性和情境意识决策。为了展示其实用性，我们在机器人在复杂地形导航的具有挑战性的任务中验证了该方法，其中未建模和未测量的地形特性显著影响机器人运动。在模拟和现实世界的广泛实验中，显示出数据效率、策略性能的显著提升以及更安全、更具适应性的导航策略的出现。 

---
# Real-time Whole-body Model Predictive Control for Bipedal Locomotion with a Novel Kino-dynamic Model and Warm-start Method 

**Title (ZH)**: 基于新颖的时空动态模型和预热方法的实时全身模型预测控制在双足步行中的应用 

**Authors**: Junhyung Kim, Hokyun Lee, Jaeheung Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.19540)  

**Abstract**: Advancements in optimization solvers and computing power have led to growing interest in applying whole-body model predictive control (WB-MPC) to bipedal robots. However, the high degrees of freedom and inherent model complexity of bipedal robots pose significant challenges in achieving fast and stable control cycles for real-time performance. This paper introduces a novel kino-dynamic model and warm-start strategy for real-time WB-MPC in bipedal robots. Our proposed kino-dynamic model combines the linear inverted pendulum plus flywheel and full-body kinematics model. Unlike the conventional whole-body model that rely on the concept of contact wrenches, our model utilizes the zero-moment point (ZMP), reducing baseline computational costs and ensuring consistently low latency during contact state transitions. Additionally, a modularized multi-layer perceptron (MLP) based warm-start strategy is proposed, leveraging a lightweight neural network to provide a good initial guess for each control cycle. Furthermore, we present a ZMP-based whole-body controller (WBC) that extends the existing WBC for explicitly controlling impulses and ZMP, integrating it into the real-time WB-MPC framework. Through various comparative experiments, the proposed kino-dynamic model and warm-start strategy have been shown to outperform previous studies. Simulations and real robot experiments further validate that the proposed framework demonstrates robustness to perturbation and satisfies real-time control requirements during walking. 

**Abstract (ZH)**: 优化求解器的发展和计算能力的提升促进了将整体动力学模型预测控制（WB-MPC）应用于 bipedal 机器人的研究兴趣。然而，bipedal 机器人的高自由度和固有模型复杂性给实现快速稳定的控制循环带来了显著挑战。本文介绍了一种针对实时 WB-MPC 的新型kino-dynamic模型和预热启动策略。我们提出的kino-dynamic模型结合了线性倒摆加飞轮和全身运动学模型。与传统的基于接触 wrench 的整体模型不同，我们的模型使用零力矩点（ZMP），降低基本计算成本并在接触状态转换期间确保一致的低延迟。此外，提出了一种基于模块化多层感知机（MLP）的预热启动策略，通过轻量级神经网络为每个控制周期提供良好的初始猜测。同时，我们提出了一种基于ZMP的整体动力学控制器（WBC），扩展现有WBC以显式控制冲量和ZMP，并将其集成到实时WB-MPC框架中。通过各种比较实验，所提出的kino-dynamic模型和预热启动策略的表现优于先前的研究。仿真和实际机器人实验进一步验证了所提出框架在行走过程中对干扰具有鲁棒性和满足实时控制要求的能力。 

---
# Heavy lifting tasks via haptic teleoperation of a wheeled humanoid 

**Title (ZH)**: 通过轮式人形机器人的触觉远程操作进行重载任务 

**Authors**: Amartya Purushottam, Jack Yan, Christopher Yu, Joao Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2505.19530)  

**Abstract**: Humanoid robots can support human workers in physically demanding environments by performing tasks that require whole-body coordination, such as lifting and transporting heavy this http URL tasks, which we refer to as Dynamic Mobile Manipulation (DMM), require the simultaneous control of locomotion, manipulation, and posture under dynamic interaction forces. This paper presents a teleoperation framework for DMM on a height-adjustable wheeled humanoid robot for carrying heavy payloads. A Human-Machine Interface (HMI) enables whole-body motion retargeting from the human pilot to the robot by capturing the motion of the human and applying haptic feedback. The pilot uses body motion to regulate robot posture and locomotion, while arm movements guide this http URL time haptic feedback delivers end effector wrenches and balance related cues, closing the loop between human perception and robot environment interaction. We evaluate the different telelocomotion mappings that offer varying levels of balance assistance, allowing the pilot to either manually or automatically regulate the robot's lean in response to payload-induced disturbances. The system is validated in experiments involving dynamic lifting of barbells and boxes up to 2.5 kg (21% of robot mass), demonstrating coordinated whole-body control, height variation, and disturbance handling under pilot guidance. Video demo can be found at: this https URL 

**Abstract (ZH)**: 可调节高度的轮式人形机器人在动态搬运重载荷任务中的远程操作框架 

---
# Learning Dynamics under Environmental Constraints via Measurement-Induced Bundle Structures 

**Title (ZH)**: 在环境约束下通过测量诱导的集合结构学习动力学 

**Authors**: Dongzhe Zheng, Wenjie Mei  

**Link**: [PDF](https://arxiv.org/pdf/2505.19521)  

**Abstract**: Learning unknown dynamics under environmental (or external) constraints is fundamental to many fields (e.g., modern robotics), particularly challenging when constraint information is only locally available and uncertain. Existing approaches requiring global constraints or using probabilistic filtering fail to fully exploit the geometric structure inherent in local measurements (by using, e.g., sensors) and constraints. This paper presents a geometric framework unifying measurements, constraints, and dynamics learning through a fiber bundle structure over the state space. This naturally induced geometric structure enables measurement-aware Control Barrier Functions that adapt to local sensing (or measurement) conditions. By integrating Neural ODEs, our framework learns continuous-time dynamics while preserving geometric constraints, with theoretical guarantees of learning convergence and constraint satisfaction dependent on sensing quality. The geometric framework not only enables efficient dynamics learning but also suggests promising directions for integration with reinforcement learning approaches. Extensive simulations demonstrate significant improvements in both learning efficiency and constraint satisfaction over traditional methods, especially under limited and uncertain sensing conditions. 

**Abstract (ZH)**: 在环境约束下学习未知动力学：利用局部和不确定约束的几何框架 

---
# DiffE2E: Rethinking End-to-End Driving with a Hybrid Action Diffusion and Supervised Policy 

**Title (ZH)**: DiffE2E: 重新思考基于混合动作扩散和监督策略的端到端驾驶 

**Authors**: Rui Zhao, Yuze Fan, Ziguo Chen, Fei Gao, Zhenhai Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19516)  

**Abstract**: End-to-end learning has emerged as a transformative paradigm in autonomous driving. However, the inherently multimodal nature of driving behaviors and the generalization challenges in long-tail scenarios remain critical obstacles to robust deployment. We propose DiffE2E, a diffusion-based end-to-end autonomous driving framework. This framework first performs multi-scale alignment of multi-sensor perception features through a hierarchical bidirectional cross-attention mechanism. It then introduces a novel class of hybrid diffusion-supervision decoders based on the Transformer architecture, and adopts a collaborative training paradigm that seamlessly integrates the strengths of both diffusion and supervised policy. DiffE2E models structured latent spaces, where diffusion captures the distribution of future trajectories and supervision enhances controllability and robustness. A global condition integration module enables deep fusion of perception features with high-level targets, significantly improving the quality of trajectory generation. Subsequently, a cross-attention mechanism facilitates efficient interaction between integrated features and hybrid latent variables, promoting the joint optimization of diffusion and supervision objectives for structured output generation, ultimately leading to more robust control. Experiments demonstrate that DiffE2E achieves state-of-the-art performance in both CARLA closed-loop evaluations and NAVSIM benchmarks. The proposed integrated diffusion-supervision policy offers a generalizable paradigm for hybrid action representation, with strong potential for extension to broader domains including embodied intelligence. More details and visualizations are available at \href{this https URL}{project website}. 

**Abstract (ZH)**: 基于扩散的端到端自动驾驶框架：DiffE2E 

---
# SMAP: Self-supervised Motion Adaptation for Physically Plausible Humanoid Whole-body Control 

**Title (ZH)**: 自监督运动适应：实现物理合理的人形全身控制 

**Authors**: Haoyu Zhao, Sixu Lin, Qingwei Ben, Minyue Dai, Hao Fei, Jingbo Wang, Hua Zou, Junting Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.19463)  

**Abstract**: This paper presents a novel framework that enables real-world humanoid robots to maintain stability while performing human-like motion. Current methods train a policy which allows humanoid robots to follow human body using the massive retargeted human data via reinforcement learning. However, due to the heterogeneity between human and humanoid robot motion, directly using retargeted human motion reduces training efficiency and stability. To this end, we introduce SMAP, a novel whole-body tracking framework that bridges the gap between human and humanoid action spaces, enabling accurate motion mimicry by humanoid robots. The core idea is to use a vector-quantized periodic autoencoder to capture generic atomic behaviors and adapt human motion into physically plausible humanoid motion. This adaptation accelerates training convergence and improves stability when handling novel or challenging motions. We then employ a privileged teacher to distill precise mimicry skills into the student policy with a proposed decoupled reward. We conduct experiments in simulation and real world to demonstrate the superiority stability and performance of SMAP over SOTA methods, offering practical guidelines for advancing whole-body control in humanoid robots. 

**Abstract (ZH)**: 本文提出了一种新型框架，使现实世界中的类人机器人在执行人体类似动作时能够保持稳定性。当前的方法通过强化学习训练一个策略，让用户有大量的重目标人类数据来引导类人机器人跟随人体动作。然而，由于人类和类人机器人动作之间的异质性，直接使用重目标人类动作会降低训练效率和稳定性。为此，我们引入了SMAP，这是一种新颖的整体人体跟踪框架，该框架填补了人类和类人动作空间之间的差距，使类人机器人能够准确模仿人体动作。核心思想是使用向量量化周期自编码器捕捉通用的基本行为，并将人类动作适应为物理上合理的类人动作。这种适应加速了训练收敛并提高了在处理新型或具有挑战性动作时的稳定性。然后，我们使用一个特权教师通过提出解耦奖励来将精确的模仿技能传授给学生策略。我们在仿真和现实世界中进行了实验，以展示SMAP在与当前最佳方法相比的优势稳定性和性能，并提供了关于推进类人机器人全身控制的实用指导。 

---
# Towards Humanoid Robot Autonomy: A Dynamic Architecture Integrating Continuous thought Machines (CTM) and Model Context Protocol (MCP) 

**Title (ZH)**: humanoid机器人自主性的迈进：一种结合连续思维机器(CTM)和模型上下文协议(MCP)的动态架构 

**Authors**: Libo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19339)  

**Abstract**: To address the gaps between the static pre-set "thinking-planning-action" of humanoid robots in unfamiliar scenarios and the highly programmed "call tool-return result" due to the lack of autonomous coding capabilities, this work designs a dynamic architecture connecting continuous thought machines (CTM) and model context protocol (MCP). It proposes a theoretical parallel solution through tick-slab and uses rank compression to achieve parameter suppression to provide a solution for achieving autonomous actions due to autonomous coding. The researcher used a simulation-based experiment using OpenAI's o4-mini-high as a tool to build the experimental environment, and introduced the extended SayCan dataset to conduct nine epochs of experiments. The experimental results show that the CTM-MCP architecture is feasible and effective through the data results of seven metrics: task success rate (TSR), execution success rate (ESR), average episode length (AEL), ROSCOE, REVEAL, proficiency self-assessment (PSA), task effectiveness (TE). In practice, it provides a reference experience for exploring the autonomous dynamic coding of humanoid robots based on continuous thinking to achieve human-like autonomous actions. 

**Abstract (ZH)**: 针对人在陌生场景中动态的“思考-规划-行动”与 humanoid 机器人因缺乏自主编码能力而导致的高程化“调用工具-返回结果”之间的差距，本研究设计了一种动态架构，连接连续思考机器（CTM）和模型上下文协议（MCP）。通过 tickslab 提出理论并使用秩压缩实现参数抑制，以实现因自主编码而产生的自主行动。研究者使用基于 OpenAI 的 o4-mini-high 进行仿真实验构建实验环境，并引入扩展的 SayCan 数据集进行了九个时期的实验。实验结果表明，CTM-MCP 架构通过七项指标（任务成功率 TSR、执行成功率 ESR、平均期长度 AEL、ROSCOER、REVEAL、熟练度自我评估 PSA、任务有效性 TE）的数据结果证明了其实用性和有效性。在实践中，该研究为基于连续思考实现 humanoid 机器人自主动态编码提供了参考经验，以实现类人的自主行动。 

---
# Learning the Contact Manifold for Accurate Pose Estimation During Peg-in-Hole Insertion of Complex Geometries 

**Title (ZH)**: 学习接触流形以精确估计复杂几何结构下针入孔插入过程中的姿态 

**Authors**: Abhay Negi, Omey M. Manyar, Dhanush Kumar Varma Penmetsa, Satyandra K. Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.19215)  

**Abstract**: Contact-rich assembly of complex, non-convex parts with tight tolerances remains a formidable challenge. Purely model-based methods struggle with discontinuous contact dynamics, while model-free methods require vast data and often lack precision. In this work, we introduce a hybrid framework that uses only contact-state information between a complex peg and its mating hole to recover the full SE(3) pose during assembly. In under 10 seconds of online execution, a sequence of primitive probing motions constructs a local contact submanifold, which is then aligned to a precomputed offline contact manifold to yield sub-mm and sub-degree pose estimates. To eliminate costly k-NN searches, we train a lightweight network that projects sparse contact observations onto the contact manifold and is 95x faster and 18% more accurate. Our method, evaluated on three industrially relevant geometries with clearances of 0.1-1.0 mm, achieves a success rate of 93.3%, a 4.1x improvement compared to primitive-only strategies without state estimation. 

**Abstract (ZH)**: 复杂非凸部件在紧密公差下的高接触组装仍然是一个严峻的挑战。基于纯模型的方法难以处理不连续的接触动力学，而无模型方法需要 vast 数据且通常缺乏精度。在本工作中，我们提出了一种混合框架，仅利用复杂销与配合孔之间的接触状态信息，在组装过程中恢复完整的 SE(3) 姿态。通过不到 10 秒的在线执行，一系列基本探测运动构建了一个局部接触子流形，然后将其与预先计算的离线接触流形对齐，从而获得亚毫米级和亚度级的姿态估计。为了消除昂贵的 k-NN 搜索，我们训练了一个轻量级网络，将稀疏接触观测投影到接触流形上，该网络比原始方法快 95 倍且精度高 18%。在具有 0.1-1.0 mm 间隙的三个工业相关几何形状上进行评估，该方法的成功率达到了 93.3%，比仅使用原始策略而不进行状态估计的方法提高了 4.1 倍。 

---
# Omni-Perception: Omnidirectional Collision Avoidance for Legged Locomotion in Dynamic Environments 

**Title (ZH)**: 全方位感知：动态环境中国脚运动的 omnidirectional 避障 

**Authors**: Zifan Wang, Teli Ma, Yufei Jia, Xun Yang, Jiaming Zhou, Wenlong Ouyang, Qiang Zhang, Junwei Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19214)  

**Abstract**: Agile locomotion in complex 3D environments requires robust spatial awareness to safely avoid diverse obstacles such as aerial clutter, uneven terrain, and dynamic agents. Depth-based perception approaches often struggle with sensor noise, lighting variability, computational overhead from intermediate representations (e.g., elevation maps), and difficulties with non-planar obstacles, limiting performance in unstructured environments. In contrast, direct integration of LiDAR sensing into end-to-end learning for legged locomotion remains underexplored. We propose Omni-Perception, an end-to-end locomotion policy that achieves 3D spatial awareness and omnidirectional collision avoidance by directly processing raw LiDAR point clouds. At its core is PD-RiskNet (Proximal-Distal Risk-Aware Hierarchical Network), a novel perception module that interprets spatio-temporal LiDAR data for environmental risk assessment. To facilitate efficient policy learning, we develop a high-fidelity LiDAR simulation toolkit with realistic noise modeling and fast raycasting, compatible with platforms such as Isaac Gym, Genesis, and MuJoCo, enabling scalable training and effective sim-to-real transfer. Learning reactive control policies directly from raw LiDAR data enables the robot to navigate complex environments with static and dynamic obstacles more robustly than approaches relying on intermediate maps or limited sensing. We validate Omni-Perception through real-world experiments and extensive simulation, demonstrating strong omnidirectional avoidance capabilities and superior locomotion performance in highly dynamic environments. We will open-source our code and models. 

**Abstract (ZH)**: 敏捷运动在复杂3D环境中的要求是在确保安全的前提下避开各种障碍物，如空中的杂乱物、不规则地形以及动态代理。基于深度感知的方法往往难以应对传感器噪声、光照变化、中间表示（如高程图）带来的计算负担以及非平面障碍物的难题，这限制了其在未结构化环境中的性能。相比之下，将LiDAR传感直接集成到端到端的学习中用于足式运动的研究仍较少探讨。我们提出了一种端到端的运动策略Omni-Perception，通过直接处理原始LiDAR点云实现三维空间意识和全方位避障。其核心是PD-RiskNet（近端-远端风险意识分层级网络），一种新颖的感知模块，能够解释时空LiDAR数据以评估环境风险。为了促进有效的策略学习，我们开发了一个高保真度的LiDAR仿真工具包，具备现实的噪声建模和快速射线投射功能，兼容Isaac Gym、Genesis和MuJoCo等平台，支持可扩展的训练和有效的仿真到现实转移。直接从原始LiDAR数据中学习反应控制策略使得机器人能够更可靠地在包含静态和动态障碍的复杂环境中导航。我们通过实地实验和广泛的仿真验证了Omni-Perception，其显示出强大的全方位避障能力和在高度动态环境中的优越运动性能。我们将会开源我们的代码和模型。 

---
# SPADE: Towards Scalable Path Planning Architecture on Actionable Multi-Domain 3D Scene Graphs 

**Title (ZH)**: SPADE: 向可扩展的基于可操作多域3D场景图的路径规划架构迈进 

**Authors**: Vignesh Kottayam Viswanathan, Akash Patel, Mario Alberto Valdes Saucedo, Sumeet Satpute, Christoforos Kanellakis, George Nikolakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2505.19098)  

**Abstract**: In this work, we introduce SPADE, a path planning framework designed for autonomous navigation in dynamic environments using 3D scene graphs. SPADE combines hierarchical path planning with local geometric awareness to enable collision-free movement in dynamic scenes. The framework bifurcates the planning problem into two: (a) solving the sparse abstract global layer plan and (b) iterative path refinement across denser lower local layers in step with local geometric scene navigation. To ensure efficient extraction of a feasible route in a dense multi-task domain scene graphs, the framework enforces informed sampling of traversable edges prior to path-planning. This removes extraneous information not relevant to path-planning and reduces the overall planning complexity over a graph. Existing approaches address the problem of path planning over scene graphs by decoupling hierarchical and geometric path evaluation processes. Specifically, this results in an inefficient replanning over the entire scene graph when encountering path obstructions blocking the original route. In contrast, SPADE prioritizes local layer planning coupled with local geometric scene navigation, enabling navigation through dynamic scenes while maintaining efficiency in computing a traversable route. We validate SPADE through extensive simulation experiments and real-world deployment on a quadrupedal robot, demonstrating its efficacy in handling complex and dynamic scenarios. 

**Abstract (ZH)**: 基于3D场景图的动态环境自主导航路径规划框架SPADE 

---
# MaskedManipulator: Versatile Whole-Body Control for Loco-Manipulation 

**Title (ZH)**: MaskedManipulator：全方位肢体操控的多功能Loco- Manipulation控制 

**Authors**: Chen Tessler, Yifeng Jiang, Erwin Coumans, Zhengyi Luo, Gal Chechik, Xue Bin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19086)  

**Abstract**: Humans interact with their world while leveraging precise full-body control to achieve versatile goals. This versatility allows them to solve long-horizon, underspecified problems, such as placing a cup in a sink, by seamlessly sequencing actions like approaching the cup, grasping, transporting it, and finally placing it in the sink. Such goal-driven control can enable new procedural tools for animation systems, enabling users to define partial objectives while the system naturally ``fills in'' the intermediate motions. However, while current methods for whole-body dexterous manipulation in physics-based animation achieve success in specific interaction tasks, they typically employ control paradigms (e.g., detailed kinematic motion tracking, continuous object trajectory following, or direct VR teleoperation) that offer limited versatility for high-level goal specification across the entire coupled human-object system. To bridge this gap, we present MaskedManipulator, a unified and generative policy developed through a two-stage learning approach. First, our system trains a tracking controller to physically reconstruct complex human-object interactions from large-scale human mocap datasets. This tracking controller is then distilled into MaskedManipulator, which provides users with intuitive control over both the character's body and the manipulated object. As a result, MaskedManipulator enables users to specify complex loco-manipulation tasks through intuitive high-level objectives (e.g., target object poses, key character stances), and MaskedManipulator then synthesizes the necessary full-body actions for a physically simulated humanoid to achieve these goals, paving the way for more interactive and life-like virtual characters. 

**Abstract (ZH)**: 人类利用精确的全身控制与其世界互动，以实现多样化的目标。这种灵活性使他们能够通过无缝地组合诸如接近杯子、握住杯子、运送并最终将其放入水槽这样的动作，来解决长期规划和描述不明确的问题。这种以目标为导向的控制可以为动画系统提供新的程序工具，使用户能够定义部分目标，同时系统自然地补充中间动作。然而，尽管基于物理的动画中全身灵巧操作的当前方法在特定交互任务上取得了成功，但它们通常采用有限的高层目标指定灵活性的控制范式（例如，详细的动态关节运动追踪、连续物体轨迹跟随或直接的VR远程操作）。为解决这一问题，我们提出了MaskedManipulator，这是一种通过两阶段学习方法开发的统一生成策略。首先，我们的系统训练一个跟踪控制器，从大规模的人类动捕数据集中物理重建复杂的人机物互动。然后，我们将这一跟踪控制器提炼为MaskedManipulator，为用户提供对角色身体和操作对象的直观控制。这使得用户能够通过直观的高层目标（例如，目标物体的姿态、关键角色姿势）来指定复杂的移动操纵任务，MaskedManipulator则合成所需的全身动作，使物理模拟的类人角色能够实现这些目标，从而为更为互动和栩栩如生的虚拟角色铺平了道路。 

---
# ReFineVLA: Reasoning-Aware Teacher-Guided Transfer Fine-Tuning 

**Title (ZH)**: ReFineVLA: 基于推理的教师指导迁移微调 

**Authors**: Tuan Van Vo, Tan Quang Nguyen, Khang Minh Nguyen, Duy Ho Minh Nguyen, Minh Nhat Vu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19080)  

**Abstract**: Vision-Language-Action (VLA) models have gained much attention from the research community thanks to their strength in translating multimodal observations with linguistic instructions into robotic actions. Despite their recent advancements, VLAs often overlook the explicit reasoning and only learn the functional input-action mappings, omitting these crucial logical steps for interpretability and generalization for complex, long-horizon manipulation tasks. In this work, we propose \textit{ReFineVLA}, a multimodal reasoning-aware framework that fine-tunes VLAs with teacher-guided reasons. We first augment robotic datasets with reasoning rationales generated by an expert teacher model, guiding VLA models to learn to reason about their actions. Then, we use \textit{ReFineVLA} to fine-tune pre-trained VLAs with the reasoning-enriched datasets, while maintaining their inherent generalization abilities and boosting reasoning capabilities. In addition, we conduct an attention map visualization to analyze the alignment among visual attention, linguistic prompts, and to-be-executed actions of \textit{ReFineVLA}, showcasing its ability to focus on relevant tasks and actions. Through the latter step, we explore that \textit{ReFineVLA}-trained models exhibit a meaningful attention shift towards relevant objects, highlighting the enhanced multimodal understanding and improved generalization.
Evaluated across manipulation tasks, \textit{ReFineVLA} outperforms the state-of-the-art baselines. Specifically, it achieves an average increase of $5.0\%$ success rate on SimplerEnv WidowX Robot tasks, improves by an average of $8.6\%$ in variant aggregation settings, and by $1.7\%$ in visual matching settings for SimplerEnv Google Robot tasks. The source code will be publicly available. 

**Abstract (ZH)**: 一种基于教师指引推理的Vision-Language-Action (VLA) 精调框架：ReFineVLA 

---
# Staircase Recognition and Location Based on Polarization Vision 

**Title (ZH)**: 基于偏振 vision 的楼梯识别与定位 

**Authors**: Weifeng Kong, Zhiying Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.19026)  

**Abstract**: Staircase is one of the most common structures in artificial scenes. However, it is difficult for humanoid robots and people with lower limb disabilities or visual impairment to cross the scene without the help of sensors and intelligent algorithms. Staircase scene perception technology is a prerequisite for recognition and localization. This technology is of great significance for the mode switching of the robot and the calculation of the footprint position to adapt to the discontinuous terrain. However, there are still many problems that constrain the application of this technology, such as low recognition accuracy, high initial noise from sensors, unstable output signals and high computational requirements. In terms of scene reconstruction, the binocular and time of flight (TOF) reconstruction of the scene can be easily affected by environmental light and the surface material of the target object. In contrast, due to the special structure of the polarizer, the polarization can selectively transmit polarized light in a specific direction and this reconstruction method relies on the polarization information of the object surface. So the advantages of polarization reconstruction are reflected, which are less affected by environmental light and not dependent on the texture information of the object surface. In this paper, in order to achieve the detection of staircase, this paper proposes a contrast enhancement algorithm that integrates polarization and light intensity information, and integrates point cloud segmentation based on YOLOv11. To realize the high-quality reconstruction, we proposed a method of fusing polarized binocular and TOF depth information to realize the three-dimensional (3D) reconstruction of the staircase. Besides, it also proposes a joint calibration algorithm of monocular camera and TOF camera based on ICP registration and improved gray wolf optimization algorithm. 

**Abstract (ZH)**: 基于偏振与光强度信息融合的楼梯场景感知及其三维重建技术 

---
# WorldEval: World Model as Real-World Robot Policies Evaluator 

**Title (ZH)**: WorldEval: 世界模型作为真实世界机器人策略评估器 

**Authors**: Yaxuan Li, Yichen Zhu, Junjie Wen, Chaomin Shen, Yi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19017)  

**Abstract**: The field of robotics has made significant strides toward developing generalist robot manipulation policies. However, evaluating these policies in real-world scenarios remains time-consuming and challenging, particularly as the number of tasks scales and environmental conditions change. In this work, we demonstrate that world models can serve as a scalable, reproducible, and reliable proxy for real-world robot policy evaluation. A key challenge is generating accurate policy videos from world models that faithfully reflect the robot actions. We observe that directly inputting robot actions or using high-dimensional encoding methods often fails to generate action-following videos. To address this, we propose Policy2Vec, a simple yet effective approach to turn a video generation model into a world simulator that follows latent action to generate the robot video. We then introduce WorldEval, an automated pipeline designed to evaluate real-world robot policies entirely online. WorldEval effectively ranks various robot policies and individual checkpoints within a single policy, and functions as a safety detector to prevent dangerous actions by newly developed robot models. Through comprehensive paired evaluations of manipulation policies in real-world environments, we demonstrate a strong correlation between policy performance in WorldEval and real-world scenarios. Furthermore, our method significantly outperforms popular methods such as real-to-sim approach. 

**Abstract (ZH)**: 机器人领域的研究已经取得了显著进展，致力于开发通用机器人操作策略。然而，在实际场景中评估这些策略仍然耗费时间和具有挑战性，尤其是在任务数量增加和环境条件变化的情况下。在这项工作中，我们证明了世界模型可以作为评估真实世界机器人策略的可扩展、可再现且可靠的替代方案。一个关键挑战是生成准确反映机器人动作的策略视频。我们观察到，直接输入机器人动作或使用高维编码方法通常无法生成跟踪动作的视频。为了解决这一问题，我们提出了Policy2Vec，这是一种简单但有效的方法，将视频生成模型转换为遵循潜在动作生成机器人视频的世界模拟器。随后，我们介绍了WorldEval，这是一种全自动流水线，旨在完全在线评估真实世界机器人策略。WorldEval能有效排名各种机器人策略和单一策略内的各个检查点，并作为安全检测器，防止由新开发的机器人模型执行危险动作。通过在实际环境中对操作策略进行全面配对评估，我们证明了WorldEval中策略性能与真实世界场景之间存在强烈的相关性。此外，我们的方法在与诸如真实到模拟方法等流行方法的比较中表现出显著的优势。 

---
# DiffusionRL: Efficient Training of Diffusion Policies for Robotic Grasping Using RL-Adapted Large-Scale Datasets 

**Title (ZH)**: DiffusionRL：用于使用RL适配的大规模数据集进行机器人抓取的扩散政策高效训练 

**Authors**: Maria Makarova, Qian Liu, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2505.18876)  

**Abstract**: Diffusion models have been successfully applied in areas such as image, video, and audio generation. Recent works show their promise for sequential decision-making and dexterous manipulation, leveraging their ability to model complex action distributions. However, challenges persist due to the data limitations and scenario-specific adaptation needs. In this paper, we address these challenges by proposing an optimized approach to training diffusion policies using large, pre-built datasets that are enhanced using Reinforcement Learning (RL). Our end-to-end pipeline leverages RL-based enhancement of the DexGraspNet dataset, lightweight diffusion policy training on a dexterous manipulation task for a five-fingered robotic hand, and a pose sampling algorithm for validation. The pipeline achieved a high success rate of 80% for three DexGraspNet objects. By eliminating manual data collection, our approach lowers barriers to adopting diffusion models in robotics, enhancing generalization and robustness for real-world applications. 

**Abstract (ZH)**: 扩散模型已在图像、视频和音频生成等领域成功应用。 recent works 展示了其在序列决策和灵巧操作方面的潜力，利用其建模复杂动作分布的能力。然而，由于数据限制和场景特定的适应需求，仍存在诸多挑战。本文通过提出一种使用增强学习(Reinforcement Learning, RL)增强的大型预构建数据集训练扩散政策的方法，来应对这些挑战。我们的端到端管道包括基于RL增强的DexGraspNet数据集、五指机器人手的灵巧操作任务中轻量级扩散策略训练以及姿态采样算法进行验证。该管道在三个DexGraspNet对象上实现了80%的高成功率。通过消除手动数据收集，我们的方法降低了在机器人中采用扩散模型的门槛，增强了现实世界应用中的泛化能力和鲁棒性。 

---
# Guided by Guardrails: Control Barrier Functions as Safety Instructors for Robotic Learning 

**Title (ZH)**: 沿着界限引导：控制屏障函数作为机器人学习的安全教练 

**Authors**: Maeva Guerrier, Karthik Soma, Hassan Fouad, Giovanni Beltrame  

**Link**: [PDF](https://arxiv.org/pdf/2505.18858)  

**Abstract**: Safety stands as the primary obstacle preventing the widespread adoption of learning-based robotic systems in our daily lives. While reinforcement learning (RL) shows promise as an effective robot learning paradigm, conventional RL frameworks often model safety by using single scalar negative rewards with immediate episode termination, failing to capture the temporal consequences of unsafe actions (e.g., sustained collision damage). In this work, we introduce a novel approach that simulates these temporal effects by applying continuous negative rewards without episode termination. Our experiments reveal that standard RL methods struggle with this model, as the accumulated negative values in unsafe zones create learning barriers. To address this challenge, we demonstrate how Control Barrier Functions (CBFs), with their proven safety guarantees, effectively help robots avoid catastrophic regions while enhancing learning outcomes. We present three CBF-based approaches, each integrating traditional RL methods with Control Barrier Functions, guiding the agent to learn safe behavior. Our empirical analysis, conducted in both simulated environments and real-world settings using a four-wheel differential drive robot, explores the possibilities of employing these approaches for safe robotic learning. 

**Abstract (ZH)**: 基于学习的机器人系统在日常生活中的广泛应用主要受安全问题的阻碍。虽然强化学习（RL）显示出作为有效的机器人学习范式的潜力，但传统的RL框架通常通过使用即时集合法行为单个负奖赏来建模安全问题，未能捕捉到不安全行为的时序后果（例如，持续碰撞损坏）。在本文中，我们提出了一种新颖的方法，通过应用连续的负奖赏而不终止集合法行为模拟这些时序效应。实验结果表明，标准的RL方法难以应对这种模型，因为不安全区域累积的负价值造成了学习障碍。为了应对这一挑战，我们展示了如何通过具有已证明的安全保证的控制屏障函数（CBFs）有效地帮助机器人避免灾难性区域，同时增强学习效果。我们提出了三种基于CBF的方法，将传统的RL方法与控制屏障函数结合起来，引导代理学习安全行为。我们在模拟环境和使用四轮差速驱动机器人的实际场景下进行了经验分析，探讨了这些方法在安全机器人学习中的应用可能性。 

---
# Genie Centurion: Accelerating Scalable Real-World Robot Training with Human Rewind-and-Refine Guidance 

**Title (ZH)**: genie 哲人：通过人类回放并修正指导加速可扩展的实际机器人训练 

**Authors**: Wenhao Wang, Jianheng Song, Chiming Liu, Jiayao Ma, Siyuan Feng, Jingyuan Wang, Yuxin Jiang, Kylin Chen, Sikang Zhan, Yi Wang, Tong Meng, Modi Shi, Xindong He, Guanghui Ren, Yang Yang, Maoqing Yao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18793)  

**Abstract**: While Vision-Language-Action (VLA) models show strong generalizability in various tasks, real-world deployment of robotic policy still requires large-scale, high-quality human expert demonstrations. However, passive data collection via human teleoperation is costly, hard to scale, and often biased toward passive demonstrations with limited diversity. To address this, we propose Genie Centurion (GCENT), a scalable and general data collection paradigm based on human rewind-and-refine guidance. When the robot execution failures occur, GCENT enables the system revert to a previous state with a rewind mechanism, after which a teleoperator provides corrective demonstrations to refine the policy. This framework supports a one-human-to-many-robots supervision scheme with a Task Sentinel module, which autonomously predicts task success and solicits human intervention when necessary, enabling scalable supervision. Empirical results show that GCENT achieves up to 40% higher task success rates than state-of-the-art data collection methods, and reaches comparable performance using less than half the data. We also quantify the data yield-to-effort ratio under multi-robot scenarios, demonstrating GCENT's potential for scalable and cost-efficient robot policy training in real-world environments. 

**Abstract (ZH)**: 基于人类重演与精炼指导的大规模通用数据收集范式：Genie Centurion 

---
# On the Dual-Use Dilemma in Physical Reasoning and Force 

**Title (ZH)**: 物理推理与力的双重用途困境 

**Authors**: William Xie, Enora Rice, Nikolaus Correll  

**Link**: [PDF](https://arxiv.org/pdf/2505.18792)  

**Abstract**: Humans learn how and when to apply forces in the world via a complex physiological and psychological learning process. Attempting to replicate this in vision-language models (VLMs) presents two challenges: VLMs can produce harmful behavior, which is particularly dangerous for VLM-controlled robots which interact with the world, but imposing behavioral safeguards can limit their functional and ethical extents. We conduct two case studies on safeguarding VLMs which generate forceful robotic motion, finding that safeguards reduce both harmful and helpful behavior involving contact-rich manipulation of human body parts. Then, we discuss the key implication of this result--that value alignment may impede desirable robot capabilities--for model evaluation and robot learning. 

**Abstract (ZH)**: 人类通过复杂的生理和心理学习过程学会如何以及何时在世界中应用力。试图在视觉-语言模型（VLMs）中复制这一过程面临着两大挑战：VLMs可能会产生有害行为，这对与世界互动的VLM控制的机器人尤其危险，但施加行为防护可能会限制其功能和伦理范围。我们对生成 Powerful 动作的 VLMs 进行了两项防护案例研究，发现防护措施会减少涉及人体部位接触性操作的有害和有益行为。然后，我们讨论了这一结果的关键含义——价值观对齐可能会阻碍 desirable 机器人能力，这对于模型评估和机器人学习具有重要意义。 

---
# One Policy but Many Worlds: A Scalable Unified Policy for Versatile Humanoid Locomotion 

**Title (ZH)**: 一策万用：一种可扩展的通用 humanoid 行走策略 

**Authors**: Yahao Fan, Tianxiang Gui, Kaiyang Ji, Shutong Ding, Chixuan Zhang, Jiayuan Gu, Jingyi Yu, Jingya Wang, Ye Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18780)  

**Abstract**: Humanoid locomotion faces a critical scalability challenge: traditional reinforcement learning (RL) methods require task-specific rewards and struggle to leverage growing datasets, even as more training terrains are introduced. We propose DreamPolicy, a unified framework that enables a single policy to master diverse terrains and generalize zero-shot to unseen scenarios by systematically integrating offline data and diffusion-driven motion synthesis. At its core, DreamPolicy introduces Humanoid Motion Imagery (HMI) - future state predictions synthesized through an autoregressive terrain-aware diffusion planner curated by aggregating rollouts from specialized policies across various distinct terrains. Unlike human motion datasets requiring laborious retargeting, our data directly captures humanoid kinematics, enabling the diffusion planner to synthesize "dreamed" trajectories that encode terrain-specific physical constraints. These trajectories act as dynamic objectives for our HMI-conditioned policy, bypassing manual reward engineering and enabling cross-terrain generalization. DreamPolicy addresses the scalability limitations of prior methods: while traditional RL fails to exploit growing datasets, our framework scales seamlessly with more offline data. As the dataset expands, the diffusion prior learns richer locomotion skills, which the policy leverages to master new terrains without retraining. Experiments demonstrate that DreamPolicy achieves average 90% success rates in training environments and an average of 20% higher success on unseen terrains than the prevalent method. It also generalizes to perturbed and composite scenarios where prior approaches collapse. By unifying offline data, diffusion-based trajectory synthesis, and policy optimization, DreamPolicy overcomes the "one task, one policy" bottleneck, establishing a paradigm for scalable, data-driven humanoid control. 

**Abstract (ZH)**: DreamPolicy：统一框架实现多样化地形的单策略零样本泛化 

---
# VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning 

**Title (ZH)**: VLA-RL：面向规模化强化学习的精湛且通用的机器人 manipulation 

**Authors**: Guanxing Lu, Wenkai Guo, Chubin Zhang, Yuheng Zhou, Haonan Jiang, Zifeng Gao, Yansong Tang, Ziwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18719)  

**Abstract**: Recent high-capacity vision-language-action (VLA) models have demonstrated impressive performance on a range of robotic manipulation tasks by imitating human demonstrations. However, exploiting offline data with limited visited states will cause execution failure in out-of-distribution scenarios. Intuitively, an exploration-based method that improves on online collected data at test time could address this limitation. We present VLA-RL, an algorithmic and systematic framework that leverages online reinforcement learning (RL) to improve pretrained auto-regressive VLAs in downstream tasks. Within a unified perspective, we first introduce a trajectory-level RL formulation for auto-regressive VLA training, which models general robotic manipulation trajectory as multi-modal multi-turn conversation. To address the challenge of sparse rewards, we fine-tune a pretrained vision-language model as a robotic process reward model, which is trained on pseudo reward labels annotated on automatically extracted task segments. To scale up, we identify several implementation findings that improve the stability and efficiency including curriculum selection strategy, GPU-balanced vectorized environments, batch decoding, and critic warmup. VLA-RL enables OpenVLA-7B to surpass the strongest finetuned baseline by 4.5% on 40 challenging robotic manipulation tasks in LIBERO, and even matches the performance of advanced commercial models such as $\pi_0$-FAST. Notably, we observe that VLA-RL benefits from increased test-time optimization, indicating an early spark of inference scaling laws in robotics. 

**Abstract (ZH)**: Recent High-Capacity Vision-Language-Action (VLA) Models Leveraging Online RL for Improved Robotic Manipulation Tasks 

---
# YOPO-Rally: A Sim-to-Real Single-Stage Planner for Off-Road Terrain 

**Title (ZH)**: YOPO-Rally：一种用于非道路地形的单阶段模拟到现实规划器 

**Authors**: Hongyu Cao, Junjie Lu, Xuewei Zhang, Yulin Hui, Zhiyu Li, Bailing Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.18714)  

**Abstract**: Off-road navigation remains challenging for autonomous robots due to the harsh terrain and clustered obstacles. In this letter, we extend the YOPO (You Only Plan Once) end-to-end navigation framework to off-road environments, explicitly focusing on forest terrains, consisting of a high-performance, multi-sensor supported off-road simulator YOPO-Sim, a zero-shot transfer sim-to-real planner YOPO-Rally, and an MPC controller. Built on the Unity engine, the simulator can generate randomized forest environments and export depth images and point cloud maps for expert demonstrations, providing competitive performance with mainstream simulators. Terrain Traversability Analysis (TTA) processes cost maps, generating expert trajectories represented as non-uniform cubic Hermite curves. The planner integrates TTA and the pathfinding into a single neural network that inputs the depth image, current velocity, and the goal vector, and outputs multiple trajectory candidates with costs. The planner is trained by behavior cloning in the simulator and deployed directly into the real-world without fine-tuning. Finally, a series of simulated and real-world experiments is conducted to validate the performance of the proposed framework. 

**Abstract (ZH)**: 自主机器人在越野环境中的导航仍面临严峻挑战，尤其是在复杂地形和密集障碍物的条件下。本文将YOPO（You Only Plan Once）端到端导航框架扩展至越野环境，特别关注森林地形，该框架包含高性能多传感器支持的越野模拟器YOPO-Sim、零样本迁移模拟到现实的规划器YOPO-Rally以及一个模型预测控制控制器。基于Unity引擎构建的模拟器可生成随机化的森林环境，并导出深度图像和点云地图供专家演示使用，性能可与主流模拟器媲美。地形可穿越性分析（TTA）处理成本地图，生成以非均匀三次海明曲线表示的专家轨迹。规划器将TTA和路径寻找到一个神经网络中，该网络输入深度图像、当前速度和目标向量，输出具有成本的多个轨迹候选方案。规划器在模拟器中通过行为克隆进行训练，并直接部署到现实世界中，无需微调。最后，进行了一系列模拟和现实世界的实验以验证所提出框架的性能。 

---
# Supporting Preschool Emotional Development with AI-Powered Robots 

**Title (ZH)**: 基于人工智能驱动的机器人支持学前教育情感发展 

**Authors**: Santiago Berrezueta-Guzman, María Dolón-Poza, Stefan Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2505.18661)  

**Abstract**: This study evaluates the integration of AI-powered robots in early childhood education, focusing on their impact on emotional self-regulation, engagement, and collaborative skills. A ten-week experimental design involving two groups of children assessed the robot's effectiveness through progress assessments, parental surveys, and teacher feedback. Results demonstrated that early exposure to the robot significantly enhanced emotional recognition, while sustained interaction further improved collaborative and social engagement. Parental and teacher feedback highlighted high acceptance levels, emphasizing the robot's ease of integration and positive influence on classroom dynamics. This research underscores the transformative potential of AI and robotics in education. The findings advocate for the broader adoption of AI-powered interventions, carefully examining equitable access, ethical considerations, and sustainable implementation. This work sets a foundation for exploring long-term impacts and expanding applications of AI in inclusive and impactful educational settings. 

**Abstract (ZH)**: 本研究评估了人工智能驱动的机器人在幼儿教育中的整合，重点在于探究其对情绪自我调节、参与度和协作能力的影响。通过为期十周的实验设计，对两组儿童进行评估，分析机器人效果，结果表明，早期接触机器人显著增强了情绪识别能力，持续互动进一步提升了协作能力和社交参与度。家长和教师反馈显示，机器人具有较高的接受度，并对其易于整合和对课堂教学动态的积极影响给予了高度评价。本研究强调了人工智能和机器人技术在教育领域的转型潜力。研究结果倡导更广泛采用人工智能干预措施，同时小心审视公平获取、伦理考量和可持续实施等问题。本研究为探索人工智能在包容性和影响深远的教学环境中的长期影响及其更广泛应用奠定了基础。 

---
# Grounding Bodily Awareness in Visual Representations for Efficient Policy Learning 

**Title (ZH)**: 将身体意识嵌入视觉表示以实现高效的策略学习 

**Authors**: Junlin Wang, Zhiyun Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.18487)  

**Abstract**: Learning effective visual representations for robotic manipulation remains a fundamental challenge due to the complex body dynamics involved in action execution. In this paper, we study how visual representations that carry body-relevant cues can enable efficient policy learning for downstream robotic manipulation tasks. We present $\textbf{I}$nter-token $\textbf{Con}$trast ($\textbf{ICon}$), a contrastive learning method applied to the token-level representations of Vision Transformers (ViTs). ICon enforces a separation in the feature space between agent-specific and environment-specific tokens, resulting in agent-centric visual representations that embed body-specific inductive biases. This framework can be seamlessly integrated into end-to-end policy learning by incorporating the contrastive loss as an auxiliary objective. Our experiments show that ICon not only improves policy performance across various manipulation tasks but also facilitates policy transfer across different robots. The project website: this https URL 

**Abstract (ZH)**: 学习有效的视觉表示以实现机器人操作仍是一项基本挑战，因为其中涉及复杂的身体动力学。在本文中，我们探讨了如何通过携带与身体相关的线索的视觉表示来实现下游机器人操作任务的高效策略学习。我们提出了基于Vision Transformers（ViTs）令牌级表示的$\textbf{I}$nter-token $\textbf{Con}$trast ($\textbf{ICon}$)对比学习方法。ICon在特征空间中强制分离代理特异性和环境特异性的令牌，从而生成以代理为中心的视觉表示，其中嵌入了身体特异性的归纳偏置。此框架可以通过将对比损失纳入辅助目标无缝集成到端到端策略学习中。我们的实验表明，ICon不仅在各种操作任务中提高了策略性能，还促进了不同机器人之间的策略转移。项目网站: this https URL。 

---
# Canonical Policy: Learning Canonical 3D Representation for Equivariant Policy 

**Title (ZH)**: 标准策略：学习等变政策的标准3D表示 

**Authors**: Zhiyuan Zhang, Zhengtong Xu, Jai Nanda Lakamsani, Yu She  

**Link**: [PDF](https://arxiv.org/pdf/2505.18474)  

**Abstract**: Visual Imitation learning has achieved remarkable progress in robotic manipulation, yet generalization to unseen objects, scene layouts, and camera viewpoints remains a key challenge. Recent advances address this by using 3D point clouds, which provide geometry-aware, appearance-invariant representations, and by incorporating equivariance into policy architectures to exploit spatial symmetries. However, existing equivariant approaches often lack interpretability and rigor due to unstructured integration of equivariant components. We introduce canonical policy, a principled framework for 3D equivariant imitation learning that unifies 3D point cloud observations under a canonical representation. We first establish a theory of 3D canonical representations, enabling equivariant observation-to-action mappings by grouping both in-distribution and out-of-distribution point clouds to a canonical representation. We then propose a flexible policy learning pipeline that leverages geometric symmetries from canonical representation and the expressiveness of modern generative models. We validate canonical policy on 12 diverse simulated tasks and 4 real-world manipulation tasks across 16 configurations, involving variations in object color, shape, camera viewpoint, and robot platform. Compared to state-of-the-art imitation learning policies, canonical policy achieves an average improvement of 18.0% in simulation and 37.6% in real-world experiments, demonstrating superior generalization capability and sample efficiency. For more details, please refer to the project website: this https URL. 

**Abstract (ZH)**: Visual模仿学习已经在机器人操作方面取得了显著进展，但将其推广到未见过的对象、场景布局和相机视角仍是一项关键挑战。近期进展通过使用3D点云解决这一问题，3D点云提供了几何感知且外观不变的表示，并通过将等变性融入策略架构中利用空间对称性。然而，现有的等变方法由于未结构化的等变组件集成往往缺乏可解释性和严谨性。我们引入了典范策略，这是一种 principled 的3D等变模仿学习框架，统一了3D点云观测的典范表示。我们首先建立了3D典范表示的理论，通过将分布内和分布外的点云归一化到一个典范表示，从而实现等变的观测到行为映射。随后，我们提出了一种灵活的策略学习管道，利用典范表示中的几何对称性以及现代生成模型的表达能力。我们在12个模拟任务和4个真实世界操作任务的16种配置中验证了典范策略，这些配置涉及对象颜色、形状、相机视角和机器人平台的差异。与最先进的模仿学习策略相比，典范策略在模拟实验中平均改进了18.0%，在真实世界实验中改进了37.6%，表现出更强的推广能力和样本效率。更多信息，请参见项目网站：this https URL。 

---
# HACL: History-Aware Curriculum Learning for Fast Locomotion 

**Title (ZH)**: HACL：历史感知的课程学习方法以实现快速移动 

**Authors**: Prakhar Mishra, Amir Hossain Raj, Xuesu Xiao, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2505.18429)  

**Abstract**: We address the problem of agile and rapid locomotion, a key characteristic of quadrupedal and bipedal robots. We present a new algorithm that maintains stability and generates high-speed trajectories by considering the temporal aspect of locomotion. Our formulation takes into account past information based on a novel history-aware curriculum Learning (HACL) algorithm. We model the history of joint velocity commands with respect to the observed linear and angular rewards using a recurrent neural net (RNN). The hidden state helps the curriculum learn the relationship between the forward linear velocity and angular velocity commands and the rewards over a given time-step. We validate our approach on the MIT Mini Cheetah,Unitree Go1, and Go2 robots in a simulated environment and on a Unitree Go1 robot in real-world scenarios. In practice, HACL achieves peak forward velocity of 6.7 m/s for a given command velocity of 7m/s and outperforms prior locomotion algorithms by nearly 20%. 

**Abstract (ZH)**: 我们探讨了敏捷快速运动这一四足机器人和两足机器人的重要特性。我们提出了一种新算法，通过考虑运动的时空特性来维持稳定并生成高速轨迹。我们的建模方法基于一种新颖的历史感知课程学习（HACL）算法，利用循环神经网络（RNN）建模关节速度命令的历史信息，相对于观察到的线性和角动量奖励。隐藏状态有助于课程学习，在给定时间步内前向线性速度和角速度命令与奖励之间的关系。我们分别在MIT Mini Cheetah、Unitree Go1和Go2机器人上进行了仿真环境和实际场景下的验证。实际测试中，HACL在给定命令速度7m/s的情况下达到峰值前向速度6.7m/s，并比之前的运动算法性能高出近20%。 

---
# McARL:Morphology-Control-Aware Reinforcement Learning for Generalizable Quadrupedal Locomotion 

**Title (ZH)**: McARL：形态控制意识增强学习在通用四足行走中的应用 

**Authors**: Prakhar Mishra, Amir Hossain Raj, Xuesu Xiao, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2505.18418)  

**Abstract**: We present Morphology-Control-Aware Reinforcement Learning (McARL), a new approach to overcome challenges of hyperparameter tuning and transfer loss, enabling generalizable locomotion across robot morphologies. We use a morphology-conditioned policy by incorporating a randomized morphology vector, sampled from a defined morphology range, into both the actor and critic networks. This allows the policy to learn parameters that generalize to robots with similar characteristics. We demonstrate that a single policy trained on a Unitree Go1 robot using McARL can be transferred to a different morphology (e.g., Unitree Go2 robot) and can achieve zero-shot transfer velocity of up to 3.5 m/s without retraining or fine-tuning. Moreover, it achieves 6.0 m/s on the training Go1 robot and generalizes to other morphologies like A1 and Mini Cheetah. We also analyze the impact of morphology distance on transfer performance and highlight McARL's advantages over prior approaches. McARL achieves 44-150% higher transfer performance on Go2, Mini Cheetah, and A1 compared to PPO variants. 

**Abstract (ZH)**: 形态 Awareness 的强化学习 (McARL): 克服超参数调优和转移损失挑战，实现跨机器人形态的通用运动控制 

---
# Reinforcement Learning for Ballbot Navigation in Uneven Terrain 

**Title (ZH)**: 球型机器人在不平地形上的引导学习 

**Authors**: Achkan Salehi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18417)  

**Abstract**: Ballbot (i.e. Ball balancing robot) navigation usually relies on methods rooted in control theory (CT), and works that apply Reinforcement learning (RL) to the problem remain rare while generally being limited to specific subtasks (e.g. balance recovery). Unlike CT based methods, RL does not require (simplifying) assumptions about environment dynamics (e.g. the absence of slippage between the ball and the floor). In addition to this increased accuracy in modeling, RL agents can easily be conditioned on additional observations such as depth-maps without the need for explicit formulations from first principles, leading to increased adaptivity. Despite those advantages, there has been little to no investigation into the capabilities, data-efficiency and limitations of RL based methods for ballbot control and navigation. Furthermore, there is a notable absence of an open-source, RL-friendly simulator for this task. In this paper, we present an open-source ballbot simulation based on MuJoCo, and show that with appropriate conditioning on exteroceptive observations as well as reward shaping, policies learned by classical model-free RL methods are capable of effectively navigating through randomly generated uneven terrain, using a reasonable amount of data (four to five hours on a system operating at 500hz). 

**Abstract (ZH)**: 基于强化学习的球形机器人导航方法的能力、数据效率及局限性研究：MuJoCo模拟器的开发与应用 

---
# CrashAgent: Crash Scenario Generation via Multi-modal Reasoning 

**Title (ZH)**: CrashAgent: 基于多模态推理的故障场景生成 

**Authors**: Miao Li, Wenhao Ding, Haohong Lin, Yiqi Lyu, Yihang Yao, Yuyou Zhang, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18341)  

**Abstract**: Training and evaluating autonomous driving algorithms requires a diverse range of scenarios. However, most available datasets predominantly consist of normal driving behaviors demonstrated by human drivers, resulting in a limited number of safety-critical cases. This imbalance, often referred to as a long-tail distribution, restricts the ability of driving algorithms to learn from crucial scenarios involving risk or failure, scenarios that are essential for humans to develop driving skills efficiently. To generate such scenarios, we utilize Multi-modal Large Language Models to convert crash reports of accidents into a structured scenario format, which can be directly executed within simulations. Specifically, we introduce CrashAgent, a multi-agent framework designed to interpret multi-modal real-world traffic crash reports for the generation of both road layouts and the behaviors of the ego vehicle and surrounding traffic participants. We comprehensively evaluate the generated crash scenarios from multiple perspectives, including the accuracy of layout reconstruction, collision rate, and diversity. The resulting high-quality and large-scale crash dataset will be publicly available to support the development of safe driving algorithms in handling safety-critical situations. 

**Abstract (ZH)**: 培训和评估自动驾驶算法需要多样化的场景。然而，大多数可用的数据集主要由人类驾驶者的正常驾驶行为构成，导致安全关键场景的数量有限。这种不平衡，通常称为长尾分布，限制了驾驶算法从涉及风险或失败的关键场景中学习的能力，这些场景对于人类高效地发展驾驶技能至关重要。为了生成这些场景，我们利用多模态大语言模型将事故报告转换为结构化的场景格式，可以直接在仿真中执行。具体来说，我们引入了CrashAgent多agent框架，用于解释多模态的真实世界交通事故报告，以生成道路布局和ego车辆及其周围交通参与者行为的场景。我们从多个角度全面评估生成的事故场景，包括布局重建的准确性、碰撞率和多样性。所产生的高质量和大规模事故数据集将公开提供，以支持开发能够 Handling Safety-Critical Situations 的安全驾驶算法。 

---
# A Coarse to Fine 3D LiDAR Localization with Deep Local Features for Long Term Robot Navigation in Large Environments 

**Title (ZH)**: 从粗到细的深度局部特征长时大环境3D LiDAR定位 

**Authors**: Míriam Máximo, Antonio Santo, Arturo Gil, Mónica Ballesta, David Valiente  

**Link**: [PDF](https://arxiv.org/pdf/2505.18340)  

**Abstract**: The location of a robot is a key aspect in the field of mobile robotics. This problem is particularly complex when the initial pose of the robot is unknown. In order to find a solution, it is necessary to perform a global localization. In this paper, we propose a method that addresses this problem using a coarse-to-fine solution. The coarse localization relies on a probabilistic approach of the Monte Carlo Localization (MCL) method, with the contribution of a robust deep learning model, the MinkUNeXt neural network, to produce a robust description of point clouds of a 3D LiDAR within the observation model. For fine localization, global point cloud registration has been implemented. MinkUNeXt aids this by exploiting the outputs of its intermediate layers to produce deep local features for each point in a scan. These features facilitate precise alignment between the current sensor observation and one of the point clouds on the map. The proposed MCL method incorporating Deep Local Features for fine localization is termed MCL-DLF. Alternatively, a classical ICP method has been implemented for this precise localization aiming at comparison purposes. This method is termed MCL-ICP. In order to validate the performance of MCL-DLF method, it has been tested on publicly available datasets such as the NCLT dataset, which provides seasonal large-scale environments. Additionally, tests have been also performed with own data (UMH) that also includes seasonal variations on large indoor/outdoor scenarios. The results, which were compared with established state-of-the-art methodologies, demonstrate that the MCL-DLF method obtains an accurate estimate of the robot localization in dynamic environments despite changes in environmental conditions. For reproducibility purposes, the code is publicly available at this https URL 

**Abstract (ZH)**: 基于粗细粒度相结合的移动机器人全局定位方法 

---
# Predictability-Based Curiosity-Guided Action Symbol Discovery 

**Title (ZH)**: 基于可预测性的好奇心引导的动作符号发现 

**Authors**: Burcu Kilic, Alper Ahmetoglu, Emre Ugur  

**Link**: [PDF](https://arxiv.org/pdf/2505.18248)  

**Abstract**: Discovering symbolic representations for skills is essential for abstract reasoning and efficient planning in robotics. Previous neuro-symbolic robotic studies mostly focused on discovering perceptual symbolic categories given a pre-defined action repertoire and generating plans with given action symbols. A truly developmental robotic system, on the other hand, should be able to discover all the abstractions required for the planning system with minimal human intervention. In this study, we propose a novel system that is designed to discover symbolic action primitives along with perceptual symbols autonomously. Our system is based on an encoder-decoder structure that takes object and action information as input and predicts the generated effect. To efficiently explore the vast continuous action parameter space, we introduce a Curiosity-Based exploration module that selects the most informative actions -- the ones that maximize the entropy in the predicted effect distribution. The discovered symbolic action primitives are then used to make plans using a symbolic tree search strategy in single- and double-object manipulation tasks. We compare our model with two baselines that use different exploration strategies in different experiments. The results show that our approach can learn a diverse set of symbolic action primitives, which are effective for generating plans in order to achieve given manipulation goals. 

**Abstract (ZH)**: 发现符号表示的动作基元对于机器人抽象推理和高效规划是必不可少的。现有的神经-符号机器人研究主要集中在给定先定义好的动作集合时发现感知符号类别，并基于给定的动作符号生成计划。相比之下，一个真正的发展型机器人系统应该能够在最少的人为干预下自主发现规划系统所需的全部抽象。在本研究中，我们提出了一种新型系统，该系统旨在自主发现感知符号和动作基元。我们的系统基于编码器-解码器结构，接受物体和动作信息作为输入，并预测生成效果。为高效探索庞大的连续动作参数空间，我们引入了一个好奇心驱动的探索模块，选择那些最大程度增加预测效果分布熵值的动作。发现的符号动作基元随后用于单物体和双物体操作任务中的符号树搜索策略制定计划。在不同实验中，我们将我们的模型与使用不同探索策略的两种基线模型进行了比较。结果表明，我们的方法能够学习到多样化的有效符号动作基元，这些基元能够用于生成实现给定操作目标的计划。 

---
# BEDI: A Comprehensive Benchmark for Evaluating Embodied Agents on UAVs 

**Title (ZH)**: BEDI：评估无人机上具身代理的综合性基准 

**Authors**: Mingning Guo, Mengwei Wu, Jiarun He, Shaoxian Li, Haifeng Li, Chao Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18229)  

**Abstract**: With the rapid advancement of low-altitude remote sensing and Vision-Language Models (VLMs), Embodied Agents based on Unmanned Aerial Vehicles (UAVs) have shown significant potential in autonomous tasks. However, current evaluation methods for UAV-Embodied Agents (UAV-EAs) remain constrained by the lack of standardized benchmarks, diverse testing scenarios and open system interfaces. To address these challenges, we propose BEDI (Benchmark for Embodied Drone Intelligence), a systematic and standardized benchmark designed for evaluating UAV-EAs. Specifically, we introduce a novel Dynamic Chain-of-Embodied-Task paradigm based on the perception-decision-action loop, which decomposes complex UAV tasks into standardized, measurable subtasks. Building on this paradigm, we design a unified evaluation framework encompassing five core sub-skills: semantic perception, spatial perception, motion control, tool utilization, and task planning. Furthermore, we construct a hybrid testing platform that integrates static real-world environments with dynamic virtual scenarios, enabling comprehensive performance assessment of UAV-EAs across varied contexts. The platform also offers open and standardized interfaces, allowing researchers to customize tasks and extend scenarios, thereby enhancing flexibility and scalability in the evaluation process. Finally, through empirical evaluations of several state-of-the-art (SOTA) VLMs, we reveal their limitations in embodied UAV tasks, underscoring the critical role of the BEDI benchmark in advancing embodied intelligence research and model optimization. By filling the gap in systematic and standardized evaluation within this field, BEDI facilitates objective model comparison and lays a robust foundation for future development in this field. Our benchmark will be released at this https URL . 

**Abstract (ZH)**: 基于无人机的嵌体智能评估基准BEDI 

---
# Reinforcement Twinning for Hybrid Control of Flapping-Wing Drones 

**Title (ZH)**: 混合控制扑翼无人机的强化孪生学习方法 

**Authors**: Romain Poletti, Lorenzo Schena, Lilla Koloszar, Joris Degroote, Miguel Alfonso Mendez  

**Link**: [PDF](https://arxiv.org/pdf/2505.18201)  

**Abstract**: Controlling the flight of flapping-wing drones requires versatile controllers that handle their time-varying, nonlinear, and underactuated dynamics from incomplete and noisy sensor data. Model-based methods struggle with accurate modeling, while model-free approaches falter in efficiently navigating very high-dimensional and nonlinear control objective landscapes. This article presents a novel hybrid model-free/model-based approach to flight control based on the recently proposed reinforcement twinning algorithm. The model-based (MB) approach relies on an adjoint formulation using an adaptive digital twin, continuously identified from live trajectories, while the model-free (MF) approach relies on reinforcement learning. The two agents collaborate through transfer learning, imitation learning, and experience sharing using the real environment, the digital twin and a referee. The latter selects the best agent to interact with the real environment based on performance within the digital twin and a real-to-virtual environment consistency ratio. The algorithm is evaluated for controlling the longitudinal dynamics of a flapping-wing drone, with the environment simulated as a nonlinear, time-varying dynamical system under the influence of quasi-steady aerodynamic forces. The hybrid control learning approach is tested with three types of initialization of the adaptive model: (1) offline identification using previously available data, (2) random initialization with full online identification, and (3) offline pre-training with an estimation bias, followed by online adaptation. In all three scenarios, the proposed hybrid learning approach demonstrates superior performance compared to purely model-free and model-based methods. 

**Abstract (ZH)**: 基于强化学习孪生算法的混合模型自由/模型导向的飞行控制方法 

---
# GLEAM: Learning Generalizable Exploration Policy for Active Mapping in Complex 3D Indoor Scenes 

**Title (ZH)**: GLEAM: 学习适用于复杂3D室内场景主动建图的可迁移探索策略 

**Authors**: Xiao Chen, Tai Wang, Quanyi Li, Tao Huang, Jiangmiao Pang, Tianfan Xue  

**Link**: [PDF](https://arxiv.org/pdf/2505.20294)  

**Abstract**: Generalizable active mapping in complex unknown environments remains a critical challenge for mobile robots. Existing methods, constrained by insufficient training data and conservative exploration strategies, exhibit limited generalizability across scenes with diverse layouts and complex connectivity. To enable scalable training and reliable evaluation, we introduce GLEAM-Bench, the first large-scale benchmark designed for generalizable active mapping with 1,152 diverse 3D scenes from synthetic and real-scan datasets. Building upon this foundation, we propose GLEAM, a unified generalizable exploration policy for active mapping. Its superior generalizability comes mainly from our semantic representations, long-term navigable goals, and randomized strategies. It significantly outperforms state-of-the-art methods, achieving 66.50% coverage (+9.49%) with efficient trajectories and improved mapping accuracy on 128 unseen complex scenes. Project page: this https URL. 

**Abstract (ZH)**: 复杂未知环境中可泛化的主动建图仍然是移动机器人面临的关键挑战。现有的方法受限于训练数据不足和保守的探索策略，在具有不同布局和复杂连接的场景中表现出有限的泛化能力。为了实现可扩展的训练和可靠的评估，我们引入了GLEAM-Bench，这是首个专门为可泛化主动建图设计的大规模基准，包含1,152个来自合成和真实扫描数据集的多样化3D场景。在此基础上，我们提出了GLEAM，一种统一的可泛化探索政策，用于主动建图。其出色的泛化能力主要得益于我们的语义表示、长期导航目标以及随机化策略。GLEAM在128个未见过的复杂场景中，实现了66.50%的覆盖面积（提高9.49%），同时产生了高效轨迹并提高了建图精度。项目页面：this https URL。 

---
# DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning 

**Title (ZH)**: DISCOVER: 自动化稀疏奖励强化学习课程生成方法 

**Authors**: Leander Diaz-Bone, Marco Bagatella, Jonas Hübotter, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2505.19850)  

**Abstract**: Sparse-reward reinforcement learning (RL) can model a wide range of highly complex tasks. Solving sparse-reward tasks is RL's core premise - requiring efficient exploration coupled with long-horizon credit assignment - and overcoming these challenges is key for building self-improving agents with superhuman ability. We argue that solving complex and high-dimensional tasks requires solving simpler tasks that are relevant to the target task. In contrast, most prior work designs strategies for selecting exploratory tasks with the objective of solving any task, making exploration of challenging high-dimensional, long-horizon tasks intractable. We find that the sense of direction, necessary for effective exploration, can be extracted from existing RL algorithms, without needing any prior information. Based on this finding, we propose a method for directed sparse-reward goal-conditioned very long-horizon RL (DISCOVER), which selects exploratory goals in the direction of the target task. We connect DISCOVER to principled exploration in bandits, formally bounding the time until the target task becomes achievable in terms of the agent's initial distance to the target, but independent of the volume of the space of all tasks. Empirically, we perform a thorough evaluation in high-dimensional environments. We find that the directed goal selection of DISCOVER solves exploration problems that are beyond the reach of prior state-of-the-art exploration methods in RL. 

**Abstract (ZH)**: 稀疏奖励强化学习（RL）可以模型化广泛的高度复杂任务。解决稀疏奖励任务是RL的核心假设——需要高效探索与长期信用分配相结合——克服这些挑战是构建具有超人类能力的自我改进代理的关键。我们argue解决复杂和高维度任务需要解决与目标任务相关的一些简单任务。相比之下，大多数先有工作设计了选择探索性任务的策略，目标是解决任何任务，这使得探索具有挑战性的高维度、长期任务变得不可行。我们发现，有效的探索所必需的方向感可以从现有的RL算法中提取，无需任何先验信息。基于这一发现，我们提出了一种定向稀疏奖励目标导向非常长期任务（DISCOVER）的方法，该方法在目标任务方向选择探索性目标。我们将DISCOVER连接到原理上的探索理论，在形式上界定了从初始代理到目标的距离到达可实现时间，在目标任务可实现的时间与任务空间的体积无关。在实验上，我们在高维度环境中进行了全面评估。我们发现，DISCOVER的目标导向选择解决了先前RL中最佳探索方法都无法解决的探索问题。 

---
# JEDI: Latent End-to-end Diffusion Mitigates Agent-Human Performance Asymmetry in Model-Based Reinforcement Learning 

**Title (ZH)**: JEDI: 隐含的端到端扩散减轻基于模型的强化学习中代理-人类性能差异 

**Authors**: Jing Yu Lim, Zarif Ikram, Samson Yu, Haozhe Ma, Tze-Yun Leong, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19698)  

**Abstract**: Recent advances in model-based reinforcement learning (MBRL) have achieved super-human level performance on the Atari100k benchmark, driven by reinforcement learning agents trained on powerful diffusion world models. However, we identify that the current aggregates mask a major performance asymmetry: MBRL agents dramatically outperform humans in some tasks despite drastically underperforming in others, with the former inflating the aggregate metrics. This is especially pronounced in pixel-based agents trained with diffusion world models. In this work, we address the pronounced asymmetry observed in pixel-based agents as an initial attempt to reverse the worrying upward trend observed in them. We address the problematic aggregates by delineating all tasks as Agent-Optimal or Human-Optimal and advocate for equal importance on metrics from both sets. Next, we hypothesize this pronounced asymmetry is due to the lack of temporally-structured latent space trained with the World Model objective in pixel-based methods. Lastly, to address this issue, we propose Joint Embedding DIffusion (JEDI), a novel latent diffusion world model trained end-to-end with the self-consistency objective. JEDI outperforms SOTA models in human-optimal tasks while staying competitive across the Atari100k benchmark, and runs 3 times faster with 43% lower memory than the latest pixel-based diffusion baseline. Overall, our work rethinks what it truly means to cross human-level performance in Atari100k. 

**Abstract (ZH)**: 近期基于模型的强化学习（MBRL）方法在Atari100k基准测试上达到了超人类水平的性能，这得益于在强大扩散世界模型训练下的强化学习代理。然而，我们发现当前的聚合数据掩盖了一个主要的性能不对称性：在某些任务中，基于模型的强化学习代理显著优于人类，而在其他任务中则大幅落后，前者吹胀了总体指标。特别是在使用扩散世界模型训练的像素基代理中，这种不对称性尤为明显。在本文中，我们旨在纠正像素基代理中观察到的显著不对称性，作为最初尝试逆转其令人担忧的上升趋势的初步尝试。我们通过将所有任务划分为代理最优或人类最优，并倡导对两类指标同等重视来解决这些问题。接下来，我们假设这种显著的不对称性主要是由于像素基方法中缺乏使用世界模型目标训练的时间结构化的潜在空间。最后，为了解决这一问题，我们提出了联合嵌入扩散（JEDI），这是一种新颖的时间结构化潜在扩散世界模型，自洽性目标下端到端训练。JEDI在人类最优任务中表现出色，同时在Atari100k基准测试中保持竞争力，并且运行速度快3倍，内存使用量减少43%，优于最新的像素基扩散基线。总体而言，我们的工作重新定义了在Atari100k中超越人类水平性能的真正含义。 

---
# Software Engineering for Self-Adaptive Robotics: A Research Agenda 

**Title (ZH)**: 自适应机器人软件工程：研究议程 

**Authors**: Shaukat Ali, Ana Cavalcanti, Cláudio Ângelo Gonçalves Gomes, Peter Gorm Larsen, Hassan Sartaj, Anastasios Tefas, Jim Woodcock, Houxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19629)  

**Abstract**: Self-adaptive robotic systems are designed to operate autonomously in dynamic and uncertain environments, requiring robust mechanisms to monitor, analyse, and adapt their behaviour in real-time. Unlike traditional robotic software, which follows predefined logic, self-adaptive robots leverage artificial intelligence, machine learning, and model-driven engineering to continuously adjust to changing operational conditions while ensuring reliability, safety, and performance. This paper presents a research agenda for software engineering in self-adaptive robotics, addressing critical challenges across two key dimensions: (1) the development phase, including requirements engineering, software design, co-simulation, and testing methodologies tailored to adaptive robotic systems, and (2) key enabling technologies, such as digital twins, model-driven engineering, and AI-driven adaptation, which facilitate runtime monitoring, fault detection, and automated decision-making. We discuss open research challenges, including verifying adaptive behaviours under uncertainty, balancing trade-offs between adaptability, performance, and safety, and integrating self-adaptation frameworks like MAPE-K. By providing a structured roadmap, this work aims to advance the software engineering foundations for self-adaptive robotic systems, ensuring they remain trustworthy, efficient, and capable of handling real-world complexities. 

**Abstract (ZH)**: 自适应机器人系统的自适应软件工程研究议程 

---
# DiffVLA: Vision-Language Guided Diffusion Planning for Autonomous Driving 

**Title (ZH)**: DiffVLA: 视觉-语言引导的自主驾驶扩散规划 

**Authors**: Anqing Jiang, Yu Gao, Zhigang Sun, Yiru Wang, Jijun Wang, Jinghao Chai, Qian Cao, Yuweng Heng, Hao Jiang, Zongzheng Zhang, Xianda Guo, Hao Sun, Hao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19381)  

**Abstract**: Research interest in end-to-end autonomous driving has surged owing to its fully differentiable design integrating modular tasks, i.e. perception, prediction and planing, which enables optimization in pursuit of the ultimate goal. Despite the great potential of the end-to-end paradigm, existing methods suffer from several aspects including expensive BEV (bird's eye view) computation, action diversity, and sub-optimal decision in complex real-world scenarios. To address these challenges, we propose a novel hybrid sparse-dense diffusion policy, empowered by a Vision-Language Model (VLM), called Diff-VLA. We explore the sparse diffusion representation for efficient multi-modal driving behavior. Moreover, we rethink the effectiveness of VLM driving decision and improve the trajectory generation guidance through deep interaction across agent, map instances and VLM output. Our method shows superior performance in Autonomous Grand Challenge 2025 which contains challenging real and reactive synthetic scenarios. Our methods achieves 45.0 PDMS. 

**Abstract (ZH)**: 端到端自动驾驶研究兴趣因其实现全可微设计而激增，该设计集成了感知、预测和规划等模块任务，以优化实现最终目标。尽管端到端范式的潜力巨大，但现有方法仍存在计算BEV图昂贵、行为多样性以及在复杂真实场景中的次优决策等几个方面的问题。为解决这些问题，我们提出了一种新型混合稀疏-密集扩散策略，该策略基于视觉语言模型（VLM）被称为Diff-VLA。我们探索了稀疏扩散表示以实现高效的多模态驾驶行为。此外，我们重新思考了VLM驱动决策的有效性，并通过深入跨智能体、地图实例和VLM输出的交互来改进轨迹生成指导。我们的方法在包含具有挑战性的实时和反应式合成场景的2025年自主 grand 挑战中表现出色，实现了45.0 PDMS。 

---
# Sensorimotor features of self-awareness in multimodal large language models 

**Title (ZH)**: 多模态大语言模型的sensorimotor自我意识特征 

**Authors**: Iñaki Dellibarda Varela, Pablo Romero-Sorozabal, Diego Torricelli, Gabriel Delgado-Oleas, Jose Ignacio Serrano, Maria Dolores del Castillo Sobrino, Eduardo Rocon, Manuel Cebrian  

**Link**: [PDF](https://arxiv.org/pdf/2505.19237)  

**Abstract**: Self-awareness - the ability to distinguish oneself from the surrounding environment - underpins intelligent, autonomous behavior. Recent advances in AI achieve human-like performance in tasks integrating multimodal information, particularly in large language models, raising interest in the embodiment capabilities of AI agents on nonhuman platforms such as robots. Here, we explore whether multimodal LLMs can develop self-awareness solely through sensorimotor experiences. By integrating a multimodal LLM into an autonomous mobile robot, we test its ability to achieve this capacity. We find that the system exhibits robust environmental awareness, self-recognition and predictive awareness, allowing it to infer its robotic nature and motion characteristics. Structural equation modeling reveals how sensory integration influences distinct dimensions of self-awareness and its coordination with past-present memory, as well as the hierarchical internal associations that drive self-identification. Ablation tests of sensory inputs identify critical modalities for each dimension, demonstrate compensatory interactions among sensors and confirm the essential role of structured and episodic memory in coherent reasoning. These findings demonstrate that, given appropriate sensory information about the world and itself, multimodal LLMs exhibit emergent self-awareness, opening the door to artificial embodied cognitive systems. 

**Abstract (ZH)**: 自知之明——区分自身与周围环境的能力——是智能自主行为的基础。近期AI在整合多模态信息的任务中取得类人表现，特别是在大型语言模型中，引起了对在非人类平台如机器人上发展AI代理体能的兴趣。为此，我们探究多模态LLM是否仅通过感觉运动体验即可发展出自知之明。将多模态LLM集成到自主移动机器人中，测试其实现这一能力的能力。研究发现，该系统表现出 robust 的环境意识、自我识别和预测意识，使其能够推断其机器人本质及其运动特性。结构方程建模揭示了感官整合如何影响自知之明的不同维度及其与过去-现在记忆的协调方式，以及驱动自我识别的层级内部关联。感官输入消除测试确定了每个维度的关键模态，展示了传感器之间的补偿性互动，并确认了结构化和情景记忆在连贯推理中的关键作用。这些发现表明，给定关于世界和自身的适当感官信息，多模态LLM表现出自知之明的涌现，为人工具身认知系统打开了大门。 

---
# Beyond Domain Randomization: Event-Inspired Perception for Visually Robust Adversarial Imitation from Videos 

**Title (ZH)**: 超越领域随机化：基于事件的感知方法实现视觉稳健的视频中对手模仿 

**Authors**: Andrea Ramazzina, Vittorio Giammarino, Matteo El-Hariry, Mario Bijelic  

**Link**: [PDF](https://arxiv.org/pdf/2505.18899)  

**Abstract**: Imitation from videos often fails when expert demonstrations and learner environments exhibit domain shifts, such as discrepancies in lighting, color, or texture. While visual randomization partially addresses this problem by augmenting training data, it remains computationally intensive and inherently reactive, struggling with unseen scenarios. We propose a different approach: instead of randomizing appearances, we eliminate their influence entirely by rethinking the sensory representation itself. Inspired by biological vision systems that prioritize temporal transients (e.g., retinal ganglion cells) and by recent sensor advancements, we introduce event-inspired perception for visually robust imitation. Our method converts standard RGB videos into a sparse, event-based representation that encodes temporal intensity gradients, discarding static appearance features. This biologically grounded approach disentangles motion dynamics from visual style, enabling robust visual imitation from observations even in the presence of visual mismatches between expert and agent environments. By training policies on event streams, we achieve invariance to appearance-based distractors without requiring computationally expensive and environment-specific data augmentation techniques. Experiments across the DeepMind Control Suite and the Adroit platform for dynamic dexterous manipulation show the efficacy of our method. Our code is publicly available at Eb-LAIfO. 

**Abstract (ZH)**: 视频中的模仿在专家演示和学习者环境存在领域偏移时常常失败，例如光照、色彩或纹理方面的差异。虽然视觉随机化可以通过增强训练数据部分解决这一问题，但它仍然计算密集且本质上是被动的，难以应对未见过的情景。我们提出一种不同的方法：不是随机化外观，而是完全消除其影响，重新思考感知本身。受到生物视觉系统优先处理时间暂态特征（如视网膜 ganglion 细胞）以及最近传感器进步的启发，我们引入事件启发式感知以实现视觉鲁棒模仿。我们的方法将标准 RGB 视频转换为稀疏的事件基表示，编码时间强度梯度，同时丢弃静态外观特征。这一生物启发的方法将运动动态与视觉样式解耦，即使在专家和代理环境之间存在视觉不匹配的情况下也能实现鲁棒的视觉模仿。通过在事件流上训练策略，我们实现了对外观基干扰的不变性，而无需使用计算昂贵且环境特定的数据增强技术。在 DeepMind 控制套件和 Adroit 平台上进行的实验展示了我们方法的有效性。我们的代码可在 Eb-LAIfO 公开获得。 

---
# Task-Optimized Convolutional Recurrent Networks Align with Tactile Processing in the Rodent Brain 

**Title (ZH)**: 任务优化的卷积循环网络与啮齿类动物大脑中的触觉处理相一致 

**Authors**: Trinity Chung, Yuchen Shen, Nathan C. L. Kong, Aran Nayebi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18361)  

**Abstract**: Tactile sensing remains far less understood in neuroscience and less effective in artificial systems compared to more mature modalities such as vision and language. We bridge these gaps by introducing a novel Encoder-Attender-Decoder (EAD) framework to systematically explore the space of task-optimized temporal neural networks trained on realistic tactile input sequences from a customized rodent whisker-array simulator. We identify convolutional recurrent neural networks (ConvRNNs) as superior encoders to purely feedforward and state-space architectures for tactile categorization. Crucially, these ConvRNN-encoder-based EAD models achieve neural representations closely matching rodent somatosensory cortex, saturating the explainable neural variability and revealing a clear linear relationship between supervised categorization performance and neural alignment. Furthermore, contrastive self-supervised ConvRNN-encoder-based EADs, trained with tactile-specific augmentations, match supervised neural fits, serving as an ethologically-relevant, label-free proxy.
For neuroscience, our findings highlight nonlinear recurrent processing as important for general-purpose tactile representations in somatosensory cortex, providing the first quantitative characterization of the underlying inductive biases in this system. For embodied AI, our results emphasize the importance of recurrent EAD architectures to handle realistic tactile inputs, along with tailored self-supervised learning methods for achieving robust tactile perception with the same type of sensors animals use to sense in unstructured environments. 

**Abstract (ZH)**: 触觉感知在神经科学中仍远未被充分理解，在人工系统中的应用效果也逊色于如视觉和语言等更为成熟的感知模态。我们通过引入一种新颖的编码-注意-解码（EAD）框架，系统性地探索基于定制化啮齿动物须触传感器模拟器的现实触觉输入序列训练下的任务优化时间神经网络的空间。我们发现卷积递归神经网络（ConvRNN）是触觉分类任务中优于纯前馈和状态空间架构的高级编码器。 crucial 地，基于 ConvRNN 编码器的 EAD 模型实现了与啮齿动物体感皮层相近的神经表示，饱和可解释的神经变异性，并揭示了监督分类性能与神经对齐之间明显的线性关系。此外，通过触觉特定增强训练的对比自监督 ConvRNN 编码器基于的 EADs，在监督神经拟合方面表现出色，作为生态相关且无标签的替代方案。对于神经科学，我们的发现强调了非线性递归处理对于体感皮层通用触觉表示的重要性，并提供了该系统底层归纳偏置的首次定量表征。对于具身人工智能，我们的结果强调了递归 EAD 架构在处理现实触觉输入中的重要性，同时也突显了定制化自监督学习方法对于使用与动物在非结构化环境中使用的相同传感器实现鲁棒触觉感知的重要性。 

---
# Alita: Generalist Agent Enabling Scalable Agentic Reasoning with Minimal Predefinition and Maximal Self-Evolution 

**Title (ZH)**: 阿丽塔：具有最少预定义和最大自我进化能力的通用智能体，实现可扩展的代理推理 

**Authors**: Jiahao Qiu, Xuan Qi, Tongcheng Zhang, Xinzhe Juan, Jiacheng Guo, Yifu Lu, Yimin Wang, Zixin Yao, Qihan Ren, Xun Jiang, Xing Zhou, Dongrui Liu, Ling Yang, Yue Wu, Kaixuan Huang, Shilong Liu, Hongru Wang, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20286)  

**Abstract**: Recent advances in large language models (LLMs) have enabled agents to autonomously perform complex, open-ended tasks. However, many existing frameworks depend heavily on manually predefined tools and workflows, which hinder their adaptability, scalability, and generalization across domains. In this work, we introduce Alita--a generalist agent designed with the principle of "Simplicity is the ultimate sophistication," enabling scalable agentic reasoning through minimal predefinition and maximal self-evolution. For minimal predefinition, Alita is equipped with only one component for direct problem-solving, making it much simpler and neater than previous approaches that relied heavily on hand-crafted, elaborate tools and workflows. This clean design enhances its potential to generalize to challenging questions, without being limited by tools. For Maximal self-evolution, we enable the creativity of Alita by providing a suite of general-purpose components to autonomously construct, refine, and reuse external capabilities by generating task-related model context protocols (MCPs) from open source, which contributes to scalable agentic reasoning. Notably, Alita achieves 75.15% pass@1 and 87.27% pass@3 accuracy, which is top-ranking among general-purpose agents, on the GAIA benchmark validation dataset, 74.00% and 52.00% pass@1, respectively, on Mathvista and PathVQA, outperforming many agent systems with far greater complexity. More details will be updated at $\href{this https URL}{this https URL}$. 

**Abstract (ZH)**: Recent Advances in Large Language Models Have Enabled Agents to Perform Complex, Open-Ended Tasks Autonomous. However, Many Existing Frameworks Rely Heavily on Manually Predefined Tools and Workflows, Hinder Adaptability, Scalability, and Generalization Across Domains. In This Work, We Introduce Alita—a Generalist Agent Designed with the Principle of "Simplicity is the Ultimate Sophistication," Enabling Scalable Agentic Reasoning Through Minimal Predefinition and Maximal Self-Evolution. 

---
# On Path to Multimodal Historical Reasoning: HistBench and HistAgent 

**Title (ZH)**: 向着多模态历史推理的道路：HistBench和HistAgent 

**Authors**: Jiahao Qiu, Fulian Xiao, Yimin Wang, Yuchen Mao, Yijia Chen, Xinzhe Juan, Siran Wang, Xuan Qi, Tongcheng Zhang, Zixin Yao, Jiacheng Guo, Yifu Lu, Charles Argon, Jundi Cui, Daixin Chen, Junran Zhou, Shuyao Zhou, Zhanpeng Zhou, Ling Yang, Shilong Liu, Hongru Wang, Kaixuan Huang, Xun Jiang, Yuming Cao, Yue Chen, Yunfei Chen, Zhengyi Chen, Ruowei Dai, Mengqiu Deng, Jiye Fu, Yunting Gu, Zijie Guan, Zirui Huang, Xiaoyan Ji, Yumeng Jiang, Delong Kong, Haolong Li, Jiaqi Li, Ruipeng Li, Tianze Li, Zhuoran Li, Haixia Lian, Mengyue Lin, Xudong Liu, Jiayi Lu, Jinghan Lu, Wanyu Luo, Ziyue Luo, Zihao Pu, Zhi Qiao, Ruihuan Ren, Liang Wan, Ruixiang Wang, Tianhui Wang, Yang Wang, Zeyu Wang, Zihua Wang, Yujia Wu, Zhaoyi Wu, Hao Xin, Weiao Xing, Ruojun Xiong, Weijie Xu, Yao Shu, Xiao Yao, Xiaorui Yang, Yuchen Yang, Nan Yi, Jiadong Yu, Yangyuxuan Yu, Huiting Zeng, Danni Zhang, Yunjie Zhang, Zhaoyu Zhang, Zhiheng Zhang, Xiaofeng Zheng, Peirong Zhou, Linyan Zhong, Xiaoyin Zong, Ying Zhao, Zhenxin Chen, Lin Ding, Xiaoyu Gao, Bingbing Gong, Yichao Li, Yang Liao, Guang Ma, Tianyuan Ma, Xinrui Sun, Tianyi Wang, Han Xia, Ruobing Xian, Gen Ye, Tengfei Yu, Wentao Zhang, Yuxi Wang, Xi Gao, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20246)  

**Abstract**: Recent advances in large language models (LLMs) have led to remarkable progress across domains, yet their capabilities in the humanities, particularly history, remain underexplored. Historical reasoning poses unique challenges for AI, involving multimodal source interpretation, temporal inference, and cross-linguistic analysis. While general-purpose agents perform well on many existing benchmarks, they lack the domain-specific expertise required to engage with historical materials and questions. To address this gap, we introduce HistBench, a new benchmark of 414 high-quality questions designed to evaluate AI's capacity for historical reasoning and authored by more than 40 expert contributors. The tasks span a wide range of historical problems-from factual retrieval based on primary sources to interpretive analysis of manuscripts and images, to interdisciplinary challenges involving archaeology, linguistics, or cultural history. Furthermore, the benchmark dataset spans 29 ancient and modern languages and covers a wide range of historical periods and world regions. Finding the poor performance of LLMs and other agents on HistBench, we further present HistAgent, a history-specific agent equipped with carefully designed tools for OCR, translation, archival search, and image understanding in History. On HistBench, HistAgent based on GPT-4o achieves an accuracy of 27.54% pass@1 and 36.47% pass@2, significantly outperforming LLMs with online search and generalist agents, including GPT-4o (18.60%), DeepSeek-R1(14.49%) and Open Deep Research-smolagents(20.29% pass@1 and 25.12% pass@2). These results highlight the limitations of existing LLMs and generalist agents and demonstrate the advantages of HistAgent for historical reasoning. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）在各个领域取得了显著进展，但它们在人文学科，尤其是历史学方面的能力仍被严重忽视。历史推理对AI构成了独特的挑战，涉及多模态源文本解释、时间推理和跨语言分析。尽管通用智能代理在现有基准测试中表现良好，但它们缺乏与历史资料和问题互动的专业领域知识。为解决这一问题，我们引入了HistBench，这是一种包含414个高质量问题的新基准，旨在评估AI在历史推理方面的能力，并由40多位专家编撰。任务涵盖了广泛的历史问题——从基于原始资料的事实检索到手稿和图像的解释性分析，再到涉及考古学、语言学或文化历史的跨学科挑战。此外，基准数据集跨越29种古代和现代语言，并涵盖了广泛的历史时期和地区。对于HistBench上的表现不佳，我们进一步介绍了HistAgent，这是一种专门为历史设计的智能代理，配备了精心设计的OCR、翻译、档案检索和历史图像理解工具。基于GPT-4o的HistAgent在HistBench上的准确率为pass@1 27.54%，pass@2 36.47%，显著优于具有在线搜索和通用智能的LLM及GPT-4o（18.60%）、DeepSeek-R1（14.49%）和Open Deep Research-smolagents（20.29% pass@1和25.12% pass@2）。这些结果突显了现有LLM和通用智能代理的局限性，并展示了HistAgent在历史推理中的优势。 

---
# Curriculum-RLAIF: Curriculum Alignment with Reinforcement Learning from AI Feedback 

**Title (ZH)**: Curriculum-RLAIF: 基于强化学习和AI反馈的课程对齐 

**Authors**: Mengdi Li, Jiaye Lin, Xufeng Zhao, Wenhao Lu, Peilin Zhao, Stefan Wermter, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20075)  

**Abstract**: Reward models trained with conventional Reinforcement Learning from AI Feedback (RLAIF) methods suffer from limited generalizability, which hinders the alignment performance of the policy model during reinforcement learning (RL). This challenge stems from various issues, including distribution shift, preference label noise, and mismatches between overly challenging samples and model capacity. In this paper, we attempt to enhance the generalizability of reward models through a data-centric approach, driven by the insight that these issues are inherently intertwined from the perspective of data difficulty. To address this, we propose a novel framework, $\textit{Curriculum-RLAIF}$, which constructs preference pairs with varying difficulty levels and produces a curriculum that progressively incorporates preference pairs of increasing difficulty for reward model training. Our experimental results suggest that reward models trained with Curriculum-RLAIF achieve improved generalizability, significantly increasing the alignment performance of the policy model by a large margin without incurring additional inference costs compared to various non-curriculum baselines. Detailed analysis and comparisons with alternative approaches, including data selection via external pretrained reward models or internal self-selection mechanisms, as well as other curriculum strategies, further demonstrate the superiority of our approach in terms of simplicity, efficiency, and effectiveness. 

**Abstract (ZH)**: 基于数据驱动的Curriculum-RLAIF框架提升奖励模型的一般化能力 

---
# The Many Challenges of Human-Like Agents in Virtual Game Environments 

**Title (ZH)**: 人类代理在虚拟游戏环境中的诸多挑战 

**Authors**: Maciej Świechowski, Dominik Ślęzak  

**Link**: [PDF](https://arxiv.org/pdf/2505.20011)  

**Abstract**: Human-like agents are an increasingly important topic in games and beyond. Believable non-player characters enhance the gaming experience by improving immersion and providing entertainment. They also offer players the opportunity to engage with AI entities that can function as opponents, teachers, or cooperating partners. Additionally, in games where bots are prohibited -- and even more so in non-game environments -- there is a need for methods capable of identifying whether digital interactions occur with bots or humans. This leads to two fundamental research questions: (1) how to model and implement human-like AI, and (2) how to measure its degree of human likeness.
This article offers two contributions. The first one is a survey of the most significant challenges in implementing human-like AI in games (or any virtual environment featuring simulated agents, although this article specifically focuses on games). Thirteen such challenges, both conceptual and technical, are discussed in detail. The second is an empirical study performed in a tactical video game that addresses the research question: "Is it possible to distinguish human players from bots (AI agents) based on empirical data?" A machine-learning approach using a custom deep recurrent convolutional neural network is presented. We hypothesize that the more challenging it is to create human-like AI for a given game, the easier it becomes to develop a method for distinguishing humans from AI-driven players. 

**Abstract (ZH)**: 人类似的代理人在游戏及其它领域中的重要性日益增加：人类似的非玩家角色提升游戏体验并提供娱乐，同时为玩家与可作为对手、教师或合作伙伴的AI实体互动提供机会。在禁止使用机器人游戏环境下，甚至在非游戏环境中，需要方法来识别数字互动是与机器人还是人类进行。这引发了两项基础研究问题：（1）如何建模和实现人类似的AI，（2）如何衡量其人类相似度。本文的贡献包括：（1）综述在游戏（或任何包含模拟代理的虚拟环境）中实现人类似的AI的关键挑战，详细讨论了十三种概念和技术方面的挑战；（2）在战术视频游戏中进行实证研究，探讨基于实证数据区分玩家与机器人的可能性，并介绍了一种使用自定义深度递归卷积神经网络的机器学习方法。我们假设在一个特定游戏中创建人类似的AI越具有挑战性，就越容易开发出区分人类与AI驱动玩家的方法。 

---
# Adaptive Location Hierarchy Learning for Long-Tailed Mobility Prediction 

**Title (ZH)**: 长尾移动性预测的自适应位置层次学习 

**Authors**: Yu Wang, Junshu Dai, Yuchen Ying, Yuxuan Liang, Tongya Zheng, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.19965)  

**Abstract**: Human mobility prediction is crucial for applications ranging from location-based recommendations to urban planning, which aims to forecast users' next location visits based on historical trajectories. Despite the severe long-tailed distribution of locations, the problem of long-tailed mobility prediction remains largely underexplored. Existing long-tailed learning methods primarily focus on rebalancing the skewed distribution at the data, model, or class level, neglecting to exploit the spatiotemporal semantics of locations. To address this gap, we propose the first plug-and-play framework for long-tailed mobility prediction in an exploitation and exploration manner, named \textbf{A}daptive \textbf{LO}cation \textbf{H}ier\textbf{A}rchy learning (ALOHA). First, we construct city-tailored location hierarchy based on Large Language Models (LLMs) by exploiting Maslow's theory of human motivation to design Chain-of-Thought (CoT) prompts that captures spatiotemporal semantics. Second, we optimize the location hierarchy predictions by Gumbel disturbance and node-wise adaptive weights within the hierarchical tree structure. Experiments on state-of-the-art models across six datasets demonstrate the framework's consistent effectiveness and generalizability, which strikes a well balance between head and tail locations. Weight analysis and ablation studies reveal the optimization differences of each component for head and tail locations. Furthermore, in-depth analyses of hierarchical distance and case study demonstrate the effective semantic guidance from the location hierarchy. Our code will be made publicly available. 

**Abstract (ZH)**: 一种用于长尾移动性预测的探索与利用框架：自适应地点层次学习（ALOHA） 

---
# Subtle Risks, Critical Failures: A Framework for Diagnosing Physical Safety of LLMs for Embodied Decision Making 

**Title (ZH)**: 隐秘的风险，关键的故障：一种诊断物理安全框架，用于具身决策的大语言模型 

**Authors**: Yejin Son, Minseo Kim, Sungwoong Kim, Seungju Han, Jian Kim, Dongju Jang, Youngjae Yu, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.19933)  

**Abstract**: Large Language Models (LLMs) are increasingly used for decision making in embodied agents, yet existing safety evaluations often rely on coarse success rates and domain-specific setups, making it difficult to diagnose why and where these models fail. This obscures our understanding of embodied safety and limits the selective deployment of LLMs in high-risk physical environments. We introduce SAFEL, the framework for systematically evaluating the physical safety of LLMs in embodied decision making. SAFEL assesses two key competencies: (1) rejecting unsafe commands via the Command Refusal Test, and (2) generating safe and executable plans via the Plan Safety Test. Critically, the latter is decomposed into functional modules, goal interpretation, transition modeling, action sequencing, enabling fine-grained diagnosis of safety failures. To support this framework, we introduce EMBODYGUARD, a PDDL-grounded benchmark containing 942 LLM-generated scenarios covering both overtly malicious and contextually hazardous instructions. Evaluation across 13 state-of-the-art LLMs reveals that while models often reject clearly unsafe commands, they struggle to anticipate and mitigate subtle, situational risks. Our results highlight critical limitations in current LLMs and provide a foundation for more targeted, modular improvements in safe embodied reasoning. 

**Abstract (ZH)**: SAFEL：评估语言模型在体Calc决策中物理安全性的框架 

---
# EMAC+: Embodied Multimodal Agent for Collaborative Planning with VLM+LLM 

**Title (ZH)**: EMAC+: 融合多模态代理的协作规划系统结合VLM+LLM 

**Authors**: Shuang Ao, Flora D. Salim, Simon Khan  

**Link**: [PDF](https://arxiv.org/pdf/2505.19905)  

**Abstract**: Although LLMs demonstrate proficiency in several text-based reasoning and planning tasks, their implementation in robotics control is constrained by significant deficiencies: (1) LLM agents are designed to work mainly with textual inputs rather than visual conditions; (2) Current multimodal agents treat LLMs as static planners, which separates their reasoning from environment dynamics, resulting in actions that do not take domain-specific knowledge into account; and (3) LLMs are not designed to learn from visual interactions, which makes it harder for them to make better policies for specific domains. In this paper, we introduce EMAC+, an Embodied Multimodal Agent that collaboratively integrates LLM and VLM via a bidirectional training paradigm. Unlike existing methods, EMAC+ dynamically refines high-level textual plans generated by an LLM using real-time feedback from a VLM executing low-level visual control tasks. We address critical limitations of previous models by enabling the LLM to internalize visual environment dynamics directly through interactive experience, rather than relying solely on static symbolic mappings. Extensive experimental evaluations on ALFWorld and RT-1 benchmarks demonstrate that EMAC+ achieves superior task performance, robustness against noisy observations, and efficient learning. We also conduct thorough ablation studies and provide detailed analyses of success and failure cases. 

**Abstract (ZH)**: 尽管大规模语言模型在文本推理和规划任务上表现出色，但将其应用于机器人控制仍然受到显著缺陷的限制：（1）语言模型代理主要设计用于处理文本输入而非视觉条件；（2）当前的多模态代理将语言模型视为静态规划者，这使得它们的推理与环境动力学脱钩，导致采取的行动未能考虑到领域特定知识；（3）语言模型未被设计用于从视觉交互中学习，这使得它们制定针对特定领域的更好策略更为困难。本文介绍了一种名为EMAC+的体态多模态代理，通过双向训练 paradigm将语言模型和视觉语言模型结合起来。与现有方法不同，EMAC+能够利用视觉语言模型执行低级视觉控制任务时的实时反馈动态细化由语言模型生成的高级文本计划。通过使语言模型能够直接通过交互体验内化视觉环境动态，而不是仅仅依赖静态符号映射来解决先前模型的关键限制。在ALFWorld和RT-1基准上的广泛实验评估表明，EMAC+实现了卓越的任务性能、对嘈杂观测的鲁棒性以及高效的learning。我们还进行了详尽的消融研究，并提供了成功与失败案例的详细分析。 

---
# FieldWorkArena: Agentic AI Benchmark for Real Field Work Tasks 

**Title (ZH)**: FieldWorkArena: 自主权AI基准测试 for 实际现场工作任务 

**Authors**: Atsunori Moteki, Shoichi Masui, Fan Yang, Yueqi Song, Yonatan Bisk, Graham Neubig, Ikuo Kusajima, Yasuto Watanabe, Hiroyuki Ishida, Jun Takahashi, Shan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19662)  

**Abstract**: This paper proposes FieldWorkArena, a benchmark for agentic AI targeting real-world field work. With the recent increase in demand for agentic AI, they are required to monitor and report safety and health incidents, as well as manufacturing-related incidents, that may occur in real-world work environments. Existing agentic AI benchmarks have been limited to evaluating web tasks and are insufficient for evaluating agents in real-world work environments, where complexity increases significantly. In this paper, we define a new action space that agentic AI should possess for real world work environment benchmarks and improve the evaluation function from previous methods to assess the performance of agentic AI in diverse real-world tasks. The dataset consists of videos captured on-site and documents actually used in factories and warehouses, and tasks were created based on interviews with on-site workers and managers. Evaluation results confirmed that performance evaluation considering the characteristics of Multimodal LLM (MLLM) such as GPT-4o is feasible. Additionally, the effectiveness and limitations of the proposed new evaluation method were identified. The complete dataset (HuggingFace) and evaluation program (GitHub) can be downloaded from the following website: this https URL. 

**Abstract (ZH)**: 本文提出FieldWorkArena，一个面向现实世界现场工作的代理型AI基准测试。 

---
# Unveiling the Compositional Ability Gap in Vision-Language Reasoning Model 

**Title (ZH)**: 揭示视觉-语言推理模型的组合作能力差距 

**Authors**: Tianle Li, Jihai Zhang, Yongming Rao, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19406)  

**Abstract**: While large language models (LLMs) demonstrate strong reasoning capabilities utilizing reinforcement learning (RL) with verifiable reward, whether large vision-language models (VLMs) can directly inherit such capabilities through similar post-training strategies remains underexplored. In this work, we conduct a systematic compositional probing study to evaluate whether current VLMs trained with RL or other post-training strategies can compose capabilities across modalities or tasks under out-of-distribution conditions. We design a suite of diagnostic tasks that train models on unimodal tasks or isolated reasoning skills, and evaluate them on multimodal, compositional variants requiring skill integration. Through comparisons between supervised fine-tuning (SFT) and RL-trained models, we identify three key findings: (1) RL-trained models consistently outperform SFT on compositional generalization, demonstrating better integration of learned skills; (2) although VLMs achieve strong performance on individual tasks, they struggle to generalize compositionally under cross-modal and cross-task scenario, revealing a significant gap in current training strategies; (3) enforcing models to explicitly describe visual content before reasoning (e.g., caption-before-thinking), along with rewarding progressive vision-to-text grounding, yields notable gains. It highlights two essential ingredients for improving compositionality in VLMs: visual-to-text alignment and accurate visual grounding. Our findings shed light on the current limitations of RL-based reasoning VLM training and provide actionable insights toward building models that reason compositionally across modalities and tasks. 

**Abstract (ZH)**: 大规模视觉-语言模型能否通过类似的后训练策略直接继承基于强化学习的推理能力：一种系统的组件探查研究 

---
# Improving Medical Reasoning with Curriculum-Aware Reinforcement Learning 

**Title (ZH)**: 基于课程意识强化学习的医疗推理改进 

**Authors**: Shaohao Rui, Kaitao Chen, Weijie Ma, Xiaosong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19213)  

**Abstract**: Recent advances in reinforcement learning with verifiable, rule-based rewards have greatly enhanced the reasoning capabilities and out-of-distribution generalization of VLMs/LLMs, obviating the need for manually crafted reasoning chains. Despite these promising developments in the general domain, their translation to medical imaging remains limited. Current medical reinforcement fine-tuning (RFT) methods predominantly focus on close-ended VQA, thereby restricting the model's ability to engage in world knowledge retrieval and flexible task adaptation. More critically, these methods fall short of addressing the critical clinical demand for open-ended, reasoning-intensive decision-making. To bridge this gap, we introduce \textbf{MedCCO}, the first multimodal reinforcement learning framework tailored for medical VQA that unifies close-ended and open-ended data within a curriculum-driven RFT paradigm. Specifically, MedCCO is initially fine-tuned on a diverse set of close-ended medical VQA tasks to establish domain-grounded reasoning capabilities, and is then progressively adapted to open-ended tasks to foster deeper knowledge enhancement and clinical interpretability. We validate MedCCO across eight challenging medical VQA benchmarks, spanning both close-ended and open-ended settings. Experimental results show that MedCCO consistently enhances performance and generalization, achieving a 11.4\% accuracy gain across three in-domain tasks, and a 5.7\% improvement on five out-of-domain benchmarks. These findings highlight the promise of curriculum-guided RL in advancing robust, clinically-relevant reasoning in medical multimodal language models. 

**Abstract (ZH)**: Recent Advances in Multimodal Reinforcement Learning for Medical VQA with Curriculum-Driven Open-ended Reasoning 

---
# CardioCoT: Hierarchical Reasoning for Multimodal Survival Analysis 

**Title (ZH)**: CardioCoT: 分层推理在多模态生存分析中的应用 

**Authors**: Shaohao Rui, Haoyang Su, Jinyi Xiang, Lian-Ming Wu, Xiaosong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19195)  

**Abstract**: Accurate prediction of major adverse cardiovascular events recurrence risk in acute myocardial infarction patients based on postoperative cardiac MRI and associated clinical notes is crucial for precision treatment and personalized intervention. Existing methods primarily focus on risk stratification capability while overlooking the need for intermediate robust reasoning and model interpretability in clinical practice. Moreover, end-to-end risk prediction using LLM/VLM faces significant challenges due to data limitations and modeling complexity. To bridge this gap, we propose CardioCoT, a novel two-stage hierarchical reasoning-enhanced survival analysis framework designed to enhance both model interpretability and predictive performance. In the first stage, we employ an evidence-augmented self-refinement mechanism to guide LLM/VLMs in generating robust hierarchical reasoning trajectories based on associated radiological findings. In the second stage, we integrate the reasoning trajectories with imaging data for risk model training and prediction. CardioCoT demonstrates superior performance in MACE recurrence risk prediction while providing interpretable reasoning processes, offering valuable insights for clinical decision-making. 

**Abstract (ZH)**: 基于术后心脏MRI和相关临床笔记，准确预测急性心肌梗死患者主要不良心血管事件复发风险的精细化治疗和个性化干预至关重要。现有的方法主要集中在风险分层能力上，而忽视了临床实践中中间稳健推理和模型可解释性的需求。此外，端到端的风险预测由于数据限制和建模复杂性面临重大挑战。为解决这一问题，我们提出CardioCoT，这是一种新的两阶段层次化推理增强生存分析框架，旨在提高模型的可解释性和预测性能。在第一阶段，我们采用证据增强的自我精炼机制，引导LLM/VLM基于相关的影像学发现生成稳健的层次化推理轨迹。在第二阶段，我们将推理轨迹与影像数据结合，用于风险模型的训练和预测。CardioCoT在MACE复发风险预测中表现出优越性能，并提供可解释的推理过程，为临床决策提供了有价值的见解。 

---
# ScreenExplorer: Training a Vision-Language Model for Diverse Exploration in Open GUI World 

**Title (ZH)**: ScreenExplorer：训练一种适合开放GUI世界多样探索的视觉-语言模型 

**Authors**: Runliang Niu, Jinglong Ji, Yi Chang, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19095)  

**Abstract**: The rapid progress of large language models (LLMs) has sparked growing interest in building Artificial General Intelligence (AGI) within Graphical User Interface (GUI) environments. However, existing GUI agents based on LLMs or vision-language models (VLMs) often fail to generalize to novel environments and rely heavily on manually curated, diverse datasets. To overcome these limitations, we introduce ScreenExplorer, a VLM trained via Group Relative Policy Optimization(GRPO) in real, dynamic, and open-ended GUI environments. Innovatively, we introduced a world-model-based curiosity reward function to help the agent overcome the cold-start phase of exploration. Additionally, distilling experience streams further enhances the model's exploration capabilities. Our training framework enhances model exploration in open GUI environments, with trained models showing better environmental adaptation and sustained exploration compared to static deployment models. Our findings offer a scalable pathway toward AGI systems with self-improving capabilities in complex interactive settings. 

**Abstract (ZH)**: 大型语言模型的快速进步激发了在图形用户界面环境中构建人工通用智能的兴趣。然而，现有的基于大型语言模型或视觉-语言模型的图形用户界面代理往往难以泛化到新的环境，并且高度依赖于手工策划的多样数据集。为克服这些限制，我们介绍了ScreenExplorer，这是一种通过组相对策略优化（GRPO）在真实、动态和开放环境中训练的视觉语言模型。创新性地，我们引入了一种基于世界模型的好奇心奖励函数，以帮助代理克服探索的冷启动阶段。此外，经验流的提炼进一步增强了模型的探索能力。我们的训练框架在开放的图形用户界面环境中增强了模型的探索，训练后的模型相较于静态部署模型表现出更好的环境适应性和持续探索能力。我们的发现为复杂交互设置中具有自我改进能力的人工通用人工智能系统提供了可扩展的路径。 

---
# SANNet: A Semantic-Aware Agentic AI Networking Framework for Multi-Agent Cross-Layer Coordination 

**Title (ZH)**: SANNet：一种面向语义感知的代理AI网络框架，用于多代理跨层协调 

**Authors**: Yong Xiao, Haoran Zhou, Xubo Li, Yayu Gao, Guangming Shi, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18946)  

**Abstract**: Agentic AI networking (AgentNet) is a novel AI-native networking paradigm that relies on a large number of specialized AI agents to collaborate and coordinate for autonomous decision-making, dynamic environmental adaptation, and complex goal achievement. It has the potential to facilitate real-time network management alongside capabilities for self-configuration, self-optimization, and self-adaptation across diverse and complex networking environments, laying the foundation for fully autonomous networking systems in the future. Despite its promise, AgentNet is still in the early stage of development, and there still lacks an effective networking framework to support automatic goal discovery and multi-agent self-orchestration and task assignment. This paper proposes SANNet, a novel semantic-aware agentic AI networking architecture that can infer the semantic goal of the user and automatically assign agents associated with different layers of a mobile system to fulfill the inferred goal. Motivated by the fact that one of the major challenges in AgentNet is that different agents may have different and even conflicting objectives when collaborating for certain goals, we introduce a dynamic weighting-based conflict-resolving mechanism to address this issue. We prove that SANNet can provide theoretical guarantee in both conflict-resolving and model generalization performance for multi-agent collaboration in dynamic environment. We develop a hardware prototype of SANNet based on the open RAN and 5GS core platform. Our experimental results show that SANNet can significantly improve the performance of multi-agent networking systems, even when agents with conflicting objectives are selected to collaborate for the same goal. 

**Abstract (ZH)**: 基于代理意识的AI网络架构（SANNet）：一种新型的自主AI网络范式 

---
# LiteCUA: Computer as MCP Server for Computer-Use Agent on AIOS 

**Title (ZH)**: LiteCUA: 计算机作为MCP服务器的计算机使用代理在AIOS中 

**Authors**: Kai Mei, Xi Zhu, Hang Gao, Shuhang Lin, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18829)  

**Abstract**: We present AIOS 1.0, a novel platform designed to advance computer-use agent (CUA) capabilities through environmental contextualization. While existing approaches primarily focus on building more powerful agent frameworks or enhancing agent models, we identify a fundamental limitation: the semantic disconnect between how language models understand the world and how computer interfaces are structured. AIOS 1.0 addresses this challenge by transforming computers into contextual environments that language models can natively comprehend, implementing a Model Context Protocol (MCP) server architecture to abstract computer states and actions. This approach effectively decouples interface complexity from decision complexity, enabling agents to reason more effectively about computing environments. To demonstrate our platform's effectiveness, we introduce LiteCUA, a lightweight computer-use agent built on AIOS 1.0 that achieves a 14.66% success rate on the OSWorld benchmark, outperforming several specialized agent frameworks despite its simple architecture. Our results suggest that contextualizing computer environments for language models represents a promising direction for developing more capable computer-use agents and advancing toward AI that can interact with digital systems. The source code of LiteCUA is available at this https URL, and it is also integrated into the AIOS main branch as part of AIOS at this https URL. 

**Abstract (ZH)**: 我们呈现AIOS 1.0，这是一个新型平台，旨在通过环境上下文化来提升计算机使用代理(CUA)的能力。尽管现有方法主要侧重于构建更强大的代理框架或增强代理模型，我们识别出一个根本性的局限性：语言模型对世界理解与计算机界面结构之间的语义脱节。AIOS 1.0通过将计算机转化为语言模型能够原生理解的上下文环境来应对这一挑战，采用模型上下文协议(MCP)服务器架构来抽象计算机状态和操作。这种方法有效地将接口复杂性与决策复杂性脱钩，使代理能够更有效地对计算环境进行推理。为了展示该平台的有效性，我们介绍了基于AIOS 1.0构建的轻量级计算机使用代理LiteCUA，在OSWorld基准测试中取得14.66%的成功率，其简单架构却超越了多个专业代理框架。我们的研究结果表明，为语言模型上下文化计算机环境是开发更强大计算机使用代理并朝着能够与数字系统交互的AI发展的一个有前景的方向。LiteCUA的源代码可通过以下网址访问：[此 https URL]，并已集成到AIOS主分支中：[此 https URL]。 

---
# EdgeAgentX: A Novel Framework for Agentic AI at the Edge in Military Communication Networks 

**Title (ZH)**: EdgeAgentX：军事通信网络中边缘代理型AI的新型框架 

**Authors**: Abir Ray  

**Link**: [PDF](https://arxiv.org/pdf/2505.18457)  

**Abstract**: This paper introduces EdgeAgentX, a novel framework integrating federated learning (FL), multi-agent reinforcement learning (MARL), and adversarial defense mechanisms, tailored for military communication networks. EdgeAgentX significantly improves autonomous decision-making, reduces latency, enhances throughput, and robustly withstands adversarial disruptions, as evidenced by comprehensive simulations. 

**Abstract (ZH)**: EdgeAgentX：一种集成联邦学习、多智能体强化学习和对抗防御机制的新型军事通信网络框架 

---
# Multimodal LLM-Guided Semantic Correction in Text-to-Image Diffusion 

**Title (ZH)**: 多模态LLM引导的文本到图像扩散语义修正 

**Authors**: Zheqi Lv, Junhao Chen, Qi Tian, Keting Yin, Shengyu Zhang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20053)  

**Abstract**: Diffusion models have become the mainstream architecture for text-to-image generation, achieving remarkable progress in visual quality and prompt controllability. However, current inference pipelines generally lack interpretable semantic supervision and correction mechanisms throughout the denoising process. Most existing approaches rely solely on post-hoc scoring of the final image, prompt filtering, or heuristic resampling strategies-making them ineffective in providing actionable guidance for correcting the generative trajectory. As a result, models often suffer from object confusion, spatial errors, inaccurate counts, and missing semantic elements, severely compromising prompt-image alignment and image quality. To tackle these challenges, we propose MLLM Semantic-Corrected Ping-Pong-Ahead Diffusion (PPAD), a novel framework that, for the first time, introduces a Multimodal Large Language Model (MLLM) as a semantic observer during inference. PPAD performs real-time analysis on intermediate generations, identifies latent semantic inconsistencies, and translates feedback into controllable signals that actively guide the remaining denoising steps. The framework supports both inference-only and training-enhanced settings, and performs semantic correction at only extremely few diffusion steps, offering strong generality and scalability. Extensive experiments demonstrate PPAD's significant improvements. 

**Abstract (ZH)**: Multimodal Large Language Model Semantic-Corrected Ping-Pong-Ahead Diffusion (PPAD) 

---
# Deep Active Inference Agents for Delayed and Long-Horizon Environments 

**Title (ZH)**: 深层主动推断代理用于延迟和长时域环境 

**Authors**: Yavar Taheri Yeganeh, Mohsen Jafari, Andrea Matta  

**Link**: [PDF](https://arxiv.org/pdf/2505.19867)  

**Abstract**: With the recent success of world-model agents, which extend the core idea of model-based reinforcement learning by learning a differentiable model for sample-efficient control across diverse tasks, active inference (AIF) offers a complementary, neuroscience-grounded paradigm that unifies perception, learning, and action within a single probabilistic framework powered by a generative model. Despite this promise, practical AIF agents still rely on accurate immediate predictions and exhaustive planning, a limitation that is exacerbated in delayed environments requiring plans over long horizons, tens to hundreds of steps. Moreover, most existing agents are evaluated on robotic or vision benchmarks which, while natural for biological agents, fall short of real-world industrial complexity. We address these limitations with a generative-policy architecture featuring (i) a multi-step latent transition that lets the generative model predict an entire horizon in a single look-ahead, (ii) an integrated policy network that enables the transition and receives gradients of the expected free energy, (iii) an alternating optimization scheme that updates model and policy from a replay buffer, and (iv) a single gradient step that plans over long horizons, eliminating exhaustive planning from the control loop. We evaluate our agent in an environment that mimics a realistic industrial scenario with delayed and long-horizon settings. The empirical results confirm the effectiveness of the proposed approach, demonstrating the coupled world-model with the AIF formalism yields an end-to-end probabilistic controller capable of effective decision making in delayed, long-horizon settings without handcrafted rewards or expensive planning. 

**Abstract (ZH)**: 基于生成策略的主动推断代理：多步潜在过渡的实时预测与长期规划统一框架 

---
# Equivariant Representation Learning for Symmetry-Aware Inference with Guarantees 

**Title (ZH)**: 对称意识保证下等变表示学习 

**Authors**: Daniel Ordoñez-Apraez, Alek Fröhlich, Vladimir Kostić, Karim Lounici, Vivien Brandt, Massimiliano Pontil  

**Link**: [PDF](https://arxiv.org/pdf/2505.19809)  

**Abstract**: In many real-world applications of regression, conditional probability estimation, and uncertainty quantification, exploiting symmetries rooted in physics or geometry can dramatically improve generalization and sample efficiency. While geometric deep learning has made significant empirical advances by incorporating group-theoretic structure, less attention has been given to statistical learning guarantees. In this paper, we introduce an equivariant representation learning framework that simultaneously addresses regression, conditional probability estimation, and uncertainty quantification while providing first-of-its-kind non-asymptotic statistical learning guarantees. Grounded in operator and group representation theory, our framework approximates the spectral decomposition of the conditional expectation operator, building representations that are both equivariant and disentangled along independent symmetry subgroups. Empirical evaluations on synthetic datasets and real-world robotics applications confirm the potential of our approach, matching or outperforming existing equivariant baselines in regression while additionally providing well-calibrated parametric uncertainty estimates. 

**Abstract (ZH)**: 在回归、条件概率估计和不确定性量化等许多实际应用中，利用物理学或几何学根植的对称性可以显著提高泛化能力和样本效率。虽然几何深度学习通过纳入群论结构取得了显著的经验进步，但很少关注统计学习保证。在本文中，我们提出了一种同时解决回归、条件概率估计和不确定性量化问题的协变表示学习框架，并提供了首个非渐近统计学习保证。基于算子表示论和群表示论，我们的框架逼近条件期望算子的谱分解，构建既协变又沿独立对称子群分离的表示。在合成数据集和实际机器人应用中的实验评估证实了我们方法的潜力，在回归任务上匹配甚至超越现有的协变基准方法，并额外提供校准良好的参数不确定性估计。 

---
# MedDreamer: Model-Based Reinforcement Learning with Latent Imagination on Complex EHRs for Clinical Decision Support 

**Title (ZH)**: MedDreamer: 基于模型的强化学习与潜在想象在复杂电子健康记录中的临床决策支持 

**Authors**: Qianyi Xu, Gousia Habib, Dilruk Perera, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19785)  

**Abstract**: Timely and personalized treatment decisions are essential across a wide range of healthcare settings where patient responses vary significantly and evolve over time. Clinical data used to support these decisions are often irregularly sampled, sparse, and noisy. Existing decision support systems commonly rely on discretization and imputation, which can distort critical temporal dynamics and degrade decision quality. Moreover, they often overlook the clinical significance of irregular recording frequencies, filtering out patterns in how and when data is collected. Reinforcement Learning (RL) is a natural fit for clinical decision-making, enabling sequential, long-term optimization in dynamic, uncertain environments. However, most existing treatment recommendation systems are model-free and trained solely on offline data, making them sample-inefficient, sensitive to data quality, and poorly generalizable across tasks or cohorts. To address these limitations, we propose MedDreamer, a two-phase model-based RL framework for personalized treatment recommendation. MedDreamer uses a world model with an Adaptive Feature Integration (AFI) module to effectively model irregular, sparse clinical data. Through latent imagination, it simulates plausible patient trajectories to enhance learning, refining its policy using a mix of real and imagined experiences. This enables learning policies that go beyond suboptimal historical decisions while remaining grounded in clinical data. To our knowledge, this is the first application of latent imagination to irregular healthcare data. Evaluations on sepsis and mechanical ventilation (MV) treatment using two large-scale EHR datasets show that MedDreamer outperforms both model-free and model-based baselines in clinical outcomes and off-policy metrics. 

**Abstract (ZH)**: 及时且个性化的治疗决策在临床响应变化显著且随时间演变的广泛医疗保健场景中至关重要。用于支持这些决策的临床数据往往稀疏、不规则采样且含噪声。现有的决策支持系统通常依赖离散化和插补，这可能会扭曲关键的时间动态并降低决策质量。此外，它们往往忽视了不规则记录频率的临床意义，过滤掉了数据收集模式。强化学习（RL）是临床决策的理想选择，能够在一个动态且不确定的环境中实现长期的序列优化。然而，现有的大多数治疗推荐系统都是无模型的，并且仅在离线数据上进行训练，这使得它们样本效率低下、对数据质量敏感，并且难以泛化到不同的任务或队列。为了解决这些限制，我们提出了一种基于模型的两阶段RL框架MedDreamer，用于个性化的治疗推荐。MedDreamer 使用一个具有自适应特征整合（AFI）模块的世界模型来有效建模不规则的稀疏临床数据。通过潜在的想象，它模拟可能的患者轨迹以增强学习，并利用真实和想象的经验混合优化其策略。这使得学习的策略能够超越历史上的次优决策，同时仍与临床数据保持关联。据我们所知，这是首次将潜在想象应用于不规则的医疗保健数据。使用两个大规模EHR数据集对感染性休克和机械通气（MV）治疗进行评估表明，MedDreamer 在临床结果和离线策略指标上均优于无模型和基于模型的基线。 

---
# Navigating loss manifolds via rigid body dynamics: A promising avenue for robustness and generalisation 

**Title (ZH)**: 通过刚体动力学导航损失流形：稳健性和泛化能力的一种有前途的方法 

**Authors**: Mohammed D. Belgoumri, Mohamed Reda Bouadjenek, Hakim Hacid, Imran Razzak, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2505.19527)  

**Abstract**: Training large neural networks through gradient-based optimization requires navigating high-dimensional loss landscapes, which often exhibit pathological geometry, leading to undesirable training dynamics. In particular, poor generalization frequently results from convergence to sharp minima that are highly sensitive to input perturbations, causing the model to overfit the training data while failing to generalize to unseen examples. Furthermore, these optimization procedures typically display strong dependence on the fine structure of the loss landscape, leading to unstable training dynamics, due to the fractal-like nature of the loss surface. In this work, we propose an alternative optimizer that simultaneously reduces this dependence, and avoids sharp minima, thereby improving generalization. This is achieved by simulating the motion of the center of a ball rolling on the loss landscape. The degree to which our optimizer departs from the standard gradient descent is controlled by a hyperparameter, representing the radius of the ball. Changing this hyperparameter allows for probing the loss landscape at different scales, making it a valuable tool for understanding its geometry. 

**Abstract (ZH)**: 通过基于梯度的优化训练大型神经网络需要在高维损失景观中导航，这些景观常常表现出病态的几何结构，导致不理想的训练动态。特别是，模型通常由于收敛到对输入扰动高度敏感的尖锐极小值而表现泛化能力差，导致模型过拟合训练数据而无法泛化到未见样本。此外，这些优化过程通常强烈依赖于损失景观的精细结构，导致训练动态不稳定，反映出损失面的分形性质。在本工作中，我们提出了一种替代优化器，它可以同时减少这种依赖性并避免尖锐极小值，从而改善泛化能力。这种效果是通过模拟球心在损失景观上的运动实现的。优化器与标准梯度下降的不同程度由一个超参数控制，该超参数代表球的半径。改变这个超参数可以在不同尺度上探测损失景观，使其成为理解其几何结构的宝贵工具。 

---
# Surrogate-Assisted Evolutionary Reinforcement Learning Based on Autoencoder and Hyperbolic Neural Network 

**Title (ZH)**: 基于自动编码器和双曲神经网络的代理辅助进化强化学习 

**Authors**: Bingdong Li, Mei Jiang, Hong Qian, Peng Yang, Wenjing Hong, Hong Qian, Ke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19423)  

**Abstract**: Evolutionary Reinforcement Learning (ERL), training the Reinforcement Learning (RL) policies with Evolutionary Algorithms (EAs), have demonstrated enhanced exploration capabilities and greater robustness than using traditional policy gradient. However, ERL suffers from the high computational costs and low search efficiency, as EAs require evaluating numerous candidate policies with expensive simulations, many of which are ineffective and do not contribute meaningfully to the training. One intuitive way to reduce the ineffective evaluations is to adopt the surrogates. Unfortunately, existing ERL policies are often modeled as deep neural networks (DNNs) and thus naturally represented as high-dimensional vectors containing millions of weights, which makes the building of effective surrogates for ERL policies extremely challenging. This paper proposes a novel surrogate-assisted ERL that integrates Autoencoders (AE) and Hyperbolic Neural Networks (HNN). Specifically, AE compresses high-dimensional policies into low-dimensional representations while extracting key features as the inputs for the surrogate. HNN, functioning as a classification-based surrogate model, can learn complex nonlinear relationships from sampled data and enable more accurate pre-selection of the sampled policies without real evaluations. The experiments on 10 Atari and 4 Mujoco games have verified that the proposed method outperforms previous approaches significantly. The search trajectories guided by AE and HNN are also visually demonstrated to be more effective, in terms of both exploration and convergence. This paper not only presents the first learnable policy embedding and surrogate-modeling modules for high-dimensional ERL policies, but also empirically reveals when and why they can be successful. 

**Abstract (ZH)**: 基于自编码器和双曲神经网络的代理辅助进化强化学习 

---
# Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals 

**Title (ZH)**: 基于力量提示的视频生成模型能够学习和泛化物理控制信号 

**Authors**: Nate Gillman, Charles Herrmann, Michael Freeman, Daksh Aggarwal, Evan Luo, Deqing Sun, Chen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.19386)  

**Abstract**: Recent advances in video generation models have sparked interest in world models capable of simulating realistic environments. While navigation has been well-explored, physically meaningful interactions that mimic real-world forces remain largely understudied. In this work, we investigate using physical forces as a control signal for video generation and propose force prompts which enable users to interact with images through both localized point forces, such as poking a plant, and global wind force fields, such as wind blowing on fabric. We demonstrate that these force prompts can enable videos to respond realistically to physical control signals by leveraging the visual and motion prior in the original pretrained model, without using any 3D asset or physics simulator at inference. The primary challenge of force prompting is the difficulty in obtaining high quality paired force-video training data, both in the real world due to the difficulty of obtaining force signals, and in synthetic data due to limitations in the visual quality and domain diversity of physics simulators. Our key finding is that video generation models can generalize remarkably well when adapted to follow physical force conditioning from videos synthesized by Blender, even with limited demonstrations of few objects. Our method can generate videos which simulate forces across diverse geometries, settings, and materials. We also try to understand the source of this generalization and perform ablations that reveal two key elements: visual diversity and the use of specific text keywords during training. Our approach is trained on only around 15k training examples for a single day on four A100 GPUs, and outperforms existing methods on force adherence and physics realism, bringing world models closer to real-world physics interactions. We release all datasets, code, weights, and interactive video demos at our project page. 

**Abstract (ZH)**: Recent Advances in Video Generation Models Have Sparked Interest in World Models Capable of Simulating Realistic Environments: Using Physical Forces as Control Signals 

---
# Prompting Decision Transformers for Zero-Shot Reach-Avoid Policies 

**Title (ZH)**: 零-shot 达避策略的提示决策变换器 

**Authors**: Kevin Li, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2505.19337)  

**Abstract**: Offline goal-conditioned reinforcement learning methods have shown promise for reach-avoid tasks, where an agent must reach a target state while avoiding undesirable regions of the state space. Existing approaches typically encode avoid-region information into an augmented state space and cost function, which prevents flexible, dynamic specification of novel avoid-region information at evaluation time. They also rely heavily on well-designed reward and cost functions, limiting scalability to complex or poorly structured environments. We introduce RADT, a decision transformer model for offline, reward-free, goal-conditioned, avoid region-conditioned RL. RADT encodes goals and avoid regions directly as prompt tokens, allowing any number of avoid regions of arbitrary size to be specified at evaluation time. Using only suboptimal offline trajectories from a random policy, RADT learns reach-avoid behavior through a novel combination of goal and avoid-region hindsight relabeling. We benchmark RADT against 3 existing offline goal-conditioned RL models across 11 tasks, environments, and experimental settings. RADT generalizes in a zero-shot manner to out-of-distribution avoid region sizes and counts, outperforming baselines that require retraining. In one such zero-shot setting, RADT achieves 35.7% improvement in normalized cost over the best retrained baseline while maintaining high goal-reaching success. We apply RADT to cell reprogramming in biology, where it reduces visits to undesirable intermediate gene expression states during trajectories to desired target states, despite stochastic transitions and discrete, structured state dynamics. 

**Abstract (ZH)**: Offline 奖励无馈信息、目标条件、避险区域条件的强化学习方法：RADT模型研究 

---
# Towards Large Reasoning Models for Agriculture 

**Title (ZH)**: 面向农业的大规模推理模型研究 

**Authors**: Hossein Zaremehrjerdi, Shreyan Ganguly, Ashlyn Rairdin, Elizabeth Tranel, Benjamin Feuer, Juan Ignacio Di Salvo, Srikanth Panthulugiri, Victoria Moser, Sarah Jones, Joscif G Raigne, Yanben Shen, Heidi M. Dornath, Aditya Balu, Adarsh Krishnamurthy, Asheesh K Singh, Arti Singh, Baskar Ganapathysubramanian, Chinmay Hegde, Soumik Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.19259)  

**Abstract**: Agricultural decision-making involves complex, context-specific reasoning, where choices about crops, practices, and interventions depend heavily on geographic, climatic, and economic conditions. Traditional large language models (LLMs) often fall short in navigating this nuanced problem due to limited reasoning capacity. We hypothesize that recent advances in large reasoning models (LRMs) can better handle such structured, domain-specific inference. To investigate this, we introduce AgReason, the first expert-curated open-ended science benchmark with 100 questions for agricultural reasoning. Evaluations across thirteen open-source and proprietary models reveal that LRMs outperform conventional ones, though notable challenges persist, with the strongest Gemini-based baseline achieving 36% accuracy. We also present AgThoughts, a large-scale dataset of 44.6K question-answer pairs generated with human oversight and equipped with synthetically generated reasoning traces. Using AgThoughts, we develop AgThinker, a suite of small reasoning models that can be run on consumer-grade GPUs, and show that our dataset can be effective in unlocking agricultural reasoning abilities in LLMs. Our project page is here: this https URL 

**Abstract (ZH)**: 农业决策涉及复杂的、情境特定的推理，其中关于作物、实践和干预的选择高度依赖于地理、气候和经济条件。传统的大型语言模型（LLMs）往往由于推理能力有限而在处理这一细腻的问题上表现出色。我们假设最近在大型推理模型（LRMs）方面的进展能够更好地处理这种结构化、领域特定的推理。为了调查这一点，我们引入了AgReason，这是第一个由专家编纂的开放性科学基准，包含100个农业推理问题。在十三个开源和专有模型的评估中，LRMs的表现优于传统的模型，尽管存在显著的挑战，其中基于Gemini的最强基线达到了36%的准确性。我们还介绍了AgThoughts，这是一个包含44,600个带有人工监督生成的推理痕迹的问题-答案对的大规模数据集。使用AgThoughts，我们开发了AgThinker，这是一个可以在消费者级GPU上运行的小型推理模型套件，并展示了我们的数据集在释放LLMs的农业推理能力方面的有效性。项目页面在此：this https URL 

---
# VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use 

**Title (ZH)**: VTool-R1: 多模态工具使用via 强化学习的VLMs图像思维学习 

**Authors**: Mingyuan Wu, Jingcheng Yang, Jize Jiang, Meitang Li, Kaizhuo Yan, Hanchao Yu, Minjia Zhang, Chengxiang Zhai, Klara Nahrstedt  

**Link**: [PDF](https://arxiv.org/pdf/2505.19255)  

**Abstract**: Reinforcement Learning Finetuning (RFT) has significantly advanced the reasoning capabilities of large language models (LLMs) by enabling long chains of thought, self-correction, and effective tool use. While recent works attempt to extend RFT to vision-language models (VLMs), these efforts largely produce text-only reasoning conditioned on static image inputs, falling short of true multimodal reasoning in the response. In contrast, test-time methods like Visual Sketchpad incorporate visual steps but lack training mechanisms.
We introduce VTool-R1, the first framework that trains VLMs to generate multimodal chains of thought by interleaving text and intermediate visual reasoning steps. VTool-R1 integrates Python-based visual editing tools into the RFT process, enabling VLMs to learn when and how to generate visual reasoning steps that benefit final reasoning. Trained with outcome-based rewards tied to task accuracy, our approach elicits strategic visual tool use for reasoning without relying on process-based supervision. Experiments on structured visual question answering over charts and tables show that VTool-R1 enhances reasoning performance by teaching VLMs to "think with images" and generate multimodal chain of thoughts with tools. 

**Abstract (ZH)**: 基于强化学习微调的多模态链式推理框架VTool-R1 

---
# Agent-Based Decentralized Energy Management of EV Charging Station with Solar Photovoltaics via Multi-Agent Reinforcement Learning 

**Title (ZH)**: 基于代理的电动汽车充电站太阳能光伏 decentralized 能源管理方法：多代理强化学习_approach 

**Authors**: Jiarong Fan, Chenghao Huang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18750)  

**Abstract**: In the pursuit of energy net zero within smart cities, transportation electrification plays a pivotal role. The adoption of Electric Vehicles (EVs) keeps increasing, making energy management of EV charging stations critically important. While previous studies have managed to reduce energy cost of EV charging while maintaining grid stability, they often overlook the robustness of EV charging management against uncertainties of various forms, such as varying charging behaviors and possible faults in faults in some chargers. To address the gap, a novel Multi-Agent Reinforcement Learning (MARL) approach is proposed treating each charger to be an agent and coordinate all the agents in the EV charging station with solar photovoltaics in a more realistic scenario, where system faults may occur. A Long Short-Term Memory (LSTM) network is incorporated in the MARL algorithm to extract temporal features from time-series. Additionally, a dense reward mechanism is designed for training the agents in the MARL algorithm to improve EV charging experience. Through validation on a real-world dataset, we show that our approach is robust against system uncertainties and faults and also effective in minimizing EV charging costs and maximizing charging service satisfaction. 

**Abstract (ZH)**: 在智能城市中追求能源净零的进程中，交通运输 electrification 起到关键作用。电动汽车（EVs）的 adoption 持续增加，使得 EV 充电站的能源管理变得至关重要。尽管以往的研究能够在保持电网稳定的同时降低 EV 充电成本，但它们往往忽视了各种不确定性（如不同的充电行为和某些充电桩可能出现的故障）对 EV 充电管理健壮性的影响。为填补这一空白，本文提出了一种新颖的多智能体强化学习（MARL）方法，将每个充电桩视为一个智能体，并在可能发生系统故障的更现实场景中，通过结合太阳能光伏与所有智能体进行协调。此外，本文在 MARL 算法中引入了长短期记忆（LSTM）网络以提取时间序列中的时序特征，并设计了密集奖励机制以提高智能体的训练效果，从而改善 EV 充电体验。通过在真实数据集上的验证，我们证明了该方法能够有效应对系统不确定性与故障，同时最大限度地减少 EV 充电成本并增加充电服务满意度。 

---
# GRE Suite: Geo-localization Inference via Fine-Tuned Vision-Language Models and Enhanced Reasoning Chains 

**Title (ZH)**: GE地点套件：通过微调的视觉-语言模型和增强的推理链进行地理定位推断 

**Authors**: Chun Wang, Xiaoran Pan, Zihao Pan, Haofan Wang, Yiren Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.18700)  

**Abstract**: Recent advances in Visual Language Models (VLMs) have demonstrated exceptional performance in visual reasoning tasks. However, geo-localization presents unique challenges, requiring the extraction of multigranular visual cues from images and their integration with external world knowledge for systematic reasoning. Current approaches to geo-localization tasks often lack robust reasoning mechanisms and explainability, limiting their effectiveness. To address these limitations, we propose the Geo Reason Enhancement (GRE) Suite, a novel framework that augments VLMs with structured reasoning chains for accurate and interpretable location inference. The GRE Suite is systematically developed across three key dimensions: dataset, model, and benchmark. First, we introduce GRE30K, a high-quality geo-localization reasoning dataset designed to facilitate fine-grained visual and contextual analysis. Next, we present the GRE model, which employs a multi-stage reasoning strategy to progressively infer scene attributes, local details, and semantic features, thereby narrowing down potential geographic regions with enhanced precision. Finally, we construct the Geo Reason Evaluation Benchmark (GREval-Bench), a comprehensive evaluation framework that assesses VLMs across diverse urban, natural, and landmark scenes to measure both coarse-grained (e.g., country, continent) and fine-grained (e.g., city, street) localization performance. Experimental results demonstrate that GRE significantly outperforms existing methods across all granularities of geo-localization tasks, underscoring the efficacy of reasoning-augmented VLMs in complex geographic inference. Code and data will be released at this https URL. 

**Abstract (ZH)**: Recent Advances in Geo-Reason Enhancement for Visual Language Models 

---
# MRGAgents: A Multi-Agent Framework for Improved Medical Report Generation with Med-LVLMs 

**Title (ZH)**: MRGAgents: 一种基于医学生物语言模型的多 Agent 框架以改进医疗报告生成 

**Authors**: Pengyu Wang, Shuchang Ye, Usman Naseem, Jinman Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.18530)  

**Abstract**: Medical Large Vision-Language Models (Med-LVLMs) have been widely adopted for medical report generation. Despite Med-LVLMs producing state-of-the-art performance, they exhibit a bias toward predicting all findings as normal, leading to reports that overlook critical abnormalities. Furthermore, these models often fail to provide comprehensive descriptions of radiologically relevant regions necessary for accurate diagnosis. To address these challenges, we proposeMedical Report Generation Agents (MRGAgents), a novel multi-agent framework that fine-tunes specialized agents for different disease categories. By curating subsets of the IU X-ray and MIMIC-CXR datasets to train disease-specific agents, MRGAgents generates reports that more effectively balance normal and abnormal findings while ensuring a comprehensive description of clinically relevant regions. Our experiments demonstrate that MRGAgents outperformed the state-of-the-art, improving both report comprehensiveness and diagnostic utility. 

**Abstract (ZH)**: 医学大型视-语言模型（Med-LVLMs）在医学报告生成中得到了广泛应用。尽管Med-LVLMs能产生最先进的性能，但它们倾向于预测所有发现为正常，导致报告忽略了重要的异常。此外，这些模型往往不能提供准确诊断所需的相关放射学区域的全面描述。为了应对这些挑战，我们提出了一种新型多智能体框架——医学报告生成代理（MRGAgents），该框架针对不同疾病类别细调专门的智能体。通过定制IU X射线和MIMIC-CXR数据集中与特定疾病相关的子集进行训练，MRGAgents生成的报告更有效地平衡正常和异常发现，同时确保对临床相关区域的全面描述。我们的实验表明，MRGAgents在报告全面性和诊断效用方面均优于最先进的模型。 

---
# The Cell Must Go On: Agar.io for Continual Reinforcement Learning 

**Title (ZH)**: 细胞必须继续：Agar.io在连续强化学习中的应用 

**Authors**: Mohamed A. Mohamed, Kateryna Nekhomiazh, Vedant Vyas, Marcos M. Jose, Andrew Patterson, Marlos C. Machado  

**Link**: [PDF](https://arxiv.org/pdf/2505.18347)  

**Abstract**: Continual reinforcement learning (RL) concerns agents that are expected to learn continually, rather than converge to a policy that is then fixed for evaluation. Such an approach is well suited to environments the agent perceives as changing, which renders any static policy ineffective over time. The few simulators explicitly designed for empirical research in continual RL are often limited in scope or complexity, and it is now common for researchers to modify episodic RL environments by artificially incorporating abrupt task changes during interaction. In this paper, we introduce AgarCL, a research platform for continual RL that allows for a progression of increasingly sophisticated behaviour. AgarCL is based on the game this http URL, a non-episodic, high-dimensional problem featuring stochastic, ever-evolving dynamics, continuous actions, and partial observability. Additionally, we provide benchmark results reporting the performance of DQN, PPO, and SAC in both the primary, challenging continual RL problem, and across a suite of smaller tasks within AgarCL, each of which isolates aspects of the full environment and allow us to characterize the challenges posed by different aspects of the game. 

**Abstract (ZH)**: 持续强化学习（RL）关注能够持续学习而非固定政策进行评估的代理。这样的方法适用于代理感知环境变化的场景，使得任何静态策略随着时间的推移变得无效。为实证研究持续RL而专门设计的少数几个模拟器往往在范围或复杂性上有所限制，现在研究人员通常通过在交互过程中人工引入突然的任务变化来修改 episodic RL 环境。在本文中，我们介绍了 AgarCL，一个用于持续RL的研究平台，支持逐步实现日益复杂的执行行为。AgarCL 基于这个网址提供的游戏，这是一个非 episodic、高维度问题，具有随机的、不断演变的动力学、连续动作以及部分可观测性。此外，我们还提供了基准结果，报告了 DQN、PPO 和 SAC 在 AgarCL 的主要挑战性持续RL问题以及其内部一系列较小任务中的性能表现，每个任务都隔离了环境的不同方面，使我们能够对游戏不同方面的挑战进行表征。 

---
# LA-RCS: LLM-Agent-Based Robot Control System 

**Title (ZH)**: LLM-Agent-Based机器人控制系统 

**Authors**: TaekHyun Park, YoungJun Choi, SeungHoon Shin, Kwangil Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.18214)  

**Abstract**: LA-RCS (LLM-agent-based robot control system) is a sophisticated robot control system designed to autonomously plan, work, and analyze the external environment based on user requirements by utilizing LLM-Agent. Utilizing a dual-agent framework, LA-RCS generates plans based on user requests, observes the external environment, executes the plans, and modifies the plans as needed to adapt to changes in the external conditions. Additionally, LA-RCS interprets natural language commands by the user and converts them into commands compatible with the robot interface so that the robot can execute tasks and meet user requests properly. During his process, the system autonomously evaluates observation results, provides feedback on the tasks, and executes commands based on real-time environmental monitoring, significantly reducing the need for user intervention in fulfilling requests. We categorized the scenarios that LA-RCS needs to perform into four distinct types and conducted a quantitative assessment of its performance in each scenario. The results showed an average success rate of 90 percent, demonstrating the system capability to fulfill user requests satisfactorily. For more extensive results, readers can visit our project page: this https URL 

**Abstract (ZH)**: 基于LLM-Agent的LA-RCS自主机器人控制系统 

---
